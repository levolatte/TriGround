"""CPU boundary checks for T's evidence view, offline labels and loss span."""

from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.visual_agent.export_trajectories import (
    decision_rows, export, revised_final_label, trajectory_composition, verify_frozen_splits,
)
from experiments.visual_agent.model import prepare_image_messages
from experiments.visual_agent.profiles import PROFILES
from experiments.visual_agent.run import build_initial_messages
from experiments.visual_agent.state import build_state_messages
from experiments.visual_agent.train import encode_decision, supervised_token_range


def candidate(candidate_id, box, role="target"):
    return {"id": candidate_id, "bbox": box, "role": role}


class TrajectorySftTest(unittest.TestCase):
    def test_latest_memory_keeps_query_and_exact_keep_without_repeating_initial_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = {}
            for modality in ("rgb", "ir", "depth_visual"):
                path = root / f"{modality}.png"
                Image.new("RGB", (64, 32), "white").save(path)
                images[modality] = str(path)
            keep_box = [0.123456, 0.234567, 0.654321, 0.765432]
            candidates = [candidate("KEEP", keep_box), candidate(2, [.3, .2, .5, .6])]
            query = "assistant，请找左侧温度更高的人"
            initial = build_initial_messages({"query": query, "images": images}, candidates,
                                             root / "global", PROFILES["sft"])
            original_text = initial[1]["content"][0]["text"]
            self.assertIn("Initial candidates (RGB normalized xyxy):", original_text)
            self.assertIn('"id":"2"', original_text)

            tool_path = root / "tool.png"
            Image.new("RGB", (32, 32), "white").save(tool_path)
            events = [{"step": 0, "action": {"action": "inspect_regions", "candidate_ids": [2]},
                       "observation": {"status": "OK", "text": "actual IR crop",
                                       "data": {"contrast": 0.37},
                                       "images": [{"path": str(tool_path), "modality": "ir",
                                                   "candidate_ids": [2],
                                                   "candidate_roles": {"2": "target"}}]}}]
            latest = build_state_messages(initial, events, candidates)
            self.assertEqual(initial[1]["content"][0]["text"], original_text)
            self.assertIn('"id":"2"', initial[1]["content"][0]["text"])
            latest_first = latest[1]["content"][0]
            self.assertNotIn("latest_memory_text", latest_first)
            self.assertIn(query, latest_first["text"])
            self.assertIn("[0.123456,0.234567,0.654321,0.765432]", latest_first["text"])
            self.assertNotIn("Initial candidates", latest_first["text"])
            visible_text = "\n".join(block["text"] for message in latest
                                     if isinstance(message["content"], list)
                                     for block in message["content"] if block["type"] == "text")
            self.assertEqual(visible_text.count("Current candidates:"), 1)
            self.assertIn('"candidate_roles":{"2":"target"}', visible_text)
            self.assertIn('"contrast":0.37', visible_text)

    def test_recent_real_images_survive_later_numeric_observation_and_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name in ("rgb", "ir", "depth", "crop_rgb", "crop_ir"):
                path = Path(directory) / f"{name}.png"
                Image.new("RGB", (64, 32), "white").save(path)
                paths.append(str(path))
            initial = [{"role": "user", "content": [
                {"type": "text", "text": "The user literally wrote assistant. Query: red car"},
                *({"type": "image", "image": path, "max_pixels": 602112, "modality": modality}
                  for path, modality in zip(paths[:3], ("rgb", "ir", "depth"))),
            ]}]
            events = [
                {"step": 0, "action": {"action": "inspect_regions"}, "observation": {
                    "status": "OK", "text": "real crops", "images": [
                        {"path": paths[3], "modality": "rgb", "candidate_ids": [1]},
                        {"path": paths[4], "modality": "ir", "candidate_ids": [1]},
                    ]}},
                {"step": 1, "action": {"action": "measure_depth"}, "observation": {
                    "status": "UNKNOWN", "text": "invalid depth", "data": {"valid_fraction": 0}, "images": []}},
                {"step": 2, "action": {"action": "analysis"}, "observation": None},
            ]
            messages = build_state_messages(initial, events, [candidate("KEEP", [0, 0, .2, .2])])
            image_blocks = [part for message in messages for part in message["content"] if part["type"] == "image"]
            self.assertEqual([part["image"] for part in image_blocks], paths)
            self.assertEqual([part["max_pixels"] for part in image_blocks[:3]], [200704] * 3)
            self.assertEqual([part["max_pixels"] for part in image_blocks[3:]], [301056] * 2)
            self.assertTrue(all(part["view"] == "tool" for part in image_blocks[3:]))
            self.assertIn('"origin_step":0', messages[-1]["content"][-4]["text"])
            prepared, metadata = prepare_image_messages(messages)
            self.assertEqual([item["modality"] for item in metadata], ["rgb", "ir", "depth", "rgb", "ir"])
            self.assertTrue(all(isinstance(part["image"], Image.Image)
                                for message in prepared for part in message["content"] if part["type"] == "image"))

    def test_gt_changes_only_terminal_label_and_never_invents_a_box(self):
        initial = [candidate("KEEP", [0, 0, .2, .2]), candidate(1, [.5, .5, .8, .8])]
        trace = {"id": "a", "memory": "latest", "manifest_row": {"images": {"rgb": "group.png"}},
                 "initial_messages": [{"role": "user", "content": [{"type": "text", "text": "Query"}]}],
                 "initial_candidates": initial,
                 "events": [{"step": 0, "messages": [{"role": "user", "content": [{"type": "text", "text": "T decision 0"}]}],
                             "action": {"action": "measure_depth", "candidate_ids": [1]},
                             "observation": {"status": "UNKNOWN", "text": "bad", "images": []},
                             "candidates_before": initial, "candidates_after": initial},
                            {"step": 1, "messages": [{"role": "user", "content": [{"type": "text", "text": "T finish hint"}]}],
                             "action": {"action": "finish", "candidate_id": "KEEP"},
                             "observation": None, "candidates_before": initial, "candidates_after": initial}]}
        rows = decision_rows(trace, [.5, .5, .8, .8])
        self.assertEqual(rows[0]["label_source"], "observed_action")
        self.assertEqual(rows[-1]["label_source"], "gt_candidate")
        self.assertEqual(rows[-1]["original_action"]["candidate_id"], "KEEP")
        self.assertEqual(rows[-1]["target_action"], '{"action":"finish","candidate_id":1}')
        self.assertEqual(rows[-1]["messages"][-1]["content"][0]["text"], "T finish hint")
        self.assertEqual(revised_final_label(trace, initial, [0, 0, .2, .2])[1], "gt_keep")
        self.assertEqual(revised_final_label(trace, initial, [.25, .25, .4, .4]), (None, "uncovered"))
        trace["memory"] = "full"
        rows_full = decision_rows(trace, [.5, .5, .8, .8])
        self.assertEqual(rows_full[-1]["messages"][-1]["content"][0]["text"], "T finish hint")
        self.assertIn("Current candidates", rows_full[-1]["messages"][-2]["content"][0]["text"])
        del trace["events"][1]["messages"]
        with self.assertRaisesRegex(ValueError, "visible messages missing"):
            decision_rows(trace, [.5, .5, .8, .8])

    def test_assistant_string_in_user_tokens_cannot_shift_loss_span(self):
        # 77091 is the old data processor's hard-coded assistant token. Here
        # it occurs in the user prefix and must stay masked.
        prefix = [1, 77091, 42, 151645, 77091, 3]
        full = prefix + [101, 102, 151645]
        self.assertEqual(supervised_token_range(prefix, full, 151645), (6, 9))
        with self.assertRaisesRegex(ValueError, "length"):
            supervised_token_range(prefix, full, 151645, max_length=8)
        self.assertEqual(supervised_token_range(prefix, full + [13], 151645,
                                                decode_tail=lambda ids: "\n" if ids == [13] else "bad"), (6, 9))
        with self.assertRaisesRegex(ValueError, "end token"):
            supervised_token_range(prefix, full[:-1] + [0], 151645)

    def test_script_continue_and_protocol_error_are_not_tool_supervision(self):
        pool = [candidate("KEEP", [0, 0, .2, .2]), candidate(1, [.5, .5, .8, .8])]
        hint = [{"role": "user", "content": [{"type": "text", "text": "decide"}]}]
        trace = {"id": "b", "memory": "latest", "final_status": "FINISHED",
                 "manifest_row": {"images": {"rgb": "group-b.png"}},
                 "initial_messages": hint, "initial_candidates": pool,
                 "events": [
                     {"action": {"action": "continue"}, "messages": hint,
                      "observation": {"status": "OK", "images": []}, "candidates_before": pool},
                     {"action": {"action": "inspect_regions"}, "messages": hint,
                      "observation": {"status": "ERROR", "text": "invalid ID", "images": []},
                      "candidates_before": pool},
                     {"action": {"action": "finish", "candidate_id": "KEEP"}, "messages": hint,
                      "observation": None, "candidates_before": pool},
                 ]}
        rows = decision_rows(trace, [.5, .5, .8, .8])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_source"], "gt_candidate")
        trace["final_status"] = "INVALID_FINAL_ACTION"
        trace["events"][-1]["observation"] = {"status": "ERROR", "images": []}
        self.assertEqual(decision_rows(trace, [.5, .5, .8, .8]), [])

    def test_real_invalid_final_repairs_only_last_visible_decision(self):
        keep = candidate("KEEP", [0, 0, .2, .2])
        seen = candidate(1, [.5, .5, .8, .8])
        future = candidate(2, [.85, .85, .95, .95])
        hint = [{"role": "user", "content": [{"type": "text", "text": "actual decision input"}]}]
        trace = {"id": "invalid-final", "memory": "latest", "final_status": "INVALID_FINAL_ACTION",
                 "manifest_row": {"images": {"rgb": "group.png"}},
                 "initial_messages": hint, "initial_candidates": [keep],
                 "events": [
                     {"action": {"action": "inspect_regions", "candidate_ids": ["KEEP"]},
                      "messages": hint, "observation": {"status": "OK", "images": []},
                      "candidates_before": [keep], "candidates_after": [keep, seen]},
                     {"action": {"action": "finish", "candidate_id": "nonexistent"},
                      "raw_output": '{"action":"finish","candidate_id":"nonexistent"}',
                      "usage": {"output_tokens": 16}, "messages": hint,
                      "observation": {"status": "ERROR", "images": []},
                      "candidates_before": [keep, seen], "candidates_after": [keep, seen, future]},
                 ]}
        rows = decision_rows(trace, seen["bbox"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["label_source"], "observed_action")
        self.assertEqual(rows[-1]["target_action"], '{"action":"finish","candidate_id":1}')
        self.assertEqual(rows[-1]["original_action"]["candidate_id"], "nonexistent")
        self.assertEqual(rows[-1]["original_raw_output"], trace["events"][-1]["raw_output"])
        self.assertEqual(rows[-1]["label_source"], "gt_candidate_repaired_invalid_final")
        self.assertEqual(rows[-1]["online_final_status"], "INVALID_FINAL_ACTION")
        self.assertEqual(rows[-1]["messages"], hint)
        self.assertTrue(rows[-1]["is_final"])

        trace["events"][-1]["action"] = None
        trace["events"][-1]["raw_output"] = 'finish: action="finish", candidate_id="KEEP"'
        keep_rows = decision_rows(trace, keep["bbox"])
        self.assertEqual(keep_rows[-1]["target_action"], '{"action":"finish","candidate_id":"KEEP"}')
        self.assertEqual(keep_rows[-1]["label_source"], "gt_keep_repaired_invalid_final")
        self.assertIsNone(keep_rows[-1]["original_action"])
        self.assertEqual(keep_rows[-1]["original_raw_output"], trace["events"][-1]["raw_output"])
        # A candidate added after the decision was not visible to the model.
        self.assertEqual(len(decision_rows(trace, future["bbox"])), 1)

        trace["events"][-1]["raw_output"] = "  "
        self.assertEqual(len(decision_rows(trace, seen["bbox"])), 1)
        trace["events"][-1]["raw_output"] = "invalid finish"
        trace["events"][-1]["usage"]["output_tokens"] = 0
        self.assertEqual(len(decision_rows(trace, seen["bbox"])), 1)
        trace["events"][-1]["usage"]["output_tokens"] = 2
        trace["final_status"] = "INPUT_BUDGET_EXHAUSTED"
        self.assertEqual(len(decision_rows(trace, seen["bbox"])), 1)

    def test_training_composition_counts_real_tool_switch_and_online_outcome(self):
        keep = candidate("KEEP", [0, 0, .2, .2])
        target = candidate(1, [.5, .5, .8, .8])
        trace = {"initial_candidates": [keep], "final_status": "FINISHED",
                 "selected_id": 1, "bbox": target["bbox"], "events": [
                     {"executed_action": {"action": "measure_depth"}, "tool_seconds": .1,
                      "observation": {"status": "UNKNOWN"}},
                     {"executed_action": {"action": "inspect_regions"}, "tool_seconds": .1,
                      "observation": {"status": "OK"}},
                     {"executed_action": {"action": "search_candidates"}, "tool_seconds": .1,
                      "observation": {"status": "OK"}, "candidates_before": [keep],
                      "candidates_after": [keep, target]},
                     {"action": {"action": "finish", "candidate_id": 1}, "observation": None},
                 ]}
        rows = [{"is_final": True, "original_action": {"action": "finish", "candidate_id": 1},
                 "target_action": '{"action":"finish","candidate_id":1}'}]
        counts = trajectory_composition(trace, rows, target["bbox"])
        self.assertTrue(counts["unknown_empty_then_other_tool"])
        self.assertTrue(counts["search_added_candidates"])
        self.assertTrue(counts["initial_c_wrong_online_final_correct"])
        self.assertFalse(counts["offline_terminal_revised"])
        trace["events"][0]["observation"]["status"] = "EMPTY"
        self.assertTrue(trajectory_composition(trace, rows, target["bbox"])["unknown_empty_then_other_tool"])
        trace["events"] = [trace["events"][0], trace["events"][-1]]
        self.assertFalse(trajectory_composition(trace, rows, target["bbox"])["unknown_empty_then_other_tool"])
        trace["selected_id"] = "KEEP"
        trace["initial_candidates"] = [candidate("KEEP", target["bbox"])]
        self.assertTrue(trajectory_composition(trace, rows, target["bbox"])["initial_c_correct_online_keep"])
        rows[0]["original_action"] = None
        self.assertTrue(trajectory_composition(trace, rows, target["bbox"])["offline_terminal_revised"])

    def test_frozen_train_holdout_export_keeps_groups_and_keep_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [candidate("KEEP", [0, 0, .2, .2]), candidate(1, [.5, .5, .8, .8])]
            cohorts = {"train": ["t1", "t2", "t3", "t4", "t5", "t6"], "holdout": ["h1"], "debug": ["d1"]}
            files = {}
            for name, ids in cohorts.items():
                path = root / f"{name}.jsonl"
                path.write_text("".join(json.dumps({"id": sample_id, "query": sample_id,
                                                    "images": {"rgb": f"{sample_id}.png"}}) + "\n"
                                        for sample_id in ids), encoding="utf-8")
                files[name] = path
            gt = {sample_id: {"bbox": ([.25, .25, .35, .35] if sample_id == "t6" else
                                      [0, 0, .2, .2] if sample_id in {"t2", "t4", "t5"} else
                                      [.5, .5, .8, .8])}
                  for ids in cohorts.values() for sample_id in ids}
            (root / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
            traces = root / "traces"
            traces.mkdir()
            (traces / "global_config.json").write_text(json.dumps({"profile": "sft"}), encoding="utf-8")
            for sample_id in ["t1", "t2", "t3", "t4", "t5", "t6", "h1", "d1"]:
                message = {"role": "user", "content": [{"type": "text", "text": sample_id + " finish"}]}
                trace = {"id": sample_id, "query": sample_id, "memory": "latest",
                         "final_status": "FINISHED", "mechanism": "D",
                         "selected_id": "KEEP", "bbox": [0, 0, .2, .2],
                         "manifest_row": {"images": {"rgb": f"{sample_id}.png"}},
                         "initial_messages": [message], "initial_candidates": candidates,
                         "events": [{"messages": [message], "action": {"action": "finish", "candidate_id": "KEEP"},
                                     "observation": None, "candidates_before": candidates}]}
                if sample_id in {"t1", "t2", "t3", "t6", "h1"}:
                    trace["events"].insert(0, {"messages": [message],
                                               "action": {"action": "measure_depth", "candidate_ids": [1]},
                                               "observation": {"status": "UNKNOWN", "images": []},
                                               "candidates_before": candidates})
                if sample_id == "t3":
                    trace["final_status"] = "INVALID_FINAL_ACTION"
                    trace["events"][-1].update({
                        "action": {"action": "finish", "candidate_id": "missing"},
                        "raw_output": '{"action":"finish","candidate_id":"missing"}',
                        "usage": {"output_tokens": 12},
                        "observation": {"status": "ERROR", "images": []}})
                if sample_id == "t1":
                    sample_dir = traces / sample_id
                    sample_dir.mkdir()
                    (sample_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
                    (sample_dir / "events.json").write_text(json.dumps({"event_count": 1}), encoding="utf-8")
                else:
                    (traces / f"{sample_id}.json").write_text(json.dumps(trace), encoding="utf-8")
            args = SimpleNamespace(traces=traces, train_manifest=files["train"],
                                   holdout_manifest=files["holdout"], debug_manifest=files["debug"],
                                   gt=root / "gt.json", output_dir=root / "out")
            inventory = export(args)
            self.assertEqual(inventory["ignored_Z_debug_traces"], 1)
            self.assertEqual(inventory["input_trajectories"], 8)
            self.assertEqual(inventory["collected_trajectories"], 7)
            self.assertEqual(inventory["collected_train_trajectories"], 6)
            self.assertEqual(inventory["train_trajectories"], 4)
            self.assertEqual(inventory["holdout_trajectories"], 1)
            self.assertEqual(inventory["collected_train_without_legal_final_trajectories"], 1)
            self.assertEqual(inventory["collected_train_terminal_only_revised_trajectories"], 2)
            self.assertEqual(inventory["collected_train_excluded_tool_decisions"], 3)
            self.assertEqual(inventory["skipped_for_keep_cap"], 1)
            self.assertEqual(inventory["collected_train_origin_trajectories"], {"autonomous": 6})
            self.assertEqual(inventory["train_origin_trajectories"], {"autonomous": 4})
            self.assertEqual(inventory["train_origin_decisions"], {"autonomous": 5})
            self.assertEqual(inventory["train_online_final_status_trajectories"],
                             {"FINISHED": 3, "INVALID_FINAL_ACTION": 1})
            self.assertEqual(inventory["collected_train_composition_trajectories"]["initial_c_correct_online_keep"], 3)
            self.assertEqual(inventory["train_composition_trajectories"]["initial_c_correct_online_keep"], 2)
            self.assertEqual(inventory["train_composition_trajectories"]["offline_terminal_revised"], 2)
            self.assertLessEqual(inventory["train_keep_fraction"], .5)
            self.assertLessEqual(inventory["train_final_keep_fraction"], .5)
            train_rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
            holdout_rows = [json.loads(line) for line in (root / "out" / "holdout.jsonl").read_text().splitlines()]
            self.assertEqual({row["id"] for row in train_rows}, {"t1", "t2", "t3", "t4"})
            self.assertEqual({row["id"] for row in holdout_rows}, set(cohorts["holdout"]))
            self.assertEqual(len(holdout_rows), 2)
            self.assertEqual([row["is_final"] for row in train_rows if row["id"] == "t1"], [True])
            self.assertEqual([row["is_final"] for row in train_rows if row["id"] == "t3"], [True])
            self.assertEqual([row["is_final"] for row in train_rows if row["id"] == "t2"], [False, True])
            self.assertNotIn("t6", {row["id"] for row in train_rows})
            self.assertEqual(next(row for row in train_rows if row["id"] == "t2" and row["is_final"])["label_source"], "gt_keep")
            overlap = {"same": {"id": "same", "images": {"rgb": "same.png"}}}
            with self.assertRaisesRegex(ValueError, "overlap"):
                verify_frozen_splits(overlap, overlap, {})
            (traces / "t1_copy.json").write_text((traces / "t1" / "trace.json").read_text(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate trajectory start"):
                export(args)

    @unittest.skipUnless(os.environ.get("QWEN_MODEL_PATH"), "set QWEN_MODEL_PATH for a real Qwen processor check")
    def test_real_qwen_processor_masks_user_assistant_word_and_all_images(self):
        from transformers import AutoProcessor
        model_path = os.environ["QWEN_MODEL_PATH"]
        processor = AutoProcessor.from_pretrained(model_path, min_pixels=1024,
                                                  max_pixels=602112, local_files_only=True)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = Path(directory) / f"{index}.png"
                Image.new("RGB", (32, 32), "white").save(path)
                paths.append(str(path))
            initial = [{"role": "user", "content": [{"type": "text", "text": "assistant is part of the user query"},
                *({"type": "image", "image": path, "max_pixels": 1024} for path in paths[:3])]}]
            events = [{"action": {"action": "inspect_regions"}, "observation": {"status": "OK", "images": [
                {"path": paths[3], "modality": "rgb", "candidate_ids": [1]}]}}]
            row = {"id": "processor-check", "decision_index": 1,
                   "messages": build_state_messages(initial, events, [candidate(1, [0, 0, 1, 1])]),
                   "target_action": '{"action":"finish","candidate_id":1}'}
            encoded = encode_decision(processor, row)
            labels = encoded["labels"][0].tolist()
            self.assertGreater(labels.count(-100), 0)
            self.assertEqual(int(encoded["image_grid_thw"].shape[0]), 4)
            supervised = [token for token in labels if token != -100]
            answer = processor.tokenizer.decode(supervised, skip_special_tokens=False)
            self.assertIn(row["target_action"], answer)
            self.assertIn("<|im_end|>", answer)
            self.assertNotIn("assistant is part of the user query", answer)


if __name__ == "__main__":
    unittest.main()
