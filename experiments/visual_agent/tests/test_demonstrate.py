"""CPU checks for GT-free scripted observations and frozen split boundaries."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.visual_agent.demonstrate import (
    DECISION_INSTRUCTION, FINAL_INSTRUCTION, demonstrate_one, frozen_train_rows, main,
)


class DemonstrateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images = {}
        for key in ("rgb", "ir", "depth_visual"):
            path = self.root / f"{key}.png"
            Image.new("RGB", (96, 64), "gray").save(path)
            images[key] = str(path)
        self.row = {"id": "train-example", "query": "the front car", "images": images,
                    "depth_encoding": "unknown"}
        self.candidate_row = {
            "id": self.row["id"], "images": images, "depth_encoding": "unknown",
            "query_info": {"scope": "single", "target_category": "car"},
            "c_bbox": [.1, .1, .3, .5],
            "candidates": [
                {"id": 1, "role": "target", "bbox": [.1, .1, .3, .5],
                 "is_baseline": True, "sources": []},
                {"id": 2, "role": "target", "bbox": [.5, .1, .7, .5],
                 "is_baseline": False, "sources": []},
            ],
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_real_ir_image_is_visible_after_scripted_inspection(self):
        actions = iter([
            {"action": "inspect_regions", "candidate_ids": ["KEEP", "2"],
             "modalities": ["rgb", "ir"], "view": "cross", "evidence_note": "compare silhouettes"},
            {"action": "measure_depth", "candidate_ids": ["KEEP", "2"]},
            None,
        ])
        trace, _ = demonstrate_one(
            self.row, self.candidate_row, self.root / "trace", policy=lambda *_: next(actions),
        )
        self.assertEqual(trace["final_status"], "OBSERVATION_COMPLETE")
        self.assertEqual(trace["trajectory_origin"], "scripted")
        self.assertIsNone(trace["selected_id"])
        self.assertEqual(len(trace["events"]), 2)
        first, second = trace["events"]
        self.assertEqual(first["observation"]["status"], "OK")
        self.assertEqual([image["modality"] for image in first["observation"]["images"]],
                         ["rgb", "ir", "rgb", "ir"])
        self.assertTrue(all(Path(image["path"]).is_file() for image in first["observation"]["images"]))
        self.assertEqual(second["observation"]["status"], "UNKNOWN")
        self.assertEqual(len([part for message in second["messages"]
                              if isinstance(message["content"], list)
                              for part in message["content"] if part["type"] == "image"]), 7)
        self.assertEqual(first["origin"], "scripted")
        self.assertEqual(first["action"], first["executed_action"])
        self.assertEqual(first["candidates_before"], trace["initial_candidates"])
        self.assertNotIn("raw_output", first)
        self.assertEqual(trace["stop_reason"], "policy_stop")
        self.assertEqual(trace["terminal_messages"][-1]["content"][0]["text"], DECISION_INSTRUCTION)
        self.assertNotIn("gt", json.dumps(trace).lower())

    def test_policy_cannot_insert_finish_or_exceed_sft_budget(self):
        with self.assertRaisesRegex(ValueError, "non-tool action"):
            demonstrate_one(self.row, self.candidate_row, self.root / "bad",
                            policy=lambda *_: {"action": "finish", "candidate_id": "KEEP"})
        with self.assertRaisesRegex(ValueError, "sft evidence-call budget"):
            demonstrate_one(self.row, self.candidate_row, self.root / "budget",
                            policy=lambda *_: None, max_steps=7)

    def test_frozen_train_manifest_excludes_holdout_and_debug_groups(self):
        def manifest(name, sample_id, rgb):
            path = self.root / name
            path.write_text(json.dumps({"id": sample_id, "query": "car",
                                        "images": {"rgb": rgb}}) + "\n", encoding="utf-8")
            return path
        train = manifest("t_train200.jsonl", "train", "train.png")
        holdout = manifest("t_holdout50.jsonl", "held", "held.png")
        debug = manifest("train32.jsonl", "debug", "debug.png")
        self.assertEqual([row["id"] for row in frozen_train_rows(train, holdout, debug)], ["train"])
        manifest("t_holdout50.jsonl", "held", "train.png")
        with self.assertRaisesRegex(ValueError, "overlap"):
            frozen_train_rows(train, holdout, debug)

    def test_cpu_cli_writes_observation_trace_without_final_label(self):
        def manifest(name, row):
            path = self.root / name
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            return path
        source_row = {**self.row, "images": {
            key: "/remote/" + Path(path).name for key, path in self.row["images"].items()}}
        source_cache = {**self.candidate_row, "images": source_row["images"]}
        train = manifest("t_train200.jsonl", source_row)
        holdout = manifest("t_holdout50.jsonl", {
            "id": "held", "query": "held", "images": {"rgb": "held.png"}})
        debug = manifest("train32.jsonl", {
            "id": "debug", "query": "debug", "images": {"rgb": "debug.png"}})
        cache = manifest("candidates.jsonl", source_cache)
        output = self.root / "observations"
        main(["--manifest", str(train), "--holdout-manifest", str(holdout),
              "--debug-manifest", str(debug), "--candidate-cache", str(cache),
              "--path-map", "/remote=" + str(self.root), "--output-dir", str(output)])
        execution = json.loads((output / "execution.json").read_text(encoding="utf-8"))
        trace = json.loads((output / "traces" / self.row["id"] / "trace.json").read_text(encoding="utf-8"))
        self.assertEqual((execution["expected"], execution["completed"], execution["complete"]),
                         (1, 1, True))
        self.assertEqual(trace["trajectory_origin"], "scripted")
        self.assertEqual(trace["manifest_row"]["images"]["rgb"], "/remote/rgb.png")
        self.assertEqual(trace["final_status"], "OBSERVATION_COMPLETE")
        self.assertEqual(trace["label_source"], "pending_offline_teacher")
        self.assertEqual(trace["stop_reason"], "policy_stop")
        self.assertEqual(trace["terminal_messages"][-1]["content"][0]["text"], DECISION_INSTRUCTION)
        self.assertTrue((output / "traces" / self.row["id"] / "events.jsonl").is_file())

    def test_evidence_limit_uses_online_final_hint(self):
        action = {"action": "inspect_regions", "candidate_ids": ["KEEP"],
                  "modalities": ["ir"], "view": "single"}
        trace, _ = demonstrate_one(self.row, self.candidate_row, self.root / "limit",
                                   policy=lambda *_: action, max_steps=1)
        self.assertEqual(trace["stop_reason"], "evidence_limit")
        self.assertEqual(trace["terminal_messages"][-1]["content"][0]["text"], FINAL_INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
