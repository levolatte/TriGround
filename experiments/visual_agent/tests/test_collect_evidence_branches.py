"""CPU checks for independent, GT-free branch images and replay context."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.visual_agent.collect_evidence_branches import collect_one, main
from experiments.visual_agent.profiles import PROFILES
from experiments.visual_agent.run import build_initial_messages
from experiments.visual_agent.vision_tools import CandidatePool


class EvidenceBranchTests(unittest.TestCase):
    def test_real_cross_images_replace_unrelated_inspect_in_replayed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = {}
            for modality in ("rgb", "ir", "depth_visual"):
                path = root / f"{modality}.png"
                Image.new("RGB", (96, 64), "gray").save(path)
                images[modality] = str(path)
            row = {"id": "sample", "query": "the white sedan in the foreground", "images": images,
                   "depth_encoding": "unknown", "gt_bbox": [.9, .9, 1, 1]}
            candidate_row = {"id": "sample", "images": images, "depth_encoding": "unknown",
                             "c_bbox": [.1, .1, .3, .6], "candidates": [
                                 {"id": 1, "role": "target", "bbox": [.1, .1, .3, .6],
                                  "is_baseline": True, "sources": []},
                                 {"id": 2, "role": "target", "bbox": [.5, .1, .7, .6],
                                  "is_baseline": False, "sources": []}]}
            pool = CandidatePool(candidate_row, (96, 64))
            initial = build_initial_messages(row, pool.public_candidates(), root / "global", PROFILES["sft"])
            inspected = {"step": 0, "action": {"action": "inspect_regions", "candidate_ids": ["KEEP"],
                                              "modalities": ["rgb"], "view": "single"},
                         "observation": {"status": "OK", "text": "unrelated old zoom", "images": []},
                         "messages": [{"role": "user", "content": [{"type": "text", "text": "old zoom"}]}]}
            depth = {"step": 1, "action": {"action": "measure_depth", "candidate_ids": ["KEEP", "2"]},
                     "executed_action": {"action": "measure_depth", "candidate_ids": ["KEEP", "2"]},
                     "observation": {"status": "UNKNOWN", "text": "depth unreliable", "images": []}}
            search = {"step": 2, "action": {"action": "search_candidates", "category": "sedan"},
                      "executed_action": {"action": "search_candidates", "category": "sedan"},
                      "observation": {"status": "EMPTY", "text": "no new candidate", "images": []}}
            trace = {"id": "sample", "query": row["query"], "manifest_row": row,
                     "trajectory_origin": "scripted", "final_status": "OBSERVATION_COMPLETE",
                     "initial_messages": initial, "events": [inspected, depth, search],
                     "final_pool_snapshot": pool.snapshot(), "final_candidates": pool.public_candidates()}
            branches, skipped = collect_one(trace, row, root / "branches", root / "source.json")
            self.assertIsNone(skipped)
            self.assertEqual([branch["candidate_id"] for branch in branches], ["KEEP", "2"])
            for branch in branches:
                self.assertEqual([event["action"]["action"] for event in branch["prefix_events"]],
                                 ["measure_depth", "search_candidates"])
                self.assertTrue(all("messages" not in event for event in branch["prefix_events"]))
                self.assertEqual(branch["event"]["observation"]["status"], "OK")
                self.assertEqual([image["modality"] for image in branch["event"]["observation"]["images"]],
                                 ["rgb", "ir", "rgb", "ir"])
                self.assertTrue(all(Path(image["path"]).is_file()
                                    for image in branch["event"]["observation"]["images"]))
                terminal_text = json.dumps(branch["terminal_messages"], ensure_ascii=False)
                self.assertNotIn("unrelated old zoom", terminal_text)
                self.assertIn("depth unreliable", terminal_text)
                self.assertIn("no new candidate", terminal_text)
                self.assertIn("Most recent actual tool image", terminal_text)
                self.assertNotIn("gt_bbox", json.dumps(branch))
            trace["events"] = [inspected] * 6
            self.assertEqual(collect_one(trace, row, root / "skipped", root / "source.json"),
                             ([], "evidence_budget"))

    def test_cli_uses_only_frozen_train_trace_and_writes_per_sample_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = {}
            for modality in ("rgb", "ir", "depth_visual"):
                path = root / f"{modality}.png"
                Image.new("RGB", (64, 32), "gray").save(path)
                images[modality] = str(path)
            row = {"id": "sample", "query": "white sedan", "images": images}
            candidate_row = {"id": "sample", "images": images, "c_bbox": [.1, .1, .3, .5],
                             "candidates": [{"id": 1, "role": "target", "bbox": [.1, .1, .3, .5],
                                             "is_baseline": True, "sources": []}]}
            pool = CandidatePool(candidate_row, (64, 32))
            trace = {"id": "sample", "query": row["query"], "manifest_row": row,
                     "trajectory_origin": "scripted", "final_status": "OBSERVATION_COMPLETE",
                     "initial_messages": build_initial_messages(row, pool.public_candidates(), root / "global",
                                                                PROFILES["sft"]),
                     "events": [], "final_pool_snapshot": pool.snapshot(),
                     "final_candidates": pool.public_candidates()}
            traces = root / "input" / "traces" / "sample"
            traces.mkdir(parents=True)
            (traces / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
            for name, value in (("t_train200.jsonl", row),
                                ("t_holdout50.jsonl", {"id": "held", "query": "held", "images": {"rgb": "held.png"}}),
                                ("train32.jsonl", {"id": "debug", "query": "debug", "images": {"rgb": "debug.png"}})):
                (root / name).write_text(json.dumps(value) + "\n", encoding="utf-8")
            output = root / "branches"
            main(["--traces", str(root / "input" / "traces"),
                  "--train-manifest", str(root / "t_train200.jsonl"),
                  "--holdout-manifest", str(root / "t_holdout50.jsonl"),
                  "--debug-manifest", str(root / "train32.jsonl"),
                  "--output-dir", str(output)])
            rows = [json.loads(line) for line in (output / "traces" / "sample" / "branches.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trajectory_origin"], "scripted_branch")
            self.assertEqual(rows[0]["event"]["action"]["candidate_ids"], ["KEEP"])
            self.assertEqual(json.loads((output / "inventory.json").read_text())["branches"], 1)


if __name__ == "__main__":
    unittest.main()
