"""CPU checks for independent real search attempts and offline branch context."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.visual_agent.collect_search_branches import collect_one, main
from experiments.visual_agent.vision_tools import VisualTools


class SearchBranchTests(unittest.TestCase):
    def test_eight_sibling_searches_record_empty_unknown_and_real_added_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = Image.new("RGB", (96, 64), "green")
            for x in range(48):
                for y in range(64):
                    rgb.putpixel((x, y), (255, 0, 0))
            rgb.save(root / "rgb.png")
            Image.new("RGB", (96, 64), "gray").save(root / "ir.png")
            row = {"id": "sample", "query": "white sedan", "images": {
                "rgb": str(root / "rgb.png"), "ir": str(root / "ir.png")},
                "depth_encoding": "city_mm",
                "dino_model_path": "unused-cpu-test-model"}
            cache = {"id": "sample", "images": row["images"], "c_bbox": [.1, .1, .3, .6],
                     "query_info": {"target_category": "sedan", "scope": "single"},
                     "candidates": [{"id": 1, "role": "target", "bbox": [.1, .1, .3, .6],
                                     "is_baseline": True, "sources": []}]}

            def fake_dino(self):
                if self.dino_model is None:
                    self.dino_model = (object(), object(), "cpu")
                return self.dino_model

            def fake_detection(image, modality, query_info, processor, model, device):
                self.assertEqual(device, "cpu")
                self.assertEqual(query_info["target_category"], "sedan")
                if modality == "rgb" and image.size == (48, 64) and image.getpixel((0, 0)) == (255, 0, 0):
                    return [{"bbox": [.6, .1, .9, .8], "score": .9}]
                return []

            with patch.object(VisualTools, "_dino", fake_dino), patch(
                    "experiments.visual_agent.vision_tools.detect_image_proposals", fake_detection):
                attempts, branches, dino, skipped = collect_one(row, cache, root / "out")
            self.assertIsNone(skipped)
            self.assertEqual(len(attempts), 8)
            self.assertEqual([event["branch_key"] for event in attempts[:4]],
                             ["rgb/left", "rgb/right", "rgb/top", "rgb/bottom"])
            self.assertEqual(sum(event["observation"]["status"] == "OK" for event in attempts), 1)
            self.assertEqual(sum(event["observation"]["status"] == "EMPTY" for event in attempts), 7)
            self.assertEqual(len(branches), 1)
            branch = branches[0]
            self.assertEqual(branch["branch_key"], "rgb/left")
            self.assertEqual(branch["branch_replay"], "offline_only")
            self.assertEqual([event["action"]["action"] for event in branch["events"]],
                             ["search_candidates", "inspect_regions"])
            self.assertEqual(branch["candidate_id"], branch["events"][0]["observation"]["data"]["appended_ids"][0])
            self.assertEqual(len(branch["events"][1]["observation"]["images"]), 4)
            self.assertTrue(all(Path(image["path"]).is_file()
                                for image in branch["events"][1]["observation"]["images"]))
            self.assertIn("Most recent actual tool image", json.dumps(branch["terminal_messages"]))
            self.assertNotIn("gt", json.dumps(branch).casefold())
            self.assertEqual(dino[2], "cpu")

            row.pop("depth_encoding", None)
            cache["depth_encoding"] = None
            with patch.object(VisualTools, "_dino", fake_dino), patch(
                    "experiments.visual_agent.vision_tools.detect_image_proposals", fake_detection):
                attempts, branches, _, _ = collect_one(row, cache, root / "unknown")
            self.assertEqual(sum(event["observation"]["status"] == "UNKNOWN" for event in attempts), 4)
            self.assertEqual(len(branches), 1)

    def test_failure_manifest_rejects_holdout_before_any_tool_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def record(sample_id, rgb):
                return {"id": sample_id, "query": "sedan", "images": {"rgb": rgb}}
            rows = {"failure": record("held", "held.png"),
                    "train": record("train", "train.png"),
                    "holdout": record("held", "held.png")}
            for filename, value in (("failures.jsonl", rows["failure"]),
                                    ("t_train200.jsonl", rows["train"]),
                                    ("t_holdout50.jsonl", rows["holdout"])):
                (root / filename).write_text(json.dumps(value) + "\n", encoding="utf-8")
            (root / "candidates.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "heldout IDs are forbidden"):
                main(["--failures-manifest", str(root / "failures.jsonl"),
                      "--train-manifest", str(root / "t_train200.jsonl"),
                      "--holdout-manifest", str(root / "t_holdout50.jsonl"),
                      "--candidate-cache", str(root / "candidates.jsonl"),
                      "--dino-model", "unused", "--output-dir", str(root / "out")])


if __name__ == "__main__":
    unittest.main()
