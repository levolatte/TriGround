import tempfile
import unittest
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.visual_agent.vision_tools import CandidatePool, VisualTools


class VisionToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rgb = self.root / "rgb.png"
        self.ir = self.root / "ir.png"
        self.depth_visual = self.root / "depth_visual.png"
        self.depth_raw = self.root / "depth_raw.png"
        Image.new("RGB", (101, 61), "red").save(self.rgb)
        Image.new("RGB", (101, 61), "blue").save(self.ir)
        Image.new("RGB", (101, 61), "gray").save(self.depth_visual)
        Image.fromarray(np.full((61, 101), 2000, np.uint16)).save(self.depth_raw)
        self.row = {
            "id": "example", "query": "red person near car",
            "images": {"rgb": str(self.rgb), "ir": str(self.ir),
                       "depth_visual": str(self.depth_visual), "depth_raw": str(self.depth_raw)},
            "depth_encoding": "city_mm", "query_info": {"scope": "single"},
        }
        self.candidate_row = {
            "id": "example", "images": self.row["images"], "depth_encoding": "city_mm",
            "c_bbox": [0.0, 0.1, 0.2, 0.6],
            "candidates": [
                {"id": 7, "role": "target", "bbox": [0.0, 0.1, 0.2, 0.6],
                 "is_baseline": True, "sources": [], "depth": {"status": "pending"}},
                {"id": 2, "role": "target", "bbox": [0.45, 0.2, 0.65, 0.7],
                 "is_baseline": False, "sources": [], "depth": {"status": "pending"}},
                {"id": 4, "role": "reference", "bbox": [0.75, 0.2, 0.9, 0.8],
                 "is_baseline": False, "sources": [], "depth": {"status": "pending"}},
            ],
        }
        self.pool = CandidatePool(self.candidate_row, (101, 61))
        self.tools = VisualTools(self.row, self.pool, self.root / "observations", 10000,
                                 dino_model=(object(), object(), "cpu"),
                                 sam_model=(object(), object(), "cpu"), max_searches=3)

    def tearDown(self):
        self.temp.cleanup()

    def test_keep_exact_and_reference_role_does_not_block_rgb_finish(self):
        public = self.pool.public_candidates()
        self.assertEqual([item["id"] for item in public], ["KEEP", "2", "4"])
        self.assertEqual(self.pool.finish("KEEP"), self.candidate_row["c_bbox"])
        with self.assertRaises(ValueError):
            self.pool.finish("7")
        self.assertEqual(self.pool.finish(4), [0.75, 0.2, 0.9, 0.8])
        finish = self.tools.execute({"action": "finish", "candidate_id": 4})
        self.assertEqual(finish["status"], "OK")
        self.assertEqual(finish["data"]["bbox"], [0.75, 0.2, 0.9, 0.8])
        self.assertEqual(self.tools.execute({"action": "inspect_regions", "view": "single",
                                             "candidate_ids": ["7"], "modalities": "rgb"})["status"], "ERROR")

    def test_cache_gt_boundary_and_public_provenance(self):
        malicious = copy.deepcopy(self.candidate_row)
        malicious["gt"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, "forbidden"):
            CandidatePool(malicious, (101, 61))
        malicious = copy.deepcopy(self.candidate_row)
        malicious["candidates"][1]["iou"] = 1.0
        with self.assertRaisesRegex(ValueError, "forbidden"):
            CandidatePool(malicious, (101, 61))
        source = {"modality": "rgb", "bbox": [.45,.2,.65,.7], "score": .8,
                  "region_px": [0,0,101,61], "projection": "normalized_shared_frame"}
        self.pool.get(2)["sources"].append(source)
        public_source = self.pool.public_candidates()[1]["sources"][0]
        self.assertEqual(public_source, {"modality": "rgb", "score": .8,
                                         "projection": "normalized_shared_frame",
                                         "bbox": [.45, .2, .65, .7],
                                         "region_px": [0, 0, 101, 61]})
        self.assertEqual(self.pool.snapshot()["source_candidates"][1]["sources"][0], source)

    def test_append_preserves_ids_and_distinct_high_iou_box_hypotheses(self):
        source = {"modality": "rgb", "score": 0.7}
        candidate_id, is_new = self.pool.append_detection(
            {"role": "target", "bbox": [0.45, 0.2, 0.65, 0.7], "source": source})
        self.assertEqual((candidate_id, is_new), ("2", False))
        new_id, is_new = self.pool.append_detection(
            {"role": "target", "bbox": [0.46, 0.2, 0.66, 0.7], "source": source})
        self.assertTrue(is_new)
        self.assertEqual(new_id, "8")
        self.assertEqual(self.pool.get(new_id)["possible_same_object_ids"], [])
        self.assertEqual(self.pool.finish(2), [0.45, 0.2, 0.65, 0.7])

    def test_all_inspect_views_save_real_labeled_images_with_shared_budget(self):
        cases = [
            {"view": "single", "candidate_ids": ["KEEP"], "modalities": "rgb"},
            {"view": "pair", "candidate_ids": ["KEEP", "2"], "modalities": "ir"},
            {"view": "cross", "candidate_ids": ["2"], "modalities": ["rgb", "ir", "depth"]},
        ]
        for case, count in zip(cases, [1, 2, 3]):
            observation = self.tools.execute({"action": "inspect_regions", **case})
            self.assertEqual(observation["status"], "OK")
            self.assertEqual(len(observation["images"]), count)
            self.assertLessEqual(observation["data"]["total_saved_pixels"], 10000)
            for item in observation["images"]:
                with Image.open(item["path"]) as image:
                    self.assertEqual(list(image.size), item["saved_size"])
                box = item["target_boxes"][0]["xyxy"]
                self.assertTrue(all(0 <= value <= 1 for value in box))
        first_bounds = self.tools.execute({"action": "inspect_regions", **cases[0]})["images"][0]["crop_xyxy"]
        self.assertEqual(first_bounds[0], 0)
        self.assertEqual(first_bounds[2] > first_bounds[0], True)
        partial = self.tools.execute({"action": "inspect_regions", "view": "cross",
                                      "candidate_ids": ["2"], "modalities": ["rgb", "ir"]})
        self.assertEqual(partial["status"], "OK")
        self.assertEqual(len(partial["images"]), 2)
        self.row["images"].pop("ir")
        missing = self.tools.execute({"action": "inspect_regions", "view": "pair",
                                      "candidate_ids": ["KEEP", "2"], "modalities": "ir"})
        self.assertEqual(missing["status"], "UNKNOWN")

    def test_search_region_maps_from_actual_integer_crop_edges_and_limits(self):
        detection = [{"bbox": [0.2, 0.25, 0.6, 0.75], "score": 0.9}]
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=detection):
            result = self.tools.execute({"action": "search_candidates", "category": "person",
                                         "region": "right", "modality": "rgb", "role": "target"})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["data"]["region_px"], [50, 0, 101, 61])
        new_id, = result["data"]["appended_ids"]
        expected_x1 = (50 + 0.2 * 51) / 101
        self.assertAlmostEqual(self.pool.get(new_id)["bbox"][0], expected_x1)
        self.assertEqual(self.pool.get(2)["bbox"], [0.45, 0.2, 0.65, 0.7])
        self.assertEqual(result["images"][0]["candidate_ids"], [new_id])
        self.assertEqual(self.tools.cost_counts["dino_calls"], 1)
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=[]):
            empty = self.tools.execute({"action": "search_candidates", "category": "car",
                                        "region": "grid:2:0",
                                        "modality": "ir", "role": "reference"})
        self.assertEqual(empty["status"], "EMPTY")
        self.assertEqual(empty["data"]["region_px"], [0, 40, 34, 61])
        self.assertEqual(self.tools.cost_counts["dino_calls"], 2)
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=[]):
            self.tools.execute({"action": "search_candidates", "category": "car",
                                "region": "full", "modality": "rgb", "role": "reference"})
        self.assertEqual(self.tools.execute({"action": "search_candidates", "category": "car",
                                             "region": "full", "modality": "rgb", "role": "reference"})["status"], "LIMIT")

    def test_depth_cache_unknown_and_unsupported_pair_hide_ordering(self):
        reliable = {"status": "reliable", "reason": "reliable", "core": {"valid_count": 100,
                    "valid_ratio": 1.0, "q1_m": 1.9, "median_m": 2.0, "q3_m": 2.1},
                    "full": {"valid_count": 100, "valid_ratio": 1.0, "median_m": 2.0}}
        self.pool.get("KEEP")["depth"] = reliable.copy()
        self.pool.get(2)["depth"] = reliable.copy()
        result = self.tools.execute({"action": "measure_depth", "candidate_ids": ["KEEP", 2]})
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["data"]["pair"], {
            "candidate_a_id": "KEEP", "candidate_b_id": "2",
            "status": "not_separated", "reason": "near_far_separation_failed",
        })
        self.assertEqual(self.tools.cost_counts["depth_cache_hits"], 2)
        self.assertEqual(self.tools.cost_counts["depth_generated"], 0)
        self.assertGreaterEqual(self.tools.cost_seconds["depth_cached"], 0.0)
        self.row["depth_encoding"] = "unknown"
        result = self.tools.execute({"action": "measure_depth", "candidate_ids": ["KEEP", 2]})
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["data"]["measurements"]["KEEP"]["reason"], "unknown_encoding")
        self.assertEqual(result["data"]["pair"]["candidate_a_id"], "KEEP")
        self.assertEqual(result["data"]["pair"]["candidate_b_id"], "2")

    def test_supported_pair_uses_keep_name_and_cached_measurement(self):
        def measurement(meters):
            return {"status": "reliable", "reason": "reliable",
                    "core": {"valid_count": 100, "valid_ratio": 1.0,
                             "q1_m": meters - 0.1, "median_m": meters,
                             "q3_m": meters + 0.1},
                    "full": {"valid_count": 100, "valid_ratio": 1.0,
                             "median_m": meters}}
        self.pool.get("KEEP")["depth"] = measurement(2.0)
        self.pool.get(2)["depth"] = measurement(4.0)
        result = self.tools.execute({"action": "measure_depth", "candidate_ids": ["KEEP", 2]})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["data"]["pair"]["near_candidate_id"], "KEEP")
        self.assertEqual(result["data"]["pair"]["far_candidate_id"], "2")

    def test_unknown_external_ir_stays_an_ir_frame_proposal(self):
        self.row["depth_encoding"] = "unknown"
        detection = [{"bbox": [.1, .2, .3, .7], "score": .9}]
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=detection):
            result = self.tools.execute({"action": "search_candidates", "category": "person",
                                         "region": "full", "modality": "ir", "role": "target"})
        self.assertEqual(result["status"], "OK")
        candidate_id, = result["data"]["appended_ids"]
        candidate = next(item for item in self.pool.public_candidates() if item["id"] == candidate_id)
        self.assertEqual(candidate["coordinate_frame"], "ir")
        self.assertFalse(candidate["finish_eligible"])
        self.assertEqual(candidate["sources"][0]["projection"], "unregistered_ir_proposal")
        with self.assertRaisesRegex(ValueError, "IR candidate"):
            self.pool.finish(candidate_id)

    def test_legacy_candidate_with_only_ir_source_is_not_rgb_eligible(self):
        row = copy.deepcopy(self.candidate_row)
        row["candidates"][1]["sources"] = [{"modality": "ir", "score": .9}]
        pool = CandidatePool(row, (101, 61))
        candidate = pool.public_candidates()[1]
        self.assertEqual(candidate["coordinate_frame"], "ir")
        self.assertFalse(candidate["finish_eligible"])

    def test_zero_valid_depth_is_unknown_without_pair_order(self):
        Image.fromarray(np.zeros((61, 101), np.uint16)).save(self.depth_raw)
        mask = np.ones((61, 101), dtype=bool)
        with patch("experiments.visual_agent.vision_tools.segment_boxes", return_value=[(mask, 0.95)]):
            result = self.tools.execute({"action": "measure_depth", "candidate_ids": [2]})
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["data"]["measurements"]["2"]["core"]["valid_count"], 0)

    def test_depth_swap_invalidates_old_candidate_and_evidence_cache(self):
        old = {"status": "reliable", "reason": "reliable", "core": {"median_m": 2.0}}
        self.pool.get(2)["depth"] = old
        self.tools.evidence_row = {
            "id": "example", "images": dict(self.row["images"]), "depth_encoding": "city_mm",
            "candidates": [{"id": 2, "bbox": self.pool.get(2)["bbox"], "depth": old}],
        }
        swapped = self.root / "swapped_raw.png"
        Image.fromarray(np.full((61, 101), 5000, np.uint16)).save(swapped)
        self.row["images"] = {**self.row["images"], "depth_raw": str(swapped)}
        mask = np.ones((61, 101), dtype=bool)
        with patch("experiments.visual_agent.vision_tools.segment_boxes", return_value=[(mask, 0.95)]) as segment:
            first = self.tools.execute({"action": "measure_depth", "candidate_ids": ["2"]})
            second = self.tools.execute({"action": "measure_depth", "candidate_ids": ["2"]})
        self.assertEqual(first["status"], "OK")
        self.assertEqual(first["data"]["measurements"]["2"]["core"]["median_m"], 5.0)
        self.assertEqual(second["data"]["measurements"]["2"]["core"]["median_m"], 5.0)
        self.assertEqual(segment.call_count, 1)
        self.assertEqual(self.tools.cost_counts["depth_generated"], 1)
        self.assertEqual(self.tools.cost_counts["depth_cache_hits"], 1)

    def test_new_candidate_gets_real_mask_depth_not_neighbor_cache(self):
        new_id, _ = self.pool.append_detection({"role": "target", "bbox": [0.3, 0.2, 0.4, 0.7],
                                                "source": {"modality": "rgb", "score": 0.8}})
        mask = np.zeros((61, 101), dtype=bool)
        mask[10:50, 20:80] = True
        with patch("experiments.visual_agent.vision_tools.segment_boxes", return_value=[(mask, 0.95)]) as segment:
            result = self.tools.execute({"action": "measure_depth", "candidate_ids": [new_id]})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["data"]["measurements"][str(new_id)]["core"]["median_m"], 2.0)
        self.assertEqual(segment.call_count, 1)
        self.assertEqual(self.tools.cost_counts["depth_generated"], 1)
        self.assertEqual(self.tools.cost_counts["depth_cache_hits"], 0)
        self.assertGreaterEqual(self.tools.cost_seconds["depth_generated"], 0.0)


if __name__ == "__main__":
    unittest.main()
