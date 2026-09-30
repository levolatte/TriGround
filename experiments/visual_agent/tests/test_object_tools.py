import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.visual_agent.object_tools import ObjectTools
from experiments.visual_agent.vision_tools import CandidatePool


class ObjectToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / f"{name}.png" for name in ("rgb", "ir", "depth_visual", "depth_raw")}
        Image.new("RGB", (120, 80), (220, 20, 30)).save(self.paths["rgb"])
        Image.new("RGB", (120, 80), (20, 30, 220)).save(self.paths["ir"])
        Image.new("RGB", (120, 80), (80, 80, 80)).save(self.paths["depth_visual"])
        Image.fromarray(np.full((80, 120), 2400, dtype=np.uint16)).save(self.paths["depth_raw"])
        self.row = {
            "id": "sample",
            "query": "the person beside the car",
            "images": {key: str(path) for key, path in self.paths.items()},
            "depth_encoding": "city_mm",
            "query_info": {"scope": "single"},
        }
        boxes = [
            [.05, .1, .2, .5], [.25, .1, .4, .5], [.45, .1, .6, .5],
            [.65, .1, .8, .5], [.1, .55, .25, .9], [.35, .55, .5, .9],
            [.6, .55, .75, .9],
        ]
        self.cache = {
            "id": "sample", "images": dict(self.row["images"]), "depth_encoding": "city_mm",
            "c_bbox": boxes[0], "query_info": {"scope": "single"},
            "candidates": [
                {"id": 11 + index, "role": "target", "bbox": box, "is_baseline": index == 0,
                 "sources": [], "depth": {"status": "pending"}}
                for index, box in enumerate(boxes)
            ],
        }
        self.pool = CandidatePool(self.cache, (120, 80))
        self.tools = self.make_tools(self.pool)

    def tearDown(self):
        self.temp.cleanup()

    def make_tools(self, pool, seed=2026):
        return ObjectTools(self.row, pool, self.root / "observations", tool_pixels=50000,
                           dino_model=(object(), object(), "cpu"),
                           sam_model=(object(), object(), "cpu"), seed=seed)

    def test_seeded_public_ids_keep_exact_bbox_and_append_without_renumbering(self):
        public = self.tools.public_candidates()
        self.assertEqual([item["id"] for item in public], [f"P{i}" for i in range(1, 8)])
        self.assertEqual(self.tools.public_candidates(), self.make_tools(self.pool).public_candidates())
        self.assertNotIn("KEEP", {item["id"] for item in public})

        baseline = next(item for item in public if item["bbox"] == self.cache["c_bbox"])
        self.assertEqual(self.tools.execute("finish", {"id": baseline["id"]})["data"]["bbox"],
                         self.cache["c_bbox"])
        old_mapping = {item["id"]: item["bbox"] for item in public}
        raw_id, is_new = self.pool.append_detection({
            "role": "target", "bbox": [.82, .2, .96, .7],
            "source": {"modality": "rgb", "score": .9, "category": "person"},
        })
        self.assertTrue(is_new)
        after = self.tools.public_candidates()
        self.assertEqual({key: value for key, value in old_mapping.items()},
                         {item["id"]: item["bbox"] for item in after if item["id"] in old_mapping})
        added = next(item for item in after if item["bbox"] == [.82, .2, .96, .7])
        self.assertEqual(added["id"], "P8")
        self.assertEqual(self.tools.execute("finish", {"id": added["id"]})["data"]["bbox"],
                         [.82, .2, .96, .7])

    def test_atlas_pages_are_unmarked_and_hold_at_most_six_candidates(self):
        observation = self.tools.atlas(modalities=["rgb"])
        self.assertEqual(observation["status"], "OK")
        self.assertEqual(len(observation["images"]), 2)
        for image_info in observation["images"]:
            self.assertLessEqual(len(image_info["candidate_ids"]), 6)
            self.assertEqual(image_info["candidate_ids"], [label["id"] for label in image_info["labels"]])
            self.assertEqual(set(image_info["candidate_ids"]), set(image_info["tile_positions"]))
            self.assertEqual(len(image_info["target_boxes"]), len(image_info["candidate_ids"]))
            with Image.open(image_info["path"]) as image:
                for target in image_info["target_boxes"]:
                    x1, y1, x2, y2 = target["xyxy"]
                    center = image.getpixel((round((x1 + x2) * image.width / 2),
                                             round((y1 + y2) * image.height / 2)))
                    self.assertEqual(center, (220, 20, 30))
            self.assertEqual(image_info["source_bbox"].keys(), set(image_info["candidate_ids"]))

    def test_inspect_supports_candidate_free_region_and_cross_modal_real_views(self):
        observation = self.tools.execute("inspect", {
            "region": "left", "modalities": ["rgb", "ir", "depth"],
        })
        self.assertEqual(observation["status"], "OK")
        self.assertEqual(len(observation["images"]), 3)
        self.assertTrue(all(image["target_boxes"] == [] for image in observation["images"]))
        self.assertTrue(all(image["crop_xyxy"] == [0, 0, 60, 80] for image in observation["images"]))
        for image_info in observation["images"]:
            with Image.open(image_info["path"]) as image:
                self.assertGreater(image.width, 0)
                self.assertGreater(image.height, 0)

        self.row["images"].pop("ir")
        missing = self.tools.inspect(candidate_ids=[self.tools.public_candidates()[0]["id"]],
                                     modalities=["rgb", "ir"])
        self.assertEqual(missing["status"], "UNKNOWN")
        self.assertEqual(missing["images"], [])

    def test_empty_full_search_runs_one_quadrant_pass_and_returns_plain_thumbnail(self):
        calls = 0

        def detect(image, modality, query_info, processor, model, device):
            nonlocal calls
            calls += 1
            if calls == 2:
                return [{"bbox": [.1, .2, .5, .8], "score": .93}]
            return []

        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", detect):
            observation = self.tools.execute("search", {
                "category": "person", "modality": "rgb", "role": "target", "region": "full",
            })
        self.assertEqual(observation["status"], "OK")
        self.assertEqual(self.tools.cost_counts["dino_calls"], 5)
        self.assertEqual([part["region"] for part in observation["data"]["regions"]],
                         ["full", "top_left", "top_right", "bottom_left", "bottom_right"])
        self.assertEqual([part["region_px"] for part in observation["data"]["regions"][1:]],
                         [[0, 0, 60, 40], [60, 0, 120, 40],
                          [0, 40, 60, 80], [60, 40, 120, 80]])
        self.assertEqual(len(observation["data"]["appended_ids"]), 1)
        candidate_id = observation["data"]["appended_ids"][0]
        self.assertTrue(candidate_id.startswith("P"))
        candidate = next(item for item in self.tools.public_candidates() if item["id"] == candidate_id)
        self.assertAlmostEqual(candidate["bbox"][0], .05)
        self.assertEqual(observation["images"][0]["candidate_ids"], [candidate_id])
        with Image.open(observation["images"][0]["path"]) as image:
            box = observation["images"][0]["target_boxes"][0]["xyxy"]
            center = image.getpixel((round((box[0] + box[2]) * image.width / 2),
                                     round((box[1] + box[3]) * image.height / 2)))
            self.assertEqual(center, (220, 20, 30))
        self.assertEqual(self.tools.execute("finish", {"id": candidate_id})["data"]["bbox"], candidate["bbox"])

    def test_depth_only_reports_cached_statistics_and_never_invents_pair_order(self):
        measurement = {
            "status": "reliable", "reason": "reliable",
            "core": {"valid_count": 100, "valid_ratio": .8,
                     "q1_m": 2.0, "median_m": 2.1, "q3_m": 2.2},
            "full": {"valid_count": 120, "valid_ratio": .9, "median_m": 2.15},
        }
        self.pool.get("KEEP")["depth"] = dict(measurement)
        self.pool.get(12)["depth"] = dict(measurement)
        first = next(item for item in self.tools.public_candidates()
                     if item["bbox"] == self.cache["c_bbox"])
        second = next(item for item in self.tools.public_candidates()
                      if item["bbox"] == self.cache["candidates"][1]["bbox"])
        observation = self.tools.depth([first["id"], second["id"]])
        self.assertEqual(observation["status"], "UNKNOWN")
        self.assertEqual(len(observation["data"]["measurements"]), 2)
        self.assertEqual(len(observation["data"]["pair_results"]), 1)
        self.assertNotIn("near_candidate_id", observation["data"]["pair_results"][0])
        self.assertNotIn("far_candidate_id", observation["data"]["pair_results"][0])
        for stats in observation["data"]["measurements"].values():
            self.assertEqual(stats["core"]["median_m"], 2.1)

    def test_depth_accepts_six_ids_and_keeps_supported_pairs_local(self):
        def measurement(median, status="reliable"):
            return {
                "status": status, "reason": status,
                "core": {"valid_count": 100, "valid_ratio": .9,
                         "q1_m": median - .1, "median_m": median, "q3_m": median + .1},
                "full": {"valid_count": 120, "valid_ratio": .95, "median_m": median},
            }

        ids = [item["id"] for item in self.tools.public_candidates()[:6]]
        raw_ids = [self.tools._raw(candidate_id) for candidate_id in ids]
        for index, raw_id in enumerate(raw_ids):
            self.pool.get(raw_id)["depth"] = measurement(2.0 + index * 1.5,
                                                          "unreliable" if index == 5 else "reliable")
        observation = self.tools.execute("depth", {"ids": ids})
        self.assertEqual(observation["status"], "UNKNOWN")
        self.assertEqual(len(observation["data"]["measurements"]), 6)
        self.assertEqual(len(observation["data"]["pair_results"]), 15)
        self.assertTrue(observation["data"]["supported_pairs"])
        self.assertTrue(any(pair["status"] == "UNKNOWN" for pair in observation["data"]["pair_results"]))
        self.assertEqual(self.tools.action_count, 1)
        self.assertEqual(self.tools.cost_counts["depth_generated"], 0)

    def test_execute_stops_after_six_high_level_actions(self):
        candidate_id = self.tools.public_candidates()[0]["id"]
        for _ in range(6):
            self.assertEqual(self.tools.execute("inspect", {"ids": [candidate_id]})["status"], "OK")
        self.assertEqual(self.tools.execute("finish", {"id": candidate_id})["status"], "OK")
        self.assertEqual(self.tools.execute("atlas", {})["status"], "LIMIT")


if __name__ == "__main__":
    unittest.main()
