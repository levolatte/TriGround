from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.depth_relation_tasks import build_depth_tasks, region_depth_evidence


class DepthRelationTasksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        raw = np.zeros((20, 60), dtype=np.uint16)
        raw[:, :20] = 1000
        raw[:, 20:40] = 2000
        raw[:, 40:] = 3000
        # A few invalid and saturated pixels must remain visible in evidence.
        raw[8:10, 8:10] = 0
        raw[10:12, 8:10] = 20_000
        self.raw = raw
        self.raw_path = root / "depth_raw.png"
        self.visual_path = root / "depth_visual.png"
        Image.fromarray(raw).save(self.raw_path)
        display = np.zeros((20, 60), dtype=np.uint8)
        display[:, :20] = 20
        display[:, 20:40] = 120
        display[:, 40:] = 220
        Image.fromarray(display).save(self.visual_path)
        self.scene_id = "city:toy_scene"
        self.scene = {"scene_id": self.scene_id, "split": "train", "location_group": "toy",
                      "images": {"rgb": "rgb.png", "infrared": "ir.png",
                                 "depth": str(self.visual_path), "depth_raw": str(self.raw_path)}}
        boxes = {"a": [0.001, 0.05, 0.332, 0.95],
                 "b": [0.334, 0.05, 0.665, 0.95],
                 "c": [0.668, 0.05, 0.999, 0.95]}
        self.records = {key: {"id": key, "visible": "visible/toy_scene.png",
                              "bbox": box, "query": f"Original query for {key}"}
                        for key, box in boxes.items()}
        self.regions = {"a": [0.08, 0.25, 0.25, 0.75],
                        "b": [0.41, 0.25, 0.59, 0.75],
                        "c": [0.75, 0.25, 0.92, 0.75]}

    def spec(self, tasks: list[dict], **extra: object) -> dict:
        return {"spec_id": "toy", "scene_id": self.scene_id, "split": "train",
                "scope_description": "Only the three marked cars on the road",
                "visual_review": {"object_binding": "human review pending"},
                "evidence_regions": self.regions, "tasks": tasks, **extra}

    @staticmethod
    def task(key: str, target: str, kind: str, competitors: list[str],
             reference: str | None = None) -> dict:
        relation = {"kind": kind, "competitors": competitors}
        if reference:
            relation["reference_id"] = reference
        return {"key": key, "target_id": target, "relation": relation,
                "query": f"Among the marked cars, locate the {key} car.",
                "query_zh": f"在标出的汽车中找出{key}。"}

    def build(self, spec: dict) -> list[dict]:
        return build_depth_tasks([spec], self.records, {self.scene_id: self.scene})

    def test_bounded_pair_keeps_original_gt_and_does_not_leak_binding(self) -> None:
        near = self.task("near", "a", "nearer", ["c"])
        far = self.task("far", "c", "farther", ["a"])
        rows = self.build(self.spec([near, far], predecessor_task_ids=["old:near", "old:far"]))
        self.assertEqual([row["task_id"] for row in rows],
                         ["city:toy_scene:depthv2:toy:near", "city:toy_scene:depthv2:toy:far"])
        self.assertEqual({row["bundle_id"] for row in rows}, {"city:toy_scene:depthv2:toy"})
        self.assertTrue(all(row["pair_required"] for row in rows))
        self.assertEqual(rows[0]["bbox"], self.records["a"]["bbox"])
        self.assertIsNot(rows[0]["bbox"], self.records["a"]["bbox"])
        self.assertEqual(rows[0]["proposed_query"], near["query"])
        self.assertEqual(rows[0]["query_zh"], near["query_zh"])
        self.assertNotIn("a", rows[0]["proposed_query"].split(" "))
        self.assertEqual(rows[0]["original_query"], self.records["a"]["query"])
        self.assertEqual(rows[0]["target_object_id"], "city:toy_scene:a")
        self.assertEqual(rows[0]["review_status"], "draft_unapproved")
        self.assertEqual(rows[0]["construction_version"], "depth_relations_v2")
        self.assertEqual(rows[0]["relation_type"], "bounded_pair")
        self.assertEqual(rows[0]["relation"]["kind"], "nearer")
        self.assertEqual(rows[0]["predecessor_task_ids"], ["old:near", "old:far"])
        a = rows[0]["depth_evidence"]["by_source_id"]["a"]
        self.assertGreater(a["raw_zero_px"], 0)
        self.assertGreater(a["raw_saturated_px"], 0)
        self.assertEqual(a["raw_median_mm"], 1000)
        self.assertEqual(a["raw_valid_iqr_mm"], 0)
        self.assertEqual(a["visual_gray_median"], 20)
        self.assertEqual(a["region_source"], "manual_subject_interior")

    def test_middle_and_reference_require_distinct_bound_objects(self) -> None:
        middle = self.task("middle", "b", "middle", ["a", "c"])
        reference = self.task("near_ref", "a", "nearer", ["c"], reference="b")
        rows = self.build(self.spec([middle, reference], pair_required=False))
        self.assertEqual([row["relation_type"] for row in rows], ["middle", "reference"])
        self.assertEqual([row["relation"]["kind"] for row in rows], ["middle", "nearer"])
        self.assertFalse(any(row["pair_required"] for row in rows))
        self.assertEqual(rows[1]["relation"]["reference_id"], "b")
        self.assertEqual(set(rows[1]["depth_evidence"]["by_source_id"]), {"a", "b", "c"})

    def test_diagnostic_split_and_farther_reference(self) -> None:
        self.scene["split"] = "diagnostic"
        spec = self.spec([self.task("far_ref", "c", "farther", ["a"], reference="b")])
        spec["split"] = "diagnostic"
        row = self.build(spec)[0]
        self.assertEqual(row["category"], "diag_depth")
        self.assertEqual(row["split"], "diagnostic")
        self.assertEqual(row["relation_type"], "reference")
        self.assertFalse(row["pair_required"])

    def test_missing_manual_region_is_explicitly_low_confidence(self) -> None:
        spec = self.spec([self.task("middle", "b", "middle", ["a", "c"])])
        spec["evidence_regions"].pop("b")
        row = self.build(spec)[0]
        self.assertFalse(row["depth_evidence"]["all_manual_regions"])
        self.assertEqual(row["depth_evidence"]["by_source_id"]["b"]["region_source"],
                         "bbox_core_low_confidence")
        self.assertFalse(row["depth_evidence"]["visible_depth_order_reviewed"])

    def test_invalid_or_saturated_depth_fails_with_spec_id(self) -> None:
        spec = self.spec([self.task("near", "a", "nearer", ["c"])])
        raw = self.raw.copy()
        raw[:, 40:] = 0
        Image.fromarray(raw).save(self.raw_path)
        with self.assertRaisesRegex(ValueError, "toy: c has no usable"):
            self.build(spec)
        raw[:, 40:] = 20_000
        raw[8, 50] = 3000
        Image.fromarray(raw).save(self.raw_path)
        with self.assertRaisesRegex(ValueError, "toy: c has no usable"):
            self.build(spec)

    def test_wrong_order_or_cross_scene_binding_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "toy: raw depth contradicts"):
            self.build(self.spec([self.task("wrong", "c", "nearer", ["a"])]))
        self.records["c"]["visible"] = "visible/other_scene.png"
        with self.assertRaisesRegex(ValueError, "toy: c belongs to another scene"):
            self.build(self.spec([self.task("near", "a", "nearer", ["c"])]))

    def test_unknown_object_fails_without_creating_relation(self) -> None:
        with self.assertRaisesRegex(ValueError, "toy: unknown source object missing"):
            self.build(self.spec([self.task("near", "a", "nearer", ["missing"])]))

    def test_region_statistics_keep_saturation_distinct_from_zero(self) -> None:
        visual = np.full(self.raw.shape, 80, dtype=np.uint8)
        result = region_depth_evidence(self.raw, visual, [0, 0, 1, 1], "manual_subject_interior")
        self.assertEqual(result["raw_zero_px"], 4)
        self.assertEqual(result["raw_saturated_px"], 4)
        self.assertEqual(result["raw_valid_px"], self.raw.size - 8)
        self.assertEqual(result["visual_gray_iqr"], 0)


if __name__ == "__main__":
    unittest.main()
