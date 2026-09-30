"""Boundary checks for the GT-free City expansion selection."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.visual_agent import prepare_expansion as expansion


def raw_row(sample_id, group, query):
    return {
        "id": sample_id,
        "image": [f"visible/{group}.png", f"infrared/{group}.png", f"depth_rgb/{group}.png"],
        "conversations": [{"from": "human", "value":
                           f"Locate the object described by this query: {query}\nReturn only JSON"},
                          {"from": "gpt", "value": '{"bbox_2d":[1,2,3,4]}'}],
    }


class ExpansionTests(unittest.TestCase):
    def test_text_tags_do_not_inherit_reference_depth(self):
        self.assertNotIn("camera_distance_text", expansion.tags_for_query(
            "The person immediately to the right of the central background figure"))
        self.assertNotIn("camera_distance_text", expansion.tags_for_query(
            "The bin closest to the bench"))
        self.assertNotIn("camera_distance_text", expansion.tags_for_query(
            "The person closest to the background figure"))
        self.assertIn("camera_distance_text", expansion.tags_for_query(
            "The white bird in the foreground"))
        self.assertIn("camera_distance_text", expansion.tags_for_query(
            "The car farthest from the camera"))
        self.assertIn("camera_distance_text", expansion.tags_for_query(
            "The nearest car"))
        self.assertIn("ordinal_text", expansion.tags_for_query("The fourth robot"))

    def test_excludes_existing_groups_and_selects_second_round_without_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [raw_row("old", "old", "The red car"),
                    raw_row("a1", "a", "The car"),
                    raw_row("a2", "a", "The closest car"),
                    raw_row("b1", "b", "The fourth robot"),
                    raw_row("b2", "b", "The blue robot"),
                    raw_row("c1", "c", "The person wearing a coat"),
                    raw_row("c2", "c", "The person")]
            train = root / "train.json"
            train.write_text(json.dumps(rows), encoding="utf-8")
            existing = root / "existing.jsonl"
            existing.write_text(json.dumps({"id": "existing", "images":
                                            {"rgb": "/remote/visible/old.png"}}) + "\n", encoding="utf-8")
            args = SimpleNamespace(train=train, existing_manifests=[existing], data_root="/data",
                                   output_dir=root / "out", count=5, seed=2029)
            inventory = expansion.prepare_expansion(args)
            manifest = [json.loads(line) for line in (args.output_dir / "manifest.jsonl").read_text().splitlines()]
            tags = [json.loads(line) for line in (args.output_dir / "selection_tags.jsonl").read_text().splitlines()]
            self.assertEqual(len(manifest), 5)
            self.assertEqual([tag["selection_round"] for tag in tags], [1, 1, 1, 2, 2])
            self.assertEqual(inventory["selected_independent_image_groups"], 3)
            self.assertEqual(inventory["exclusion"]["existing_manifest_queries_removed_after_eligibility"], 1)
            self.assertTrue(all(set(row) == {"id", "query", "images", "depth_encoding"} for row in manifest))
            self.assertTrue(all("bbox" not in json.dumps(row) for row in manifest))
            self.assertTrue(all(not tag["image_evidence_verified"] for tag in tags))
            args.output_dir = root / "too_many"
            args.count = 7
            with self.assertRaisesRegex(ValueError, "at most two"):
                expansion.prepare_expansion(args)


if __name__ == "__main__":
    unittest.main()
