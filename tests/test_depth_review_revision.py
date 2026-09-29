"""Small review-package regression tests for the Depth v2 revision."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.prepare_triground_abv_data import _native_task, read_jsonl, revise_depth, write_jsonl


def candidate(task_id: str, category: str, scene: str, source_id: str,
              image: Path, *, relation_type: str | None = None) -> dict:
    row = {"task_id": task_id, "bundle_id": f"bundle:{task_id}", "scene_id": scene,
           "split": "diagnostic" if category.startswith("diag_") else "train",
           "category": category, "source": "city" if category in {"depth_relation", "diag_depth"} else "fixture",
           "images": {"rgb": str(image), "infrared": str(image), "depth": str(image)},
           "bbox": [.1, .2, .7, .8], "source_id": source_id,
           "target_object_id": f"target:{task_id}", "depth_policy": "millimeter",
           "proposed_query": f"Find {task_id}.", "query_zh": f"寻找{task_id}。",
           "review_status": "draft_unapproved", "fixture_note": {"nested": [task_id]}}
    if relation_type:
        row.update(construction_version="depth_relations_v2", pair_required=False,
                   relation_type=relation_type, relation={"kind": "nearer"})
    return row


class DepthReviewRevisionTest(unittest.TestCase):
    def test_local_revision_keeps_gt_splits_and_translations(self) -> None:
        from tools.prepare_triground_abv_data import OUT, DEPTH_V2_OUT, city_raw_labels
        path = DEPTH_V2_OUT / "data/candidates.jsonl"
        if not path.is_file():
            self.skipTest("local Depth v2 review package is not built")
        old = read_jsonl(OUT / "data/candidates.jsonl")
        rows = read_jsonl(path)
        categories = {"depth_relation", "diag_depth"}
        self.assertEqual([r for r in rows if r["category"] not in categories],
                         [r for r in old if r["category"] not in categories])
        scenes = {r["scene_id"]: r["split"] for r in old}
        originals = city_raw_labels()
        depth = [r for r in rows if r["category"] in categories]
        self.assertEqual({r["relation_type"] for r in depth}, {"middle", "reference", "bounded_pair"})
        for row in depth:
            self.assertEqual(row["bbox"], originals[row["source_id"]]["bbox"])
            self.assertEqual(row["split"], scenes[row["scene_id"]])
            self.assertTrue(row["depth_evidence"]["all_manual_regions"])
            self.assertNotRegex(row["proposed_query"].lower(), r"\b(nearest|farthest)\b")
            self.assertEqual(row["review_status"], "draft_unapproved")
        page = (DEPTH_V2_OUT / "review/index.html").read_text(encoding="utf-8")
        payload = json.loads(re.search(r'<script id="review-data" type="application/json">(.*?)</script>', page, re.S).group(1))
        for row in rows:
            for key in ("proposed_query", "original_query"):
                if row.get(key):
                    self.assertRegex(payload["translations"][row[key].strip()], r"[\u4e00-\u9fff]")

    def test_revision_preserves_source_and_unchanged_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (16, 12), "white").save(image)
            source, revised = root / "original", root / "revised"
            (source / "data").mkdir(parents=True)
            (source / "review").mkdir()
            old_depth = [candidate("old-depth-train", "depth_relation", "city:train", "src:train", image),
                         candidate("old-depth-diag", "diag_depth", "city:diag", "src:diag", image)]
            unchanged = [candidate("robo-stays", "robo_competition", "robo:1", "src:robo", image),
                         candidate("ir-stays", "ir_complement", "rgbt:1", "src:ir", image)]
            original_rows = old_depth + unchanged
            write_jsonl(source / "data/candidates.jsonl", original_rows)
            (source / "data/ancestor_overlap.json").write_text('{"kept": true}\n', encoding="utf-8")
            (source / "review/decisions.csv").write_text("original human decisions\n", encoding="utf-8")
            (source / "review/index.html").write_text("original atlas\n", encoding="utf-8")
            before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            new_depth = [candidate("new-train", "depth_relation", "city:train", "src:train", image,
                                   relation_type="bounded_pair"),
                         candidate("new-diag", "diag_depth", "city:diag", "src:diag", image,
                                   relation_type="middle")]
            specs = root / "depth_specs.json"
            specs.write_text("{}\n", encoding="utf-8")
            with patch("tools.prepare_triground_abv_data.city_candidates", return_value=new_depth):
                summary = revise_depth(source, revised, specs)

            self.assertEqual({path.relative_to(source): path.read_bytes()
                              for path in source.rglob("*") if path.is_file()}, before)
            rows = read_jsonl(revised / "data/candidates.jsonl")
            self.assertEqual(rows[2:], unchanged)
            self.assertEqual({row["task_id"] for row in rows[:2]}, {"new-train", "new-diag"})
            self.assertTrue({row["task_id"] for row in rows[:2]}.isdisjoint(
                {row["task_id"] for row in old_depth}))
            self.assertEqual(rows[0]["predecessor_task_ids"], ["old-depth-train"])
            self.assertEqual(rows[1]["predecessor_task_ids"], ["old-depth-diag"])
            self.assertEqual(summary["unchanged_tasks"], 2)
            self.assertEqual(summary["new_depth_tasks"], 2)
            self.assertEqual((revised / "data/depth_specs.json").read_text(encoding="utf-8"), "{}\n")
            self.assertEqual((revised / "data/ancestor_overlap.json").read_bytes(),
                             before[Path("data/ancestor_overlap.json")])

            revision = json.loads((revised / "data/review_revision.json").read_text(encoding="utf-8"))
            source_bundles = list(dict.fromkeys(row["bundle_id"] for row in original_rows))
            expected_key = (f"aic-abv-review-v2:{len(source_bundles)}:"
                            f"{source_bundles[0]}:{source_bundles[-1]}")
            self.assertEqual(revision["source_package_key"], expected_key)
            self.assertEqual(revision["unchanged_task_ids"], [row["task_id"] for row in unchanged])
            self.assertEqual(revision["retired_depth_task_ids"], [row["task_id"] for row in old_depth])
            self.assertEqual(revision["new_depth_task_ids"], [row["task_id"] for row in new_depth])
            html = (revised / "review/index.html").read_text(encoding="utf-8")
            embedded = re.search(r'<script id="review-data" type="application/json">(.*?)</script>',
                                 html, re.S).group(1)
            payload = json.loads(embedded)
            self.assertEqual(payload["revision"], revision)
            self.assertEqual(payload["translations"][new_depth[0]["proposed_query"]],
                             new_depth[0]["query_zh"])
            js = (revised / "review/review.js").read_text(encoding="utf-8")
            self.assertIn("aic-abv-review-v2:${bundles.length}", js)
            self.assertIn("data.revision?.source_package_key", js)

            approved = {**rows[0], "query": rows[0]["proposed_query"]}
            sample = _native_task(approved, "sample:new-train", (), revised / "assets")
            self.assertEqual(sample["source_task_id"], "new-train")
            self.assertTrue({"predecessor_task_ids", "relation_type", "relation", "pair_required",
                             "query_zh", "review_status", "fixture_note"}.isdisjoint(sample))
            self.assertNotIn("predecessor_task_ids", json.dumps(sample))

    def test_revision_refuses_source_and_existing_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            (source / "data").mkdir(parents=True)
            (source / "data/candidates.jsonl").write_text("", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                revise_depth(source, source)
            existing = root / "existing"
            (existing / "review").mkdir(parents=True)
            atlas = existing / "review/index.html"
            atlas.write_text("preexisting review\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                revise_depth(source, existing)
            self.assertEqual(atlas.read_text(encoding="utf-8"), "preexisting review\n")


if __name__ == "__main__":
    unittest.main()
