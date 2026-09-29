"""Tests for independent rejected-candidate repair review packages."""
from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.prepare_review_repairs import prepare_review_repairs
from tools.prepare_triground_abv_data import read_jsonl, write_jsonl


class PrepareReviewRepairsTest(unittest.TestCase):
    def make_source(self, root: Path) -> tuple[Path, list[dict]]:
        source = root / "source"
        (source / "data").mkdir(parents=True)
        (source / "review").mkdir()
        image = root / "toy.png"
        Image.new("RGB", (40, 30), "white").save(image)
        rows = []
        for index in range(3):
            task_id = f"toy:{index:02d}"
            rows.append({
                "task_id": task_id, "bundle_id": f"bundle:{task_id}",
                "scene_id": f"scene:{index}", "split": "diagnostic" if index == 1 else "train",
                "category": "reliability", "source": "fixture",
                "source_id": f"source:{index}", "target_object_id": f"object:{index}",
                "images": {"rgb": str(image)}, "bbox": [.125, .2, .6, .8],
                "proposed_query": f"Old query {index}.", "original_query": f"Original {index}.",
                "original_query_zh": f"原始问题{index}", "query_zh": f"旧中文{index}",
                "review_status": "draft_unapproved",
            })
        write_jsonl(source / "data/candidates.jsonl", rows)
        with (source / "review/decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task_id", "decision", "note"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"task_id": row["task_id"], "decision": "reject",
                                 "note": f"人工拒绝{row['task_id']}"})
        return source, rows

    def make_specs(self, root: Path, source_rows: list[dict], *, invalid_id: bool = False) -> Path:
        specs = []
        for index, row in enumerate(source_rows):
            spec = {
                "task_id": "unknown:99" if invalid_id and index == 0 else row["task_id"],
                "action": "repair" if index < 2 else "drop",
                "reason_zh": f"修订原因{index}",
                "visual_evidence": f"图中目标证据{index}",
                "inspected_paths": [row["images"]["rgb"]],
            }
            if index < 2:
                spec.update(query=f"Repaired query {index}.", query_zh=f"修订问题{index}。")
            if index == 1:
                spec["bbox"] = [.125001, .2, .7, .8]
            specs.append(spec)
        path = root / "specs.json"
        path.write_text(json.dumps(specs, ensure_ascii=False), encoding="utf-8")
        return path

    def test_builds_new_draft_package_without_migrating_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, original_rows = self.make_source(root)
            specs = self.make_specs(root, original_rows)
            before = {path.relative_to(source): path.read_bytes()
                      for path in source.rglob("*") if path.is_file()}
            output = root / "repaired"

            summary = prepare_review_repairs(source, [specs], output)

            self.assertEqual({path.relative_to(source): path.read_bytes()
                              for path in source.rglob("*") if path.is_file()}, before)
            rows = read_jsonl(output / "data/candidates.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["task_id"], "toy:00:repair1")
            self.assertEqual(rows[0]["bundle_id"], "bundle:toy:00:repair1")
            self.assertEqual(rows[0]["review_status"], "draft_unapproved")
            self.assertEqual(rows[0]["split"], original_rows[0]["split"])
            self.assertEqual(rows[0]["source"], original_rows[0]["source"])
            self.assertEqual(rows[0]["images"], original_rows[0]["images"])
            self.assertEqual(rows[0]["source_id"], original_rows[0]["source_id"])
            self.assertEqual(rows[0]["original_query"], original_rows[0]["original_query"])
            self.assertEqual(rows[0]["original_query_zh"], original_rows[0]["original_query_zh"])
            self.assertEqual(rows[0]["repair_info"]["user_note"], "人工拒绝toy:00")
            self.assertEqual(rows[0]["repair_info"]["previous_bbox"], original_rows[0]["bbox"])
            self.assertFalse(rows[0]["repair_info"]["box_changed"])

            changed = rows[1]
            self.assertEqual(changed["bbox"], [.125001, .2, .7, .8])
            self.assertEqual(changed["repair_info"]["previous_bbox"], original_rows[1]["bbox"])
            self.assertTrue(changed["repair_info"]["box_changed"])
            self.assertEqual(changed["target_object_id"], "object:1:repair1")
            self.assertEqual(changed["source_id"], original_rows[1]["source_id"])
            old_box_image = changed["repair_info"]["previous_answer_image"]
            self.assertTrue((output / "review" / old_box_image).is_file())
            html = (output / "review/index.html").read_text(encoding="utf-8")
            payload = json.loads(re.search(
                r'<script id="review-data" type="application/json">(.*?)</script>', html, re.S
            ).group(1))
            rendered = [task for bundle in payload["bundles"] for task in bundle["tasks"]
                        if task["task_id"] == changed["task_id"]][0]
            self.assertTrue((output / "review" / rendered["answer_image"]).is_file())

            self.assertEqual(summary["repair_candidates"], 2)
            self.assertEqual(summary["dropped_candidates"], 1)
            self.assertFalse(summary["release_created"])
            self.assertFalse((output / "review/decisions.csv").exists())
            self.assertFalse((output / "release.json").exists())
            with (output / "review/decisions_blank.csv").open(
                    encoding="utf-8-sig", newline="") as stream:
                blank_csv = list(csv.DictReader(stream))
            self.assertEqual(len(blank_csv), 2)
            self.assertTrue(all(not row["decision"] for row in blank_csv))
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(audit["source_rejected_ids"]), 3)
            self.assertFalse(audit["original_decisions_migrated"])
            self.assertEqual(len(audit["drop_records"]), 1)

    def test_unknown_spec_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, rows = self.make_source(root)
            specs = self.make_specs(root, rows, invalid_id=True)
            with self.assertRaisesRegex(ValueError, "unknown repair spec task ID"):
                prepare_review_repairs(source, [specs], root / "repaired")


if __name__ == "__main__":
    unittest.main()
