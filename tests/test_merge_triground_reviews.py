from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.merge_triground_reviews import M3FD_ROLE_CORRECTION, merge_review_packages
from tools.prepare_triground_abv_data import DECISION_FIELDS, accepted_candidates, read_jsonl


def _candidate(task_id: str, *, scene: str, split: str = "train",
               category: str = "ir_complement", bbox: list[float] | None = None) -> dict:
    return {"task_id": task_id, "bundle_id": task_id, "scene_id": scene,
            "split": split, "category": category, "source": "rgbt_groundbench_m3fd",
            "source_id": task_id.split(":")[0], "original_query": "The original query",
            "proposed_query": "The reviewed query", "bbox": bbox or [.125, .2, .375, .5],
            "target_object_id": "object-1", "images": {"rgb": "rgb.png", "infrared": "ir.png"},
            "depth_policy": "visual", "review_status": "draft_unapproved"}


def _package(path: Path, rows: list[dict], *, decisions: dict[str, dict] | None = None) -> None:
    (path / "data").mkdir(parents=True)
    (path / "review").mkdir()
    (path / "data/candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with (path / "review/decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        for row in rows:
            decision = {key: row.get(key, "") for key in DECISION_FIELDS}
            decision.update({"decision": "approve", "review_mode": "quick"})
            decision.update((decisions or {}).get(row["task_id"], {}))
            writer.writerow(decision)


class MergeReviewsTest(unittest.TestCase):
    def test_preserves_repaired_box_query_and_provenance_and_corrects_m3fd_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "original"
            repair = root / "repair"
            role_row = _candidate(M3FD_ROLE_CORRECTION, scene="m3fd:01169",
                                  category="reliability")
            role_row["reliability_variants"] = ["normal", "ir_missing"]
            repaired = _candidate("repaired:one", scene="m3fd:01200",
                                  bbox=[.123456789, .2, .376543211, .5])
            repaired["repair_info"] = {"previous_bbox": [.1, .2, .3, .5],
                                       "box_changed": True}
            _package(source, [role_row], decisions={M3FD_ROLE_CORRECTION: {
                "note": "IR makes this person visible"}})
            _package(repair, [repaired], decisions={"repaired:one": {
                "approved_query": "The revised English query"}})
            original_csv = (source / "review/decisions.csv").read_bytes()
            output = root / "merged"
            with patch("tools.merge_triground_reviews.ancestor_overlap", return_value={"tested": True}):
                result = merge_review_packages([source, repair], output)

            self.assertEqual(result["approved_total"], 2)
            merged = {row["task_id"]: row for row in read_jsonl(output / "data/candidates.jsonl")}
            corrected = merged["rgbt_m3fd_train_002847:ir_complement"]
            self.assertEqual(corrected["category"], "ir_complement")
            self.assertNotIn("reliability_variants", corrected)
            self.assertEqual(corrected["predecessor_task_ids"], [M3FD_ROLE_CORRECTION])
            self.assertEqual(corrected["approval_origin"]["source_task_id"], M3FD_ROLE_CORRECTION)
            self.assertEqual(merged["repaired:one"]["bbox"], repaired["bbox"])
            self.assertEqual(merged["repaired:one"]["repair_info"], repaired["repair_info"])
            self.assertEqual(merged["repaired:one"]["query"], "The revised English query")
            released = accepted_candidates(list(merged.values()), output / "review/decisions.csv")
            self.assertEqual({row["query"] for row in released},
                             {"The reviewed query", "The revised English query"})
            self.assertEqual((source / "review/decisions.csv").read_bytes(), original_csv)
            self.assertEqual((output / "audit/source_packages/00_original/decisions.csv").read_bytes(),
                             original_csv)

    def test_rejects_source_csv_mismatch_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            row = _candidate("one", scene="scene-a")
            _package(source, [row], decisions={"one": {"proposed_query": "wrong target"}})
            output = root / "merged"
            with self.assertRaisesRegex(ValueError, "source proposed_query mismatch"):
                merge_review_packages([source], output)
            self.assertFalse(output.exists())

    def test_rejects_scene_and_sequence_split_collisions(self) -> None:
        for collision in ("scene", "sequence"):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                a = _candidate("one", scene="scene-a")
                b = _candidate("two", scene="scene-a" if collision == "scene" else "scene-b",
                               split="diagnostic", category="diag_ir")
                if collision == "sequence":
                    a["sequence_group"] = b["sequence_group"] = "m3fd:011"
                _package(root / "a", [a])
                _package(root / "b", [b])
                with self.assertRaisesRegex(ValueError, "collision"):
                    merge_review_packages([root / "a", root / "b"], root / "merged")
                self.assertFalse((root / "merged").exists())

    def test_rejected_half_of_required_pair_cannot_be_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = _candidate("pair:near", scene="scene-a", category="depth_relation")
            second = _candidate("pair:far", scene="scene-a", category="depth_relation")
            for row in (first, second):
                row.update(bundle_id="pair", construction_version="depth_relations_v2",
                           pair_required=True)
            _package(root / "source", [first, second], decisions={"pair:far": {"decision": "reject"}})
            with self.assertRaisesRegex(ValueError, "approved required pair was dropped"):
                merge_review_packages([root / "source"], root / "merged")
            self.assertFalse((root / "merged").exists())


if __name__ == "__main__":
    unittest.main()
