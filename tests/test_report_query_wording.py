from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools import report_query_wording


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


class QueryWordingReportTests(unittest.TestCase):
    def test_partial_full_denominator_and_paired_rescues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            tmp_path = Path(temporary_dir)
            image_path = tmp_path / "scene.png"
            Image.new("RGB", (40, 30), (90, 100, 110)).save(image_path)
            pairs = [
                {"task_id": f"q{index}", "long_query": f"long {index}",
                 "long_query_zh": f"长{index}", "short_query": f"short {index}",
                 "short_query_zh": f"短{index}"}
                for index in range(3)
            ]
            candidates = {
                pair["task_id"]: {
                    "task_id": pair["task_id"], "bbox": [0.25, 0.2, 0.75, 0.8],
                    "images": {"rgb": str(image_path)}, "scene_id": f"scene-{index // 2}",
                    "relation_type": "reference" if index == 0 else "bounded_pair",
                }
                for index, pair in enumerate(pairs)
            }
            predictions = {
                "q0::long": {"id": "q0::long", "prediction": [0.0, 0.0, 0.2, 0.2],
                              "raw_text": "wrong long"},
                "q0::short": {"id": "q0::short", "prediction": [0.25, 0.2, 0.75, 0.8],
                               "raw_text": "exact short"},
                "q1::long": {"id": "q1::long", "prediction": [0.25, 0.2, 0.75, 0.8],
                              "raw_text": "exact long"},
                "q1::short": {"id": "q1::short", "prediction": [0.0, 0.0, 0.2, 0.2],
                               "raw_text": "wrong short"},
                "q2::long": {"id": "q2::long", "prediction": [0.8, 0.2, 0.1, 0.7],
                              "raw_text": "invalid box"},
            }
            pairs_path = tmp_path / "pairs.json"
            pairs_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
            candidate_path = tmp_path / "candidates.jsonl"
            _jsonl(candidate_path, list(candidates.values()))
            prediction_path = tmp_path / "predictions.jsonl"
            _jsonl(prediction_path, list(predictions.values()))

            loaded_pairs = report_query_wording._load_pairs(pairs_path)
            loaded_candidates = report_query_wording._index_candidates(candidate_path, loaded_pairs)
            loaded_predictions, missing = report_query_wording._prediction_rows(
                prediction_path, loaded_pairs
            )
            invalid = sorted(sample_id for sample_id, row in loaded_predictions.items()
                             if row.get("_failure") is not None)
            report = report_query_wording.build_report(
                loaded_pairs, loaded_candidates, loaded_predictions, missing, invalid
            )

            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["task_count"], 3)
            self.assertEqual(report["overall"]["long"]["denominator"], 3)
            self.assertEqual(report["overall"]["long"]["hits_0.5"], 1)
            self.assertEqual(report["overall"]["short"]["denominator"], 3)
            self.assertEqual(report["overall"]["short"]["hits_0.5"], 1)
            self.assertEqual(report["missing_prediction_ids"], ["q2::short"])
            self.assertEqual(report["invalid_prediction_ids"], ["q2::long"])
            self.assertEqual(report["paired"]["short_rescued_ids_at_0.5"], ["q0"])
            self.assertEqual(report["paired"]["short_harmed_ids_at_0.5"], ["q1"])
            self.assertEqual(report["paired"]["per_task_iou_wins"], {
                "long": ["q1"], "short": ["q0"], "tie": ["q2"]
            })

            output_dir = tmp_path / "report"
            report_query_wording.write_outputs(report, output_dir)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["partial"])
            report_text = (output_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("不据此判断哪种措辞更好", report_text)
            with (output_dir / "paired.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)
            self.assertTrue((output_dir / "index.html").is_file())
            self.assertEqual(len(list((output_dir / "images").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
