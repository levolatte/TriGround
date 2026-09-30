"""Manifest-restricted inputs retain all score IDs without inventing boxes."""

from pathlib import Path
import sys
import json
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.prepare_evaluation_subset import subset_predictions
from experiments.evidence_decision.evaluate import load_run, score_subset


class PrepareEvaluationSubsetTest(unittest.TestCase):
    def test_filters_extras_and_marks_missing_predictions_with_null_bbox(self):
        manifest = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        predictions = [
            {"id": "c", "bbox": [.1, .2, .3, .4]},
            {"id": "outside", "bbox": [.2, .3, .4, .5]},
            {"id": "a", "bbox": None},
        ]

        rows, summary = subset_predictions(manifest, predictions)

        self.assertEqual([row["id"] for row in rows], ["a", "b", "c"])
        self.assertIsNone(rows[1]["bbox"])
        self.assertTrue(rows[1]["prediction_missing"])
        self.assertEqual(rows[2]["bbox"], [.1, .2, .3, .4])
        self.assertEqual(summary, {"expected": 3, "present": 2, "missing": 1,
                                  "ignored_outside_manifest": 1})
        with tempfile.TemporaryDirectory() as directory:
            prediction_path = Path(directory) / "subset.jsonl"
            prediction_path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                                       encoding="utf-8")
            loaded = load_run(prediction_path, {"a", "b", "c"})
            ground_truth = {sample_id: {"bbox": [.1, .2, .3, .4]} for sample_id in ("a", "b", "c")}
            score = score_subset(loaded, ground_truth, ["a", "b", "c"])
            self.assertEqual(score["n"], 3)
            self.assertEqual(score["valid_final_bbox"], 1)
            self.assertEqual(score["hits_0.5"], 1)

    def test_duplicate_manifest_or_prediction_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "manifest contains duplicate IDs"):
            subset_predictions([{"id": "a"}, {"id": "a"}], [])
        with self.assertRaisesRegex(ValueError, "duplicate ID a"):
            subset_predictions([{"id": "a"}], [{"id": "a"}, {"id": "a"}])


if __name__ == "__main__":
    unittest.main()
