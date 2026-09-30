"""Boundary checks for frozen offline prediction selection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.visual_agent.subset_predictions import select_predictions


class SubsetPredictionsTest(unittest.TestCase):
    def test_manifest_order_and_extra_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, predictions = root / "manifest.jsonl", root / "predictions.jsonl"
            manifest.write_text('{"id":"b"}\n{"id":"a"}\n', encoding="utf-8")
            predictions.write_text('{"id":"a","bbox":[0,0,1,1]}\n'
                                   '{"id":"c","bbox":[0,0,1,1]}\n'
                                   '{"id":"b","bbox":[0,0,1,1]}\n', encoding="utf-8")
            self.assertEqual([row["id"] for row in select_predictions(manifest, predictions)], ["b", "a"])

    def test_duplicate_or_missing_prediction_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, predictions = root / "manifest.jsonl", root / "predictions.jsonl"
            manifest.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
            predictions.write_text('{"id":"a"}\n{"id":"a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate prediction ID a"):
                select_predictions(manifest, predictions)
            predictions.write_text('{"id":"a"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 1 frozen IDs"):
                select_predictions(manifest, predictions)


if __name__ == "__main__":
    unittest.main()
