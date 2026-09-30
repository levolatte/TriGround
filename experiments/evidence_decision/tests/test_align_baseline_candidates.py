"""Frozen-C cache alignment preserves the existing non-baseline proposals."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.align_baseline_candidates import align_candidates
from experiments.evidence_decision.vision_tools import CandidatePool


class AlignBaselineCandidatesTest(unittest.TestCase):
    def test_updates_only_c_box_and_records_frozen_source(self):
        candidate_row = {
            "id": "sample",
            "c_bbox": [.1, .1, .3, .3],
            "candidates": [
                {"id": 7, "role": "target", "bbox": [.1, .1, .3, .3], "is_baseline": True,
                 "sources": [{"modality": "c", "bbox": [.1, .1, .3, .3], "score": None},
                             {"modality": "rgb", "bbox": [.11, .1, .31, .3], "score": .8}]},
                {"id": 3, "role": "reference", "bbox": [.5, .4, .7, .8], "is_baseline": False,
                 "sources": [{"modality": "ir", "bbox": [.5, .4, .7, .8], "score": .7}]},
            ],
        }
        original_nonbaseline = deepcopy(candidate_row["candidates"][1])
        fresh_box = [.2, .12, .42, .36]

        aligned = align_candidates(
            [candidate_row], [{"id": "sample", "bbox": fresh_box}],
            model="Qwen3-VL-8B-Instruct", adapter="checkpoint-500",
            prediction_source="reference_holdout/predictions.jsonl",
        )[0]

        self.assertEqual(aligned["c_bbox"], fresh_box)
        self.assertEqual(aligned["candidates"][0]["id"], 7)
        self.assertEqual(aligned["candidates"][0]["bbox"], fresh_box)
        self.assertEqual(aligned["candidates"][0]["sources"][0]["bbox"], fresh_box)
        self.assertEqual(aligned["candidates"][0]["sources"][1]["bbox"], [.11, .1, .31, .3])
        self.assertEqual(aligned["candidates"][1], original_nonbaseline)
        self.assertEqual(aligned["frozen_c_baseline_source"]["adapter"], "checkpoint-500")
        self.assertEqual(aligned["frozen_c_baseline_source"]["prediction_file"],
                         "reference_holdout/predictions.jsonl")
        pool = CandidatePool(aligned, (100, 100))
        self.assertEqual(pool.finish("KEEP"), fresh_box)

    def test_requires_exact_ids_unique_baseline_and_legal_boxes(self):
        row = {"id": "a", "c_bbox": [.1, .1, .2, .2], "candidates": [
            {"id": 4, "role": "target", "bbox": [.1, .1, .2, .2], "is_baseline": True,
             "sources": [{"modality": "c", "bbox": [.1, .1, .2, .2]}]},
        ]}
        kwargs = {"model": "model", "adapter": "adapter", "prediction_source": "baseline.jsonl"}
        with self.assertRaisesRegex(ValueError, "ID coverage differs"):
            align_candidates([row], [{"id": "b", "bbox": [.1, .1, .2, .2]}], **kwargs)
        with self.assertRaisesRegex(ValueError, "invalid normalized xyxy"):
            align_candidates([row], [{"id": "a", "bbox": [.4, .1, .2, .2]}], **kwargs)
        duplicate_baseline = [{"id": "a", "bbox": [.1, .1, .2, .2]},
                              {"id": "a", "bbox": [.1, .1, .2, .2]}]
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            align_candidates([row], duplicate_baseline, **kwargs)


if __name__ == "__main__":
    unittest.main()
