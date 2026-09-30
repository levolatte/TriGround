"""Offline slices and evidence-recovery summaries stay outside score files."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "code"))
from experiments.evidence_decision import diagnose_capabilities as diagnostic


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CapabilityDiagnosticTests(unittest.TestCase):
    def test_slices_bbox_rescue_status_recovery_and_probe_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = {"a": {"visible": "scene-a.png", "bbox": [.1, .1, .3, .3]},
                  "b": {"visible": "scene-b.png", "bbox": [.6, .6, .8, .8]}}
            gt_path = root / "gt.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            eval_manifest = root / "eval.jsonl"
            write_jsonl(eval_manifest, [{"id": "a"}, {"id": "b"}])
            candidates = [{"id": "P1", "role": "target", "finish_eligible": True,
                           "coordinate_frame": "rgb", "bbox": [.1, .1, .3, .3]}]
            predictions = root / "predictions.jsonl"
            write_jsonl(predictions, [
                {"id": "a", "final_status": "FINISHED", "finish_source": "candidate",
                 "selected_id": "P1", "selected_public_id": "P1", "bbox": [.1, .1, .3, .3],
                 "initial_bbox": [.5, .5, .7, .7], "initial_candidates": [],
                 "final_candidates": candidates},
                {"id": "b", "final_status": "FINISHED", "finish_source": "predicted_bbox",
                 "selected_id": None, "selected_public_id": None, "bbox": [.6, .6, .8, .8],
                 "initial_bbox": [.1, .1, .3, .3], "initial_candidates": [], "final_candidates": []},
            ])
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({"gt": str(gt_path), "manifest": str(eval_manifest),
                                              "denominator": 2, "runs": {"trained": {"all": {}}},
                                              "run_sources": {"trained": str(predictions)}}), encoding="utf-8")
            slices = root / "slices.jsonl"
            write_jsonl(slices, [{"id": "a", "capability_slices": ["condition_binding"]},
                                 {"id": "b", "capability_slices": ["missing_pool", "bbox_finish"]}])
            trace = root / "trace.json"
            trace.write_text(json.dumps({"id": "b", "events": [
                {"action": {"name": "depth", "arguments": {"ids": ["P1", "P2"]}},
                 "observation": {"status": "UNKNOWN", "data": {}}},
                {"action": {"name": "inspect", "arguments": {"ids": ["P1"], "modalities": ["rgb"]}},
                 "observation": {"status": "OK", "data": {"no_new_evidence": True}}},
                {"action": {"name": "finish", "arguments": {"bbox": [.6, .6, .8, .8]}},
                 "observation": {"status": "OK", "data": {}}},
            ]}), encoding="utf-8")
            paired = root / "paired.jsonl"
            write_jsonl(paired, [{"full": {"predicted_action": {"name": "finish", "arguments": {"id": "P1"}}},
                                 "masked": {"predicted_action": {"name": "inspect", "arguments": {}}}}])
            output = root / "diagnostic"
            result = diagnostic.diagnose(evaluation, slices, output, trace, "trained", paired)
            self.assertEqual(result["capability_slices"]["bbox_finish"]["trained"]["hits_0.5"], 1)
            self.assertEqual(result["capability_slices"]["missing_pool"]["trained"]
                             ["search_and_bbox_coverage"]["predicted_bbox_unique_rescues_without_final_candidate_coverage"], 1)
            self.assertEqual(result["evidence_transitions"]["trigger_next_action_counts"]["UNKNOWN"],
                             {"inspect": 1})
            self.assertEqual(result["evidence_transitions"]["trigger_next_action_counts"]["no_new_evidence"],
                             {"finish": 1})
            self.assertEqual(result["paired_image_mask_sensitivity"]["action_name_changed"], 1)
            self.assertTrue(result["diagnostic_only"])
            self.assertTrue((output / "capability_diagnostics.md").exists())

    def test_error_recovery_counts_next_tool_and_final_accuracy(self):
        trace = {"id": "recover", "events": [
            {"action": {"name": "inspect", "arguments": {"ids": ["P1"], "modalities": ["depth"]}},
             "observation": {"status": "ERROR", "data": {}}},
            {"action": {"name": "search", "arguments": {"region": "left", "modality": "rgb"}},
             "observation": {"status": "OK", "data": {}}},
            {"action": {"name": "finish", "arguments": {"id": "P2"}},
             "observation": {"status": "OK", "data": {}}},
        ]}
        result = diagnostic.evidence_transitions(
            [trace], {"recover": {"bbox": [.1, .1, .3, .3]}},
            {"recover": {"bbox": [.1, .1, .3, .3]}},
        )
        self.assertEqual(result["trigger_next_action_counts"]["ERROR"], {"search": 1})
        self.assertEqual(result["affected_episode_final_accuracy"]["ERROR"],
                         {"n": 1, "hits_0.5": 1, "acc_0.5": 1.0})


if __name__ == "__main__":
    unittest.main()
