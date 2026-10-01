"""Small boundary checks for offline alignment to real demonstration branches."""

import json
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.visual_agent.export_aligned_demonstrations import align_rows


def _row(sample_id, decision_index, action, *, final=False, label_source="scripted"):
    return {
        "id": sample_id, "decision_index": decision_index,
        "image_group": f"{sample_id}.png", "messages": [f"original input {decision_index}"],
        "target_action": json.dumps(action, separators=(",", ":")),
        "original_action": action, "label_source": label_source,
        "is_final": final, "origin": "scripted", "original_raw_output": None,
    }


def _branch(sample_id="sample", candidate_id="7", *, action=None, status="OK",
            observation_path="target_ir.png", terminal_path="target_ir.png"):
    action = action or {"action": "inspect_regions", "candidate_ids": [candidate_id],
                        "modalities": ["ir"], "view": "single"}
    images = [] if observation_path is None else [
        {"path": observation_path, "candidate_ids": [candidate_id], "modality": "ir"}]
    return {
        "id": sample_id, "candidate_id": candidate_id,
        "prefix_events": [{"action": {"action": "measure_depth"}}],
        "event": {"action": action, "messages": ["branch action input"],
                  "observation": {"status": status, "images": images}},
        "terminal_messages": [{"role": "user", "content": [
            {"type": "image", "image": terminal_path}] if terminal_path else [
            {"type": "text", "text": "no target image"}]}],
    }


class ExportAlignedDemonstrationsTest(unittest.TestCase):
    def test_keep_preserves_original_rows_and_manual_exclusions_remove_positive_row(self):
        keep_reviewed = [
            _row("keep-reviewed", 0, {"action": "inspect_regions", "candidate_ids": ["KEEP"]}),
            _row("keep-reviewed", 1, {"action": "finish", "candidate_id": "KEEP"},
                 final=True, label_source="offline_gt_keep"),
        ]
        keep_unchanged = [
            _row("keep-unchanged", 0, {"action": "measure_depth", "candidate_ids": ["KEEP"]}),
            _row("keep-unchanged", 1, {"action": "finish", "candidate_id": "KEEP"},
                 final=True, label_source="offline_gt_keep"),
        ]
        selected, revised = align_rows(keep_reviewed + keep_unchanged, {},
                                       {("keep-reviewed", 0)})
        self.assertEqual([row["id"] for row in selected],
                         ["keep-reviewed", "keep-unchanged", "keep-unchanged"])
        self.assertEqual(selected[0]["target_action"], '{"action":"finish","candidate_id":"KEEP"}')
        self.assertEqual(selected[1:], keep_unchanged)
        self.assertEqual(revised, [])

    def test_nonkeep_replaces_old_inspections_with_real_target_branch_and_terminal(self):
        rows = [
            _row("sample", 0, {"action": "inspect_regions", "candidate_ids": ["KEEP"]}),
            _row("sample", 1, {"action": "finish", "candidate_id": "7"}, final=True,
                 label_source="offline_gt_candidate"),
        ]
        branch = _branch()
        selected, revised = align_rows(rows, {("sample", "7"): branch}, set())
        self.assertEqual(len(selected), 2)
        self.assertEqual([row["target_action"] for row in selected], [
            json.dumps(branch["event"]["action"], separators=(",", ":")),
            rows[-1]["target_action"],
        ])
        self.assertEqual([row["decision_index"] for row in selected], [1, 2])
        self.assertEqual(selected[0]["messages"], branch["event"]["messages"])
        self.assertEqual(selected[1]["messages"], branch["terminal_messages"])
        self.assertTrue(selected[1]["is_final"])
        self.assertTrue(all(row["origin"] == "scripted_offline_branch_selection" for row in selected))
        self.assertTrue(all(row["original_raw_output"] is None for row in selected))
        self.assertEqual(revised, [{"id": "sample", "candidate_id": "7", "prior_decisions": 2,
                                    "new_decisions": 2, "target_observation_images": ["target_ir.png"]}])

    def test_invalid_or_unseen_branch_evidence_is_rejected(self):
        rows = [_row("sample", 0, {"action": "finish", "candidate_id": "7"}, final=True)]
        branches = [
            (_branch(action={"action": "inspect_regions", "candidate_ids": ["8"]}),
             "does not inspect its final candidate"),
            (_branch(status="UNKNOWN"), "does not inspect its final candidate"),
            (_branch(terminal_path="other.png"), "final-target images missing"),
        ]
        for branch, message in branches:
            with self.subTest(message=message, branch=branch):
                with self.assertRaisesRegex(ValueError, message):
                    align_rows(rows, {("sample", "7"): branch}, set())


if __name__ == "__main__":
    unittest.main()
