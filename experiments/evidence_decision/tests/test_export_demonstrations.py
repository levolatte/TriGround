"""Boundary checks for GT-audited scripted demonstration export."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.evidence_decision.export_demonstrations import export


KEEP_BOX = [0.05, 0.05, 0.15, 0.15]
TARGET_A = [0.5, 0.5, 0.7, 0.7]
TARGET_B = [0.75, 0.5, 0.9, 0.7]


def _candidate(candidate_id, bbox, role="target"):
    return {"id": candidate_id, "bbox": bbox, "role": role}


def _manifest_row(sample_id):
    return {"id": sample_id, "query": f"query for {sample_id}",
            "images": {"rgb": f"{sample_id}.png"}}


def _trace(sample_id, *, final_candidates=None, events=None, origin="scripted"):
    pool = [_candidate("KEEP", KEEP_BOX), _candidate(7, TARGET_A), _candidate(8, TARGET_B)]
    return {
        "id": sample_id, "query": f"query for {sample_id}",
        "manifest_row": _manifest_row(sample_id), "trajectory_origin": origin,
        "final_status": "OBSERVATION_COMPLETE", "events": events or [],
        "terminal_messages": [{"role": "user", "content": [
            {"type": "text", "text": f"actual final decision input for {sample_id}"}]}],
        "initial_candidates": pool,
        "final_candidates": final_candidates if final_candidates is not None else pool,
    }


def _event(action, status, visible_text):
    return {
        "action": action,
        "observation": {"status": status, "text": f"actual tool return: {status}"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": visible_text}]}],
    }


class ExportDemonstrationsTest(unittest.TestCase):
    def _inputs(self, root, train_ids, traces, gt):
        root.mkdir(parents=True, exist_ok=True)
        manifests = {}
        cohorts = {"train": train_ids, "holdout": ["holdout"], "debug": ["debug"]}
        for cohort, ids in cohorts.items():
            path = root / f"{cohort}.jsonl"
            path.write_text("".join(json.dumps(_manifest_row(sample_id)) + "\n" for sample_id in ids),
                            encoding="utf-8")
            manifests[cohort] = path
        trace_path = root / "traces.jsonl"
        trace_path.write_text("".join(json.dumps(trace) + "\n" for trace in traces), encoding="utf-8")
        gt_path = root / "gt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        return SimpleNamespace(traces=trace_path, train_manifest=manifests["train"],
                               holdout_manifest=manifests["holdout"], debug_manifest=manifests["debug"],
                               gt=gt_path, output_dir=root / "out")

    def test_real_ok_unknown_tools_survive_error_is_excluded_and_gt_only_labels_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actions = [
                {"action": "measure_depth", "candidate_ids": ["KEEP", "7"]},
                {"action": "inspect_regions", "candidate_ids": ["KEEP", "7"],
                 "modalities": ["ir"], "view": "pair"},
                {"action": "search_candidates", "category": "person", "region": "full",
                 "modality": "ir", "role": "target"},
            ]
            trace = _trace("train", events=[
                _event(actions[0], "UNKNOWN", "depth unavailable before measurement"),
                _event(actions[1], "OK", "actual UNKNOWN depth observation; inspect IR next"),
                _event(actions[2], "ERROR", "failed search must not become supervision"),
            ])
            trace_path = root / "traces.jsonl"
            args = self._inputs(root, ["train"], [trace], {"train": {"bbox": TARGET_A}})
            before = args.traces.read_bytes()
            export(args)
            self.assertEqual(args.traces.read_bytes(), before)

            rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
            self.assertEqual([row["target_action"] for row in rows[:2]],
                             [json.dumps(action, separators=(",", ":")) for action in actions[:2]])
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["target_action"], '{"action":"finish","candidate_id":7}')
            self.assertTrue(rows[-1]["is_final"])
            self.assertEqual([row["origin"] for row in rows], ["scripted"] * 3)
            self.assertTrue(all(row["original_raw_output"] is None for row in rows))
            self.assertEqual(rows[0]["label_source"], "scripted_query_and_actual_observation_policy")
            self.assertEqual(rows[-1]["label_source"], "offline_gt_candidate")
            self.assertIn("actual UNKNOWN depth observation", rows[1]["messages"][0]["content"][0]["text"])

    def test_uncovered_gt_is_skipped_even_if_a_reference_candidate_overlaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            covered = _trace("covered")
            uncovered = _trace("uncovered", final_candidates=[
                _candidate("KEEP", KEEP_BOX), _candidate(9, TARGET_A, role="reference")])
            args = self._inputs(root, ["covered", "uncovered"], [covered, uncovered], {
                "covered": {"bbox": TARGET_A}, "uncovered": {"bbox": TARGET_A},
            })
            inventory = export(args)
            rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
            self.assertEqual(inventory["uncovered_trajectories"], 1)
            self.assertEqual({row["id"] for row in rows}, {"covered"})
            self.assertEqual(rows[0]["target_action"], '{"action":"finish","candidate_id":7}')

    def test_keep_terminal_supervision_is_capped_at_half(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = [_trace("keep1"), _trace("keep2"), _trace("target")]
            gt = {"keep1": {"bbox": KEEP_BOX}, "keep2": {"bbox": KEEP_BOX},
                  "target": {"bbox": TARGET_A}}
            args = self._inputs(root, ["keep1", "keep2", "target"], traces, gt)
            inventory = export(args)
            rows = [json.loads(line) for line in (root / "out" / "train.jsonl").read_text().splitlines()]
            keep_rows = [row for row in rows if row["target_action"] == '{"action":"finish","candidate_id":"KEEP"}']
            self.assertEqual(inventory["keep_trajectories"], 1)
            self.assertEqual(inventory["training_trajectories"], 2)
            self.assertLessEqual(len(keep_rows) / len(rows), 0.5)

    def test_nontraining_and_duplicate_trace_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root, ["train"], [_trace("holdout")],
                                {"holdout": {"bbox": TARGET_A}})
            with self.assertRaisesRegex(ValueError, "frozen training start"):
                export(args)

            args = self._inputs(root, ["train"], [_trace("train"), _trace("train")],
                                {"train": {"bbox": TARGET_A}})
            with self.assertRaisesRegex(ValueError, "frozen training start"):
                export(args)

    def test_query_must_match_frozen_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field in ("query", "manifest_query"):
                trace = _trace("train")
                if field == "query":
                    trace["query"] = "a different question on the same image"
                else:
                    trace["manifest_row"]["query"] = "a different question on the same image"
                args = self._inputs(root / field, ["train"], [trace], {"train": {"bbox": TARGET_A}})
                with self.assertRaisesRegex(ValueError, "training query changed"):
                    export(args)

    def test_autonomous_model_trace_cannot_be_exported_as_scripted_demonstration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root, ["train"], [_trace("train", origin="autonomous")],
                                {"train": {"bbox": TARGET_A}})
            with self.assertRaisesRegex(ValueError, "explicitly scripted"):
                export(args)


if __name__ == "__main__":
    unittest.main()
