"""CPU checks for deliberate protocol probes and their real tool feedback."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.evidence_decision.collect_protocol_feedback import collect_one, main
from experiments.evidence_decision.vision_tools import CandidatePool


def _candidate(candidate_id, box, baseline=False):
    return {"id": candidate_id, "role": "target", "bbox": box,
            "is_baseline": baseline, "sources": []}


def _sample(root, sample_id="sample", target_count=4):
    images = {}
    for modality in ("rgb", "ir"):
        path = root / f"{sample_id}_{modality}.png"
        Image.new("RGB", (80, 48), "gray").save(path)
        images[modality] = str(path.resolve())
    row = {"id": sample_id, "query": f"query for {sample_id}", "images": images,
           "depth_encoding": "unknown"}
    candidates = [_candidate(1, [.05, .05, .2, .3], baseline=True)]
    for candidate_id in range(2, target_count + 1):
        left = .2 + .15 * (candidate_id - 2)
        candidates.append(_candidate(candidate_id, [left, .1, left + .08, .3]))
    source = {"id": sample_id, "images": images, "c_bbox": candidates[0]["bbox"],
              "candidates": candidates, "depth_encoding": "unknown"}
    pool = CandidatePool(source, (80, 48))
    old_inspect = {"step": 0, "action": {"action": "inspect_regions", "candidate_ids": ["KEEP"]},
                   "observation": {"status": "OK", "text": "old irrelevant zoom", "images": []},
                   "candidates_before": pool.public_candidates(),
                   "candidates_after": pool.public_candidates()}
    depth = {"step": 1, "action": {"action": "measure_depth", "candidate_ids": ["KEEP", "2"]},
             "executed_action": {"action": "measure_depth", "candidate_ids": ["KEEP", "2"]},
             "observation": {"status": "UNKNOWN", "text": "real prior depth result", "images": []},
             "candidates_before": pool.public_candidates(),
             "candidates_after": pool.public_candidates()}
    trace = {"id": sample_id, "query": row["query"], "manifest_row": row,
             "trajectory_origin": "scripted", "final_status": "OBSERVATION_COMPLETE",
             "initial_messages": [{"role": "user", "content": [
                 {"type": "text", "text": "full query and global views"}]}],
             "events": [old_inspect, depth], "final_pool_snapshot": pool.snapshot(),
             "final_candidates": pool.public_candidates()}
    return row, trace


class CollectProtocolFeedbackTest(unittest.TestCase):
    def test_cli_records_real_three_id_error_without_pool_change_or_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, trace = _sample(root)
            trace_dir = root / "observations_v2" / "traces" / "sample"
            trace_dir.mkdir(parents=True)
            trace_path = trace_dir / "trace.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            original_trace = trace_path.read_bytes()
            for name, manifest_row in (
                    ("train.jsonl", row),
                    ("holdout.jsonl", {"id": "held", "query": "held",
                                        "images": {"rgb": "held.png"}}),
                    ("debug.jsonl", {"id": "debug", "query": "debug",
                                     "images": {"rgb": "debug.png"}})):
                (root / name).write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            output = root / "out"
            summary = main([
                "--traces", str(trace_dir.parent),
                "--train-manifest", str(root / "train.jsonl"),
                "--holdout-manifest", str(root / "holdout.jsonl"),
                "--debug-manifest", str(root / "debug.jsonl"),
                "--output-dir", str(output),
            ])
            record = json.loads((output / "protocol_feedback.jsonl").read_text().strip())
            self.assertEqual(summary["protocol_errors"], 1)
            self.assertFalse(summary["gt_used"])
            self.assertEqual(record["origin"], "scripted_protocol_probe")
            self.assertTrue(record["intentional_invalid_action"])
            self.assertFalse(record["supervised"])
            self.assertEqual(record["input_action"], {
                "action": "inspect_regions", "candidate_ids": ["KEEP", "2", "3"],
                "modalities": ["rgb", "ir"], "view": "cross"})
            self.assertEqual(record["observation"]["status"], "ERROR")
            self.assertEqual(record["observation"]["data"]["kind"], "protocol")
            self.assertEqual(record["observation"]["images"], [])
            self.assertEqual(record["pool_before"], record["pool_after"])
            self.assertEqual(record["pool_before"], record["event"]["candidates_before"])
            self.assertEqual(record["prefix_events"][-1], record["event"])
            self.assertEqual(record["prefix_events"][-1]["observation"], record["observation"])
            self.assertEqual(record["trace_path"], str(trace_path.resolve()))
            self.assertIn("real prior depth result", json.dumps(record["event"]["messages"]))
            self.assertNotIn("old irrelevant zoom", json.dumps(record["event"]["messages"]))
            self.assertEqual(trace_path.read_bytes(), original_trace)
            tool_output = output / "tool_output" / "sample"
            self.assertFalse(any(path.is_file() for path in tool_output.rglob("*")))

    def test_fewer_than_three_targets_is_recorded_as_skip_without_tool_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, trace = _sample(root, target_count=2)
            output = root / "tool_output" / "sample"
            result = collect_one(trace, row, output, root / "source_trace.json")
            self.assertEqual(result["status"], "SKIP")
            self.assertFalse(result["intentional_invalid_action"])
            self.assertIsNone(result["input_action"])
            self.assertIsNone(result["observation"])
            self.assertIsNone(result["event"])
            self.assertEqual(result["pool_before"], result["pool_after"])
            self.assertFalse(output.exists())

    def test_cli_summarizes_mixed_protocol_error_and_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error_row, error_trace = _sample(root, "error", target_count=4)
            skip_row, skip_trace = _sample(root, "skip", target_count=2)
            traces_root = root / "traces"
            for trace in (error_trace, skip_trace):
                sample_dir = traces_root / trace["id"]
                sample_dir.mkdir(parents=True)
                (sample_dir / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
            train_rows = (error_row, skip_row)
            (root / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
            for name, sample_id in (("holdout.jsonl", "held"), ("debug.jsonl", "debug")):
                row = {"id": sample_id, "query": sample_id, "images": {"rgb": f"{sample_id}.png"}}
                (root / name).write_text(json.dumps(row) + "\n", encoding="utf-8")

            output = root / "out"
            summary = main([
                "--traces", str(traces_root),
                "--train-manifest", str(root / "train.jsonl"),
                "--holdout-manifest", str(root / "holdout.jsonl"),
                "--debug-manifest", str(root / "debug.jsonl"),
                "--output-dir", str(output),
            ])
            records = [json.loads(line) for line in
                       (output / "protocol_feedback.jsonl").read_text().splitlines()]
            self.assertEqual(summary["input_traces"], 2)
            self.assertEqual(summary["protocol_errors"], 1)
            self.assertEqual(summary["skipped_too_few_targets"], 1)
            self.assertEqual({record["id"]: record["status"] for record in records},
                             {"error": "ERROR", "skip": "SKIP"})


if __name__ == "__main__":
    unittest.main()
