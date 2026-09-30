"""Checks for GT-free recovery from real native protocol-error prefixes."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.visual_agent.collect_prefix_recovery import _choose_prefix, collect_one, main
from experiments.visual_agent.vision_tools import CandidatePool


def sample(root, sample_id="sample", rgb=None):
    images = {}
    for modality in ("rgb", "ir"):
        path = rgb if modality == "rgb" and rgb else root / f"{sample_id}_{modality}.png"
        if not path.exists():
            Image.new("RGB", (80, 48), "gray").save(path)
        images[modality] = str(path.resolve())
    row = {"id": sample_id, "query": "the second car", "images": images}
    candidates = [{"id": key, "role": "target", "bbox": [.05 + key * .18, .1, .17 + key * .18, .5],
                   "is_baseline": key == 1, "sources": []} for key in (1, 2, 3)]
    cache = {"id": sample_id, "images": images, "c_bbox": candidates[0]["bbox"],
             "candidates": candidates}
    pool = CandidatePool(cache, (80, 48))
    before = pool.public_candidates()
    bad = {"action": "inspect_regions", "candidate_ids": ["bogus", "KEEP", "2", "3"],
           "modalities": ["rgb", "ir"], "view": "cross"}
    event = {"step": 0, "action": bad, "executed_action": bad,
             "observation": {"status": "ERROR", "text": "invalid inspect view, IDs, or modality list",
                             "images": [], "data": {"kind": "protocol"}},
             "candidates_before": before, "candidates_after": before, "tool_seconds": .01}
    trace = {"id": sample_id, "query": row["query"], "manifest_row": row,
             "memory": "latest", "profile": {"name": "sft"},
             "initial_messages": [{"role": "system", "content": "Return one legal action."},
                                  {"role": "user", "content": [{"type": "text", "text": row["query"]}]}],
             "initial_candidates": before, "initial_pool_snapshot": pool.snapshot(),
             "events": [event]}
    return row, cache, trace


class PrefixRecoveryTests(unittest.TestCase):
    def test_real_recovery_keeps_prefix_and_exposes_new_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, cache, trace = sample(root)
            result = collect_one(trace, row, cache, root / "out", root / "native.json")
            self.assertEqual(result["status"], "OK")
            self.assertEqual(result["prefix_events"], trace["events"])
            self.assertEqual(result["recovery_action"], {
                "action": "inspect_regions", "candidate_ids": ["KEEP", "2"],
                "modalities": ["rgb", "ir"], "view": "cross"})
            self.assertFalse(result["gt_used"])
            self.assertFalse(result["autonomous_success"])
            self.assertFalse(result["processor_consumed"])
            pictures = result["event"]["observation"]["images"]
            self.assertEqual(len(pictures), 4)
            self.assertTrue(all(Path(picture["path"]).is_file() for picture in pictures))
            terminal = {block["image"] for message in result["terminal_messages"]
                        if isinstance(message["content"], list)
                        for block in message["content"] if block["type"] == "image"}
            self.assertTrue({picture["path"] for picture in pictures} <= terminal)
            self.assertIn("invalid inspect", json.dumps(result["event"]["messages"]))

    def test_only_first_error_and_tool_budget_are_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            row, cache, trace = sample(Path(directory))
            first = dict(trace["events"][0])
            first["action"] = {"action": "compare"}
            first["executed_action"] = None
            trace["events"].insert(0, first)
            self.assertEqual(_choose_prefix(trace)[1], "prior_protocol_error")
            trace["events"] = [trace["events"][-1]]
            ok = {"action": {"action": "measure_depth", "candidate_ids": ["KEEP"]},
                  "executed_action": {"action": "measure_depth", "candidate_ids": ["KEEP"]},
                  "observation": {"status": "UNKNOWN", "data": {}, "images": []},
                  "candidates_before": trace["initial_candidates"],
                  "candidates_after": trace["initial_candidates"], "tool_seconds": .01}
            trace["events"] = [dict(ok) for _ in range(6)] + trace["events"]
            self.assertEqual(_choose_prefix(trace)[1], "evidence_budget")

    def test_changed_pool_and_single_modality_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row, cache, trace = sample(root)
            trace["events"][0]["action"]["modalities"] = ["rgb"]
            choice, reason = _choose_prefix(trace)
            self.assertIsNone(reason)
            self.assertEqual(choice[1]["view"], "pair")
            trace["events"][0]["candidates_after"] = []
            with self.assertRaisesRegex(ValueError, "candidate pool changed"):
                collect_one(trace, row, cache, root / "out", root / "native.json")
            search = {"action": {"action": "search_candidates"},
                      "executed_action": {"action": "search_candidates"},
                      "observation": {"status": "OK", "data": {"appended_ids": ["4"]}},
                      "candidates_before": trace["initial_candidates"],
                      "candidates_after": [*trace["initial_candidates"], {"id": "4"}],
                      "tool_seconds": .01}
            trace["events"] = [search, trace["events"][0]]
            self.assertEqual(_choose_prefix(trace)[1], "search_changed_pool")

    def test_cli_full_cohort_same_image_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_row, first_cache, first_trace = sample(root, "one")
            second_row, second_cache, second_trace = sample(root, "two",
                                                           Path(first_row["images"]["rgb"]))
            traces_root = root / "native"
            for trace in (first_trace, second_trace):
                path = traces_root / trace["id"]
                path.mkdir(parents=True)
                (path / "trace.json").write_text(json.dumps(trace), encoding="utf-8")
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in (first_row, second_row)), encoding="utf-8")
            (root / "cache.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in (first_cache, second_cache)), encoding="utf-8")
            original = (traces_root / "one" / "trace.json").read_bytes()
            inventory = main(["--traces", str(traces_root), "--manifest", str(root / "manifest.jsonl"),
                              "--candidate-cache", str(root / "cache.jsonl"),
                              "--output-dir", str(root / "out"), "--limit", "1"])
            self.assertEqual(inventory["input_traces"], 2)
            self.assertEqual(inventory["processed_starts"], 1)
            self.assertEqual(inventory["selected_recoveries"], 1)
            self.assertEqual(len((root / "out" / "records.jsonl").read_text().splitlines()), 1)
            self.assertEqual((traces_root / "one" / "trace.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
