"""The paired probe must preserve a real UNKNOWN prefix and expose new crops."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments.evidence_decision import evaluate
from experiments.evidence_decision.model import InputBudgetExceeded
from experiments.evidence_decision.probe_unknown_evidence import (
    freeze_selection, prefix_choice, prediction_row, probe_branch,
)
from experiments.evidence_decision.vision_tools import CandidatePool


class FakeBackend:
    def __init__(self):
        self.messages = []
        self.raw_output = '{"action":"finish","candidate_id":"7"}'
        self.budget_error = False

    def begin_sample(self):
        pass

    def peak_memory(self):
        return 1234

    def generate(self, messages, max_new_tokens, profile, remaining_visual):
        self.messages.append(messages)
        if self.budget_error:
            raise InputBudgetExceeded("context_token_budget",
                                      {"input_tokens": 4090, "visual_tokens": 700,
                                       "preprocess_seconds": .2})
        images = [block["image"] for message in messages
                  for block in (message.get("content") if isinstance(message.get("content"), list) else [])
                  if block.get("type") == "image"]
        return {"raw_output": self.raw_output,
                "usage": {"input_tokens": 200, "visual_tokens": 80,
                          "output_tokens": 12,
                          "image_geometry": [{"path": path} for path in images]}}


class UnknownEvidenceProbeTests(unittest.TestCase):
    def test_first_inspect_error_requires_no_prior_local_observation(self):
        from experiments.evidence_decision.tests.test_collect_prefix_recovery import sample
        with tempfile.TemporaryDirectory() as directory:
            row, cache, trace = sample(Path(directory))
            choice, reason = prefix_choice(trace, "first_inspect_error")
            self.assertIsNone(reason)
            self.assertEqual(choice[1], ["KEEP", "2"])
            prior = {**trace["events"][0], "action": {"action": "inspect_regions"},
                     "executed_action": {"action": "inspect_regions"},
                     "observation": {"status": "OK", "images": [{"path": "prior.png"}]}}
            trace["events"].insert(0, prior)
            self.assertEqual(prefix_choice(trace, "first_inspect_error")[1],
                             "prior_local_visual_observation")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        paths = {}
        for modality in ("rgb", "ir"):
            path = self.root / f"{modality}.png"
            Image.new("RGB", (128, 128), "white").save(path)
            paths[modality] = str(path)
        self.row = {"id": "a", "query": "The nearest white car", "images": paths,
                    "depth_encoding": "city_mm"}
        candidate_row = {"id": "a", "images": paths, "depth_encoding": "city_mm",
                         "c_bbox": [.1, .1, .3, .3], "candidates": [
                             {"id": 0, "bbox": [.1, .1, .3, .3], "role": "target",
                              "is_baseline": True, "sources": []},
                             {"id": 7, "bbox": [.6, .6, .8, .8], "role": "target",
                              "is_baseline": False, "sources": []}]}
        self.pool = CandidatePool(candidate_row, (128, 128))
        candidates = self.pool.public_candidates()
        self.event = {"step": 0, "action": {"action": "measure_depth",
                       "candidate_ids": ["KEEP", "7"]},
                      "executed_action": {"action": "measure_depth",
                                          "candidate_ids": ["KEEP", "7"]},
                      "observation": {"status": "UNKNOWN", "text": "Depth unavailable",
                                      "images": [], "data": {}},
                      "usage": {"output_tokens": 20, "visual_tokens": 100},
                      "tool_seconds": .1, "candidates_before": candidates,
                      "candidates_after": candidates}
        self.trace = {"id": "a", "query": self.row["query"], "manifest_row": self.row,
                      "initial_messages": [{"role": "system", "content": "system"},
                           {"role": "user", "content": [{"type": "text", "text":
                             "Full query: The nearest white car", "latest_memory_text":
                             "Full query: The nearest white car; KEEP fixed"},
                             {"type": "image", "image": paths["rgb"], "max_pixels": 1024}]}],
                      "initial_candidates": candidates,
                      "initial_pool_snapshot": self.pool.snapshot(),
                      "events": [self.event], "memory": "latest", "profile": {"name": "sft"}}

    def test_first_unknown_and_seeded_group_selection(self):
        trace_b = {**self.trace, "id": "b", "manifest_row":
                   {**self.row, "id": "b"}, "events": []}
        choice, reason = prefix_choice(self.trace)
        self.assertIsNone(reason)
        self.assertEqual(choice[1], ["KEEP", "7"])
        self.assertEqual(prefix_choice(trace_b)[1], "no_measure_depth_unknown")
        bad = {**self.trace, "events": [{**self.event,
               "action": {"action": "measure_depth", "candidate_ids": ["0"]},
               "executed_action": {"action": "measure_depth", "candidate_ids": ["0"]}}]}
        self.assertEqual(prefix_choice(bad)[1], "invalid_depth_candidate_ids")
        selection = freeze_selection({"a": (self.trace, self.root / "a/trace.json"),
                                      "b": (trace_b, self.root / "b/trace.json")},
                                     {"a": self.row, "b": trace_b["manifest_row"]}, 1, 2030)
        self.assertEqual(selection["selected_ids"], ["a"])
        self.assertEqual(selection["exclusion_counts"]["no_measure_depth_unknown"], 1)

    def test_real_ir_rgb_crop_reaches_finish_messages_and_limit_keeps_row(self):
        backend = FakeBackend()
        records = {branch: probe_branch(self.trace, self.row, self.pool, [self.event],
                                        ["KEEP", "7"], branch, self.root / branch, backend)
                   for branch in ("direct", "ir", "rgb")}
        self.assertEqual([records[b]["status"] for b in ("direct", "ir", "rgb")],
                         ["FINISHED"] * 3)
        self.assertIsNone(records["direct"]["observation"])
        prediction_path = self.root / "predictions.jsonl"
        prediction_path.write_text(json.dumps(prediction_row(records["direct"], self.trace)) + "\n",
                                   encoding="utf-8")
        prediction = evaluate.load_run(prediction_path, {"a"})["a"]
        self.assertEqual(prediction["bbox"], [.6, .6, .8, .8])
        self.assertEqual(prediction["metrics"]["model_calls"], 1)
        self.assertEqual(prediction["metrics"]["output_tokens"], 12)
        for branch in ("ir", "rgb"):
            observation = records[branch]["observation"]
            self.assertEqual(observation["status"], "OK")
            self.assertTrue(all(image["modality"] == branch for image in observation["images"]))
            shown = {block["image"] for message in records[branch]["messages"]
                     for block in (message.get("content") if isinstance(message.get("content"), list) else [])
                     if block.get("type") == "image"}
            self.assertTrue({image["path"] for image in observation["images"]} <= shown)
            self.assertEqual(records[branch]["bbox"], [.6, .6, .8, .8])
            self.assertTrue(records[branch]["new_images_processed"])
            self.assertTrue(records[branch]["model_generation_executed"])
            self.assertEqual(records[branch]["peak_memory_bytes"], 1234)
        exhausted = {**self.event, "usage": {"output_tokens": 1600, "visual_tokens": 100}}
        limited = probe_branch(self.trace, self.row, self.pool, [exhausted],
                               ["KEEP", "7"], "ir", self.root / "limited", backend)
        self.assertEqual((limited["status"], limited["limit_reason"]),
                         ("LIMIT", "finish_output_reservation"))
        self.assertIsNone(limited["observation"])
        self.assertEqual(prediction_row(limited, self.trace)["metrics"]["model_calls"], 0)
        crowded = [self.event] * 6
        slot_limit = probe_branch(self.trace, self.row, self.pool, crowded,
                                  ["KEEP", "7"], "rgb", self.root / "slot_limit", backend)
        self.assertEqual((slot_limit["status"], slot_limit["limit_reason"]),
                         ("LIMIT", "decision_slots"))
        prior = {**self.event, "action": {"action": "inspect_regions"},
                 "executed_action": {"action": "inspect_regions"},
                 "observation": {"status": "OK", "text": "prior IR", "data": {}, "images": [
                     {"path": self.row["images"]["ir"], "modality": "ir", "candidate_ids": ["7"]}]}}
        repeated = probe_branch(self.trace, self.row, self.pool, [prior, self.event],
                                ["KEEP", "7"], "ir", self.root / "repeated", backend)
        self.assertEqual(repeated["repeated_observation_ids"], ["7"])
        self.assertTrue(repeated["latest_window_replaces_prior_tool_images"])
        backend.raw_output = '{"action":"finish","candidate_id":"999"}'
        invalid = probe_branch(self.trace, self.row, self.pool, [self.event],
                               ["KEEP", "7"], "direct", self.root / "invalid", backend)
        self.assertEqual(invalid["status"], "INVALID_FINAL_ACTION")
        self.assertIsNone(invalid["bbox"])
        self.assertIsNone(prediction_row(invalid, self.trace)["selected_id"])
        backend.budget_error = True
        over_budget = probe_branch(self.trace, self.row, self.pool, [self.event],
                                   ["KEEP", "7"], "direct", self.root / "over_budget", backend)
        costs = prediction_row(over_budget, self.trace)["metrics"]
        self.assertEqual((over_budget["status"], costs["model_calls"],
                          costs["rejected_input_tokens"]), ("LIMIT", 0, 4090))


if __name__ == "__main__":
    unittest.main()
