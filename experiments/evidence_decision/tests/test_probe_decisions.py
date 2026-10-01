"""Classification and full-denominator checks without loading Qwen or tools."""

import time
import unittest

from experiments.evidence_decision.model import InputBudgetExceeded
from experiments.evidence_decision.probe_decisions import (
    prepare_evidence_pairs, probe_evidence_pairs, probe_rows, summarize,
)


class FakeBackend:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.seen = []

    def begin_sample(self):
        pass

    def generate(self, messages, max_new_tokens, profile, remaining_visual):
        self.seen.append(messages)
        assert max_new_tokens == 128 and profile.name == "sft"
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return {"raw_output": outcome, "usage": {"input_tokens": 100, "output_tokens": 8}}


class FakeNativeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []
        self.load_seconds = 0

    def begin_sample(self):
        pass

    def generate(self, messages, max_new_tokens, profile, remaining_visual, *, tools=None):
        self.inputs.append((messages, max_new_tokens, profile.context_tokens, tools))
        return {"raw_output": next(self.outputs), "usage": {"input_tokens": 100, "output_tokens": 8}}


class ProbeDecisionTests(unittest.TestCase):
    def test_exact_action_and_finish_id_have_full_denominators(self):
        rows = [
            {"id": "a", "origin": "scripted_branch", "label_source": "offline_final",
             "messages": [{"role": "user", "content": "first"}],
             "target_action": '{"action":"finish","candidate_id":"KEEP"}'},
            {"id": "b", "origin": "scripted_branch", "label_source": "offline_tool",
             "messages": [{"role": "user", "content": "second"}],
             "target_action": '{"action":"inspect_regions","candidate_ids":["KEEP"]}'},
            {"id": "c", "origin": "scripted", "label_source": "offline_final",
             "messages": [{"role": "user", "content": "third"}],
             "target_action": '{"action":"finish","candidate_id":"2"}'},
        ]
        backend = FakeBackend([
            '{"candidate_id":"KEEP","action":"finish"}',
            '{"action":"inspect_regions","candidate_ids":["KEEP"],"evidence_note":"Check appearance."}',
            InputBudgetExceeded("context_token_budget", {"input_tokens": 4100}),
        ])
        records = list(probe_rows(rows, backend))
        self.assertEqual(backend.seen, [row["messages"] for row in rows])
        self.assertEqual([record["status"] for record in records],
                         ["GENERATED", "GENERATED", "INPUT_BUDGET_EXCEEDED"])
        self.assertEqual([record["exact_canonical_match"] for record in records],
                         [True, False, False])
        self.assertEqual([record["action_arguments_match"] for record in records],
                         [True, True, False])
        self.assertEqual([record["action_name_match"] for record in records],
                         [True, True, False])
        self.assertEqual([record["finish_id_match"] for record in records],
                         [True, None, False])
        summary = summarize(records)
        self.assertEqual(summary["all"]["rows"], 3)
        self.assertEqual(summary["all"]["action_arguments_matches"], 2)
        self.assertEqual(summary["all"]["action_arguments_rate"], 2 / 3)
        self.assertEqual(summary["all"]["finish_targets"], 2)
        self.assertEqual(summary["all"]["finish_id_matches"], 1)
        self.assertEqual(summary["all"]["input_budget_exceeded"], 1)
        self.assertEqual(summary["by_target_action"]["finish"]["rows"], 2)
        self.assertEqual(summary["by_origin"]["scripted_branch"]["rows"], 2)

    def test_time_limit_marks_every_unrun_row_as_failure(self):
        rows = [{"id": str(index), "messages": [],
                 "target_action": '{"action":"finish","candidate_id":"KEEP"}'}
                for index in range(2)]
        backend = FakeBackend([])
        records = list(probe_rows(rows, backend, started=time.perf_counter() - 1,
                                  max_run_seconds=0))
        self.assertEqual(len(records), 2)
        self.assertEqual(backend.seen, [])
        self.assertEqual(summarize(records)["all"]["not_run_time_limit"], 2)
        self.assertEqual(summarize(records)["all"]["finish_id_rate"], 0)

    def test_prepare_paired_tool_image_mask_preserves_tool_text_and_nonlocal_images(self):
        tools = [{"type": "function", "function": {"name": "finish", "parameters": {}}}]
        trace = {"id": "sample", "query": "red car", "events": [
            {"step": 0, "action": {"name": "inspect", "arguments": {"ids": ["P1"]}},
             "observation": {"status": "OK", "images": [{"path": "tool.png", "modality": "rgb",
                                                                   "candidate_ids": ["P1"]}]}},
            {"step": 1, "available_tools": ["finish"], "tools": tools,
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": "Original query and global text"},
                 {"type": "image", "image": "global.png", "view": "global"},
                 {"type": "image", "image": "atlas.png", "view": "atlas"},
                 {"role": "tool", "type": "text", "text": "Inspect fact: P1 red, P2 blue"},
                 {"type": "image", "image": "tool.png", "view": "tool", "modality": "rgb",
                  "candidate_ids": ["P1"]},
             ]}],
        }]}
        selection = prepare_evidence_pairs([trace], [{"id": "sample", "image_group": "scene-a"}], 1, 1)
        pair = selection["pairs"][0]
        full = pair["messages_full"]
        masked = pair["messages_masked"]
        full_images = [block.get("image") for _, _, block in _iter_images(full)]
        masked_images = [block.get("image") for _, _, block in _iter_images(masked)]
        self.assertIn("tool.png", full_images)
        self.assertNotIn("tool.png", masked_images)
        self.assertEqual(set(masked_images), {"global.png", "atlas.png"})
        masked_text = " ".join(block.get("text", "") for message in masked
                                for block in (message.get("content") or []) if isinstance(block, dict))
        self.assertIn("Inspect fact: P1 red, P2 blue", masked_text)
        self.assertIn("pixel content of this one returned tool image is withheld", masked_text)
        self.assertTrue(pair["tool_text_preserved"])
        self.assertFalse(pair["gt_used"])

    def test_paired_probe_has_explicit_qwen_entry_and_is_not_scored(self):
        row = {"id": "sample", "image_group": "scene-a", "decision_index": 1,
               "after_tool": {"name": "inspect", "arguments": {}, "status": "OK"},
               "key_tool_image": {"modality": "rgb"}, "mask_scope": "one image",
               "tool_text_preserved": True, "gt_used": False, "formal_score_row": False,
               "available_tools": ["finish"], "tools": [{"type": "function", "function": {
                   "name": "finish", "parameters": {}}}],
               "messages_full": [], "messages_masked": []}
        backend = FakeNativeBackend([
            '<tool_call>{"name":"finish","arguments":{"id":"P1"}}</tool_call>',
            '<tool_call>{"name":"finish","arguments":{"bbox":[0.1,0.2,0.3,0.4]}}</tool_call>',
        ])
        summary = probe_evidence_pairs([row], backend)
        self.assertEqual(summary["pairs"], 1)
        self.assertEqual(summary["both_generated"], 1)
        self.assertEqual(summary["action_name_changed"], 0)
        self.assertEqual(summary["action_arguments_changed"], 1)
        self.assertTrue(summary["diagnostic_only"])
        self.assertFalse(summary["formal_scoring"])
        self.assertEqual([item[2] for item in backend.inputs], [8192, 8192])


def _iter_images(messages):
    for mi, message in enumerate(messages):
        for bi, block in enumerate(message.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "image":
                yield mi, bi, block


if __name__ == "__main__":
    unittest.main()
