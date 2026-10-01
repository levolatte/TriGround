"""Classification and full-denominator checks without loading Qwen or tools."""

import time
import unittest

from experiments.visual_agent.model import InputBudgetExceeded
from experiments.visual_agent.probe_decisions import probe_rows, summarize


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


if __name__ == "__main__":
    unittest.main()
