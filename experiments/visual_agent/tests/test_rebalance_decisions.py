import json
import unittest
from collections import Counter

from experiments.visual_agent.rebalance_decisions import rebalance


class RebalanceDecisionsTest(unittest.TestCase):
    def test_frozen_count_and_provenance(self):
        def row(sample_id, action, *, label_source=None, replay=None):
            return {
                "id": sample_id,
                "target_action": json.dumps({"action": action}),
                "is_final": action == "finish",
                "label_source": label_source,
                "replay": replay,
                "source_path": "actual/tool/trace.json",
            }

        ids = [f"sample_{i:02d}" for i in range(46)]
        rows = [row(sample_id, "finish", label_source="base") for sample_id in ids]
        rows += [row(ids[i], "finish", label_source="offline_gt_existing_target_protocol_replay",
                     replay={"kind": "protocol_error_then_actual_inspect"}) for i in range(20)]
        rows += [row(ids[i], "finish", label_source="offline_gt_existing_target_after_search",
                     replay={"kind": "empty_then_successful_search"}) for i in range(20, 23)]
        rows += [row(ids[i % 46], "inspect_regions") for i in range(69)]
        rows += [row(ids[i], "measure_depth") for i in range(11)]
        rows += [row(ids[i % 15], "search_candidates") for i in range(21)]

        output, inventory = rebalance(rows)
        self.assertEqual(len(output), 170)
        self.assertEqual(Counter(json.loads(r["target_action"])["action"] for r in output),
                         Counter({"finish": 46, "inspect_regions": 69,
                                  "measure_depth": 22, "search_candidates": 33}))
        self.assertEqual(len({r["id"] for r in output}), 46)
        self.assertEqual(inventory["removed_finish_counts"],
                         {"protocol_replay_finish": 20, "empty_replay_finish": 3})
        repeated = [r for r in output if "resampling" in r]
        self.assertEqual(Counter(r["resampling"]["action"] for r in repeated),
                         Counter({"measure_depth": 11, "search_candidates": 12}))
        self.assertEqual(len({r["id"] for r in repeated if r["resampling"]["action"] == "search_candidates"}), 12)
        self.assertTrue(all(r["source_path"] == "actual/tool/trace.json" for r in repeated))


if __name__ == "__main__":
    unittest.main()
