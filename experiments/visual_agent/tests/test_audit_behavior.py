"""Checks that IR consumption requires the real next Processor input."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "code"))
from experiments.visual_agent.audit_behavior import audit


class BehaviorAuditTests(unittest.TestCase):
    def test_real_ir_return_matches_only_the_next_processor(self):
        default_trace = (ROOT / "results/visual_agent/implementation_20260928/dev_native_v3_records"
                         / "native_D_fast/traces/city_hehe_186_000002_028_00000050_002/trace.json")
        trace_path = Path(os.environ["VISUAL_AGENT_TEST_TRACE"]) if os.environ.get("VISUAL_AGENT_TEST_TRACE") else default_trace
        if not trace_path.is_file():
            self.skipTest(f"real IR trace fixture unavailable: {trace_path}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        actual = audit([trace])["summary"]
        self.assertEqual(actual["returned_tool_images_by_modality"]["ir"], 1)
        self.assertEqual(actual["next_processor_consumed_images_by_modality"]["ir"], 1)
        self.assertEqual(actual["ir_next_processor_consumed_queries"], 1)

        broken = deepcopy(trace)
        for geometry in broken["events"][1]["usage"]["image_geometry"]:
            if geometry.get("view") == "tool" and geometry.get("modality") == "ir":
                geometry["path"] += ".different"
        rejected = audit([broken])["summary"]
        self.assertEqual(rejected["returned_tool_images_by_modality"]["ir"], 1)
        self.assertEqual(rejected["next_processor_consumed_images_by_modality"].get("ir", 0), 0)
        self.assertEqual(rejected["ir_next_processor_consumed_queries"], 0)


if __name__ == "__main__":
    unittest.main()
