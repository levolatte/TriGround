"""Audit actual trace geometry rather than assuming saved tool PNGs were read."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "code"))
from experiments.visual_agent import report


class BehaviorAuditTests(unittest.TestCase):
    def test_exact_later_path_and_first_consumption_geometry(self):
        trace = {"id": "scene_1", "events": [
            {"action": {"action": "inspect_regions", "view": "single"},
             "executed_action": {"action": "inspect_regions", "view": "single"},
             "tool_seconds": .1,
             "usage": {"image_geometry": [{"path": "global.png"}]},
             "observation": {"status": "OK", "images": [
                 {"path": "/a/obs.png"}, {"path": "/b/obs.png"}, {"path": "/a/obs.png"}]}},
            {"action": {"action": "measure_depth"}, "executed_action": {"action": "measure_depth"},
             "usage": {"model_seconds": .1, "image_geometry": [{"path": "/a/obs.png", "modality": "rgb", "targets": [
                 {"id": "KEEP", "width_px": 64, "height_px": 32, "area_px": 2048,
                  "area_ratio_to_global": .8, "effective_zoom": False},
                 {"id": "new", "width_px": 20, "height_px": 10, "area_px": 200,
                  "area_ratio_to_global": None, "effective_zoom": False}]}]},
             "observation": {"status": "UNKNOWN", "images": []}},
            {"action": {"action": "finish", "candidate_id": "KEEP"},
             "usage": {"image_geometry": [{"path": "/a/obs.png", "targets": [
                 {"id": "KEEP", "width_px": 999, "height_px": 999, "area_px": 999,
                  "area_ratio_to_global": 9, "effective_zoom": True}]}]},
             "observation": None},
        ]}
        audited = report.audit_traces([trace])
        self.assertEqual((audited["returned_tool_images"], audited["consumed_tool_images"],
                          audited["unconsumed_tool_images"]), (2, 1, 1))
        self.assertEqual(audited["tool_invocation_counts"], {"inspect_regions": 1})
        self.assertEqual(audited["trace_ids_with_unconsumed_images"], ["scene_1"])
        self.assertEqual(audited["per_trace"][0]["unconsumed_images"], ["/b/obs.png"])
        self.assertEqual(audited["processed_target_geometry"]["width_px"]["max"], 64)
        self.assertEqual(audited["processed_target_geometry"]["unknown_area_ratio_count"], 1)
        self.assertEqual(audited["processed_target_geometry"]["effective_zoom_le_1_count"], 1)
        self.assertEqual(audited["processed_target_geometry"]["effective_zoom_le_1_fraction_of_known"], 1)

    def test_nonfinal_generation_without_tool_path_is_unconsumed_and_errors_counted(self):
        trace = {"id": "scene_2", "events": [
            {"action": {"action": "inspect_regions", "candidate_ids": ["3"]},
             "executed_action": {"action": "inspect_regions", "candidate_ids": ["3"]},
             "observation": {"status": "ERROR", "text": "invalid inspect view", "images": [],
                             "data": {"kind": "protocol"}}},
            {"action": {"action": "inspect_regions", "view": "cross"},
             "executed_action": {"action": "inspect_regions", "view": "cross"},
             "observation": {"status": "OK", "images": [{"path": "/x/crop.png"}]}},
            {"action": {"action": "measure_depth"}, "executed_action": {"action": "measure_depth"},
             "usage": {"model_seconds": .1, "image_geometry": [{"path": "/x/other.png"}]},
             "observation": {"status": "UNKNOWN", "images": []}},
            {"action": {"action": "finish", "candidate_id": "KEEP"},
             "usage": {"image_geometry": []}, "observation": None},
        ]}
        audited = report.audit_traces([trace])
        self.assertEqual(audited["inspect_view_counts"], {"<missing>": 1, "cross": 1})
        self.assertEqual(audited["selected_tool_action_counts"],
                         {"inspect_regions": 2, "measure_depth": 1})
        self.assertEqual(audited["tool_invocation_counts"], {})
        self.assertEqual(audited["error_reasons"], {"invalid inspect view": 1})
        self.assertEqual(audited["error_kinds"], {"protocol": 1})
        self.assertEqual((audited["returned_tool_images"], audited["consumed_tool_images"]), (1, 0))
        self.assertIsNone(audited["processed_target_geometry"]["effective_zoom_le_1_fraction_of_known"])

    def test_processor_geometry_without_model_forward_does_not_count_as_consumed(self):
        trace = {"id": "budget", "events": [
            {"action": {"action": "inspect_regions", "view": "pair"},
             "tool_seconds": .1, "executed_action": {"action": "inspect_regions", "view": "pair"},
             "observation": {"status": "OK", "images": [{"path": "/x/tool.png"}]}},
            {"action": None, "raw_output": "", "usage": {"model_seconds": 0,
             "image_geometry": [{"path": "/x/tool.png", "targets": []}]},
             "observation": {"status": "LIMIT", "images": []}},
        ]}
        audited = report.audit_traces([trace])
        self.assertEqual((audited["processor_matched_tool_images"],
                          audited["processor_only_rejected_images"], audited["consumed_tool_images"]), (1, 1, 0))
        self.assertEqual(audited["trace_ids_with_unconsumed_images"], ["budget"])

    def test_late_image_does_not_rescue_missing_next_decision(self):
        trace = {"id": "late", "events": [
            {"action": {"action": "inspect_regions", "view": "single"},
             "observation": {"status": "OK", "images": [{"path": "/x/tool.png"}]}},
            {"action": {"action": "measure_depth"}, "usage": {"model_seconds": .1, "image_geometry": []},
             "observation": {"status": "UNKNOWN", "images": []}},
            {"action": {"action": "finish"}, "usage": {"model_seconds": .1,
             "image_geometry": [{"path": "/x/tool.png", "targets": []}]}, "observation": None},
        ]}
        audited = report.audit_traces([trace])
        self.assertEqual((audited["processor_matched_tool_images"], audited["consumed_tool_images"]), (0, 0))
        self.assertEqual(audited["trace_ids_with_unconsumed_images"], ["late"])

    def test_loads_complete_trace_files_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(("one", "two")):
                location = root / name
                location.mkdir()
                (location / "trace.json").write_text(json.dumps({"id": str(index), "events": []}), encoding="utf-8")
            self.assertEqual([trace["id"] for trace in report.load_traces(root)], ["0", "1"])
            (root / "two" / "trace.json").write_text(json.dumps({"id": "0", "events": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate trace ID"):
                report.load_traces(root)

    def test_audit_only_cli_needs_no_evaluation_or_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace.json"
            target = root / "behavior_audit.json"
            source.write_text(json.dumps({"id": "q", "events": []}), encoding="utf-8")
            report.main(["--traces", str(source), "--output", str(target)])
            self.assertEqual(json.loads(target.read_text())["traces"], 1)


if __name__ == "__main__":
    unittest.main()
