"""Matched, GT-free behavior comparison uses only executed evidence."""

from copy import deepcopy
import unittest

from experiments.visual_agent.compare_recovery import compare


def event(step, action, status, *, images=None, data=None, geometry=None):
    result = {"step": step, "action": action, "executed_action": action,
              "tool_seconds": 0.1, "observation": {
                  "status": status, "text": status, "images": images or [], "data": data or {}},
              "usage": {"model_seconds": 1, "image_geometry": geometry or []}}
    return result


class CompareRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.query = "The pole to the left of the sign"
        ir_path = "/actual/tool_ir.png"
        self.before = {"id": "sample", "query": self.query, "mechanism": "D",
                       "prompt_version": "before", "profile": {"name": "sft"},
                       "initial_candidates": [{"id": "KEEP"}],
                       "final_status": "FINISHED", "selected_id": "KEEP",
                       "events": [
                           event(0, {"action": "inspect_regions", "candidate_ids": ["KEEP"],
                                     "modalities": ["ir"], "view": "single"}, "OK",
                                 images=[{"path": ir_path, "modality": "ir"}]),
                           event(1, {"action": "measure_depth", "candidate_ids": ["KEEP"]},
                                 "UNKNOWN", data={"pair": {"status": "unreliable"}},
                                 geometry=[{"path": ir_path, "modality": "ir", "view": "tool"}]),
                           event(2, {"action": "inspect_regions", "candidate_ids": ["KEEP"],
                                     "modalities": ["rgb"], "view": "single"}, "OK"),
                           {"step": 3, "action": {"action": "finish", "candidate_id": "KEEP"},
                            "observation": None, "usage": {"model_seconds": 1}},
                       ]}
        self.after = deepcopy(self.before)
        self.after["prompt_version"] = "after"
        self.after["events"] = [self.after["events"][0], self.after["events"][3]]

    def test_matched_changes_and_real_ir_processor_match(self):
        result = compare([self.before], [self.after], [{"id": "sample", "query": self.query}])
        old, new = result["before_summary"], result["after_summary"]
        self.assertEqual(old["adjacent_inspect_then_depth_calls"], 1)
        self.assertEqual(old["depth_possible_non_camera_relation_review"], 1)
        self.assertEqual(old["unknown_empty_next_decision"], {"changed_tool": 1})
        self.assertEqual(old["ir_next_processor_consumed_images"], 1)
        self.assertEqual(new["ir_returned_images"], 1)
        self.assertEqual(new["ir_next_processor_consumed_images"], 0)
        self.assertEqual(result["samples"][0]["change"]["depth_calls"], -1)
        self.assertEqual(result["samples"][0]["before"]["action_sequence"][1]["status"], "UNKNOWN")

    def test_requires_full_matching_manifest_and_query(self):
        with self.assertRaisesRegex(ValueError, "full fixed manifest"):
            compare([self.before], [self.after], [{"id": "another", "query": self.query}])
        changed = deepcopy(self.after)
        changed["query"] = "changed"
        with self.assertRaisesRegex(ValueError, "query differs"):
            compare([self.before], [changed], [{"id": "sample", "query": self.query}])
        with self.assertRaisesRegex(ValueError, "RGB image group differs"):
            compare([self.before], [self.after], [{"id": "sample", "query": self.query,
                                                  "images": {"rgb": "/fixed/rgb.png"}}])


if __name__ == "__main__":
    unittest.main()
