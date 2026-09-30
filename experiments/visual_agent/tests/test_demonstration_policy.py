"""Narrow checks for GT-free scripted evidence choices and recovery."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.visual_agent.demonstration_policy import next_action


KEEP = {"id": "KEEP", "role": "target", "bbox": [.1, .1, .3, .6]}
OTHER = {"id": "2", "role": "target", "bbox": [.6, .1, .8, .6]}
REFERENCE = {"id": "3", "role": "reference", "bbox": [.4, .1, .5, .7]}


def row(query):
    return {"query": query, "images": {"rgb": "rgb.png", "ir": "ir.png",
                                      "depth_raw": "depth.png"}, "depth_encoding": "city_mm",
            "dino_model_path": "dino", "ir_rgb_registration": "normalized_shared_frame",
            "gt_bbox": [.9, .9, 1, 1]}


def event(action, status, data=None):
    return {"origin": "scripted", "executed_action": action,
            "observation": {"status": status, "data": data or {}, "images": []}}


class DemonstrationPolicyTest(unittest.TestCase):
    def test_only_explicit_camera_order_uses_depth_and_unknown_recovers(self):
        camera = {"target_category": "person", "relation_type": "camera_near", "scope": "single"}
        start = next_action(row("the person nearest to the camera"), [KEEP, OTHER], [], camera)
        self.assertEqual(start["action"], "measure_depth")
        self.assertEqual(start["candidate_ids"], ["KEEP", "2"])
        after_unknown = next_action(row("the person nearest to the camera"), [KEEP, OTHER],
                                    [event(start, "UNKNOWN")], camera)
        self.assertEqual(after_unknown["action"], "inspect_regions")
        self.assertEqual(after_unknown["modalities"], ["rgb", "ir"])
        self.assertIn("UNKNOWN", after_unknown["evidence_note"])
        self.assertIsNone(next_action(row("the person nearest to the camera"), [KEEP, OTHER],
                                      [event(start, "UNKNOWN"), event(after_unknown, "OK")], camera))
        relative = {"target_category": "person", "relation_type": "other", "scope": "single"}
        action = next_action(row("the person nearest to the tree"), [KEEP, OTHER, REFERENCE], [], relative)
        self.assertEqual(action["action"], "inspect_regions")
        self.assertIn("rgb", action["modalities"])
        self.assertEqual(action["candidate_ids"], ["KEEP", "3"])
        self.assertIsNone(next_action(row("the leftmost person"), [KEEP, OTHER], [],
                                      {"target_category": "person", "relation_type": "none", "scope": "single"}))
        self.assertIsNone(next_action(row("the person farther left"), [KEEP, OTHER], [],
                                      {"target_category": "person", "relation_type": "camera_far", "scope": "single"}))
        high = {"id": "9", "role": "target", "bbox": [.3, .1, .4, .6],
                "sources": [{"modality": "ir", "score": .9}]}
        low = {"id": "8", "role": "target", "bbox": [.5, .1, .6, .6],
               "sources": [{"modality": "rgb", "score": .1}]}
        foreground = next_action(row("the person in the foreground"),
                                 [KEEP, low, high, REFERENCE], [],
                                 {"target_category": "person", "relation_type": "other", "scope": "single"})
        self.assertEqual(foreground["action"], "measure_depth")
        self.assertEqual(foreground["candidate_ids"], ["KEEP", "9"])
        reference_background = next_action(
            row("The person standing immediately to the right of the central background figure"),
            [KEEP, OTHER, REFERENCE], [],
            {"target_category": "person", "relation_type": "other", "scope": "single"})
        self.assertNotEqual(reference_background["action"], "measure_depth")
        self.assertEqual(reference_background["modalities"], ["rgb"])
        bird_foreground = next_action(
            row("The white bird in the foreground"), [KEEP, OTHER], [],
            {"target_category": "bird", "relation_type": "other", "scope": "single"})
        self.assertEqual(bird_foreground["action"], "measure_depth")
        lone_sedan = next_action(
            row("The white sedan in the foreground"), [KEEP], [],
            {"target_category": "sedan", "relation_type": "other", "scope": "single"})
        self.assertEqual((lone_sedan["action"], lone_sedan["category"]),
                         ("search_candidates", "sedan"))
        self.assertIn("No external target hypothesis", lone_sedan["evidence_note"])

    def test_rgb_ir_identity_and_ordinal_search_use_only_real_status(self):
        semantic = {"target_category": "person", "relation_type": "none", "scope": "single"}
        visual = next_action(row("the blurred person in a red shirt"), [KEEP, OTHER], [], semantic)
        self.assertEqual((visual["action"], visual["view"], visual["modalities"]),
                         ("inspect_regions", "cross", ["rgb", "ir"]))
        self.assertIsNone(next_action(row("the blurred person in a red shirt"), [KEEP, OTHER],
                                      [event(visual, "OK")], semantic))
        search = next_action(row("the second hidden person"), [KEEP], [], {"query_info": semantic})
        self.assertEqual({key: search[key] for key in ("action", "category", "region", "modality", "role")},
                         {"action": "search_candidates", "category": "person", "region": "full",
                          "modality": "ir", "role": "target"})
        after_empty = next_action(row("the second hidden person"), [KEEP],
                                  [event(search, "EMPTY")], semantic)
        self.assertEqual(after_empty["action"], "inspect_regions")
        self.assertIn("uninformative", after_empty["evidence_note"])
        self.assertIsNone(next_action(row("the second hidden person"), [KEEP],
                                      [event(search, "EMPTY"), event(after_empty, "OK")], semantic))
        after_added = next_action(row("the second hidden person"), [KEEP, OTHER],
                                  [event(search, "OK", {"appended_ids": ["2"]})], semantic)
        self.assertEqual(after_added["action"], "inspect_regions")
        self.assertEqual(after_added["candidate_ids"], ["2"])
        rgb_search = next_action(row("the second person"), [KEEP], [], semantic)
        self.assertEqual(rgb_search["modality"], "rgb")
        fourth = next_action(row("the fourth robot"), [KEEP, OTHER], [],
                             {"target_category": "robot", "relation_type": "none", "scope": "single"})
        self.assertEqual((fourth["action"], fourth["category"]), ("search_candidates", "robot"))
        numeric = next_action(row("the 4th robot"), [KEEP, OTHER], [],
                              {"target_category": "robot", "relation_type": "none", "scope": "single"})
        self.assertEqual(numeric["action"], "search_candidates")
        ir_retry = next_action(row("the second person"), [KEEP], [event(rgb_search, "EMPTY")], semantic)
        self.assertEqual((ir_retry["action"], ir_retry["modality"]), ("search_candidates", "ir"))
        self.assertIn("EMPTY", ir_retry["evidence_note"])
        after_retry_empty = next_action(row("the second person"), [KEEP],
                                        [event(rgb_search, "EMPTY"), event(ir_retry, "EMPTY")], semantic)
        self.assertEqual(after_retry_empty["action"], "inspect_regions")

    def test_budget_and_missing_search_model_do_not_invent_actions(self):
        semantic = {"target_category": "person", "relation_type": "none", "scope": "single"}
        sample = row("the second person")
        sample["dino_model_path"] = None
        self.assertIsNone(next_action(sample, [KEEP], [], semantic))
        sample = row("the blurred person")
        past = [event({"action": "inspect_regions"}, "OK") for _ in range(6)]
        self.assertIsNone(next_action(sample, [KEEP, OTHER], past, semantic))


if __name__ == "__main__":
    unittest.main()
