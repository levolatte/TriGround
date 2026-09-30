import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.evidence_decision.controller import run_episode
from experiments.evidence_decision.model import prepare_image_messages, geometry_after_processor, resize_for_budget
from experiments.evidence_decision.profiles import PROFILES
from experiments.evidence_decision.run import build_initial_messages
from experiments.evidence_decision.vision_tools import CandidatePool, VisualTools


class ScriptedBackend:
    """Protocol test double, never used to report model ACC."""
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []

    def begin_sample(self):
        pass

    def peak_memory(self):
        return 0

    def generate(self, messages, max_new_tokens, profile, remaining_visual):
        self.inputs.append(copy.deepcopy(messages))
        prepared, metadata = prepare_image_messages(messages, profile.global_pixels)
        grids = [[1, item["prepared_size"][1] // 16, item["prepared_size"][0] // 16] for item in metadata]
        geometry = geometry_after_processor(metadata, grids, 16, 2)
        return {"raw_output": next(self.actions), "usage": {
            "input_tokens": 180, "input_text_tokens": 80, "visual_tokens": 100,
            "output_tokens": 20, "model_seconds": 0, "preprocess_seconds": 0,
            "image_grid_thw": grids, "image_geometry": geometry}}


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images = {}
        for key in ("rgb", "ir", "depth_visual"):
            image = Image.new("RGB", (641, 321), "gray")
            path = self.root / f"{key}.png"
            image.save(path)
            images[key] = str(path)
        self.row = {"id": "test", "query": "the person on the left, not the crop's left panel",
                    "images": images, "depth_encoding": "unknown"}
        self.cache = {"id": "test", "c_bbox": [.2, .3, .4, .8], "query_info": {"scope": "single"},
            "candidates": [
                {"id": 1, "bbox": [.2, .3, .4, .8], "is_baseline": True, "role": "target", "sources": []},
                {"id": 2, "bbox": [.5, .3, .7, .8], "is_baseline": False, "role": "target", "sources": []},
                {"id": 3, "bbox": [.1, .1, .8, .2], "is_baseline": False, "role": "reference", "sources": []}]}

    def tearDown(self):
        self.temp.cleanup()

    def episode(self, actions, mechanism="C", memory="full"):
        pool = CandidatePool(self.cache, (641, 321))
        profile = PROFILES["sft" if memory == "latest" else "fast"]
        tools = VisualTools(self.row, pool, self.root / "tools", profile.tool_pixels, allow_search=mechanism == "D")
        backend = ScriptedBackend(actions)
        initial = build_initial_messages(self.row, pool.public_candidates(), self.root / "global", profile)
        result = run_episode(self.row, pool, tools, backend, initial, mechanism=mechanism, profile=profile, memory=memory)
        return result, backend

    def test_actual_tool_pixels_enter_next_decision_and_keep_is_exact(self):
        result, backend = self.episode([
            json.dumps({"action": "inspect_regions", "candidate_ids": ["KEEP", "2"], "modalities": ["rgb"], "view": "pair"}),
            json.dumps({"action": "finish", "candidate_id": "KEEP"})])
        self.assertEqual(result["bbox"], self.cache["c_bbox"])
        self.assertEqual(result["metrics"]["tool_calls"], 1)
        self.assertEqual(len(result["events"][0]["usage"]["image_geometry"]), 3)
        self.assertEqual(len(result["events"][1]["usage"]["image_geometry"]), 5)
        self.assertIn(self.row["query"], json.dumps(backend.inputs[1]))
        self.assertTrue(all(Path(p["path"]).is_file() for p in result["events"][0]["observation"]["images"]))

    def test_unknown_depth_is_valid_observation(self):
        result, _ = self.episode([
            '{"action":"measure_depth","candidate_ids":["KEEP","2"]}',
            '{"action":"finish","candidate_id":"2"}'])
        self.assertEqual(result["metrics"]["tool_status_counts"], {"UNKNOWN": 1})
        self.assertEqual(result["final_status"], "FINISHED")

    def test_two_object_cross_modal_observations_enter_next_processor(self):
        result, _ = self.episode([
            '{"action":"inspect_regions","candidate_ids":["KEEP","2"],"modalities":["rgb","ir"],"view":"cross","evidence_note":"Compare silhouettes"}',
            '{"action":"finish","candidate_id":"KEEP"}'])
        observation = result["events"][0]["observation"]
        self.assertEqual(observation["status"], "OK")
        self.assertEqual([(image["candidate_ids"], image["modality"]) for image in observation["images"]],
                         [(["KEEP"], "rgb"), (["KEEP"], "ir"), (["2"], "rgb"), (["2"], "ir")])
        self.assertLessEqual(observation["data"]["total_saved_pixels"], PROFILES["fast"].tool_pixels)
        self.assertEqual(len(result["events"][1]["usage"]["image_geometry"]), 7)
        self.assertEqual(result["metrics"]["invalid_actions"], 0)

    def test_invalid_final_does_not_fall_back_to_keep(self):
        result, _ = self.episode(['{"action":"finish","candidate_id":"3"}'], mechanism="A")
        self.assertIsNone(result["bbox"])
        self.assertFalse(result["metrics"]["legal_finish"])

    def test_malformed_action_gets_only_one_feedback_then_finish(self):
        result, _ = self.episode(['{"action":', '{"action":"inspect_regions"}'])
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertIsNone(result["bbox"])

    def test_tool_protocol_error_allows_one_bounded_tool_correction(self):
        result, backend = self.episode([
            '{"action":"inspect_regions","candidate_ids":["99999"],"modalities":["rgb"],"view":"single"}',
            '{"action":"inspect_regions","candidate_ids":["KEEP"],"modalities":["rgb"],"view":"single"}',
            '{"action":"finish","candidate_id":"KEEP"}'])
        self.assertEqual(result["metrics"]["invalid_actions"], 1)
        self.assertIn("observation", backend.inputs[1][-1]["content"][0]["text"])
        self.assertEqual(result["events"][1]["observation"]["status"], "OK")
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(result["final_status"], "FINISHED")

    def test_b_uses_text_only_two_stage_budget(self):
        result, backend = self.episode(['{"evidence_notes":"Compare positions"}', '{"action":"finish","candidate_id":"KEEP"}'], mechanism="B")
        self.assertEqual(result["metrics"]["tool_calls"], 0)
        self.assertEqual(len(backend.inputs), 2)

    def test_fixed_schedule_can_finish_early_or_advance(self):
        result, _ = self.episode(['{"action":"continue"}', '{"action":"finish","candidate_id":"KEEP"}'], mechanism="S")
        self.assertEqual(result["events"][0]["executed_action"]["action"], "inspect_regions")
        self.assertEqual(result["metrics"]["tool_calls"], 1)

    def test_exact_fixed_step_alias_cannot_change_the_schedule(self):
        result, _ = self.episode([
            '{"action":"inspect_regions","candidate_ids":["KEEP","2"],"modalities":["ir"],"view":"pair","evidence_note":"compare"}',
            '{"action":"measure_depth","candidate_ids":["KEEP","2"]}',
            '{"action":"finish","candidate_id":"KEEP"}'], mechanism="S")
        self.assertEqual(result["metrics"]["invalid_actions"], 0)
        self.assertEqual(result["metrics"]["tool_calls"], 2)
        self.assertTrue(result["events"][0]["fixed_step_alias"])
        changed, _ = self.episode([
            '{"action":"inspect_regions","candidate_ids":["KEEP","2"],"modalities":["rgb"],"view":"pair"}',
            '{"action":"finish","candidate_id":"KEEP"}'], mechanism="S")
        self.assertEqual(changed["metrics"]["invalid_actions"], 1)
        self.assertEqual(changed["metrics"]["tool_calls"], 0)

    def test_compressed_history_preserves_latest_image_after_depth(self):
        result, _ = self.episode([
            '{"action":"inspect_regions","candidate_ids":["KEEP"],"modalities":["rgb"],"view":"single"}',
            '{"action":"measure_depth","candidate_ids":["KEEP"]}',
            '{"action":"finish","candidate_id":"KEEP"}'], memory="latest")
        self.assertEqual(len(result["events"][2]["usage"]["image_geometry"]), 4)

    def test_pixel_budget_and_actual_grid_geometry(self):
        image = resize_for_budget(Image.new("RGB", (1901, 1033)), 602112)
        self.assertLessEqual(image.width * image.height, 602112)
        self.assertEqual(image.width % 32, 0)
        self.assertEqual(image.height % 32, 0)
        meta = [{"view": "global", "modality": "rgb", "target_boxes": [{"id": "KEEP", "bbox": [.1,.1,.2,.2]}]},
                {"view": "tool", "modality": "rgb", "target_boxes": [{"candidate_id": "KEEP", "xyxy": [.1,.1,.8,.8]}]}]
        geometry = geometry_after_processor(meta, [[1, 20, 40], [1, 20, 40]], 16, 2)
        self.assertGreater(geometry[1]["targets"][0]["area_ratio_to_global"], 1)
        meta[1]["target_boxes"] = [{"id": "12", "bbox": [.1,.1,.8,.8]}]
        meta[1]["source_bbox"] = {"12": [.4,.4,.5,.5]}
        dynamic = geometry_after_processor(meta, [[1,20,40], [1,20,40]], 16, 2)
        self.assertAlmostEqual(dynamic[1]["targets"][0]["area_ratio_to_global"], 49)


if __name__ == "__main__":
    unittest.main()
