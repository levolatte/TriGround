import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.visual_agent.diagnostic import diagnose
from experiments.visual_agent.profiles import PROFILES


class FakeProcessor:
    def __init__(self, grid_multiplier=1):
        self.image_processor = type("ImageProcessor", (), {"patch_size": 16, "merge_size": 2})()
        self.grid_multiplier = grid_multiplier

    def apply_chat_template(self, messages, **kwargs):
        images = [part["image"] for message in messages
                  if isinstance(message.get("content"), list)
                  for part in message["content"] if part.get("type") == "image"]
        grids = np.asarray([[1, image.height // 16 * self.grid_multiplier,
                             image.width // 16 * self.grid_multiplier] for image in images], dtype=np.int64)
        tokens = 100 + sum(int(np.prod(grid) // 4) for grid in grids)
        return {"image_grid_thw": grids, "input_ids": np.zeros((1, tokens), dtype=np.int64)}


class FakeBackend:
    load_seconds = 0.1

    def __init__(self, grid_multiplier=1):
        self.processor = FakeProcessor(grid_multiplier)
        self.generate_calls = 0

    def begin_sample(self):
        pass

    def peak_memory(self):
        return 123

    def generate(self, messages, max_new_tokens, profile, remaining_visual):
        self.generate_calls += 1
        assert max_new_tokens == 128
        assert remaining_visual >= 0
        return {"raw_output": '{"action":"finish","candidate_id":"KEEP"}',
                "usage": {"output_tokens": 12, "visual_tokens": 3000}}


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images = {}
        for key in ("rgb", "ir", "depth_visual"):
            path = self.root / f"{key}.png"
            Image.new("RGB", (64, 64), "gray").save(path)
            images[key] = str(path)
        self.row = {"id": "example", "query": "a person next to a car", "images": images,
                    "depth_encoding": "unknown"}
        self.cache = {"id": "example", "images": images, "depth_encoding": "unknown",
                      "c_bbox": [.1, .1, .3, .8],
                      "query_info": {"scope": "single", "target_category": "person"},
                      "candidates": [
                          {"id": 1, "role": "target", "bbox": [.1,.1,.3,.8],
                           "is_baseline": True, "sources": []},
                          {"id": 2, "role": "target", "bbox": [.5,.1,.7,.8],
                           "is_baseline": False, "sources": [{"modality": "rgb", "score": .8}]},
                      ]}

    def tearDown(self):
        self.temp.cleanup()

    def test_full_six_tool_images_reach_one_final_generation(self):
        backend = FakeBackend()
        detection = [{"bbox": [.75,.1,.9,.8], "score": .9}]
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=detection):
            result = diagnose(self.row, self.cache, None, backend, self.root / "full",
                              PROFILES["capacity24"], dino_model=(object(), object(), "cpu"))
        self.assertTrue(result["coverage"]["all_six_executed"])
        self.assertTrue(result["coverage"]["search_nonempty"])
        self.assertEqual(backend.generate_calls, 1)
        self.assertEqual(result["final_generation"]["status"], "GENERATED")
        self.assertEqual(len(result["decision_inputs"][-1]["image_grid_thw"]), 11)
        self.assertTrue(Path(self.root / "full" / "diagnostic.json").is_file())
        self.assertFalse(result["autonomous_agent_result"])

    def test_missing_competitor_reports_incomplete_coverage(self):
        cache = copy.deepcopy(self.cache)
        cache["candidates"] = cache["candidates"][:1]
        backend = FakeBackend()
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=[]):
            result = diagnose(self.row, cache, None, backend, self.root / "missing",
                              PROFILES["capacity24"], dino_model=(object(), object(), "cpu"))
        self.assertFalse(result["coverage"]["all_six_executed"])
        self.assertEqual(result["coverage"]["real_tool_calls"], 5)
        self.assertEqual(result["coverage"]["skipped"][0]["slot"], "pair_rgb")
        self.assertEqual(backend.generate_calls, 1)

    def test_reference_candidate_can_fill_pair_view(self):
        cache = copy.deepcopy(self.cache)
        cache["candidates"][1]["role"] = "reference"
        backend = FakeBackend()
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=[]):
            result = diagnose(self.row, cache, None, backend, self.root / "reference",
                              PROFILES["capacity24"], dino_model=(object(), object(), "cpu"))
        self.assertTrue(result["coverage"]["all_six_executed"])
        pair = next(item for item in result["observations"] if item["slot"] == "pair_rgb")
        self.assertEqual(pair["action"]["candidate_ids"], ["KEEP", "2"])

    def test_processor_budget_limit_prevents_model_forward(self):
        backend = FakeBackend(grid_multiplier=8)
        with patch("experiments.visual_agent.vision_tools.detect_image_proposals", return_value=[]):
            result = diagnose(self.row, self.cache, None, backend, self.root / "limit",
                              PROFILES["capacity24"], dino_model=(object(), object(), "cpu"))
        self.assertEqual(result["final_generation"]["status"], "LIMIT")
        self.assertEqual(backend.generate_calls, 0)
        self.assertFalse(result["decision_inputs"][-1]["single_visual_within_profile"])

    def test_sft_short_history_profile_is_not_misreported_as_full_history(self):
        with self.assertRaisesRegex(ValueError, "full-history"):
            diagnose(self.row, self.cache, None, FakeBackend(), self.root / "sft",
                     PROFILES["sft"])


if __name__ == "__main__":
    unittest.main()
