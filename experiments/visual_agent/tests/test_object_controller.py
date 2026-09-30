import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.visual_agent.object_controller import (
    OBJECT_PROFILE,
    TOOL_SCHEMAS,
    build_initial_messages,
    run_episode,
)
from experiments.visual_agent.evaluate import load_run
from experiments.visual_agent.model import InputBudgetExceeded


class FakeObjectTools:
    def __init__(self, root):
        self.root = Path(root)
        self.pool = self
        self.executed = []
        self.ids = {
            "P9": {"role": "target", "bbox": [.1, .2, .3, .6]},
            "P2": {"role": "target", "bbox": [.5, .2, .8, .7]},
            "P4": {"role": "reference", "bbox": [.2, .1, .4, .2]},
        }
        self.raw_ids = {"P9": "KEEP", "P2": "2", "P4": "3"}
        self.cost_counts = {"dino_calls": 0}
        self.cost_seconds = {"loads": 0.0}
        self.image_paths = {}
        for index in range(7):
            path = self.root / f"atlas_{index}.png"
            Image.new("RGB", (80, 60), (index * 20, 80, 120)).save(path)
            self.image_paths[f"atlas-{index}"] = path
        self.inspect_path = self.root / "inspect_P2_rgb.png"
        Image.new("RGB", (96, 64), "orange").save(self.inspect_path)

    def finish(self, candidate_id):
        return list(self.ids[candidate_id]["bbox"])

    def _raw(self, candidate_id):
        return self.raw_ids[candidate_id]

    def public_candidates(self):
        return [{"id": candidate_id, **copy.deepcopy(item), "sources": []}
                for candidate_id, item in self.ids.items()]

    def atlas(self, candidate_ids=None, modalities=None):
        images = []
        for index, path in enumerate(self.image_paths.values()):
            images.append({"path": str(path), "modality": "rgb", "candidate_ids": ["P2"],
                           "candidate_roles": {"P2": "target"}, "source_size": [80, 60],
                           "tile_positions": {"P2": {"row": index, "col": 0}},
                           "labels": [{"id": "P2", "role": "target", "bbox": [.5, .2, .8, .7]}]})
        return {"status": "OK", "text": "Atlas pages", "images": images,
                "data": {"pages": [], "total_saved_pixels": 7 * 80 * 60}}

    def execute(self, name, args):
        self.executed.append((name, copy.deepcopy(args)))
        if name == "inspect":
            image = {"path": str(self.inspect_path), "modality": "rgb", "candidate_ids": ["P2"],
                     "candidate_roles": {"P2": "target"}, "source_size": [96, 64],
                     "crop_xyxy": [0, 0, 96, 64], "source_bbox": {"P2": self.ids["P2"]["bbox"]}}
            return {"status": "OK", "text": "P2 appearance inspected.", "images": [image],
                    "data": {"regions": [{"modality": "rgb", "candidate_ids": ["P2"],
                                          "region_px": [0, 0, 96, 64]}]}}
        if name == "depth":
            return {"status": "OK", "text": "Camera-depth comparison.", "images": [],
                    "data": {"measurements": {"P2": {"median_m": 12.4}},
                             "pair": {"status": "supported", "median_difference_m": 4.2}}}
        if name == "finish":
            candidate_id = args["id"]
            return {"status": "OK", "text": f"Finished {candidate_id}.", "images": [],
                    "data": {"candidate_id": candidate_id, "bbox": self.finish(candidate_id)}}
        raise AssertionError(name)


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.inputs = []
        self.schemas = []
        self.max_new_tokens = []

    def begin_sample(self):
        pass

    def peak_memory(self):
        return 0

    def generate(self, messages, max_new_tokens, profile, remaining_visual, *, tools=None):
        self.inputs.append(copy.deepcopy(messages))
        self.schemas.append(copy.deepcopy(tools))
        self.max_new_tokens.append(max_new_tokens)
        raw_output = next(self.outputs)
        return {"raw_output": raw_output, "usage": {
            "input_tokens": 300, "input_text_tokens": 200, "visual_tokens": 100,
            "output_tokens": min(30, max_new_tokens), "model_seconds": 0.0,
            "preprocess_seconds": 0.0,
        }}


class ObjectControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images = {}
        for modality in ("rgb", "ir", "depth_visual"):
            path = self.root / f"global_{modality}.png"
            Image.new("RGB", (160, 96), "gray").save(path)
            images[modality] = str(path)
        self.row = {"id": "object-test", "query": "the nearest red car to the pedestrian",
                    "images": images}
        self.tools = FakeObjectTools(self.root)
        self.initial = build_initial_messages(self.row, self.tools.public_candidates(),
                                              self.tools.atlas(), OBJECT_PROFILE)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_tool_calls_keep_relevant_images_and_finish_public_id(self):
        backend = FakeBackend([
            'The target is a red car near the pedestrian; P2 and P9 are the main competitors. '
            '<tool_call>{"name":"inspect","arguments":{"ids":["P2"],"modalities":["rgb"]}}</tool_call>',
            'P2 looks red; distance remains uncertain against P9. '
            '<tool_call>{"name":"depth","arguments":{"ids":["P2"]}}</tool_call>',
            'The query asks for the nearer red car; P9 is the baseline candidate. '
            '<tool_call>{"name":"finish","arguments":{"id":"P9"}}</tool_call>',
        ])

        result = run_episode(self.row, self.tools, backend, self.initial)

        self.assertEqual(result["final_status"], "FINISHED")
        self.assertEqual(result["selected_public_id"], "P9")
        self.assertTrue(result["selected_is_initial"])
        self.assertEqual(result["selected_id"], "KEEP")
        self.assertEqual(result["bbox"], [.1, .2, .3, .6])
        self.assertEqual(self.tools.executed[-1], ("finish", {"id": "P9"}))
        self.assertEqual(result["metrics"]["tool_calls"], 2)
        self.assertEqual(result["metrics"]["finish_calls"], 1)
        self.assertEqual([len(schema) for schema in backend.schemas], [len(TOOL_SCHEMAS)] * 3)

        for turn_index in (1, 2):
            tool_messages = [message for message in backend.inputs[turn_index] if message["role"] == "tool"]
            retained = [part for message in tool_messages for part in message["content"]
                        if part.get("type") == "image" and part.get("image") == str(self.tools.inspect_path)
                        and part.get("candidate_ids") == ["P2"]]
            self.assertTrue(retained, "the relevant inspect image should persist after the depth-only call")
            self.assertTrue(Path(retained[0]["image"]).is_file())

        history_calls = [message for message in backend.inputs[2]
                         if message["role"] == "assistant" and message.get("tool_calls")]
        self.assertEqual(len(history_calls), 2)
        self.assertEqual(history_calls[0]["content"], "")
        self.assertIn("P2 looks red", history_calls[-1]["content"])
        inspection_facts = next(message for message in backend.inputs[2]
                                if message["role"] == "tool" and message["name"] == "inspect")
        fact_text = inspection_facts["content"][0]["text"]
        self.assertIn('"candidate_ids":["P2"]', fact_text)
        self.assertNotIn("region_px", fact_text)

        atlas_image_count = sum(part.get("type") == "image" and part.get("view") == "atlas"
                                for part in self.initial[1]["content"])
        self.assertEqual(atlas_image_count, 6)
        self.assertEqual([call[0] for call in self.tools.executed], ["inspect", "depth", "finish"])

    def test_text_without_tool_call_does_not_fall_back_to_keep(self):
        backend = FakeBackend(["I think the first candidate is probably correct."])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result["final_status"], "INVALID_MODEL_OUTPUT")
        self.assertIsNone(result["bbox"])
        self.assertIsNone(result["selected_id"])
        self.assertEqual(self.tools.executed, [])

    def test_other_initial_candidate_keeps_public_id_for_evaluation(self):
        backend = FakeBackend([
            '<tool_call>{"name":"finish","arguments":{"id":"P2"}}</tool_call>'
        ])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result["selected_public_id"], "P2")
        self.assertEqual(result["selected_id"], "P2")
        self.assertFalse(result["selected_is_initial"])
        prediction = {
            "id": result["id"], "final_status": result["final_status"],
            "selected_id": result["selected_id"], "selected_public_id": result["selected_public_id"],
            "selected_is_initial": result["selected_is_initial"], "bbox": result["bbox"],
            "initial_bbox": result["initial_bbox"], "final_candidates": result["final_candidates"],
        }
        path = self.root / "prediction.jsonl"
        path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
        checked = load_run(path, {self.row["id"]})
        self.assertEqual(checked[self.row["id"]]["selected_id"], "P2")

    def test_six_evidence_calls_still_reserve_separate_finish(self):
        inspect = '<tool_call>{"name":"inspect","arguments":{"ids":["P2"],"modalities":["rgb"]}}</tool_call>'
        finish = '<tool_call>{"name":"finish","arguments":{"id":"P9"}}</tool_call>'
        backend = FakeBackend([inspect] * 6 + [finish])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result["final_status"], "FINISHED")
        self.assertEqual(result["metrics"]["tool_calls"], 6)
        self.assertEqual(result["metrics"]["finish_calls"], 1)
        self.assertEqual(backend.schemas[-1], [TOOL_SCHEMAS[-1]])
        self.assertEqual(backend.max_new_tokens[-1], OBJECT_PROFILE.finish_tokens)

    def test_input_limit_still_allows_one_explicit_final_model_choice(self):
        class NearLimitBackend(FakeBackend):
            def generate(self, messages, max_new_tokens, profile, remaining_visual, *, tools=None):
                if len(tools) > 1:
                    raise InputBudgetExceeded('context_token_budget', {'input_tokens': 3911})
                return super().generate(messages, max_new_tokens, profile, remaining_visual, tools=tools)
        backend = NearLimitBackend(['<tool_call>{"name":"finish","arguments":{"id":"P2"}}</tool_call>'])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result['selected_id'], 'P2')
        self.assertEqual(result['metrics']['model_calls'], 1)
        self.assertEqual(result['events'][0]['available_tools'], ['finish'])
        self.assertEqual(result['events'][0]['tools'], [TOOL_SCHEMAS[-1]])
        self.assertIn('budget_transition', result['events'][0])


if __name__ == "__main__":
    unittest.main()
