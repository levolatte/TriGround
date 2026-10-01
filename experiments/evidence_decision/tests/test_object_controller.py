import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from experiments.evidence_decision.object_controller import (
    OBJECT_PROFILE,
    TOOL_SCHEMAS,
    build_initial_messages,
    build_messages,
    run_episode,
    _tool_message,
    _tool_facts,
)
from experiments.evidence_decision.evaluate import load_run
from experiments.evidence_decision.model import InputBudgetExceeded


class FakeObjectTools:
    def __init__(self, root):
        self.root = Path(root)
        self.pool = self
        self.c_bbox = [.1, .2, .3, .6]
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
        return [{"id": candidate_id, **copy.deepcopy(item), "sources": [],
                 "coordinate_frame": "rgb", "finish_eligible": True}
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
            if "bbox" in args:
                return {"status": "OK", "text": "Finished predicted bbox.", "images": [],
                        "data": {"candidate_id": None, "bbox": list(args["bbox"]),
                                 "finish_source": "predicted_bbox"}}
            candidate_id = args["id"]
            return {"status": "OK", "text": f"Finished {candidate_id}.", "images": [],
                    "data": {"candidate_id": candidate_id, "bbox": self.finish(candidate_id),
                             "finish_source": "candidate"}}
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
        self.assertEqual(result["selected_id"], "P9")
        self.assertEqual(result["finish_source"], "candidate")
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
        self.assertIn("main competitors", history_calls[0]["content"])
        self.assertIn("P2 looks red", history_calls[-1]["content"])
        inspection_facts = next(message for message in backend.inputs[2]
                                if message["role"] == "tool" and message["name"] == "inspect")
        fact_text = inspection_facts["content"][0]["text"]
        self.assertIn('"candidate_ids":["P2"]', fact_text)
        self.assertIn('"region_px":[0,0,96,64]', fact_text)

        atlas_image_count = sum(part.get("type") == "image" and part.get("view") == "atlas"
                                for part in self.initial[1]["content"])
        self.assertEqual(atlas_image_count, 6)
        self.assertEqual([call[0] for call in self.tools.executed], ["inspect", "depth", "finish"])

    def test_text_without_tool_call_does_not_fall_back_to_keep(self):
        backend = FakeBackend(["I think the first candidate is probably correct.",
                               "I still have no valid tool call."])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result["final_status"], "INVALID_MODEL_OUTPUT")
        self.assertIsNone(result["bbox"])
        self.assertIsNone(result["selected_id"])
        self.assertEqual(self.tools.executed, [])
        self.assertEqual(result["metrics"]["model_calls"], 2)
        self.assertEqual(result["metrics"]["invalid_actions"], 2)

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
            "finish_source": result["finish_source"],
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

    def test_input_limit_retries_compressed_prompt_without_removing_evidence_tools(self):
        class NearLimitBackend(FakeBackend):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.failed_once = False

            def generate(self, messages, max_new_tokens, profile, remaining_visual, *, tools=None):
                if len(tools) > 1 and not self.failed_once:
                    self.failed_once = True
                    raise InputBudgetExceeded('context_token_budget', {'input_tokens': 3911})
                return super().generate(messages, max_new_tokens, profile, remaining_visual, tools=tools)
        backend = NearLimitBackend(['<tool_call>{"name":"finish","arguments":{"id":"P2"}}</tool_call>'])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result['selected_id'], 'P2')
        self.assertEqual(result['metrics']['model_calls'], 1)
        self.assertEqual(result['events'][0]['available_tools'], [item['function']['name'] for item in TOOL_SCHEMAS])
        self.assertEqual(result['events'][0]['tools'], TOOL_SCHEMAS)
        self.assertIn('budget_transition', result['events'][0])
        self.assertEqual(len(backend.schemas), 1)
        self.assertEqual(backend.schemas[0], TOOL_SCHEMAS)

    def test_predicted_bbox_finish_has_no_synthetic_candidate_id(self):
        bbox = [.12, .22, .42, .62]
        backend = FakeBackend([
            '<tool_call>{"name":"finish","arguments":{"bbox":[0.12,0.22,0.42,0.62]}}</tool_call>'
        ])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result['final_status'], 'FINISHED')
        self.assertEqual(result['bbox'], bbox)
        self.assertEqual(result['finish_source'], 'predicted_bbox')
        self.assertIsNone(result['selected_id'])
        self.assertIsNone(result['selected_public_id'])
        self.assertIsNone(result['selected_is_initial'])
        path = self.root / "predicted_bbox.jsonl"
        path.write_text(json.dumps({
            "id": result["id"], "final_status": result["final_status"],
            "finish_source": result["finish_source"], "selected_id": result["selected_id"],
            "selected_public_id": result["selected_public_id"], "bbox": result["bbox"],
            "initial_bbox": result["initial_bbox"], "final_candidates": result["final_candidates"],
        }) + "\n", encoding="utf-8")
        self.assertEqual(load_run(path, {self.row["id"]})[self.row["id"]]["finish_source"],
                         "predicted_bbox")

    def test_protocol_recovery_does_not_reduce_six_evidence_slots(self):
        inspect = '<tool_call>{"name":"inspect","arguments":{"ids":["P2"],"modalities":["rgb"]}}</tool_call>'
        finish = '<tool_call>{"name":"finish","arguments":{"id":"P9"}}</tool_call>'
        backend = FakeBackend(['malformed'] + [inspect] * 6 + [finish])
        result = run_episode(self.row, self.tools, backend, self.initial)
        self.assertEqual(result['final_status'], 'FINISHED')
        self.assertEqual(result['metrics']['tool_calls'], 6)
        self.assertEqual(result['metrics']['model_calls'], 8)
        self.assertTrue(result['events'][0]['recoverable'])
        self.assertEqual(result['events'][-1]['available_tools'], ['finish'])

    def test_profile_separates_context_and_visual_token_limits(self):
        self.assertEqual(OBJECT_PROFILE.context_tokens, 8192)
        self.assertEqual(OBJECT_PROFILE.visual_tokens, 4096)

    def test_schemas_separate_six_id_batches_from_three_active_detail_competitors(self):
        for schema in TOOL_SCHEMAS[:2]:
            self.assertEqual(schema["function"]["parameters"]["properties"]["ids"]["maxItems"], 6)
        finish_parameters = TOOL_SCHEMAS[-1]["function"]["parameters"]
        self.assertEqual(len(finish_parameters["oneOf"]), 2)
        self.assertIn("id", finish_parameters["properties"])
        self.assertIn("bbox", finish_parameters["properties"])

    def test_joint_is_inspect_only_and_unmapped_region_facts_are_preserved(self):
        inspect_region = TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["region"]["description"]
        search_region = TOOL_SCHEMAS[2]["function"]["parameters"]["properties"]["region"]["description"]
        self.assertIn("region='joint'", inspect_region)
        self.assertNotIn("joint", search_region)
        facts = _tool_facts({"status": "OK", "data": {"regions": [{
            "modality": "ir", "unmapped_candidate_ids": ["P1"], "view": "unregistered_global",
        }]}}, "inspect")
        self.assertEqual(facts["data"]["regions"][0]["unmapped_candidate_ids"], ["P1"])

    def test_build_messages_refreshes_candidate_snapshot_without_rewriting_tool_history(self):
        call = {"name": "inspect", "arguments": {"ids": ["P2"], "modalities": ["rgb"]}}
        assistant = {"role": "assistant", "content": "Checking P2.", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "inspect", "arguments": '{"ids":["P2"],"modalities":["rgb"]}'}}]}
        reply = {"role": "tool", "name": "inspect", "tool_call_id": "call-1",
                 "content": [{"type": "text", "text": '{"status":"UNKNOWN"}'}]}
        candidates = self.tools.public_candidates() + [{
            "id": "P8", "role": "target", "bbox": [.7, .1, .9, .4], "sources": [],
            "coordinate_frame": "rgb", "finish_eligible": True,
        }]
        messages = build_messages(self.initial, [{"assistant_message": assistant, "tool_message": reply}],
                                  candidates, [], {}, "Choose or continue.")
        initial_text = next(part["text"] for part in messages[1]["content"] if part.get("type") == "text")
        self.assertIn('"id":"P8"', initial_text)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], messages[3]["tool_call_id"])
        self.assertEqual(messages[3]["content"][0]["text"], '{"status":"UNKNOWN"}')

    def test_compacted_search_facts_keep_no_new_evidence_state(self):
        message = _tool_message("call-search", "search", {
            "status": "OK", "text": "Only existing evidence was returned.", "images": [],
            "data": {"appended_ids": [], "found_ids": ["P2"], "no_new_evidence": True,
                     "new_evidence": False, "cached": True, "cross_modal_identity": "unknown"},
        })
        facts = message["content"][0]["text"]
        self.assertIn('"no_new_evidence":true', facts)
        self.assertIn('"new_evidence":false', facts)
        self.assertIn('"status":"OK"', facts)


if __name__ == "__main__":
    unittest.main()
