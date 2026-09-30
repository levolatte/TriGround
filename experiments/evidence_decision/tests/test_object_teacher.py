import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.evidence_decision.object_teacher import (
    export_episode,
    init_episode,
    step_episode,
)
from experiments.evidence_decision.object_controller import PROMPT_VERSION
from experiments.evidence_decision.object_tools import ObjectTools


class ObjectTeacherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rgb = self.root / "rgb.png"
        Image.new("RGB", (160, 96), (214, 18, 36)).save(self.rgb)
        self.manifest = self.root / "manifest.jsonl"
        self.cache = self.root / "candidates.jsonl"
        self.row = {
            "id": "teacher-sample", "query": "the red object beside the car",
            "images": {"rgb": str(self.rgb), "ir": None, "depth_visual": None, "depth_raw": None},
            "depth_encoding": "unknown", "ir_rgb_registration": "normalized_shared_frame",
        }
        self.cache_row = {
            "id": "teacher-sample", "images": self.row["images"],
            "depth_encoding": "unknown", "c_bbox": [.1, .15, .3, .8],
            "query_info": {"scope": "single"},
            "candidates": [
                {"id": 1, "role": "target", "bbox": [.1, .15, .3, .8],
                 "is_baseline": True, "sources": [], "depth": {"status": "pending"}},
                {"id": 2, "role": "reference", "bbox": [.55, .2, .78, .78],
                 "is_baseline": False, "sources": [], "depth": {"status": "pending"}},
            ],
        }
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        self.cache.write_text(json.dumps(self.cache_row) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_teacher_steps_use_real_images_persist_pool_and_export_only_valid_labels(self):
        episode_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "episode")
        state = json.loads(episode_path.read_text(encoding="utf-8"))
        self.assertEqual(state["row"]["ir_rgb_registration"], "normalized_shared_frame")
        self.assertTrue((episode_path.parent / "current_messages.json").is_file())
        initial_ids = [candidate["id"] for candidate in state["initial_candidates"]]
        target_id = next(candidate["id"] for candidate in state["initial_candidates"]
                         if candidate["role"] == "target")

        first = step_episode(episode_path, "Check this object's appearance.", "inspect",
                             {"ids": [target_id], "modalities": ["rgb"]})
        self.assertEqual(first["observation"]["status"], "OK")
        self.assertTrue(first["protocol_valid"])
        self.assertEqual(first["tools"][0]["function"]["name"], "inspect")
        self.assertTrue(any(part.get("type") == "image"
                            for message in json.loads(episode_path.read_text())["current_messages"]
                            for part in (message.get("content") if isinstance(message.get("content"), list) else [])))
        with Image.open(first["observation"]["images"][0]["path"]) as view:
            self.assertGreater(view.width, 0)
            self.assertGreater(view.height, 0)

        detection = [{"bbox": [.6, .2, .85, .8], "score": .94}]
        with patch("experiments.evidence_decision.object_tools._ObjectVisualTools._dino",
                   return_value=(object(), object(), "cpu")), patch(
                       "experiments.evidence_decision.vision_tools.detect_image_proposals",
                       return_value=detection):
            search = step_episode(episode_path, "Check for a missed red object.", "search",
                                  {"category": "object", "modality": "rgb", "role": "target"})
        self.assertEqual(search["observation"]["status"], "OK")
        added_id = search["observation"]["data"]["appended_ids"][0]
        self.assertTrue(added_id.startswith("P"))
        after_search = json.loads(episode_path.read_text(encoding="utf-8"))
        self.assertEqual(after_search["tool_state"]["searches"], 1)
        self.assertEqual(after_search["tool_state"]["new_candidates"], 1)
        self.assertEqual(set(after_search["tool_state"]["public_to_raw"]), set(initial_ids) | {added_id})

        invalid = step_episode(episode_path, "Try to finish with an absent ID.", "finish", {"id": "P99"})
        self.assertEqual(invalid["observation"]["status"], "ERROR")
        self.assertFalse(invalid["protocol_valid"])
        final = step_episode(episode_path, "This candidate matches the query.", "finish", {"id": added_id})
        self.assertEqual(final["observation"]["status"], "OK")
        for actual, expected in zip(final["observation"]["data"]["bbox"], [.6, .2, .85, .8]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(json.loads(episode_path.read_text())["complete"])

        output = self.root / "teacher.jsonl"
        rows = export_episode(episode_path, output)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["schema_version"], "visual-agent-object-teacher-v3")
        self.assertEqual(rows[-1]["finish_source"], "candidate")
        self.assertTrue(all("<tool_call>" in row["target_action"] for row in rows))
        self.assertTrue(all(row["label_source"] == "gpt-6-luna_max_teacher"
                            and row["supervision_status"] == "candidate_teacher_supervision"
                            for row in rows))
        self.assertFalse(any('"id":"P99"' in row["target_action"] for row in rows))
        self.assertEqual(rows[-1]["tools"][-1]["function"]["name"], "finish")
        self.assertTrue(any(part.get("type") == "image"
                            for message in rows[-1]["messages"]
                            for part in (message.get("content") if isinstance(message.get("content"), list) else [])))

    def test_resume_prefix_rebuilds_public_pool_and_keeps_student_errors_unlabeled(self):
        original_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "student")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        with patch("experiments.evidence_decision.object_tools._ObjectVisualTools._dino",
                   return_value=(object(), object(), "cpu")), patch(
                       "experiments.evidence_decision.vision_tools.detect_image_proposals",
                       return_value=[{"bbox": [.62, .2, .84, .75], "score": .9}]):
            tool_step = step_episode(original_path, "Find a missed candidate.", "search",
                                     {"category": "object", "modality": "rgb"})
        student_state = json.loads(original_path.read_text(encoding="utf-8"))
        search_event = {
            "step": 0,
            "action": {"name": "search", "arguments": {"category": "object", "modality": "rgb"}},
            "raw_output": 'Find a missed candidate. <tool_call>{"name":"search","arguments":{"category":"object","modality":"rgb"}}</tool_call>',
            "observation": tool_step["observation"],
            "candidates_after": tool_step["candidates_after"],
            "tool_seconds": .01,
        }
        bad_event = {"step": 1, "action": None, "raw_output": "malformed student output",
                     "observation": {"status": "ERROR", "text": "expected one tool call", "images": [], "data": {}}}
        trace = self.root / "student_trace.json"
        trace.write_text(json.dumps({
            "schema_version": "visual-agent-object-v3", "prompt_version": PROMPT_VERSION,
            "id": "teacher-sample", "initial_candidates": original["initial_candidates"],
            "initial_messages": original["initial_messages"], "events": [search_event, bad_event],
        }), encoding="utf-8")

        resumed_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "resumed",
                                    resume_prefix=trace)
        resumed = json.loads(resumed_path.read_text(encoding="utf-8"))
        self.assertEqual(resumed["tool_state"]["searches"], 1)
        self.assertEqual(resumed["tool_state"]["new_candidates"], 1)
        self.assertEqual(set(resumed["tool_state"]["public_to_raw"]),
                         {candidate["id"] for candidate in tool_step["candidates_after"]})
        self.assertEqual(len(export_episode(resumed_path)), 0)
        self.assertIn("malformed student output", json.dumps(resumed["current_messages"]))

        added_id = tool_step["observation"]["data"]["appended_ids"][0]
        final = step_episode(resumed_path, "Finish with the candidate found in the search.", "finish",
                             {"id": added_id})
        self.assertEqual(final["observation"]["status"], "OK")
        self.assertEqual(len(export_episode(resumed_path)), 1)

    def test_teacher_accepts_predicted_bbox_finish_without_candidate_id(self):
        episode_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "bbox-episode")
        bbox = [.22, .25, .52, .75]
        final = step_episode(episode_path, "The target occupies this RGB region.", "finish", {"bbox": bbox})
        state = json.loads(episode_path.read_text(encoding="utf-8"))
        self.assertEqual(final["observation"]["status"], "OK")
        self.assertEqual(state["finish_source"], "predicted_bbox")
        self.assertTrue(state["complete"])
        self.assertIsNone(final["observation"]["data"]["candidate_id"])
        rows = export_episode(episode_path)
        self.assertEqual(rows[-1]["finish_source"], "predicted_bbox")
        self.assertIn('"bbox":[0.22,0.25,0.52,0.75]', rows[-1]["target_action"])

    def test_resume_rejects_old_message_history_version(self):
        original_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "old-trace-source")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        trace = self.root / "old_trace.json"
        trace.write_text(json.dumps({
            "schema_version": "visual-agent-object-v2", "prompt_version": "paired-evidence-capability-training",
            "id": "teacher-sample", "initial_candidates": original["initial_candidates"],
            "initial_messages": original["initial_messages"], "events": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "history version"):
            init_episode(self.manifest, self.cache, "teacher-sample", self.root / "old-trace-target",
                         resume_prefix=trace)

        trace.write_text(json.dumps({
            "schema_version": "visual-agent-object-v1", "prompt_version": "unknown-prompt",
            "id": "teacher-sample", "initial_candidates": original["initial_candidates"], "events": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "history version"):
            init_episode(self.manifest, self.cache, "teacher-sample", self.root / "unknown-trace-target",
                         resume_prefix=trace, allow_legacy_prefix_migration=True)

    def test_explicit_legacy_trace_migration_rebuilds_messages_from_real_tool_events(self):
        source_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "legacy-source")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        target_id = next(item["id"] for item in source["initial_candidates"] if item["role"] == "target")
        step = step_episode(source_path, "Inspect this object.", "inspect",
                            {"ids": [target_id], "modalities": ["rgb"]})
        trace = self.root / "legacy_prefix.json"
        trace.write_text(json.dumps({
            "schema_version": "visual-agent-object-v2",
            "prompt_version": "paired-evidence-capability-training",
            "id": "teacher-sample",
            "initial_candidates": source["initial_candidates"],
            "initial_messages": [{"role": "system", "content": "OLD_PROMPT_MUST_NOT_SURVIVE"}],
            "events": [{
                "step": 0,
                "action": step["call"],
                "raw_output": step["target_action"],
                "observation": step["observation"],
                "candidates_after": step["candidates_after"],
                "tool_seconds": .01,
            }, {
                "step": 1, "action": None, "raw_output": "malformed legacy prose",
                "observation": {"status": "ERROR", "text": "bad protocol", "images": [], "data": {}},
            }, {
                "step": 2, "action": None, "raw_output": "",
                "error": {"type": "InputBudgetExceeded", "message": "context_token_budget"},
            }],
        }), encoding="utf-8")
        migrated_path = init_episode(
            self.manifest, self.cache, "teacher-sample", self.root / "legacy-migrated",
            resume_prefix=trace, allow_legacy_prefix_migration=True,
        )
        migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["prefix_source_version"],
                         "visual-agent-object-v2/paired-evidence-capability-training")
        self.assertEqual(len(migrated["prefix_events"]), 2)
        self.assertNotIn("OLD_PROMPT_MUST_NOT_SURVIVE", json.dumps(migrated["initial_messages"]))
        self.assertIn("malformed legacy prose", json.dumps(migrated["current_messages"]))
        self.assertIn("Protocol error: bad protocol", json.dumps(migrated["current_messages"]))
        self.assertNotIn("context_token_budget", json.dumps(migrated["current_messages"]))
        calls = [message for message in migrated["current_messages"]
                 if message.get("role") == "assistant" and message.get("tool_calls")]
        replies = [message for message in migrated["current_messages"] if message.get("role") == "tool"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(replies), 1)
        self.assertEqual(calls[0]["tool_calls"][0]["id"], replies[0]["tool_call_id"])
        self.assertEqual(migrated["steps"], [])
        self.assertEqual(export_episode(migrated_path), [])

    def test_explicit_v1_student_trace_migration_uses_events_and_real_candidate_records(self):
        source_path = init_episode(self.manifest, self.cache, "teacher-sample", self.root / "v1-source")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        target_id = next(item["id"] for item in source["initial_candidates"]
                         if item["role"] == "target")
        real_step = step_episode(source_path, "Inspect the target candidate.", "inspect",
                                 {"ids": [target_id], "modalities": ["rgb"]})
        self.assertEqual(real_step["observation"]["status"], "OK")
        self.assertIn("coordinate_frame", real_step["candidates_after"][0])

        legacy_event = {
            "step": 0,
            "action": real_step["call"],
            "raw_output": real_step["target_action"],
            "observation": real_step["observation"],
            "candidates_before": real_step["candidates_before"],
            "candidates_after": real_step["candidates_after"],
            "available_tools": [tool["function"]["name"] for tool in real_step["tools"]],
            "tools": real_step["tools"],
            "protocol_valid": real_step["protocol_valid"],
            "tool_seconds": .01,
            "usage": {},
        }
        trace = self.root / "student200_v1_trace.json"
        trace.write_text(json.dumps({
            "schema_version": "visual-agent-object-v1",
            "prompt_version": "paired-evidence-capability-training",
            "id": "teacher-sample",
            "initial_candidates": source["initial_candidates"],
            "initial_messages": [{"role": "system", "content": "old v1 prompt"}],
            "events": [legacy_event],
        }), encoding="utf-8")

        migrated_path = init_episode(
            self.manifest, self.cache, "teacher-sample", self.root / "v1-migrated",
            resume_prefix=trace, allow_legacy_prefix_migration=True,
        )
        migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["prefix_source_version"],
                         "visual-agent-object-v1/paired-evidence-capability-training")
        self.assertEqual(len(migrated["prefix_events"]), 1)
        self.assertEqual(migrated["turns"][-1]["assistant_message"]["tool_calls"][0]["function"]["name"],
                         real_step["call"]["name"])
        self.assertEqual(migrated["turns"][-1]["tool_message"]["name"], "inspect")
        self.assertEqual(migrated["tool_state"]["action_count"], 1)
        self.assertEqual(len(migrated["initial_candidates"]), len(source["initial_candidates"]))
        self.assertNotIn("old v1 prompt", json.dumps(migrated["initial_messages"]))


if __name__ == "__main__":
    unittest.main()
