import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from experiments.evidence_decision import codex_teacher


class CodexTeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fresh = self.root / "teachers" / "fresh"
        self.output = self.root / "actions"
        self.fresh.mkdir(parents=True)
        self.states = {}
        for sample_id in ("sample-a", "sample-b"):
            sample_dir = self.fresh / sample_id
            sample_dir.mkdir()
            image_path = self.root / f"{sample_id}.png"
            Image.new("RGB", (20, 12), (120, 80, 40)).save(image_path)
            messages = [
                {"role": "system", "content": "Native system instruction stays verbatim."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"Query for {sample_id}."},
                    {"type": "image", "image": str(image_path), "modality": "rgb", "view": "tool",
                     "candidate_ids": ["P1"]},
                    {"type": "text", "text": "Observed target description."},
                ]},
            ]
            tools = [{"type": "function", "function": {"name": "inspect", "description": "Inspect.",
                      "parameters": {"type": "object", "properties": {"ids": {"type": "array"}}}}},
                     {"type": "function", "function": {"name": "finish", "description": "Finish.",
                      "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}}]
            (sample_dir / "current_messages.json").write_text(json.dumps(messages), encoding="utf-8")
            (sample_dir / "current_tools.json").write_text(json.dumps(tools), encoding="utf-8")
            self.states[sample_id] = (messages, tools)

    def tearDown(self):
        self.temp.cleanup()

    def _codex_response(self, command, **kwargs):
        output_path = Path(command[command.index("-o") + 1])
        if "review only the supplied current state" not in kwargs["input"]:
            response = {"actions": [
                {"id": "sample-a", "note": "Compare the two boxes.", "name": "inspect",
                 "arguments": '{"ids":["P1"]}'},
                {"id": "sample-b", "note": "The visible evidence is sufficient.", "name": "finish",
                 "arguments": {"id": "P1"}},
            ]}
        else:
            response = {"reviews": [
                {"id": "sample-a", "use_action": True, "note_supported": True,
                 "reason": "The requested comparison matches the open uncertainty."},
                {"id": "sample-b", "use_action": False, "note_supported": False,
                 "reason": "The note claims evidence absent from the state."},
            ]}
        output_path.write_text(json.dumps(response), encoding="utf-8")
        self.command = command
        self.prompt = kwargs["input"]
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def test_teacher_reads_prepared_state_and_writes_one_object_action_each(self):
        with patch.object(codex_teacher.subprocess, "run", side_effect=self._codex_response):
            self.assertEqual(codex_teacher.main([
                "--fresh-dir", str(self.fresh), "--output-dir", str(self.output),
            ]), 0)

        action_a = json.loads((self.output / "sample-a.action.json").read_text(encoding="utf-8"))
        action_b = json.loads((self.output / "sample-b.action.json").read_text(encoding="utf-8"))
        log = json.loads((self.output / "codex_teacher_batch_001.log.json").read_text(encoding="utf-8"))
        self.assertEqual(action_a["arguments"], {"ids": ["P1"]})
        self.assertEqual(action_b["name"], "finish")
        self.assertEqual(log["returncode"], 0)
        self.assertIn("--output-schema", log["argv"])
        self.assertIn("Native system instruction stays verbatim.", self.prompt)
        self.assertIn("Observed target description.", self.prompt)
        self.assertIn("IMAGE 1", self.prompt)
        self.assertIn("sample=sample-a", self.prompt)
        self.assertIn("sample=sample-b", self.prompt)
        self.assertEqual(self.command[self.command.index("-m") + 1], "gpt-6-luna")
        self.assertIn("--ephemeral", self.command)
        self.assertIn("--ignore-user-config", self.command)
        self.assertIn("--sandbox", self.command)
        self.assertIn('forced_login_method="chatgpt"', self.command)
        self.assertIn('model_reasoning_effort="max"', self.command)
        image_args = [self.command[index + 1] for index, value in enumerate(self.command[:-1]) if value == "-i"]
        self.assertEqual(image_args, [str((self.root / "sample-a.png").resolve()),
                                      str((self.root / "sample-b.png").resolve())])

    def test_blind_review_is_pending_and_does_not_auto_accept(self):
        self.output.mkdir()
        for sample_id in self.states:
            action = {"id": sample_id, "note": "Use only visible facts.", "name": "finish", "arguments": {"id": "P1"}}
            (self.output / f"{sample_id}.action.json").write_text(json.dumps(action), encoding="utf-8")
        review_output = self.root / "reviews"
        with patch.object(codex_teacher.subprocess, "run", side_effect=self._codex_response):
            codex_teacher.main([
                "--fresh-dir", str(self.fresh), "--mode", "review", "--actions-dir", str(self.output),
                "--output-dir", str(review_output),
            ])
        review = json.loads((review_output / "sample-a.review.json").read_text(encoding="utf-8"))
        log = json.loads((review_output / "codex_reviewer_batch_001.log.json").read_text(encoding="utf-8"))
        self.assertEqual(review["acceptance"], "pending")
        self.assertTrue(review["use_action"])
        self.assertTrue(review["note_supported"])
        self.assertNotIn("accepted", review)
        self.assertIn("current state and the selected action", self.prompt)
        self.assertEqual(log["returncode"], 0)

    def test_native_calls_and_reply_binding_reach_teacher_and_reviewer(self):
        messages = [
            {"role": "assistant", "content": "Find missing robots.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "search", "arguments": '{"category":"robot","region":"bottom"}'}}]},
            {"role": "tool", "name": "search", "tool_call_id": "call_1",
             "content": [{"type": "text", "text": "No new evidence."}]},
        ]
        state = {"id": "robot", "messages": messages, "tools": self.states["sample-a"][1]}
        for mode in ("teacher", "review"):
            prompt, _ = codex_teacher._prompt([state], mode, {"robot": {
                "id": "robot", "name": "search", "arguments": {"region": "right"}}})
            self.assertIn(json.dumps(messages[0]["tool_calls"], ensure_ascii=False), prompt)
            self.assertIn('"tool_call_id": "call_1"', prompt)
            self.assertIn('"name": "search"', prompt)

    def test_shared_prepared_image_is_attached_once_with_both_sample_bindings(self):
        picture = str(self.root / 'sample-a.png')
        states = [{"id": sample_id, "messages": [{"role": "user", "content": [
            {"type": "image", "image": picture, "modality": "rgb"}]}],
                   "tools": self.states["sample-a"][1]} for sample_id in ('one', 'two')]
        prompt, images = codex_teacher._prompt(states, 'teacher')
        self.assertEqual(images, [picture])
        self.assertIn('CLI image 1: sample=one', prompt)
        self.assertIn('CLI image 1: sample=two', prompt)
        self.assertEqual(prompt.count('[ATTACHED IMAGE 1;'), 2)


if __name__ == "__main__":
    unittest.main()
