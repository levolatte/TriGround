"""Run profile and frozen prompt-version configuration checks."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "code"))
from experiments.evidence_decision import object_controller, run_objects


class RunObjectsTests(unittest.TestCase):
    def test_context_default_and_override_are_saved_with_current_prompt_version(self):
        base_args = ["--manifest", "manifest.jsonl", "--candidate-cache", "candidates.jsonl",
                     "--model", "model", "--adapter", "adapter", "--output-dir", "out"]
        default = run_objects.parser().parse_args(base_args)
        explicit = run_objects.parser().parse_args(base_args + ["--context-tokens", "12288"])
        default_config = run_objects.run_config(
            default, replace(run_objects.OBJECT_PROFILE, context_tokens=default.context_tokens))
        explicit_config = run_objects.run_config(
            explicit, replace(run_objects.OBJECT_PROFILE, context_tokens=explicit.context_tokens))
        self.assertEqual(default.context_tokens, 8192)
        self.assertEqual(explicit_config["profile"]["context_tokens"], 12288)
        self.assertEqual(default_config["prompt_version"], object_controller.PROMPT_VERSION)
        self.assertEqual(explicit_config["prompt_version"], object_controller.PROMPT_VERSION)


if __name__ == "__main__":
    unittest.main()
