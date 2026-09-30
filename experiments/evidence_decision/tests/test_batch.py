import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.evidence_decision import batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.common = ["--run-root", str(self.root / "runs"), "--code-root", str(self.root),
                       "--manifests", str(self.root / "manifests"),
                       "--candidate-cache", str(self.root / "candidates.jsonl"),
                       "--model", "base", "--c-adapter", "adapter",
                       "--budget-hours", "12", "--python", "python"]

    def tearDown(self):
        self.temp.cleanup()

    def args(self, phase, *extra):
        return batch.parser().parse_args(["--phase", phase, *self.common, *extra])

    def test_frozen_phase_lists_and_dry_run_no_files(self):
        expected = {"debug": 4, "dev-fast": 10, "dev-capacity": 4, "full": 2, "t-generate": 1}
        options = {
            "debug": ["--limit", "4"],
            "dev-fast": [],
            "dev-capacity": ["--controllers", "native", "--profile", "capacity24"],
            "full": ["--mechanisms", "native:C:capacity24,c-lora:D:fast"],
            "t-generate": ["--mechanisms", "D"],
        }
        for phase, count in expected.items():
            preview = batch.execute(self.args(phase, *options[phase], "--dry-run"))
            self.assertEqual(preview["status"], "dry_run")
            self.assertEqual(len(preview["tasks"]), count)
            self.assertTrue(all(task["argv"][:3] == ["python", "-m", "experiments.evidence_decision.run"]
                                for task in preview["tasks"]))
        self.assertFalse((self.root / "runs").exists())

    def test_shared_ledger_counts_each_invocation_once_and_resumes_partial(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "run_config.json").write_text("{}", encoding="utf-8")
            complete = len(calls) != 1
            (directory / "execution.json").write_text(json.dumps({"complete": complete}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        args = self.args("debug", "--limit", "4", "--budget-ledger", "z_budget.jsonl")
        with patch.object(batch.subprocess, "run", side_effect=fake_run):
            first = batch.execute(args)
            self.assertEqual(first["status"], "budget_interrupted")
            args.resume = True
            second = batch.execute(args)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(len(calls), 5)  # first task attempted twice, then three others
        self.assertIn("--resume", calls[1])
        ledger = batch._jsonl(self.root / "runs" / "z_budget.jsonl")
        self.assertEqual(len(ledger), 5)
        self.assertEqual(sum(item["outcome"] == "budget_interrupted" for item in ledger), 1)
        self.assertAlmostEqual(sum(item["elapsed_seconds"] for item in ledger),
                               second["used_hours"] * 3600)

    def test_shared_budget_is_not_reset_by_new_phase(self):
        ledger = self.root / "runs" / "z_budget.jsonl"
        batch._append_jsonl(ledger, {"elapsed_seconds": 12 * 3600, "budget_hours": 12,
                                      "phase": "debug", "outcome": "budget_interrupted"})
        result = batch.execute(self.args("dev-fast"))
        self.assertEqual(result["status"], "budget_interrupted")
        self.assertEqual(len(batch._jsonl(ledger)), 1)

    def test_resume_rejects_changed_frozen_config(self):
        args = self.args("debug", "--limit", "4")
        with patch.object(batch.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            self.assertEqual(batch.execute(args)["status"], "failed")
        args.resume = True
        args.model = "different-base"
        with self.assertRaisesRegex(ValueError, "resume batch config differs"):
            batch.execute(args)

    def test_full_native_sft_is_untrained_control_with_latest_memory(self):
        preview = batch.execute(self.args("full", "--mechanisms", "native:D:sft,t-lora:D:sft",
                                          "--t-adapter", "t-adapter", "--dry-run"))
        self.assertEqual(len(preview["tasks"]), 2)
        native, trained = (task["argv"] for task in preview["tasks"])
        self.assertIn("--memory", native)
        self.assertEqual(native[native.index("--memory") + 1], "latest")
        self.assertNotIn("--adapter", native)
        self.assertIn("--adapter", trained)
        self.assertEqual(trained[trained.index("--adapter") + 1], "t-adapter")
        with self.assertRaisesRegex(ValueError, "t-lora full entry must use sft"):
            batch.task_specs(self.args("full", "--mechanisms", "t-lora:D:fast"))


if __name__ == "__main__":
    unittest.main()
