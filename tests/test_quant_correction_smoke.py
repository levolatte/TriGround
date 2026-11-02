"""CPU 夹具验证 GPU smoke 入口、四分流检查与检查点恢复闭环。"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_quant_correction_runtime import TinyPeft, TinyTemplate
from tools.quant_correction import QuestionSampler, classify
from tools.quant_correction_runtime import CorrectionRuntime
from tools.smoke_quant_correction import (main, run_smoke, synthetic_completions,
                                           _cache_equivalence, _retention_rng)


class SavedTinyPeft(TinyPeft):
    def save_pretrained(self, path, **kwargs):
        super().save_pretrained(path, **kwargs)
        state = {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters()
                 if ".default." in name}
        torch.save(state, Path(path) / "mock_adapter.pt")


def factory_calls():
    calls = []

    def factory(config, mode, adapter_path=None):
        calls.append(adapter_path)
        # 每次重新建立相同冻结底座与 C550 参考；只从检查点覆盖 default adapter。
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = SavedTinyPeft()
        if adapter_path:
            saved = torch.load(Path(adapter_path) / "mock_adapter.pt", weights_only=False)
            named = dict(model.named_parameters())
            with torch.no_grad():
                for name, tensor in saved.items():
                    named[name].copy_(tensor)
        with patch("tools.quant_correction_runtime._load_components", return_value=(model, TinyTemplate())):
            return CorrectionRuntime(config, mode, adapter_path)

    return factory, calls


class CorrectionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = {
            "model": "cpu-mock", "model_type": "qwen3_5", "model_revision": None,
            "init_adapter": str(self.root / "C550"), "media_root": str(self.root),
            "train_index": str(self.root / "train.jsonl"), "seed": 2026,
            "gradient_checkpointing": True, "max_length": 128, "learning_rate": 1e-3,
            "beta": 0.01, "keep_probability": 0.2,
            "stop_presentations": 4400, "checkpoint_every": 400,
        }
        self.rows = [{"id": f"q{index}", "images": ["rgb.png", "ir.png"],
                      "messages": [{"role": "user", "content": "<image><image> locate"}],
                      "solution": [0.0, 0.0, 1.0, 1.0]} for index in range(8)]
        first = {row["id"] for row in QuestionSampler(self.rows, 2026).next_batch()}
        # 保证正式四题夹具之外才有三图，检查 smoke 会先补三图生成和真实反向。
        extra = next(row for row in self.rows if row["id"] not in first)
        extra["images"].append("depth.png")
        extra["messages"][0]["content"] = "<image><image><image> locate"
        self._write_rows()

    def _write_rows(self):
        Path(self.config["train_index"]).write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_cpu_fixture_runs_four_routes_all_skip_cache_and_reload(self):
        factory, calls = factory_calls()
        output = self.root / "smoke"
        report = run_smoke(self.config, output, runtime_factory=factory, require_cuda=False)
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["counts_toward_formal_4400"])
        self.assertEqual(report["four_question_step"]["routes"], ["mixed", "gt", "retain", "skip"])
        self.assertEqual(report["four_question_step"]["optimizer_updates"], 1)
        self.assertGreater(report["four_question_step"]["parameter_change"]["max_abs_change"], 0)
        self.assertEqual(report["all_skip_step"]["optimizer_updates"], 0)
        self.assertFalse(report["all_skip_step"]["affects_checkpoint_sampler"])
        self.assertTrue(report["restore"]["mixed_pool"])
        self.assertEqual(report["real_image_counts"], [2, 3])
        self.assertEqual(report["real_backward_image_counts"], [2, 3])
        self.assertEqual(report["preflight_gt_backward"][0]["image_count"], 3)
        self.assertTrue(report["preflight_gt_backward"][0]["finite_nonzero_gradients"])
        self.assertTrue(report["cache_equivalence"]["all_trainable_gradients_exact"])
        self.assertEqual((report["cache_equivalence"]["atol"], report["cache_equivalence"]["rtol"]), (0, 0))
        self.assertEqual(len(report["cache_equivalence"]["runs"]), 4)
        self.assertEqual(set(report["cache_equivalence"]["comparisons"]), {
            "cached_repeat", "native_repeat", "cached_vs_native_0", "cached_vs_native_1"})
        self.assertGreater(report["sft_four_question_step"]["seconds"], 0)
        self.assertGreater(report["sft_four_question_step"]["parameter_change"]["max_abs_change"], 0)
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(progress["stage"], "passed")
        self.assertGreater(progress["four_question_step"]["seconds"], 0)
        self.assertGreater(progress["sft_four_question_step"]["seconds"], 0)
        self.assertTrue(all(audit["visual_forwards"] == 1 for audit in report["prompt_and_visual_audits"]))
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0])
        self.assertEqual(calls[1], report["checkpoint"])
        self.assertTrue(all(report["restore"][key] for key in (
            "adapter_exact", "reference_exact", "optimizer_exact", "rng_exact", "random_draws_exact",
            "retention_rng_exact", "previous_model_released")))
        self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8")), report)
        actual = [json.loads(line) for line in (output / "real_generations.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(actual), 5)
        self.assertTrue(all(record["source"] == "actual_hf_g2" and not record["synthetic"] for record in actual))
        fixtures = [json.loads(line) for line in (output / "synthetic_branch_records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(fixtures), 8)
        self.assertTrue(all(record["synthetic"] for record in fixtures))
        self.assertFalse(Path(self.config["init_adapter"]).exists())

    def test_synthetic_fixtures_keep_raw_json_tokens_and_form_expected_classes(self):
        factory, _ = factory_calls()
        runtime = factory(self.config, "correction")
        for kind, expected in (("mixed", "mixed"), ("gt", "all_wrong"), ("retain", "all_hit"), ("skip", "all_hit")):
            completions = synthetic_completions(runtime, self.rows[0], kind)
            self.assertEqual(classify(self.rows[0], completions)["route"], expected)
            self.assertTrue(all(item.token_ids[-1] == runtime.answer_eos_id for item in completions))

    def test_retention_rng_uses_original_probability_for_both_branch_patterns(self):
        rng, seed = _retention_rng(2027, 0.2)
        self.assertIsInstance(rng, random.Random)
        self.assertEqual(rng.getstate(), random.Random(seed).getstate())
        self.assertLess(rng.random(), 0.2)
        rng.randrange(2)
        self.assertGreaterEqual(rng.random(), 0.2)
        skip, _ = _retention_rng(2027, 0.2, all_skip=True)
        self.assertTrue(all(skip.random() >= 0.2 for _ in range(4)))

    def test_cache_checker_rejects_any_logp_change_without_relaxing_tolerance(self):
        factory, _ = factory_calls()
        runtime = factory(self.config, "correction")
        original = runtime.multimodal.get_image_features
        count = 0

        def changing_features(*args, **kwargs):
            nonlocal count
            count += 1
            value = original(*args, **kwargs)
            if count == 2:
                value.pooler_output = tuple(tensor + 0.01 for tensor in value.pooler_output)
            return value

        runtime.multimodal.get_image_features = changing_features
        output = self.root / "cache_failure"
        output.mkdir()
        with self.assertRaises(AssertionError):
            _cache_equivalence(runtime, self.rows[0], [7, 8, 2], output)
        evidence = json.loads((output / "cache_diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(evidence["runs"]), 4)
        noise = evidence["comparisons"]["native_repeat"]
        self.assertGreater(noise["logps"]["nonzero_differences"], 0)
        self.assertGreater(noise["gradients"]["max_abs"], 0)
        self.assertGreater(noise["gradients"]["global_relative_l2"], 0)
        self.assertIsNotNone(noise["gradients"]["first_module"])

    def test_step_timings_are_preserved_when_cache_gate_later_fails(self):
        factory, _ = factory_calls()
        output = self.root / "after_step_failure"
        with patch("tools.smoke_quant_correction._cache_equivalence", side_effect=AssertionError("strict cache failure")):
            with self.assertRaisesRegex(AssertionError, "strict cache"):
                run_smoke(self.config, output, runtime_factory=factory, require_cuda=False)
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        self.assertGreater(progress["four_question_step"]["seconds"], 0)
        self.assertGreater(progress["sft_four_question_step"]["seconds"], 0)
        self.assertEqual(len(progress["real_g2_timings"]), 5)
        self.assertFalse((output / "report.json").exists())

    def test_cli_writes_failed_report_and_nonzero_exit_on_runtime_error(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        output = self.root / "failure"
        with patch("tools.smoke_quant_correction.run_smoke", side_effect=RuntimeError("mock GPU failure")), redirect_stderr(StringIO()):
            code = main(["--config", str(config_path), "--output-dir", str(output)])
        self.assertEqual(code, 1)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "mock GPU failure")

    def test_c550_output_is_rejected_before_any_write(self):
        factory, _ = factory_calls()
        for output in (Path(self.config["init_adapter"]), Path(self.config["init_adapter"]) / "smoke"):
            with self.assertRaisesRegex(ValueError, "C550"):
                run_smoke(self.config, output, runtime_factory=factory, require_cuda=False)
        self.assertFalse(Path(self.config["init_adapter"]).exists())


if __name__ == "__main__":
    unittest.main()
