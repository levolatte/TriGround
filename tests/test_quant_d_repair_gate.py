"""CPU checks for the four-query D repair learning-signal gate."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.run_quant_d_repair_gate import (_new_reward_items, eval_groups, gate_summary,
                                           formal_training_hook, main, training_hook)

_PREPARE_PATH = (Path(__file__).resolve().parents[1] /
                 "docs/research/2026-10-26-quant-rl/run-scripts/prepare_d_repair_inputs.py")
_PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_d_repair_inputs", _PREPARE_PATH)
prepare_inputs = importlib.util.module_from_spec(_PREPARE_SPEC)
_PREPARE_SPEC.loader.exec_module(prepare_inputs)


def group(name: str, *, geometric: bool = False, contrast_only: bool = False,
          format_only: bool = False) -> dict:
    samples = []
    for index in range(8):
        if format_only:
            valid = index == 0
            iou = 0.3 if valid else None
            contrast = 0.0
            reward = 0.3 if valid else -0.2
        else:
            valid = True
            iou = 0.3 if not geometric or index < 4 else 0.5
            contrast = 0.0 if not contrast_only or index < 4 else 0.3
            reward = iou + contrast
        samples.append({"score": {"valid": valid, "iou": iou,
                                   "contrast": contrast, "reward": reward,
                                   "truncated": False}})
    return {"id": name, "samples": samples}


def test_two_groups_with_valid_geometry_reward_difference_pass():
    rows = [group("city", geometric=True), group("rgbdt", geometric=True),
            group("robo"), group("range")]
    summary = gate_summary(rows)
    assert summary["passed"]
    assert summary["geometry_reward_groups"] == 2
    assert summary["valid"] == 32


def test_negative_box_contrast_can_supply_geometry_signal():
    rows = [group("city", contrast_only=True), group("rgbdt", contrast_only=True),
            group("robo"), group("range")]
    assert gate_summary(rows)["passed"]


def test_format_only_reward_variation_does_not_pass():
    rows = [group(str(index), format_only=True) for index in range(4)]
    summary = gate_summary(rows)
    assert not summary["passed"]
    assert summary["valid"] == 4
    assert summary["geometry_reward_groups"] == 0


def test_one_geometric_group_is_insufficient():
    rows = [group("city", geometric=True), group("rgbdt"), group("robo"), group("range")]
    assert not gate_summary(rows)["passed"]


def test_twelve_group_extension_rate_gates():
    rows = [group(str(index), geometric=index < 2) for index in range(12)]
    assert gate_summary(rows, expected_queries=12,
                        min_valid_rate=0.95, max_truncated_rate=0.05)["passed"]
    rows[2]["samples"][0]["score"]["valid"] = False
    rows[2]["samples"][0]["score"]["truncated"] = True
    for index in range(3, 8):
        rows[2]["samples"][index]["score"]["valid"] = False
        rows[2]["samples"][index]["score"]["truncated"] = True
    summary = gate_summary(rows, expected_queries=12,
                           min_valid_rate=0.95, max_truncated_rate=0.05)
    assert not summary["passed"]
    assert summary["geometry_reward_groups"] == 2
    assert summary["truncated_rate"] > 0.05


def test_requires_four_distinct_g8_groups():
    rows = [group("duplicate") for _ in range(4)]
    with pytest.raises(ValueError, match="distinct"):
        gate_summary(rows)
    rows = [group(str(index)) for index in range(4)]
    rows[0]["samples"].pop()
    with pytest.raises(ValueError, match="eight"):
        gate_summary(rows)


def test_eval_predictions_cli_and_reward_log_reader(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    rows = [group("city", geometric=True), group("rgbdt", geometric=True),
            group("robo"), group("range")]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = tmp_path / "gate.json"
    assert main(["generate-check", "--predictions", str(predictions),
                 "--result", str(result)])["passed"]
    assert json.loads(result.read_text(encoding="utf-8"))["geometry_reward_groups"] == 2
    assert len(eval_groups(rows)) == 4

    metrics = tmp_path / "rewards.jsonl"
    payload = {"items": [{"id": "city", "valid": True, "iou": 0.3,
                          "contrast": 0.0, "reward": 0.3} for _ in range(8)]}
    metrics.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    items, offset = _new_reward_items(metrics, 0)
    assert len(items) == 8 and offset == metrics.stat().st_size


def test_prepare_fixed_hard4_and_untrained_train12_excludes_hard_scenes(tmp_path):
    evidence = (Path(__file__).resolve().parents[1] /
                "docs/research/2026-10-26-quant-rl/evidence/d-rootcause-formal-diagnostic128.jsonl")
    rows = [json.loads(line) for line in evidence.read_text(encoding="utf-8").splitlines()]
    output = tmp_path / "prepared"
    prepare_inputs.main(["--diagnostic", str(evidence), "--output-dir", str(output)])
    train4 = [json.loads(line) for line in (output / "train4.jsonl").read_text(encoding="utf-8").splitlines()]
    train12 = [json.loads(line) for line in (output / "train12.jsonl").read_text(encoding="utf-8").splitlines()]
    hard_ids = set(prepare_inputs.HARD4_IDS)
    hard_scenes = {row["group_id"] for row in train4}
    assert {row["id"] for row in train4} == hard_ids
    assert len(train12) == 12
    assert {source: sum(row["source"] == source for row in train12)
            for source in ("city", "rgbdt", "robo")} == {"city": 6, "rgbdt": 3, "robo": 3}
    assert not ({row["id"] for row in train12} & hard_ids)
    assert not ({row["group_id"] for row in train12} & hard_scenes)
    source_positions = {row["id"]: index for index, row in enumerate(rows)}
    assert [source_positions[row["id"]] for row in train12] == sorted(
        source_positions[row["id"]] for row in train12)


def _fake_training_hook(tmp_path, monkeypatch, *, invalid_count=0, truncated_count=0):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    peft = ModuleType("peft")
    peft.get_peft_model_state_dict = lambda model: {
        "language_model.lora_A.weight": model.weight.detach().clone()
    }
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    import tools.train_quant_grounding as training
    monkeypatch.setattr(training, "audit_gradients", lambda model, lib: True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

    class Trainer:
        def __init__(self):
            self.state = SimpleNamespace(global_step=0)
            self.args = SimpleNamespace(per_device_train_batch_size=1)
            self.model = Model()
            self.callbacks = []
            self._rewards_per_func = None

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def _score_completions(self, samples):
            group_name = samples[0].extra["id"]
            geometric = samples[0].extra["geometric"]
            items = []
            for index in range(8):
                iou = 0.4 + (0.1 if geometric and index >= 4 else 0.0)
                invalid = group_name == "0" and index < invalid_count + truncated_count
                truncated = group_name == "0" and index < truncated_count
                items.append({"id": group_name, "valid": not invalid,
                              "truncated": truncated, "iou": iou if not invalid else None,
                              "contrast": 0.0, "reward": iou if not invalid else -0.2,
                              "generated_text": f'{{"bbox_2d":[{index},0,500,500]}}',
                              "finish_reason": "length" if truncated else "stop"})
            self._rewards_per_func = torch.tensor([[item["reward"]] for item in items])
            with (tmp_path / "reward_metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"items": items}) + "\n")
            return samples

        def _generate_and_score_completions(self, _inputs):
            return [{"grpo_batch": SimpleNamespace(
                completion_mask=torch.ones(1, 2, dtype=torch.bool),
                advantages=torch.ones(1, 2), old_per_token_logps=torch.zeros(1, 2),
                ref_per_token_logps=torch.zeros(1, 2), num_items_in_batch=torch.tensor(16))}
                    for _ in range(8)]

    trainer = Trainer()
    report = {}
    dataset_rows = {str(index): {"id": str(index), "source": "city",
                                 "group_id": f"city:{index}", "solution": [0, 0, 1, 1],
                                 "negative_boxes": [], "messages": []}
                    for index in range(4)}
    cleanup = training_hook(tmp_path, report, dataset_rows)(trainer)
    for index in range(4):
        samples = [SimpleNamespace(
            extra={"id": str(index), "geometric": index < 2},
            response_token_ids=[100 + sample_index, 200 + sample_index],
            rollout_infos={"bounded_thinking": {"token_ids": [100 + sample_index,
                                                                  200 + sample_index],
                                                   "thought_boundary": "natural",
                                                   "sampled_logprobs": [-0.1, -0.2],
                                                   "generation_processors": ["RepetitionPenaltyLogitsProcessor"]}},
        ) for sample_index in range(8)]
        trainer._score_completions(samples)
        trainer._generate_and_score_completions(None)
    return torch, safetensors, trainer, report, cleanup


def test_training_gate_requires_valid_rate_and_zero_truncation(tmp_path, monkeypatch):
    _, _, trainer, report, cleanup = _fake_training_hook(
        tmp_path, monkeypatch, invalid_count=2)
    callback = trainer.callbacks[0]
    with pytest.raises(RuntimeError, match="failed the output/geometry/advantage gate"):
        callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace(should_save=False))
    assert report["train_gate"]["valid_rate"] < 0.95
    cleanup()

    truncated_dir = tmp_path / "truncated"
    truncated_dir.mkdir()
    _, _, trainer, report, cleanup = _fake_training_hook(
        truncated_dir, monkeypatch, truncated_count=1)
    callback = trainer.callbacks[0]
    with pytest.raises(RuntimeError, match="failed the output/geometry/advantage gate"):
        callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace(should_save=False))
    assert report["train_gate"]["truncated"] == 1
    cleanup()


def test_one_step_hook_checks_real_update_and_saved_adapter_tensors(tmp_path, monkeypatch):
    torch, safetensors, trainer, report, cleanup = _fake_training_hook(tmp_path, monkeypatch)
    callback = trainer.callbacks[0]
    control = SimpleNamespace(should_save=False)
    callback.on_pre_optimizer_step(None, trainer.state, control)
    with torch.no_grad():
        trainer.model.weight.add_(1)
    trainer.state.global_step = 1
    callback.on_step_end(None, trainer.state, control)
    assert control.should_save
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    safetensors.save_file(sys.modules["peft"].get_peft_model_state_dict(trainer.model),
                          checkpoint / "adapter_model.safetensors")
    callback.on_save(None, trainer.state, control)
    cleanup()
    assert report["train_gate"]["passed"]
    assert report["parameter_update"]["changed_elements"] == 2
    assert report["saved_adapter_tensor_check"]["saved_values_equal"]
    assert not report["saved_adapter_tensor_check"]["adapter_reloaded"]
    rollout_rows = [json.loads(line) for line in
                    (tmp_path / "first_rollout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rollout_rows) == 32
    assert report["first_rollout_groups"] == 4
    assert rollout_rows[0]["dataset_record"]["solution"] == [0, 0, 1, 1]
    assert rollout_rows[0]["token_ids"] == [100, 200]
    assert rollout_rows[0]["thinking_trace"]["sampled_logprobs"] == [-0.1, -0.2]
    assert rollout_rows[0]["scored"]["reward"] == 0.4


def _fake_formal_trainer(tmp_path, monkeypatch, *, gradients_ok=True,
                         invalid_count=0, truncated_count=0):
    torch = pytest.importorskip("torch")
    transformers = ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    import tools.train_quant_grounding as training
    calls = []
    monkeypatch.setattr(training, "audit_gradients",
                        lambda model, lib: calls.append(True) or gradients_ok)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Trainer:
        def __init__(self):
            self.state = SimpleNamespace(global_step=0)
            self.args = SimpleNamespace(per_device_train_batch_size=1)
            self.model = Model()
            self.callbacks = []
            self._rewards_per_func = None
            self.group_index = 0
            self.current_group_count = 0
            self.current_geometry = False

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def _score_completions(self, samples):
            group_index = self.group_index
            group_id = samples[0].extra["id"]
            geometric = group_index < 2
            self.current_geometry = geometric
            items = []
            for sample_index in range(8):
                iou = 0.3 if not geometric or sample_index < 4 else 0.5
                invalid = group_index == 0 and sample_index < invalid_count + truncated_count
                truncated = group_index == 0 and sample_index < truncated_count
                items.append({"id": group_id, "occurrence_id": f"presentation:{group_index}",
                              "valid": not invalid, "truncated": truncated,
                              "iou": iou if not invalid else None,
                              "contrast": 0.0, "reward": iou if not invalid else -0.2})
            self._rewards_per_func = torch.tensor([[item["reward"]] for item in items])
            with (tmp_path / "reward_metrics.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps({"items": items}) + "\n")
            self.group_index += 1
            self.current_group_count += 1
            return samples

        def _generate_and_score_completions(self, _inputs):
            groups = self.current_group_count
            geometry = self.current_geometry
            self.current_group_count = 0
            old = 0.0 if self.state.global_step == 0 else 0.02
            return [{"grpo_batch": SimpleNamespace(
                completion_mask=torch.ones(1, 2, dtype=torch.bool),
                advantages=torch.ones(1, 2) if geometry else torch.zeros(1, 2),
                old_per_token_logps=torch.full((1, 2), old),
                ref_per_token_logps=torch.zeros(1, 2),
                num_items_in_batch=torch.tensor(groups * 16))}
                    for _ in range(groups * 8)]

    trainer = Trainer()
    report = {}
    cleanup = formal_training_hook(tmp_path, report)(trainer)
    return torch, trainer, report, cleanup, calls


def test_formal_gate_uses_occurrences_and_allows_zero_variance_groups(tmp_path, monkeypatch):
    torch, trainer, report, cleanup, gradient_calls = _fake_formal_trainer(tmp_path, monkeypatch)
    callback = trainer.callbacks[0]
    for step in range(3):
        for _ in range(4):
            query_id = f"query:{trainer.group_index % 11}"
            occurrence_id = f"presentation:{trainer.group_index}"
            samples = [SimpleNamespace(extra={"id": query_id,
                                             "occurrence_id": occurrence_id}) for _ in range(8)]
            trainer._score_completions(samples)
            trainer._generate_and_score_completions(None)
        callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace(),)
        if report["optimizer_step_gates"][-1]["nonzero_advantages"]:
            with torch.no_grad():
                trainer.model.weight.add_(1)
        trainer.state.global_step += 1
        callback.on_step_end(None, trainer.state, SimpleNamespace())
    cleanup()
    gate = json.loads((tmp_path / "first12_gate.json").read_text(encoding="utf-8"))
    assert gate["passed"]
    assert gate["queries"] == 12
    assert gate["geometry_reward_groups"] == 2
    assert gate["zero_variance_groups"] == 10
    assert len({group["id"] for group in gate["groups"]}) == 12
    assert len(gradient_calls) == 1
    assert report["optimizer_step_gates"][0]["answers"] == 32
    assert report["optimizer_step_gates"][0]["valid_rate"] == 1.0
    assert report["optimizer_step_gates"][0]["truncated"] == 0
    assert report["optimizer_step_gates"][0]["parameter_update"]["changed_elements"] == 1
    assert report["optimizer_step_gates"][1]["max_old_ref_logp_gap"] == pytest.approx(0.02)


def test_formal_gate_stops_on_missing_gradient_for_localization_signal(tmp_path, monkeypatch):
    _, trainer, _, cleanup, _ = _fake_formal_trainer(tmp_path, monkeypatch, gradients_ok=False)
    callback = trainer.callbacks[0]
    for _ in range(4):
        occurrence_id = f"presentation:{trainer.group_index}"
        samples = [SimpleNamespace(extra={"id": "query:0",
                                         "occurrence_id": occurrence_id}) for _ in range(8)]
        trainer._score_completions(samples)
        trainer._generate_and_score_completions(None)
    with pytest.raises(RuntimeError, match="despite nonzero advantages"):
        callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace())
    cleanup()


@pytest.mark.parametrize("invalid_count,truncated_count,valid,truncated", [
    (2, 0, 30, 0),
    (0, 1, 31, 1),
])
def test_formal_step_stops_on_validity_or_truncation_failure(
        tmp_path, monkeypatch, invalid_count, truncated_count, valid, truncated):
    _, trainer, report, cleanup, _ = _fake_formal_trainer(
        tmp_path, monkeypatch, invalid_count=invalid_count,
        truncated_count=truncated_count)
    callback = trainer.callbacks[0]
    for _ in range(4):
        occurrence_id = f"presentation:{trainer.group_index}"
        samples = [SimpleNamespace(extra={"id": "query:0",
                                         "occurrence_id": occurrence_id}) for _ in range(8)]
        trainer._score_completions(samples)
        trainer._generate_and_score_completions(None)
    with pytest.raises(RuntimeError, match="32-answer validity/truncation gate"):
        callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace())
    step = report["optimizer_step_gates"][0]
    assert step["valid"] == valid
    assert step["truncated"] == truncated
    assert step["passed"] is False
    cleanup()


def test_formal_step_stops_when_nonzero_advantages_do_not_change_parameters(tmp_path, monkeypatch):
    torch, trainer, _, cleanup, _ = _fake_formal_trainer(tmp_path, monkeypatch)
    callback = trainer.callbacks[0]
    for _ in range(4):
        occurrence_id = f"presentation:{trainer.group_index}"
        samples = [SimpleNamespace(extra={"id": "query:0",
                                         "occurrence_id": occurrence_id}) for _ in range(8)]
        trainer._score_completions(samples)
        trainer._generate_and_score_completions(None)
    callback.on_pre_optimizer_step(None, trainer.state, SimpleNamespace())
    with pytest.raises(RuntimeError, match="left all trainable parameters unchanged"):
        callback.on_step_end(None, trainer.state, SimpleNamespace())
    cleanup()


def test_train_formal_uses_first_236_and_disables_gpu_timer(tmp_path, monkeypatch):
    from tools import quant_grounding_config, train_quant_grounding
    import tools.run_quant_d_repair_gate as gate_tool

    for name in ("MAX_PIXELS", "MIN_PIXELS", "QUANT_REWARD_METRICS_JSONL"):
        monkeypatch.delenv(name, raising=False)
    schedule = tmp_path / "schedule.jsonl"
    rows = [{"id": f"query:{index % 227}", "occurrence_id": f"presentation:{index}"}
            for index in range(240)]
    schedule.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    base_config = {"arm": "D", "bounded_thinking": True,
                   "format_policy": "single_box", "adapters": "warmup50-checkpoint",
                   "max_gpu_hours": 1.0}
    monkeypatch.setattr(quant_grounding_config, "load_config",
                        lambda path, **overrides: {**base_config, **overrides})

    def prepare(source, destination, data_root, arm, *, bounded_thinking):
        assert arm == "D" and bounded_thinking
        prepared_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        assert len(prepared_rows) == 236
        return len(prepared_rows)

    monkeypatch.setattr(quant_grounding_config, "prepare_dataset", prepare)
    calls = {}

    def fake_hook(output_dir, report):
        report["first12_gate"] = {"passed": True}
        report["optimizer_steps_completed"] = 59
        return lambda trainer: lambda: None

    def run_swift(config, dataset, **kwargs):
        calls.update(config=config, dataset=dataset, kwargs=kwargs)

    monkeypatch.setattr(gate_tool, "formal_training_hook", fake_hook)
    monkeypatch.setattr(train_quant_grounding, "run_swift", run_swift)
    output = tmp_path / "formal-run"
    report = gate_tool.main(["train-formal", "--config", str(tmp_path / "config.json"),
                             "--schedule", str(schedule), "--data-root", str(tmp_path),
                             "--output-dir", str(output)])
    assert report["status"] == "passed"
    assert report["schedule_rows_available"] == 240
    assert report["schedule_rows_used"] == 236
    assert calls["config"]["max_steps"] == 59
    assert calls["config"]["max_gpu_hours"] is None
    assert calls["kwargs"]["max_steps"] == calls["kwargs"]["save_steps"] == 59
    assert "resume" not in calls["kwargs"]
