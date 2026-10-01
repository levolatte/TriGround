from __future__ import annotations

import json
import random
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from tools import compare_lora_resume
from tools.compare_lora_resume import compare


NAME = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
LANGUAGE_B = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
VISUAL = "base_model.model.model.visual.merger.fc1.lora_B.weight"
VISUAL_A = "base_model.model.model.visual.merger.fc1.lora_A.weight"


def _checkpoint(root, step, value, moment, *, dtype=torch.float32, samples=None, horizon=600):
    path = root / f"checkpoint-{step}"
    path.mkdir(parents=True, exist_ok=True)
    save_file({NAME: torch.tensor([[value]], dtype=dtype),
               LANGUAGE_B: torch.tensor([[value]], dtype=dtype)},
              str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": 1, "target_modules": ["q_proj"]}), encoding="utf-8")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": horizon}), encoding="utf-8")
    state = {parameter_id: {
        "step": torch.tensor(float(step)),
        "exp_avg": torch.tensor([[moment]], dtype=torch.float32),
        "exp_avg_sq": torch.tensor([[moment]], dtype=torch.float32),
    } for parameter_id in (0, 1)}
    group = {"params": [0, 1], "param_names": [
        NAME.replace(".lora_A.", ".lora_A.default."),
        LANGUAGE_B.replace(".lora_B.", ".lora_B.default."),
    ],
             "lr": 5e-6 * (1 - step / horizon), "initial_lr": 5e-6, "fused": True}
    torch.save({"state": state, "param_groups": [group]}, path / "optimizer.pt")
    torch.save({"last_epoch": step, "_last_lr": [group["lr"]]}, path / "scheduler.pt")
    torch.save({
        "python": random.Random(step).getstate(),
        "numpy": np.random.RandomState(step).get_state(),
        "cpu": torch.arange(8, dtype=torch.uint8) + step,
        "cuda": [torch.arange(8, dtype=torch.uint8) + step],
    }, path / "rng_state.pth")
    if samples is not None:
        (root / "consumed_samples.jsonl").write_text(
            "\n".join(json.dumps({"id": item}) for item in samples) + "\n", encoding="utf-8"
        )
    return path


def _four(tmp_path, *, resumed_value=0.0001, resumed_moment=0.001, horizon=600):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    ids = [f"sample-{i}" for i in range(32)]
    c16 = _checkpoint(continuous, 16, 0.0, 0.0, horizon=horizon)
    c32 = _checkpoint(continuous, 32, 0.0001, 0.001, samples=ids, horizon=horizon)
    r16 = _checkpoint(resumed, 16, 0.0, 0.0, horizon=horizon)
    r32 = _checkpoint(resumed, 32, resumed_value, resumed_moment, samples=ids, horizon=horizon)
    return c16, c32, r16, r32


def test_exact_resume_passes_with_state_and_sample_trace(tmp_path):
    result = compare(*_four(tmp_path), expected_groups=1, microbatches_per_step=1)
    assert result["status"] == "pass"
    assert result["adapter32"]["exact_tensors"] == 2
    assert result["optimizer32"]["summary"]["exact_tensors"] == 4
    assert result["sample_trace"]["entries"] == 32
    assert result["checkpoints"]["continuous32"]["optimizer"]["group_membership"] == "verified_by_param_names"
    assert result["rng_state16"]["equal"] is True
    assert result["rng_state32"]["files"]["rng_state.pth"]["cuda"] is True


def test_four_step_resume_with_ten_microbatches(tmp_path):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    ids = [f"sample-{i}" for i in range(40)]
    paths = (
        _checkpoint(continuous, 2, 0.0, 0.0),
        _checkpoint(continuous, 4, 0.0001, 0.001, samples=ids),
        _checkpoint(resumed, 2, 0.0, 0.0),
        _checkpoint(resumed, 4, 0.0001, 0.001, samples=ids),
    )
    result = compare(*paths, expected_groups=1, microbatches_per_step=10,
                     split_step=2, final_step=4)
    assert result["status"] == "pass"
    assert result["sample_trace"]["entries"] == 40
    assert (result["split_step"], result["final_step"]) == (2, 4)


def test_four_step_cli_aliases(tmp_path, monkeypatch):
    continuous = tmp_path / "continuous"
    resumed = tmp_path / "resumed"
    ids = [str(i) for i in range(40)]
    paths = (
        _checkpoint(continuous, 2, 0.0, 0.0),
        _checkpoint(continuous, 4, 0.0001, 0.001, samples=ids),
        _checkpoint(resumed, 2, 0.0, 0.0),
        _checkpoint(resumed, 4, 0.0001, 0.001, samples=ids),
    )
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_lora_resume", "--continuous-split", str(paths[0]),
        "--continuous-final", str(paths[1]), "--resumed-split", str(paths[2]),
        "--resumed-final", str(paths[3]), "--split-step", "2", "--final-step", "4",
        "--microbatches-per-step", "10", "--expected-groups", "1",
        "--expected-steps", "600", "--output", str(output),
    ])
    compare_lora_resume.main()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_small_fp32_difference_is_reported_separately(tmp_path):
    result = compare(*_four(tmp_path, resumed_value=0.0001001, resumed_moment=0.0010001),
                     expected_groups=1, microbatches_per_step=1)
    assert result["status"] == "pass"
    assert result["adapter32"]["exact_tensors"] == 0
    assert result["adapter32"]["all_within_tolerance"] is True


def test_peft_target_module_order_is_semantic_set(tmp_path):
    paths = _four(tmp_path)
    for index, path in enumerate(paths):
        modules = ["q_proj", "k_proj"] if index < 2 else ["k_proj", "q_proj"]
        (path / "adapter_config.json").write_text(
            json.dumps({"r": 1, "target_modules": modules}), encoding="utf-8")
    assert compare(*paths, expected_groups=1, microbatches_per_step=1)["status"] == "pass"
    (paths[-1] / "adapter_config.json").write_text(
        json.dumps({"r": 1, "target_modules": ["q_proj", "v_proj"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="PEFT adapter config differs"):
        compare(*paths, expected_groups=1, microbatches_per_step=1)


def test_large_resume_difference_fails(tmp_path):
    result = compare(*_four(tmp_path, resumed_value=0.00011), expected_groups=1, microbatches_per_step=1)
    assert result["status"] == "fail"
    assert NAME in result["adapter32"]["failures"]


def test_bf16_adapter_is_rejected(tmp_path):
    c16, c32, r16, r32 = _four(tmp_path)
    replacement = r16 / "adapter_model.new.safetensors"
    save_file({NAME: torch.tensor([[0.0]], dtype=torch.bfloat16),
               LANGUAGE_B: torch.tensor([[0.0]], dtype=torch.bfloat16)}, str(replacement))
    replacement.replace(r16 / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="non-FP32"):
        compare(c16, c32, r16, r32, expected_groups=1, microbatches_per_step=1)


def test_trace_mismatch_fails(tmp_path):
    paths = _four(tmp_path)
    root = paths[3].parent
    ids = [f"sample-{i}" for i in range(32)]
    ids[20] = "wrong"
    (root / "consumed_samples.jsonl").write_text(
        "\n".join(json.dumps({"id": item}) for item in ids) + "\n", encoding="utf-8"
    )
    result = compare(*paths, expected_groups=1, microbatches_per_step=1)
    assert result["status"] == "fail"
    assert result["sample_trace"]["first_mismatch_index"] == 20


def test_rng_mismatch_fails_even_when_weights_match(tmp_path):
    paths = _four(tmp_path)
    state = torch.load(paths[3] / "rng_state.pth", map_location="cpu", weights_only=False)
    state["numpy"] = np.random.RandomState(999).get_state()
    torch.save(state, paths[3] / "rng_state.pth")
    result = compare(*paths, expected_groups=1, microbatches_per_step=1)
    assert result["status"] == "fail"
    assert result["rng_state16"]["equal"] is True
    assert result["rng_state32"]["files"]["rng_state.pth"]["numpy"] is False


def test_missing_rng_file_is_rejected(tmp_path):
    paths = _four(tmp_path)
    (paths[2] / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match=r"missing rng_state\*\.pth"):
        compare(*paths, expected_groups=1, microbatches_per_step=1)


def test_strict_optimizer_group_audit_requires_param_names(tmp_path):
    paths = _four(tmp_path)
    checkpoint = paths[3]
    optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    del optimizer["param_groups"][0]["param_names"]
    torch.save(optimizer, checkpoint / "optimizer.pt")
    with pytest.raises(ValueError, match="requires optimizer param_names"):
        compare(*paths, expected_groups=1, microbatches_per_step=1)


def test_explicit_400_step_scheduler_horizon(tmp_path):
    paths = _four(tmp_path, horizon=400)
    assert compare(*paths, expected_groups=1, expected_steps=400,
                   microbatches_per_step=1)["status"] == "pass"
    with pytest.raises(ValueError, match="600-step"):
        compare(*paths, expected_groups=1, microbatches_per_step=1)


def test_two_named_groups_map_to_language_and_visual(tmp_path):
    paths = _four(tmp_path)
    for path in paths:
        step = int(path.name.split("-")[-1])
        weights = {name: tensor.clone() for name, tensor in
                   load_file(str(path / "adapter_model.safetensors")).items()}
        weights[VISUAL_A] = torch.tensor([[0.01 if step == 16 else 0.0101]])
        weights[VISUAL] = torch.tensor([[0.0 if step == 16 else 0.0002]])
        new_file = path / "adapter_model.new.safetensors"
        save_file(weights, str(new_file))
        new_file.replace(path / "adapter_model.safetensors")
        optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        optimizer["state"][2] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.tensor([[0.0 if step == 16 else 0.002]]),
            "exp_avg_sq": torch.tensor([[0.0 if step == 16 else 0.002]]),
        }
        optimizer["state"][3] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.tensor([[0.0 if step == 16 else 0.002]]),
            "exp_avg_sq": torch.tensor([[0.0 if step == 16 else 0.002]]),
        }
        optimizer["param_groups"].append({
            "params": [2, 3], "param_names": [
                VISUAL_A.replace(".lora_A.", ".lora_A.default."),
                VISUAL.replace(".lora_B.", ".lora_B.default."),
            ],
            "lr": 2e-5 * (1 - step / 600), "initial_lr": 2e-5, "fused": True,
        })
        torch.save(optimizer, path / "optimizer.pt")
        scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
        scheduler["_last_lr"].append(optimizer["param_groups"][1]["lr"])
        torch.save(scheduler, path / "scheduler.pt")
    result = compare(*paths, expected_groups=2, microbatches_per_step=1)
    assert result["status"] == "pass"
    assert result["checkpoints"]["continuous32"]["module_counts"]["visual_modules"] == 1
    assert result["checkpoints"]["continuous32"]["optimizer"]["group_modules"] == [["language"], ["visual"]]


def test_cli_writes_pass_json(tmp_path, monkeypatch):
    c16, c32, r16, r32 = _four(tmp_path)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_lora_resume", "--continuous16", str(c16), "--continuous32", str(c32),
        "--resumed16", str(r16), "--resumed32", str(r32),
        "--expected-groups", "1", "--expected-steps", "600",
        "--microbatches-per-step", "1", "--output", str(output),
    ])
    compare_lora_resume.main()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
