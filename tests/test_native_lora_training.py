"""CPU checks for the native FP32 LoRA handoff and optimizer contract."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import native_lora_training as native


def _parameter_model(scope):
    named = []
    for module in sorted(native.expected_modules(scope)):
        rank = 8 if module.startswith("model.visual.") else 32
        for side in ("A", "B"):
            parameter = torch.nn.Parameter(torch.ones((rank, 4) if side == "A" else (4, rank)))
            named.append((f"base_model.model.{module}.lora_{side}.default.weight", parameter))
    return SimpleNamespace(
        named_parameters=lambda: iter(named),
        parameters=lambda: (parameter for _, parameter in named),
    )


@pytest.mark.parametrize("scope,modules,tensors,groups", [
    ("language", 144, 288, 1),
    ("language_merger", 152, 304, 2),
])
def test_fp32_lora_modules_and_optimizer_groups(scope, modules, tensors, groups):
    model = _parameter_model(scope)
    assert native.audit_lora_model(model, scope) == {
        "modules": modules, "tensors": tensors, "dtype": "float32"
    }
    grouped = native.optimizer_parameter_groups(
        model, scope=scope, language_lr=5e-6, visual_lr=2e-5
    )
    assert len(grouped) == groups
    assert [group["group_name"] for group in grouped] == (
        ["language", "visual"] if groups == 2 else ["language"]
    )
    assert [len(group["params"]) for group in grouped] == (
        [288, 16] if groups == 2 else [288]
    )
    assert [group["lr"] for group in grouped] == (
        [5e-6, 2e-5] if groups == 2 else [5e-6]
    )
    for group in grouped:
        assert len(group["param_names"]) == len(group["params"])
        assert group["weight_decay"] == 0
    optimizer = torch.optim.AdamW(grouped)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert native.assert_fp32_optimizer(optimizer, model, require_state=True)["moment_states"] == tensors


def test_fp32_optimizer_resume_is_exact_and_rejects_changed_moments(tmp_path):
    model = _parameter_model("language")
    optimizer = torch.optim.AdamW(native.optimizer_parameter_groups(
        model, scope="language", language_lr=5e-6, visual_lr=2e-5
    ))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    restored_model = _parameter_model("language")
    restored = torch.optim.AdamW(native.optimizer_parameter_groups(
        restored_model, scope="language", language_lr=5e-6, visual_lr=2e-5
    ))
    restored.load_state_dict(torch.load(tmp_path / "optimizer.pt", weights_only=True))
    native.compare_optimizer_checkpoint(restored, tmp_path)
    native.assert_fp32_optimizer(restored, restored_model, require_state=True)
    first = next(iter(restored_model.parameters()))
    restored.state[first]["exp_avg"][0, 0] += 1
    with pytest.raises(RuntimeError, match="exp_avg changed"):
        native.compare_optimizer_checkpoint(restored, tmp_path)


def test_low_rank_change_matches_dense_product():
    generator = torch.Generator().manual_seed(17)
    old_a = torch.randn(3, 5, generator=generator)
    old_b = torch.randn(4, 3, generator=generator)
    new_a = old_a + torch.randn(3, 5, generator=generator) * 0.01
    new_b = old_b + torch.randn(4, 3, generator=generator) * 0.01
    expected = torch.linalg.vector_norm(new_b @ new_a - old_b @ old_a).item()
    assert native.low_rank_delta_norm(old_a, old_b, new_a, new_b) == pytest.approx(expected, rel=1e-4)


def test_composite_target_regex_and_visual_rank_pattern():
    config = SimpleNamespace(target_modules=set(native.LANGUAGE_TARGETS),
                             rank_pattern={}, alpha_pattern={})
    configured = native.configure_lora(config, "language_merger")
    import re
    targets = native.expected_modules("language_merger")
    assert all(re.fullmatch(configured.target_modules, target) for target in targets)
    assert len(targets) == 152
    assert not re.fullmatch(configured.target_modules, "model.visual.blocks.0.attn.q_proj")
    assert configured.rank_pattern == {native.VISUAL_RANK_PATTERN: 8}
    assert configured.alpha_pattern == {native.VISUAL_RANK_PATTERN: 16}
    assert config.target_modules == set(native.LANGUAGE_TARGETS)


def test_real_peft_transplant_on_tiny_qwen_shaped_model(tmp_path):
    peft = pytest.importorskip("peft")

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in native.LANGUAGE_TARGETS:
                setattr(self, name, torch.nn.Linear(4, 4, bias=False))

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Merger(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_fc1 = torch.nn.Linear(4, 4, bias=False)
            self.linear_fc2 = torch.nn.Linear(4, 4, bias=False)

    class Visual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.merger = Merger()
            self.deepstack_merger_list = torch.nn.ModuleList([Merger() for _ in range(3)])

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([Block() for _ in range(36)])
            self.visual = Visual()

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    source_config = peft.LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=list(native.LANGUAGE_TARGETS), bias="none"
    )
    source = native.initialize_lora(TinyQwen(), source_config, scope="language",
                                    init_adapter="", peft=peft,
                                    get_peft_model=peft.get_peft_model)
    source.save_pretrained(str(tmp_path))
    composite = native.initialize_lora(TinyQwen(), source_config,
                                       scope="language_merger", init_adapter=str(tmp_path),
                                       peft=peft, get_peft_model=peft.get_peft_model)
    assert native.audit_lora_model(composite, "language_merger")["tensors"] == 304
    weights = peft.get_peft_model_state_dict(composite)
    assert len(weights) == 304
    assert sum(".visual." in name and ".lora_B." in name
               and torch.count_nonzero(value).item() == 0
               for name, value in weights.items()) == 8
    assert set(source_config.target_modules) == set(native.LANGUAGE_TARGETS)
