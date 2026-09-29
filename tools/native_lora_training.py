"""Small, explicit PEFT and optimizer helpers for the native Qwen3-VL launcher."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re

import torch


LANGUAGE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
VISUAL_TARGETS = tuple(
    f"model.visual.{merger}.{linear}"
    for merger in ("merger", *(f"deepstack_merger_list.{index}" for index in range(3)))
    for linear in ("linear_fc1", "linear_fc2")
)
COMPOSITE_TARGET_REGEX = (
    r"^model\.(?:language_model\.layers\.[0-9]+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)|visual\."
    r"(?:merger|deepstack_merger_list\.[0-2])\.linear_fc[12])$"
)
VISUAL_RANK_PATTERN = r"visual\.(?:merger|deepstack_merger_list\.[0-2])\.linear_fc[12]"
_LORA_PARAMETER = re.compile(r"^(.*)\.lora_([AB])\.default\.weight$")


def expected_modules(scope: str, language_layers: int = 36) -> set[str]:
    if scope not in {"language", "language_merger"}:
        raise ValueError(f"unknown LORA_SCOPE: {scope}")
    modules = {
        f"model.language_model.layers.{layer}.self_attn.{target}"
        for layer in range(language_layers)
        for target in LANGUAGE_TARGETS
    }
    if scope == "language_merger":
        modules.update(VISUAL_TARGETS)
    return modules


def configure_lora(expected, scope: str):
    """Preserve upstream language settings; add only the eight visual exits."""
    expected_modules(scope)
    config = deepcopy(expected)
    if scope == "language_merger":
        config.target_modules = COMPOSITE_TARGET_REGEX
        config.rank_pattern = {VISUAL_RANK_PATTERN: 8}
        config.alpha_pattern = {VISUAL_RANK_PATTERN: 16}
    return config


def _source_config_mismatch(saved, expected) -> dict:
    fields = ("r", "lora_alpha", "lora_dropout", "bias", "task_type")
    mismatch = {
        field: (getattr(saved, field), getattr(expected, field))
        for field in fields
        if getattr(saved, field) != getattr(expected, field)
    }
    if set(saved.target_modules) != set(LANGUAGE_TARGETS):
        mismatch["source_target_modules"] = saved.target_modules
    if set(expected.target_modules) != set(LANGUAGE_TARGETS):
        mismatch["expected_target_modules"] = expected.target_modules
    if getattr(saved, "rank_pattern", {}) or getattr(saved, "alpha_pattern", {}):
        mismatch["source_rank_alpha_patterns"] = (saved.rank_pattern, saved.alpha_pattern)
    return mismatch


def _language_key(key: str) -> bool:
    return ".language_model.layers." in key and ".self_attn." in key and ".lora_" in key


def _visual_key(key: str) -> bool:
    return ".visual." in key and ".lora_" in key


def initialize_lora(base_model, upstream_config, *, scope: str, init_adapter: str,
                    peft, get_peft_model):
    """Create one adapter, strictly transplanting the old language tensors."""
    config = configure_lora(upstream_config, scope)
    if not init_adapter:
        model = get_peft_model(base_model, config, autocast_adapter_dtype=True)
        audit_lora_model(model, scope)
        return model

    adapter_path = Path(init_adapter).expanduser()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"INIT_ADAPTER is not a directory: {adapter_path}")
    saved_config = peft.PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    mismatch = _source_config_mismatch(saved_config, upstream_config)
    if mismatch:
        raise RuntimeError(f"INIT_ADAPTER language LoRA config differs: {mismatch}")
    saved = peft.load_peft_weights(str(adapter_path), device="cpu")
    if len(saved) != 288 or not all(_language_key(key) for key in saved):
        raise RuntimeError(f"INIT_ADAPTER must contain exactly 288 language LoRA tensors; got {len(saved)}")

    visual_initial = {}
    if scope == "language":
        model = peft.PeftModel.from_pretrained(
            base_model, str(adapter_path), is_trainable=True,
            autocast_adapter_dtype=True, local_files_only=True,
        )
    else:
        model = get_peft_model(base_model, config, autocast_adapter_dtype=True)
        initial = peft.get_peft_model_state_dict(model)
        if len(initial) != 304 or set(saved) != {key for key in initial if _language_key(key)}:
            raise RuntimeError("composite adapter does not contain exactly the C language keys and 16 visual keys")
        if not all(bool(torch.count_nonzero(value).item() == 0)
                   for key, value in initial.items() if _visual_key(key) and ".lora_B." in key):
            raise RuntimeError("new visual LoRA B must be zero at initialization")
        visual_initial = {key: value.detach().cpu().clone()
                          for key, value in initial.items() if _visual_key(key)}
        peft.set_peft_model_state_dict(model, saved)

    loaded = peft.get_peft_model_state_dict(model)
    if set(saved) != {key for key in loaded if _language_key(key)}:
        raise RuntimeError("C language LoRA keys differ after loading")
    altered = [key for key, value in saved.items()
               if loaded[key].dtype != torch.float32
               or not torch.equal(value.float(), loaded[key].detach().cpu())]
    if altered:
        raise RuntimeError(f"C language LoRA tensor values/dtype changed: {altered[:5]}")
    changed_visual = [key for key, value in visual_initial.items()
                      if not torch.equal(value, loaded[key].detach().cpu())]
    if changed_visual:
        raise RuntimeError(f"new visual LoRA changed while loading C: {changed_visual[:5]}")
    audit_lora_model(model, scope)
    return model


def audit_lora_model(model, scope: str, language_layers: int = 36) -> dict:
    expected = expected_modules(scope, language_layers)
    found: dict[str, set[str]] = {}
    unexpected = []
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    for name, parameter in trainable:
        match = _LORA_PARAMETER.match(name)
        module = match.group(1) if match else ""
        matched = next((target for target in expected if module.endswith(target)), None)
        rank = 8 if matched and matched.startswith("model.visual.") else 32
        rank_axis = 0 if match and match.group(2) == "A" else 1
        if (matched is None or parameter.dtype != torch.float32
                or parameter.ndim != 2 or parameter.shape[rank_axis] != rank):
            unexpected.append((name, str(parameter.dtype)))
        else:
            found.setdefault(matched, set()).add(match.group(2))
    missing = sorted(target for target in expected if found.get(target) != {"A", "B"})
    if missing or unexpected or len(trainable) != 2 * len(expected):
        raise RuntimeError(
            f"LoRA trainable audit failed: count={len(trainable)}, expected={2 * len(expected)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return {"modules": len(expected), "tensors": len(trainable), "dtype": "float32"}


def optimizer_parameter_groups(model, *, scope: str, language_lr: float,
                               visual_lr: float) -> list[dict]:
    """Weight decay is zero in this launcher, so two groups suffice."""
    expected_modules(scope)
    language, visual = [], []
    language_names, visual_names = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".language_model." in name:
            language.append(parameter)
            language_names.append(name)
        elif ".visual." in name:
            visual.append(parameter)
            visual_names.append(name)
        else:
            raise RuntimeError(f"trainable parameter outside LoRA groups: {name}")
    expected_visual = 16 if scope == "language_merger" else 0
    if len(language) != 288 or len(visual) != expected_visual:
        raise RuntimeError(f"optimizer groups expected 288/{expected_visual} tensors, got {len(language)}/{len(visual)}")
    groups = [{"params": language, "param_names": language_names,
               "group_name": "language", "lr": language_lr, "weight_decay": 0.0}]
    if expected_visual:
        groups.append({"params": visual, "param_names": visual_names,
                       "group_name": "visual", "lr": visual_lr, "weight_decay": 0.0})
    return groups


def assert_fp32_optimizer(optimizer, model, *, require_state: bool) -> dict:
    named_trainable = {id(p): (name, p) for name, p in model.named_parameters() if p.requires_grad}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    if len(grouped) != len(named_trainable) or {id(p) for p in grouped} != set(named_trainable):
        raise RuntimeError("optimizer parameter groups do not cover every trainable LoRA exactly once")
    for group in optimizer.param_groups:
        if len(group["params"]) != len(group.get("param_names", [])):
            raise RuntimeError("optimizer param_names do not match params")
        group_name = group.get("group_name")
        if group_name not in {"language", "visual"}:
            raise RuntimeError(f"unexpected optimizer group_name: {group_name}")
        for name, parameter in zip(group["param_names"], group["params"]):
            expected_name = named_trainable[id(parameter)][0]
            if name != expected_name or f".{('language_model' if group_name == 'language' else 'visual')}." not in name:
                raise RuntimeError(f"optimizer group/name mismatch: {group_name}: {name} != {expected_name}")
    state_count = 0
    for parameter in grouped:
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"trainable LoRA parameter is {parameter.dtype}, expected float32")
        state = optimizer.state.get(parameter, {})
        if not state:
            if require_state:
                raise RuntimeError("restored optimizer lacks LoRA moments")
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is None or value.dtype != torch.float32 or value.shape != parameter.shape:
                raise RuntimeError(f"optimizer {key} missing or not float32/parameter-shaped")
        state_count += 1
    return {"parameters": len(grouped), "moment_states": state_count,
            "learning_rates": [group["lr"] for group in optimizer.param_groups]}


def compare_optimizer_checkpoint(optimizer, checkpoint: Path) -> None:
    saved = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    current = optimizer.state_dict()
    if len(saved["param_groups"]) != len(current["param_groups"]):
        raise RuntimeError("resumed optimizer parameter-group count differs from checkpoint")
    for old, now in zip(saved["param_groups"], current["param_groups"]):
        if old != now:
            raise RuntimeError("resumed optimizer parameter groups differ from checkpoint")
    if set(saved["state"]) != set(current["state"]):
        raise RuntimeError("resumed optimizer state keys differ from checkpoint")
    for index, old_state in saved["state"].items():
        new_state = current["state"][index]
        if set(old_state) != set(new_state):
            raise RuntimeError(f"resumed optimizer fields differ for parameter {index}")
        for key, old_value in old_state.items():
            new_value = new_state[key]
            if isinstance(old_value, torch.Tensor):
                if not torch.equal(old_value, new_value.detach().cpu()):
                    raise RuntimeError(f"resumed optimizer {key} changed for parameter {index}")
            elif old_value != new_value:
                raise RuntimeError(f"resumed optimizer {key} changed for parameter {index}")


def low_rank_delta_norm(old_a: torch.Tensor, old_b: torch.Tensor,
                        new_a: torch.Tensor, new_b: torch.Tensor) -> float:
    """Compute ||B1 A1 - B0 A0||_F using only rank-sized Gram matrices."""
    old_a, old_b = old_a.double(), old_b.double()
    new_a, new_b = new_a.double(), new_b.double()
    aa0 = old_a @ old_a.T
    aa1 = new_a @ new_a.T
    bb0 = old_b.T @ old_b
    bb1 = new_b.T @ new_b
    cross = (new_b.T @ old_b * (old_a @ new_a.T).T).sum()
    squared = (bb0 * aa0).sum() + (bb1 * aa1).sum() - 2 * cross
    return math.sqrt(max(float(squared), 0.0))
