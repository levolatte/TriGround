"""CPU-only checkpoint audit: continuous 32 steps versus 16-to-32 resume.

Reads PEFT adapter, Trainer optimizer/scheduler/state, and consumed sample IDs.
It never loads a base model. Exact equality is preferred; a bounded FP32
comparison is reported separately and never masks dtype or state failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


MOMENT_KEYS = ("exp_avg", "exp_avg_sq")
RNG_KEYS = ("python", "numpy", "cpu", "cuda")


def _read_rng_states(path: Path) -> dict[str, dict[str, Any]]:
    files = sorted(path.glob("rng_state*.pth"))
    if not files:
        raise ValueError(f"{path}: missing rng_state*.pth")
    states = {}
    for file in files:
        # Trainer serializes Python and NumPy RNG tuples, which need pickle loading.
        state = torch.load(file, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or any(key not in state for key in RNG_KEYS):
            raise ValueError(f"{file}: RNG state needs python/numpy/cpu/cuda entries")
        states[file.name] = state
    return states


def _rng_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _rng_value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(
            _rng_value_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _compare_rng_states(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = first["rng_states"]
    right = second["rng_states"]
    files_match = left.keys() == right.keys()
    entries = {
        filename: {
            key: _rng_value_equal(left[filename][key], right[filename][key])
            for key in RNG_KEYS
        }
        for filename in left.keys() & right.keys()
    }
    return {
        "equal": files_match and all(all(values.values()) for values in entries.values()),
        "files_match": files_match,
        "files": entries,
    }


def _canonical_name(name: str) -> str:
    name = name.replace(".default.", ".")
    for anchor in ("language_model.", "visual."):
        if anchor in name:
            return anchor + name.split(anchor, 1)[1]
    return name


def _tensor_stats(left: torch.Tensor, right: torch.Tensor,
                  before: torch.Tensor | None = None) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"shape_dtype_match": False, "exact": False, "within_tolerance": False}
    diff = left.to(torch.float64) - right.to(torch.float64)
    diff_norm = float(torch.linalg.vector_norm(diff))
    max_abs = float(diff.abs().max()) if diff.numel() else 0.0
    exact = bool(torch.equal(left, right))
    update_norm = None
    if before is not None:
        if before.shape != left.shape:
            raise ValueError("reference checkpoint tensor shape differs")
        update_norm = float(torch.linalg.vector_norm(left.to(torch.float64) - before.to(torch.float64)))
    relative = diff_norm / update_norm if update_norm else (0.0 if exact else None)
    return {
        "shape_dtype_match": True,
        "dtype": str(left.dtype), "shape": list(left.shape),
        "exact": exact, "max_abs": max_abs, "difference_l2": diff_norm,
        "reference_update_l2": update_norm, "difference_over_update": relative,
        "within_tolerance": exact or (relative is not None and max_abs <= 1e-6 and relative <= 0.01),
    }


def _aggregate(stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "tensors": len(stats),
        "exact_tensors": sum(value["exact"] for value in stats.values()),
        "max_abs": max((value.get("max_abs", 0.0) for value in stats.values()), default=0.0),
        "all_within_tolerance": all(value["within_tolerance"] for value in stats.values()),
        "failures": [key for key, value in stats.items() if not value["within_tolerance"]][:20],
    }


def _module_counts(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    modules = {name.split(".lora_")[0] for name in weights if ".lora_" in name}
    a = {name.split(".lora_A.")[0] for name in weights if ".lora_A." in name}
    b = {name.split(".lora_B.")[0] for name in weights if ".lora_B." in name}
    if not modules or a != b or len(weights) != len(a) + len(b):
        raise ValueError("adapter must contain exactly one LoRA A/B tensor pair per module")
    visual = {name for name in modules if ".visual." in name}
    language = {name for name in modules if ".language_model." in name}
    return {
        "adapter_tensors": len(weights), "modules": len(modules),
        "language_modules": len(language), "visual_modules": len(visual),
        "other_modules": len(modules - visual - language),
        "lora_A_tensors": len(a),
        "lora_B_tensors": len(b),
    }


def _read_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    weights = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if state["global_step"] != expected_step:
        raise ValueError(f"{path}: global_step {state['global_step']} != {expected_step}")
    optimizer = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    rng_states = _read_rng_states(path)
    adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    # PEFT serializes target_modules from a set; order can differ by process.
    # A string is a regex and must retain its exact text.
    if isinstance(adapter_config.get("target_modules"), list):
        adapter_config["target_modules"] = sorted(adapter_config["target_modules"])
    config_path = path.parent / "native_train_config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
    return {
        "path": str(path.resolve()), "weights": weights, "trainer_state": state,
        "optimizer": optimizer, "scheduler": scheduler, "rng_states": rng_states,
        "adapter_config": adapter_config, "run_config": run_config,
        "module_counts": _module_counts(weights),
    }


def _optimizer_audit(checkpoint: dict[str, Any], expected_groups: int | None) -> dict[str, Any]:
    optimizer = checkpoint["optimizer"]
    groups = optimizer["param_groups"]
    ids = [parameter_id for group in groups for parameter_id in group["params"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{checkpoint['path']}: optimizer parameter ID occurs more than once")
    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError(f"{checkpoint['path']}: expected {expected_groups} optimizer groups, got {len(groups)}")
    if len(ids) != len(checkpoint["weights"]):
        raise ValueError(f"{checkpoint['path']}: optimizer parameter count differs from adapter tensors")
    if set(optimizer["state"]) != set(ids):
        raise ValueError(f"{checkpoint['path']}: optimizer state does not cover every adapter parameter")
    if any(tensor.dtype != torch.float32 for tensor in checkpoint["weights"].values()):
        raise ValueError(f"{checkpoint['path']}: adapter contains non-FP32 tensors")
    for parameter_id, state in optimizer["state"].items():
        for key in MOMENT_KEYS:
            if key not in state or state[key].dtype != torch.float32:
                raise ValueError(f"{checkpoint['path']}: optimizer {parameter_id}.{key} is not FP32")
            if not bool(torch.isfinite(state[key]).all()):
                raise ValueError(f"{checkpoint['path']}: optimizer {parameter_id}.{key} is non-finite")
        if int(state["step"].item()) != checkpoint["trainer_state"]["global_step"]:
            raise ValueError(f"{checkpoint['path']}: optimizer {parameter_id} step differs from Trainer step")
        if state["step"].dtype != torch.float32:
            raise ValueError(f"{checkpoint['path']}: optimizer {parameter_id}.step is not FP32")
    listed_names = [name for group in groups for name in group.get("param_names", [])]
    has_names = all("param_names" in group for group in groups)
    if expected_groups is not None and not has_names:
        raise ValueError(f"{checkpoint['path']}: strict group audit requires optimizer param_names")
    if has_names:
        normalized = [_canonical_name(name) for name in listed_names]
        saved = {_canonical_name(name): tensor for name, tensor in checkpoint["weights"].items()}
        if len(normalized) != len(ids) or len(saved) != len(checkpoint["weights"]) or set(normalized) != set(saved):
            raise ValueError(f"{checkpoint['path']}: optimizer param_names do not match adapter keys")
        for parameter_id, name in zip(ids, normalized):
            for moment_key in MOMENT_KEYS:
                if optimizer["state"][parameter_id][moment_key].shape != saved[name].shape:
                    raise ValueError(f"{checkpoint['path']}: optimizer {parameter_id}.{moment_key} shape differs from {name}")
    group_modules = None
    if has_names:
        group_modules = []
        for group in groups:
            if group.get("fused") is not True:
                raise ValueError(f"{checkpoint['path']}: expected fused AdamW optimizer group")
            kinds = {"visual" if ".visual." in name else "language"
                     if ".language_model." in name else "other" for name in group["param_names"]}
            group_modules.append(sorted(kinds))
        if expected_groups == 2 and sorted(group_modules) != [["language"], ["visual"]]:
            raise ValueError(f"{checkpoint['path']}: expected distinct language and visual optimizer groups")
    return {
        "group_count": len(groups), "parameter_count": len(ids),
        "group_sizes": [len(group["params"]) for group in groups],
        "group_learning_rates": [group["lr"] for group in groups],
        "group_initial_learning_rates": [group.get("initial_lr") for group in groups],
        "group_names": [[name for name in group.get("param_names", [])] for group in groups]
        if has_names else None,
        "group_modules": group_modules,
        "group_membership": "verified_by_param_names" if has_names else "unverified_group_membership",
        "state_dtype": "torch.float32", "all_states_at_trainer_step": True,
    }


def _trace(path: Path, step: int, microbatches_per_step: int) -> list[str]:
    trace_path = path.parent / "consumed_samples.jsonl"
    lines = [line for line in trace_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    expected = step * microbatches_per_step
    if len(lines) < expected:
        raise ValueError(f"{trace_path}: {len(lines)} sample entries, expected at least {expected}")
    return [str(json.loads(line)["id"]) for line in lines[:expected]]


def _compare_optimizer(
    before: dict[str, Any], continuous: dict[str, Any], resumed: dict[str, Any]
) -> dict[str, Any]:
    left = continuous["optimizer"]
    right = resumed["optimizer"]
    baseline = before["optimizer"]
    if left["param_groups"] != right["param_groups"]:
        # Direct dict comparison is safe here: groups only hold scalars and parameter IDs.
        raise ValueError("continuous and resumed optimizer param_groups differ")
    if set(left["state"]) != set(right["state"]) or set(left["state"]) != set(baseline["state"]):
        raise ValueError("optimizer parameter state IDs differ")
    stats: dict[str, dict[str, Any]] = {}
    for parameter_id in left["state"]:
        for key in MOMENT_KEYS:
            label = f"{parameter_id}.{key}"
            stats[label] = _tensor_stats(
                left["state"][parameter_id][key],
                right["state"][parameter_id][key],
                baseline["state"][parameter_id][key],
            )
        for key in set(left["state"][parameter_id]) - set(MOMENT_KEYS):
            lvalue = left["state"][parameter_id][key]
            rvalue = right["state"][parameter_id][key]
            equal = torch.equal(lvalue, rvalue) if isinstance(lvalue, torch.Tensor) else lvalue == rvalue
            if not equal:
                raise ValueError(f"optimizer scalar/state {parameter_id}.{key} differs")
    return {"summary": _aggregate(stats), "tensors": stats}


def _compare_optimizer16(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = first["optimizer"]
    right = second["optimizer"]
    if left["param_groups"] != right["param_groups"] or set(left["state"]) != set(right["state"]):
        raise ValueError("16-step optimizer groups or state IDs differ")
    stats = {}
    for parameter_id in left["state"]:
        if set(left["state"][parameter_id]) != set(right["state"][parameter_id]):
            raise ValueError(f"16-step optimizer state fields differ for {parameter_id}")
        for key, tensor in left["state"][parameter_id].items():
            if isinstance(tensor, torch.Tensor):
                stats[f"{parameter_id}.{key}"] = _tensor_stats(tensor, right["state"][parameter_id][key])
            elif tensor != right["state"][parameter_id][key]:
                raise ValueError(f"16-step optimizer scalar {parameter_id}.{key} differs")
    return {"summary": _aggregate(stats), "tensors": stats}


def compare(
    continuous16: Path, continuous32: Path, resumed16: Path, resumed32: Path,
    *, expected_groups: int | None = None, expected_steps: int = 600,
    microbatches_per_step: int = 8,
) -> dict[str, Any]:
    if microbatches_per_step <= 0:
        raise ValueError("microbatches_per_step must be positive")
    if expected_steps not in (400, 600):
        raise ValueError("expected_steps must be 400 or 600")
    checkpoints = {
        "continuous16": _read_checkpoint(continuous16, 16),
        "continuous32": _read_checkpoint(continuous32, 32),
        "resumed16": _read_checkpoint(resumed16, 16),
        "resumed32": _read_checkpoint(resumed32, 32),
    }
    reference_config = checkpoints["continuous16"]["adapter_config"]
    reference_count = checkpoints["continuous16"]["module_counts"]
    for label, checkpoint in checkpoints.items():
        if checkpoint["adapter_config"] != reference_config:
            raise ValueError(f"{label}: PEFT adapter config differs")
        if checkpoint["module_counts"] != reference_count:
            raise ValueError(f"{label}: LoRA module count differs")
        if checkpoint["trainer_state"].get("max_steps") != expected_steps:
            raise ValueError(f"{label}: expected the common {expected_steps}-step scheduler horizon")
    audits = {label: _optimizer_audit(checkpoint, expected_groups)
              for label, checkpoint in checkpoints.items()}
    first = checkpoints["continuous16"]["weights"]
    second = checkpoints["continuous32"]["weights"]
    split_first = checkpoints["resumed16"]["weights"]
    split_second = checkpoints["resumed32"]["weights"]
    if not (set(first) == set(second) == set(split_first) == set(split_second)):
        raise ValueError("adapter tensor keys differ across checkpoints")
    init_stats = {name: _tensor_stats(first[name], split_first[name]) for name in first}
    weight_stats = {name: _tensor_stats(second[name], split_second[name], first[name])
                    for name in first}
    first_summary = _aggregate(init_stats)
    weights_summary = _aggregate(weight_stats)
    optimizer16 = _compare_optimizer16(checkpoints["continuous16"], checkpoints["resumed16"])
    optimizer_comparison = _compare_optimizer(
        checkpoints["continuous16"], checkpoints["continuous32"], checkpoints["resumed32"]
    )
    rng16 = _compare_rng_states(checkpoints["continuous16"], checkpoints["resumed16"])
    rng32 = _compare_rng_states(checkpoints["continuous32"], checkpoints["resumed32"])
    trace_continuous = _trace(continuous32, 32, microbatches_per_step)
    trace_resumed = _trace(resumed32, 32, microbatches_per_step)
    traces_equal = trace_continuous == trace_resumed
    scheduler_equal = checkpoints["continuous32"]["scheduler"] == checkpoints["resumed32"]["scheduler"]
    scheduler16_equal = checkpoints["continuous16"]["scheduler"] == checkpoints["resumed16"]["scheduler"]
    trainer_fields = ("global_step", "max_steps", "epoch", "num_train_epochs", "train_batch_size")
    trainer32 = {
        field: {
            "continuous": checkpoints["continuous32"]["trainer_state"].get(field),
            "resumed": checkpoints["resumed32"]["trainer_state"].get(field),
        }
        for field in trainer_fields
    }
    trainer32_equal = all(value["continuous"] == value["resumed"] for value in trainer32.values())
    # LambdaLR state is a plain dictionary for this launcher.
    passed = (first_summary["exact_tensors"] == first_summary["tensors"]
              and optimizer16["summary"]["exact_tensors"] == optimizer16["summary"]["tensors"]
              and weights_summary["all_within_tolerance"]
              and optimizer_comparison["summary"]["all_within_tolerance"]
              and traces_equal and scheduler16_equal and scheduler_equal and trainer32_equal
              and rng16["equal"] and rng32["equal"])
    return {
        "status": "pass" if passed else "fail",
        "criteria": "16 adapter/optimizer/RNG exact; 32 adapter/optimizer exact preferred, otherwise max_abs<=1e-6 AND L2 difference<=1% of 16-to-32 update; 32 RNG exact",
        "checkpoints": {label: {"path": value["path"], "step": value["trainer_state"]["global_step"],
                                "module_counts": value["module_counts"], "optimizer": audits[label]}
                        for label, value in checkpoints.items()},
        "adapter16": first_summary,
        "optimizer16": optimizer16,
        "adapter32": weights_summary,
        "adapter32_tensors": weight_stats,
        "optimizer32": optimizer_comparison,
        "sample_trace": {"equal": traces_equal, "entries": len(trace_continuous),
                         "first_mismatch_index": next((i for i, (a, b) in enumerate(zip(trace_continuous, trace_resumed)) if a != b), None)},
        "scheduler32_equal": scheduler_equal,
        "scheduler16_equal": scheduler16_equal,
        "rng_state16": rng16,
        "rng_state32": rng32,
        "trainer_state32": {"equal": trainer32_equal, "fields": trainer32},
        "limitations": [
            "A passing tolerance is numerical concordance, not bitwise reproducibility.",
            "Without optimizer param_names, group membership cannot be mapped offline to language/visual modules.",
            "This tool reads no base-model weights and does not establish downstream accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous16", type=Path, required=True)
    parser.add_argument("--continuous32", type=Path, required=True)
    parser.add_argument("--resumed16", type=Path, required=True)
    parser.add_argument("--resumed32", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int)
    parser.add_argument("--expected-steps", type=int, choices=(400, 600), default=600)
    parser.add_argument("--microbatches-per-step", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.continuous16, args.continuous32, args.resumed16, args.resumed32,
        expected_groups=args.expected_groups, expected_steps=args.expected_steps,
        microbatches_per_step=args.microbatches_per_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
