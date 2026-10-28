"""Four-query D repair gate and one real GRPO optimizer update.

Generation is performed by eval_quant_grounding.py. This tool checks its scored
predictions, or trains once on four fixed hard queries with fresh on-policy
samples. The training hook records its first on-policy rollouts and gates the
actual optimizer update.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HARD4_IDS = {
    "city_shuming_78_00000046_010",
    "city_002041_001",
    "rgbdt500_323_00000118::q01",
    "roborefit_train_000919",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def _signal(sample: dict) -> dict:
    return sample.get("score", sample)


def gate_summary(groups: list[dict], *, expected_queries: int = 4,
                 min_valid_rate: float = 0.0, max_truncated_rate: float = 1.0,
                 min_geometry_groups: int = 2) -> dict:
    """Require geometry-dependent reward variation within complete G8 groups."""
    if expected_queries < 1 or not 0 <= min_valid_rate <= 1 or not 0 <= max_truncated_rate <= 1:
        raise ValueError("invalid query count or rate gate")
    if len(groups) != expected_queries or len({group["id"] for group in groups}) != expected_queries:
        raise ValueError(f"D repair gate requires {expected_queries} distinct queries")
    details = []
    valid_total = truncated_total = 0
    for group in groups:
        samples = group["samples"]
        if len(samples) != 8:
            raise ValueError(f"{group['id']}: expected eight completions")
        signals = [_signal(sample) for sample in samples]
        rewards = [float(signal["reward"]) for signal in signals]
        if not all(math.isfinite(value) for value in rewards):
            raise ValueError(f"{group['id']}: non-finite reward")
        valid = [signal for signal in signals if signal["valid"]]
        valid_total += len(valid)
        truncated_total += sum(bool(signal.get("truncated", False)) for signal in signals)
        geometry_pairs = 0
        for index, left in enumerate(valid):
            for right in valid[index + 1:]:
                reward_diff = abs(float(left["reward"]) - float(right["reward"])) > 1e-6
                iou_diff = abs(float(left["iou"]) - float(right["iou"])) > 1e-6
                # The reviewed-negative contrast is computed from max negative IoU.
                contrast_diff = abs(float(left.get("contrast", 0.0)) -
                                    float(right.get("contrast", 0.0))) > 1e-6
                geometry_pairs += bool(reward_diff and (iou_diff or contrast_diff))
        details.append({"id": group["id"], "valid": len(valid),
                        "reward_difference": max(rewards) - min(rewards),
                        "geometry_reward_pairs": geometry_pairs,
                        "geometry_reward_signal": geometry_pairs > 0})
    geometry_groups = sum(group["geometry_reward_signal"] for group in details)
    answer_count = expected_queries * 8
    valid_rate = valid_total / answer_count
    truncated_rate = truncated_total / answer_count
    return {"passed": geometry_groups >= min_geometry_groups and
            valid_rate >= min_valid_rate and truncated_rate <= max_truncated_rate,
            "queries": expected_queries, "answers": answer_count,
            "valid": valid_total, "valid_rate": valid_rate,
            "truncated": truncated_total, "truncated_rate": truncated_rate,
            "min_valid_rate": min_valid_rate, "max_truncated_rate": max_truncated_rate,
            "min_geometry_groups": min_geometry_groups,
            "geometry_reward_groups": geometry_groups,
            "groups": details}


def eval_groups(rows: list[dict]) -> list[dict]:
    return [{"id": row["id"], "samples": row["samples"]} for row in rows]


def _new_reward_items(path: Path, offset: int) -> tuple[list[dict], int]:
    with path.open("rb") as source:
        source.seek(offset)
        data = source.read()
        position = source.tell()
    payloads = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    items = [item for payload in payloads for item in payload["items"]]
    if len(items) != 8:
        raise RuntimeError(f"Swift reward log emitted {len(items)} items for one G8 rollout")
    return items, position


def _check_saved_adapter_tensors(path: Path, expected: dict, torch) -> dict:
    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"saved adapter missing at {path}")
    observed = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as source:
            for key in source.keys():
                if key in observed or key not in expected:
                    raise RuntimeError(f"unexpected or duplicate saved adapter tensor: {key}")
                if not torch.equal(source.get_tensor(key), expected[key]):
                    raise RuntimeError(f"saved adapter differs from live post-step tensor: {key}")
                observed.add(key)
    if observed != set(expected):
        raise RuntimeError(f"saved adapter lacks {len(set(expected) - observed)} live tensors")
    return {"checkpoint": str(path), "saved_tensor_count": len(observed),
            "saved_values_equal": True, "adapter_reloaded": False,
            "description": "Saved safetensors equal live post-step tensors; no reload or extra optimizer step."}


def _response_token_ids(sample) -> list[int]:
    token_ids = sample.response_token_ids
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise RuntimeError("D repair expects one assistant response per rollout sample")
        token_ids = token_ids[0]
    return [int(token) for token in token_ids]


def _thinking_trace(sample) -> dict:
    from dataclasses import asdict, is_dataclass

    trace = sample.rollout_infos["bounded_thinking"]
    return asdict(trace) if is_dataclass(trace) else trace


def training_hook(output_dir: Path, report: dict, dataset_rows: dict[str, dict]):
    """Observe actual Swift rewards and stop before optimizer if signal is absent."""
    def install(trainer):
        import torch
        from peft import get_peft_model_state_dict
        from transformers import TrainerCallback
        from tools.train_quant_grounding import audit_gradients

        if trainer.state.global_step != 0:
            raise RuntimeError("D repair one-step run must start at optimizer step zero")
        if trainer.args.per_device_train_batch_size != 1:
            raise RuntimeError("D repair gate assumes one completion per microbatch")
        parameters = {name: parameter for name, parameter in trainer.model.named_parameters()
                      if parameter.requires_grad}
        if not parameters:
            raise RuntimeError("no trainable LoRA parameters")
        initial = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
        reward_path = output_dir / "reward_metrics.jsonl"
        if reward_path.exists() and reward_path.stat().st_size:
            raise RuntimeError("one-step reward log must start empty")
        rollout_path = output_dir / "first_rollout.jsonl"
        if rollout_path.exists() and rollout_path.stat().st_size:
            raise RuntimeError("first rollout export must start empty")
        report["first_rollout_path"] = str(rollout_path)
        report["first_rollout_groups"] = 0
        original_score = trainer._score_completions
        original_batch = trainer._generate_and_score_completions
        captured = {"groups": [], "reward_offset": 0, "nonzero_advantages": 0,
                    "active_tokens": 0, "max_old_ref_logp_gap": 0.0,
                    "pre_optimizer_checked": False, "step_finished": False,
                    "saved_verified": False, "saved_state": None}

        def score(samples):
            scored = original_score(samples)
            items, captured["reward_offset"] = _new_reward_items(reward_path, captured["reward_offset"])
            if len(samples) != 8:
                raise RuntimeError(f"D repair expected one G8 score call, got {len(samples)} samples")
            ids = {item["id"] for item in items}
            if len(ids) != 1:
                raise RuntimeError(f"Swift mixed queries in one G8 group: {ids}")
            actual = trainer._rewards_per_func[:, 0].detach().float().cpu().tolist()
            if len(actual) != 8 or any(abs(a - float(item["reward"])) > 1e-5
                                       for a, item in zip(actual, items)):
                raise RuntimeError("Swift advantage rewards differ from grounded reward log")
            sample_id = next(iter(ids))
            if any(sample.extra.get("id") != sample_id for sample in samples):
                raise RuntimeError("Swift sample IDs differ from grounded reward log")
            if len(captured["groups"]) < 4:
                with rollout_path.open("a", encoding="utf-8") as stream:
                    for index, (sample, item) in enumerate(zip(samples, items)):
                        stream.write(json.dumps({
                            "id": sample_id,
                            "sample_index": index,
                            "dataset_record": dataset_rows[sample_id],
                            "raw_text": item["generated_text"],
                            "token_ids": _response_token_ids(sample),
                            "thinking_trace": _thinking_trace(sample),
                            "finish_reason": item["finish_reason"],
                            "scored": item,
                        }, ensure_ascii=False, allow_nan=False) + "\n")
            captured["groups"].append({"id": sample_id, "samples": items})
            report["first_rollout_groups"] = min(len(captured["groups"]), 4)
            return scored

        def batch(inputs):
            batches = original_batch(inputs)
            active = 0
            denominator = None
            for item in batches:
                signal = item["grpo_batch"]
                mask = signal.completion_mask.bool()
                active += int(mask.sum().item())
                if not mask.any().item():
                    raise RuntimeError("GRPO microbatch has no sampled action tokens")
                advantages = signal.advantages[mask]
                if not torch.isfinite(advantages).all().item():
                    raise RuntimeError("non-finite GRPO advantage")
                captured["nonzero_advantages"] += int(torch.count_nonzero(advantages).item())
                if signal.ref_per_token_logps is None:
                    raise RuntimeError("D repair expected a frozen reference distribution")
                gap = (signal.old_per_token_logps[mask] -
                       signal.ref_per_token_logps[mask]).abs().max().item()
                captured["max_old_ref_logp_gap"] = max(captured["max_old_ref_logp_gap"], gap)
                current_denominator = float(signal.num_items_in_batch)
                if denominator is None:
                    denominator = current_denominator
                elif not math.isclose(denominator, current_denominator, abs_tol=1e-6):
                    raise RuntimeError("DAPO denominator differs within G8 group")
            if not math.isclose(denominator, active, abs_tol=1e-6):
                raise RuntimeError(f"DAPO denominator {denominator} != sampled-action tokens {active}")
            captured["active_tokens"] += active
            return batches

        class GateCallback(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                summary = gate_summary(captured["groups"], min_valid_rate=0.95,
                                       max_truncated_rate=0.0, min_geometry_groups=2)
                report["train_gate"] = summary
                report["active_tokens"] = captured["active_tokens"]
                report["nonzero_advantages"] = captured["nonzero_advantages"]
                report["max_old_ref_logp_gap"] = captured["max_old_ref_logp_gap"]
                if not summary["passed"] or not captured["nonzero_advantages"]:
                    raise RuntimeError(
                        "actual train rollout failed the output/geometry/advantage gate: "
                        f"valid={summary['valid']}/{summary['answers']}, "
                        f"truncated={summary['truncated']}, "
                        f"geometry_groups={summary['geometry_reward_groups']}, "
                        f"nonzero_advantages={captured['nonzero_advantages']}")
                if captured["max_old_ref_logp_gap"] > 1e-3:
                    raise RuntimeError("initial policy differs from reference; policy gradient is ambiguous")
                if not audit_gradients(trainer.model, torch):
                    raise RuntimeError("no finite nonzero LoRA gradient before optimizer step")
                captured["pre_optimizer_checked"] = True

            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step != 1 or not captured["pre_optimizer_checked"]:
                    raise RuntimeError("D repair expected exactly one gated optimizer step")
                changed_tensors = changed_elements = 0
                max_abs_delta = 0.0
                for name, parameter in parameters.items():
                    current = parameter.detach().cpu()
                    if not torch.isfinite(current).all().item():
                        raise RuntimeError(f"non-finite post-step LoRA parameter: {name}")
                    delta = current.float() - initial[name].float()
                    changed = int(torch.count_nonzero(delta).item())
                    changed_tensors += bool(changed)
                    changed_elements += changed
                    max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
                if not changed_elements:
                    raise RuntimeError("optimizer step left all LoRA values unchanged")
                report["parameter_update"] = {"changed_tensors": changed_tensors,
                                               "changed_elements": changed_elements,
                                               "max_abs_delta": max_abs_delta}
                initial.clear()
                captured["saved_state"] = {
                    key: value.detach().cpu().clone()
                    for key, value in get_peft_model_state_dict(trainer.model).items()
                }
                captured["step_finished"] = True
                control.should_save = True
                return control

            def on_save(self, args, state, control, **kwargs):
                if state.global_step == 1 and captured["step_finished"]:
                    checkpoint = output_dir / "checkpoint-1"
                    report["saved_adapter_tensor_check"] = _check_saved_adapter_tensors(
                        checkpoint, captured["saved_state"], torch)
                    captured["saved_state"] = None
                    captured["saved_verified"] = True

        trainer._score_completions = score
        trainer._generate_and_score_completions = batch
        callback = GateCallback()
        trainer.add_callback(callback)

        def cleanup():
            trainer._score_completions = original_score
            trainer._generate_and_score_completions = original_batch

        return cleanup

    return install


def formal_training_hook(output_dir: Path, report: dict):
    """Gate the first 12 formal G8 queries before optimizer step three."""
    def install(trainer):
        import torch
        from transformers import TrainerCallback
        from tools.train_quant_grounding import audit_gradients

        if trainer.state.global_step != 0:
            raise RuntimeError("formal D schedule must start at optimizer step zero")
        if trainer.args.per_device_train_batch_size != 1:
            raise RuntimeError("formal D schedule assumes one completion per microbatch")
        parameters = {name: parameter for name, parameter in trainer.model.named_parameters()
                      if parameter.requires_grad}
        if not parameters:
            raise RuntimeError("no trainable LoRA parameters")
        reward_path = output_dir / "reward_metrics.jsonl"
        if reward_path.exists() and reward_path.stat().st_size:
            raise RuntimeError("formal reward log must start empty")
        first12_path = output_dir / "first12_gate.json"
        report["first12_gate_path"] = str(first12_path)
        original_score = trainer._score_completions
        original_batch = trainer._generate_and_score_completions
        captured = {"first12_groups": [], "reward_offset": 0, "score_groups": 0,
                    "zero_variance_groups": 0, "step_groups": 0,
                    "step_geometry_groups": 0, "step_zero_variance_groups": 0,
                    "step_answers": 0, "step_valid": 0, "step_truncated": 0,
                    "step_nonzero_advantages": 0, "step_active_tokens": 0,
                    "step_localization_nonzero_advantages": 0,
                    "step_max_old_ref_logp_gap": 0.0, "pending_geometry": [],
                    "step_initial_parameters": None, "current_step_record": None,
                    "pre_optimizer_checked": False}

        def score(samples):
            result = original_score(samples)
            items, captured["reward_offset"] = _new_reward_items(
                reward_path, captured["reward_offset"])
            if len(samples) != 8:
                raise RuntimeError(f"formal D schedule expected one G8 score call, got {len(samples)}")
            ids = {item["id"] for item in items}
            if len(ids) != 1:
                raise RuntimeError(f"formal D rollout mixed queries in one G8 group: {ids}")
            occurrences = {item["occurrence_id"] for item in items}
            if len(occurrences) != 1 or None in occurrences:
                raise RuntimeError(f"formal D rollout lacks one presentation identity: {occurrences}")
            actual = trainer._rewards_per_func[:, 0].detach().float().cpu().tolist()
            if len(actual) != 8 or any(abs(value - float(item["reward"])) > 1e-5
                                       for value, item in zip(actual, items)):
                raise RuntimeError("formal Swift rewards differ from grounded reward log")
            sample_id = next(iter(ids))
            occurrence_id = next(iter(occurrences))
            if any(sample.extra.get("id") != sample_id
                   or sample.extra.get("occurrence_id") != occurrence_id for sample in samples):
                raise RuntimeError("formal Swift sample IDs differ from grounded reward log")
            group = {"id": occurrence_id, "samples": items}
            one = gate_summary([group], expected_queries=1, min_geometry_groups=0)
            geometry = one["geometry_reward_groups"]
            zero_variance = one["groups"][0]["reward_difference"] <= 1e-6
            captured["score_groups"] += 1
            captured["step_groups"] += 1
            captured["step_geometry_groups"] += geometry
            captured["zero_variance_groups"] += zero_variance
            captured["step_zero_variance_groups"] += zero_variance
            captured["step_answers"] += len(items)
            captured["step_valid"] += sum(bool(item["valid"]) for item in items)
            captured["step_truncated"] += sum(bool(item["truncated"]) for item in items)
            captured["pending_geometry"].append(bool(geometry))
            if len(captured["first12_groups"]) < 12:
                captured["first12_groups"].append(group)
            report["score_groups_seen"] = captured["score_groups"]
            report["zero_variance_groups"] = captured["zero_variance_groups"]
            return result

        def batch(inputs):
            batches = original_batch(inputs)
            if len(captured["pending_geometry"]) != 1:
                raise RuntimeError("formal G8 rollout and prepared GRPO batch ownership differs")
            geometry_signal = captured["pending_geometry"].pop()
            active = nonzero = 0
            denominator = None
            max_gap = 0.0
            for item in batches:
                signal = item["grpo_batch"]
                mask = signal.completion_mask.bool()
                active += int(mask.sum().item())
                if not mask.any().item():
                    raise RuntimeError("formal GRPO microbatch has no sampled action tokens")
                advantages = signal.advantages[mask]
                if not torch.isfinite(advantages).all().item():
                    raise RuntimeError("non-finite formal GRPO advantage")
                nonzero += int(torch.count_nonzero(advantages).item())
                if signal.ref_per_token_logps is None:
                    raise RuntimeError("formal D requires a frozen reference distribution")
                gap = (signal.old_per_token_logps[mask] -
                       signal.ref_per_token_logps[mask]).abs().max().item()
                max_gap = max(max_gap, gap)
                current_denominator = float(signal.num_items_in_batch)
                if denominator is None:
                    denominator = current_denominator
                elif not math.isclose(denominator, current_denominator, abs_tol=1e-6):
                    raise RuntimeError("formal DAPO denominator differs within optimizer step")
            if not math.isclose(denominator, active, abs_tol=1e-6):
                raise RuntimeError(f"formal DAPO denominator {denominator} != action tokens {active}")
            captured["step_nonzero_advantages"] += nonzero
            captured["step_active_tokens"] += active
            if geometry_signal and nonzero:
                captured["step_localization_nonzero_advantages"] += nonzero
            captured["step_max_old_ref_logp_gap"] = max(
                captured["step_max_old_ref_logp_gap"], max_gap)
            return batches

        class FormalGateCallback(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, **kwargs):
                step = int(state.global_step)
                if captured["step_groups"] != 4:
                    expected_groups = False
                else:
                    expected_groups = True
                valid_rate = (captured["step_valid"] / captured["step_answers"]
                              if captured["step_answers"] else None)
                has_nonzero_advantages = captured["step_nonzero_advantages"] > 0
                step_record = {
                    "optimizer_step": step + 1,
                    "queries": captured["step_groups"],
                    "answers": captured["step_answers"],
                    "valid": captured["step_valid"],
                    "valid_rate": valid_rate,
                    "truncated": captured["step_truncated"],
                    "geometry_reward_groups": captured["step_geometry_groups"],
                    "zero_variance_groups": captured["step_zero_variance_groups"],
                    "active_tokens": captured["step_active_tokens"],
                    "nonzero_advantages": captured["step_nonzero_advantages"],
                    "localization_nonzero_advantages": captured[
                        "step_localization_nonzero_advantages"],
                    "max_old_ref_logp_gap": captured["step_max_old_ref_logp_gap"],
                    "gradient_checked": has_nonzero_advantages,
                    "parameter_change_checked": has_nonzero_advantages,
                    "passed": False,
                }
                report.setdefault("optimizer_step_gates", []).append(step_record)
                if (not expected_groups or captured["step_answers"] != 32
                        or valid_rate is None or valid_rate < 0.95
                        or captured["step_truncated"] != 0):
                    step_record["failure"] = "step requires four G8 groups, valid rate >= 0.95, and zero truncation"
                    raise RuntimeError(
                        f"formal D optimizer step {step + 1} failed the 32-answer validity/truncation gate"
                    )
                if step == 0 and captured["step_max_old_ref_logp_gap"] > 1e-3:
                    step_record["failure"] = "initial policy differs from reference"
                    raise RuntimeError("initial policy differs from reference; first-step gradient is ambiguous")
                if has_nonzero_advantages and not audit_gradients(trainer.model, torch):
                    step_record["failure"] = "non-finite or zero LoRA gradient despite nonzero advantages"
                    raise RuntimeError("non-finite or zero LoRA gradient despite nonzero advantages")
                if step == 2:
                    if len(captured["first12_groups"]) != 12:
                        summary = {"passed": False, "queries": 12,
                                   "observed_queries": len(captured["first12_groups"]),
                                   "stage": "before_optimizer_step_3"}
                    else:
                        summary = gate_summary(
                            captured["first12_groups"], expected_queries=12,
                            min_valid_rate=0.95, max_truncated_rate=0.0,
                            min_geometry_groups=2,
                        )
                        summary["zero_variance_groups"] = sum(
                            detail["reward_difference"] <= 1e-6 for detail in summary["groups"])
                        summary["stage"] = "before_optimizer_step_3"
                    report["first12_gate"] = summary
                    first12_path.parent.mkdir(parents=True, exist_ok=True)
                    first12_path.write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if not summary["passed"]:
                        step_record["failure"] = "first12 gate failed"
                        raise RuntimeError("first 12 formal D queries lack the required learning signal")
                if has_nonzero_advantages:
                    captured["step_initial_parameters"] = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in parameters.items()
                    }
                step_record["passed"] = True
                captured["current_step_record"] = step_record
                captured["pre_optimizer_checked"] = True
                return control

            def on_step_end(self, args, state, control, **kwargs):
                if not captured["pre_optimizer_checked"] or captured["current_step_record"] is None:
                    raise RuntimeError("formal D optimizer step bypassed its gradient gate")
                record = captured["current_step_record"]
                if record["nonzero_advantages"]:
                    initial = captured["step_initial_parameters"]
                    changed_tensors = changed_elements = 0
                    max_abs_delta = 0.0
                    non_finite_name = None
                    for name, parameter in parameters.items():
                        current = parameter.detach().cpu()
                        if not torch.isfinite(current).all().item():
                            non_finite_name = name
                            break
                        delta = current.float() - initial[name].float()
                        changed = int(torch.count_nonzero(delta).item())
                        changed_tensors += bool(changed)
                        changed_elements += changed
                        max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
                    initial.clear()
                    captured["step_initial_parameters"] = None
                    if non_finite_name is not None:
                        record["passed"] = False
                        record["failure"] = f"non-finite post-step parameter: {non_finite_name}"
                        raise RuntimeError(f"non-finite post-step LoRA parameter: {non_finite_name}")
                    record["parameter_update"] = {"changed_tensors": changed_tensors,
                                                   "changed_elements": changed_elements,
                                                   "max_abs_delta": max_abs_delta}
                    if not changed_elements:
                        record["passed"] = False
                        record["failure"] = "optimizer step left trainable parameters unchanged"
                        raise RuntimeError("optimizer step left all trainable parameters unchanged")
                elif captured["step_initial_parameters"] is not None:
                    raise RuntimeError("zero-advantage step unexpectedly retained a parameter snapshot")
                report["optimizer_steps_completed"] = int(state.global_step)
                captured["current_step_record"] = None
                captured["step_groups"] = 0
                captured["step_geometry_groups"] = 0
                captured["step_zero_variance_groups"] = 0
                captured["step_answers"] = 0
                captured["step_valid"] = 0
                captured["step_truncated"] = 0
                captured["step_nonzero_advantages"] = 0
                captured["step_active_tokens"] = 0
                captured["step_localization_nonzero_advantages"] = 0
                captured["step_max_old_ref_logp_gap"] = 0.0
                captured["pre_optimizer_checked"] = False
                return control

        trainer._score_completions = score
        trainer._generate_and_score_completions = batch
        trainer.add_callback(FormalGateCallback())

        def cleanup():
            trainer._score_completions = original_score
            trainer._generate_and_score_completions = original_batch

        return cleanup

    return install


def _train_formal(args) -> dict:
    from tools.quant_grounding_config import load_config, prepare_dataset
    from tools.train_quant_grounding import run_swift

    schedule = read_jsonl(args.schedule)
    if len(schedule) < 236:
        raise ValueError(f"formal D schedule needs at least 236 presentations, got {len(schedule)}")
    rows = schedule[:236]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule_path = args.output_dir / "formal_schedule_236.jsonl"
    schedule_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    config = load_config(args.config, dataset=str(schedule_path), data_root=str(args.data_root),
                         output_dir=str(args.output_dir))
    if config["arm"] != "D" or config.get("bounded_thinking") is not True:
        raise ValueError("train-formal requires D with bounded_thinking=true")
    if config.get("format_policy") != "single_box":
        raise ValueError("train-formal requires the shared single_box reward format")
    if not config.get("adapters"):
        raise ValueError("train-formal must start from the shared warmup50 adapter")
    if config.get("resume_from_checkpoint"):
        raise ValueError("train-formal starts fresh from warmup50; resume_from_checkpoint must be unset")
    config["max_steps"] = 59
    config["save_steps"] = 59
    config["max_gpu_hours"] = None
    destination = args.output_dir / "prepared_train.jsonl"
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["QUANT_REWARD_METRICS_JSONL"] = str(args.output_dir / "reward_metrics.jsonl")
    prepared = prepare_dataset(schedule_path, destination, args.data_root, "D", bounded_thinking=True)
    if prepared != 236:
        raise RuntimeError(f"formal D schedule prepared {prepared}/236 presentations")
    report = {"status": "running", "schedule_source": str(args.schedule),
              "schedule_rows_available": len(schedule), "schedule_rows_used": 236,
              "prepared_dataset": str(destination), "output_dir": str(args.output_dir),
              "warmup_adapter": config["adapters"], "max_steps": 59,
              "max_gpu_hours": None}
    result_path = args.output_dir / "gate_result.json"
    try:
        run_swift(config, destination, max_steps=59, save_steps=59,
                  trainer_hook=formal_training_hook(args.output_dir, report))
        if not report.get("first12_gate", {}).get("passed"):
            raise RuntimeError("first 12 formal D presentations did not pass before optimizer step three")
        if report.get("optimizer_steps_completed") != 59:
            raise RuntimeError("formal D training did not complete 59 optimizer steps")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    generated = modes.add_parser("generate-check", help="CPU check of scored G8 eval predictions")
    generated.add_argument("--predictions", required=True, type=Path)
    generated.add_argument("--result", required=True, type=Path)
    generated.add_argument("--queries", type=int, default=4)
    generated.add_argument("--min-valid-rate", type=float, default=0.0)
    generated.add_argument("--max-truncated-rate", type=float, default=1.0)
    train = modes.add_parser("train-one", help="one on-policy GRPO optimizer update")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--data-root", required=True, type=Path)
    train.add_argument("--output-dir", required=True, type=Path)
    formal = modes.add_parser("train-formal", help="59-step bounded D training over 236 presentations")
    formal.add_argument("--config", required=True, type=Path)
    formal.add_argument("--schedule", required=True, type=Path)
    formal.add_argument("--data-root", required=True, type=Path)
    formal.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "generate-check":
        result = gate_summary(eval_groups(read_jsonl(args.predictions)),
                              expected_queries=args.queries, min_valid_rate=args.min_valid_rate,
                              max_truncated_rate=args.max_truncated_rate)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not result["passed"]:
            raise RuntimeError("four-query generation lacks geometry-dependent reward differences")
        return result
    if args.mode == "train-formal":
        return _train_formal(args)

    from tools.quant_grounding_config import load_config, prepare_dataset
    from tools.train_quant_grounding import run_swift

    source = read_jsonl(args.dataset)
    if len(source) != 4 or {row["id"] for row in source} != HARD4_IDS:
        raise ValueError("train-one requires the four fixed hard training queries")
    config = load_config(args.config, dataset=str(args.dataset), data_root=str(args.data_root),
                         output_dir=str(args.output_dir))
    if config["arm"] != "D" or config.get("bounded_thinking") is not True:
        raise ValueError("train-one requires D with bounded_thinking=true")
    if config.get("format_policy") != "single_box":
        raise ValueError("train-one requires the shared single_box reward format")
    destination = args.output_dir / "prepared_train.jsonl"
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["QUANT_REWARD_METRICS_JSONL"] = str(args.output_dir / "reward_metrics.jsonl")
    prepare_dataset(args.dataset, destination, args.data_root, "D", bounded_thinking=True)
    report: dict = {"status": "running", "dataset": str(args.dataset),
                    "output_dir": str(args.output_dir)}
    result_path = args.output_dir / "gate_result.json"
    try:
        run_swift(config, destination, max_steps=1, save_steps=1,
                  trainer_hook=training_hook(args.output_dir, report,
                                             {row["id"]: row for row in source}))
        if "saved_adapter_tensor_check" not in report:
            raise RuntimeError("checkpoint-1 saved adapter tensor comparison was not observed")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
