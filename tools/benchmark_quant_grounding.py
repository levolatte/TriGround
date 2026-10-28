"""Fixed-answer, one-update GRPO speed and numerical comparison.

Prepare on a CPU host, then run each variant in a separate fresh GPU process::

  python tools/benchmark_quant_grounding.py prepare --formal FORMAL/schedule.jsonl --data-root DATA --output BUNDLE
  python tools/benchmark_quant_grounding.py run --bundle BUNDLE --config arm_a.json --adapter checkpoint-50 --variant baseline --output RUNS
  python tools/benchmark_quant_grounding.py run --bundle BUNDLE --config arm_a.json --adapter checkpoint-50 --variant audit --output RUNS
  python tools/benchmark_quant_grounding.py run --bundle BUNDLE --config arm_a.json --adapter checkpoint-50 --variant mb2 --output RUNS
  python tools/benchmark_quant_grounding.py compare --runs RUNS

This deliberately uses Swift's rollout, reward, encoding, old/reference forward,
loss and optimizer. Only sampled responses are replaced. Candidate runs use the
baseline's masked old/reference log probabilities after Swift has prepared them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SELECTED_IDS = (
    "rgbdt500_110_00000078::q01",  # hit/miss split, reviewed negative
    "rgbdt500_323_00000118::q01",  # hit/miss split, reviewed negative
    "rgbdt500_083_00000299::q01",  # graded contrast values
    "rgbdt500_072_00000064::q01",  # equal-reward control
)
VARIANTS = {
    "baseline": (1, 1),
    "baseline_repeat": (1, 1),
    "audit": (1, 1),
    "gc2": (1, 2),
    "mb2": (2, 1),
    "mb2_gc2": (2, 2),
    "audit_conv": (1, 1),
    "gc2_conv": (1, 2),
    "mb2_conv": (2, 1),
    "mb2_gc2_conv": (2, 2),
}


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def prepare(formal: Path, data_root: Path, raw: Path, scored: Path, output: Path) -> dict:
    from tools.quant_grounding_config import prepare_dataset
    from tools.quant_grounding_reward import score_completion

    output.mkdir(parents=True, exist_ok=True)
    wanted = set(SELECTED_IDS)
    rows = {}
    for row in _jsonl(formal):
        if row["id"] in wanted and row["id"] not in rows:
            rows[row["id"]] = row
    raw_rows = {row["id"]: row for row in _jsonl(raw) if row["id"] in wanted}
    scored_rows = {row["id"]: row for row in _jsonl(scored) if row["id"] in wanted}
    if set(rows) != wanted or set(raw_rows) != wanted or set(scored_rows) != wanted:
        raise ValueError("四个固定问题必须同时存在于正式数据和 warmup50 full-vocab 原始/评分记录")

    manifest_rows = []
    source_path = output / "selected_source.jsonl"
    with source_path.open("w", encoding="utf-8") as stream:
        for sample_id in SELECTED_IDS:
            row, raw_row, scored_row = rows[sample_id], raw_rows[sample_id], scored_rows[sample_id]
            if len(row["images"]) != 3 or len(raw_row["samples"]) != 8 or len(scored_row["samples"]) != 8:
                raise ValueError(f"{sample_id}: 必须是真实三图问题且有完整 G8 回答")
            rewards = []
            responses = []
            for original, evaluated in zip(raw_row["samples"], scored_row["samples"]):
                if (original["text"], original["finish_reason"]) != (evaluated["text"], evaluated["finish_reason"]):
                    raise ValueError(f"{sample_id}: raw/scored 回答不一致")
                score = score_completion(
                    original["text"], row["solution"], row.get("negative_boxes"),
                    reward_mode="contrast", answer_mode="short",
                    finish_reason=original["finish_reason"], require_termination=True,
                )["reward"]
                if not math.isclose(score, evaluated["score"]["reward"], abs_tol=1e-7):
                    raise ValueError(f"{sample_id}: 正式数据奖励与诊断评分不一致")
                rewards.append(score)
                responses.append({"text": original["text"], "finish_reason": original["finish_reason"]})
            manifest_rows.append({"id": sample_id, "source": row["source"],
                                  "occurrence_id": row.get("occurrence_id"),
                                  "images": row["images"], "rewards": rewards,
                                  "responses": responses, "mixed_reward": len(set(rewards)) > 1})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if sum(row["mixed_reward"] for row in manifest_rows) < 2:
        raise ValueError("固定四组中至少两组奖励必须有差异")
    dataset_path = output / "swift_dataset.jsonl"
    if prepare_dataset(source_path, dataset_path, data_root, "A") != 4:
        raise RuntimeError("固定数据集不是四组")
    manifest = {"source_formal": str(formal.resolve()), "source_raw": str(raw.resolve()),
                "source_scored": str(scored.resolve()), "dataset": str(dataset_path.resolve()),
                "selected": manifest_rows, "answer_count": 32, "group_count": 4}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _masked_rows(batches, rewards):
    """Put all target signals into sample order, independent of microbatch padding."""
    items = []
    for batch in batches:
        signal = batch["grpo_batch"]
        mask = signal.completion_mask.bool()
        for i in range(mask.shape[0]):
            active = mask[i]
            items.append({
                "completion_tokens": int(active.sum().item()),
                "old": signal.old_per_token_logps[i][active].detach().float().cpu(),
                "ref": signal.ref_per_token_logps[i][active].detach().float().cpu(),
                "advantage": signal.advantages[i][active].detach().float().cpu(),
                "truncated": bool(signal.truncated_mask[i].item()),
            })
    if len(items) != len(rewards):
        raise RuntimeError("Swift 编码样本数与奖励数不一致")
    for item, reward in zip(items, rewards):
        item["reward"] = float(reward)
    return items


def _replay_targets(batches, native_rows, targets):
    import torch

    if len(native_rows) != len(targets):
        raise RuntimeError("固定答案数量与基线不一致")
    deviations = {"old_max_abs": 0.0, "ref_max_abs": 0.0}
    index = 0
    for batch in batches:
        signal = batch["grpo_batch"]
        mask = signal.completion_mask.bool()
        for row_index in range(mask.shape[0]):
            current, target = native_rows[index], targets[index]
            if (current["sample_id"] != target["sample_id"]
                    or current["token_ids"] != target["token_ids"]
                    or current["response_index"] != target["response_index"]
                    or current["completion_tokens"] != target["completion_tokens"]
                    or current["truncated"] != target["truncated"]
                    or not math.isclose(current["reward"], target["reward"], abs_tol=1e-7)
                    or not torch.allclose(current["advantage"], target["advantage"], rtol=0, atol=1e-6)):
                raise RuntimeError(f"固定目标在回答 {index} 发生变化")
            for key in ("old", "ref"):
                native = current[key]
                golden = target[key]
                deviations[f"{key}_max_abs"] = max(deviations[f"{key}_max_abs"],
                                                       float((native - golden).abs().max().item()))
                signal_name = "old_per_token_logps" if key == "old" else "ref_per_token_logps"
                getattr(signal, signal_name)[row_index][mask[row_index]] = golden.to(
                    device=mask.device, dtype=getattr(signal, signal_name).dtype)
            index += 1
    return deviations


def install_preclip_snapshot(trainer, model, destination: Path, state: dict, torch):
    """Capture the accumulated LoRA gradient before HF applies norm clipping."""
    original = trainer._clip_grad_norm
    state["preclip_calls"] = 0

    def capture(*args, **kwargs):
        if state["preclip_calls"]:
            raise RuntimeError("固定基准只允许一次梯度裁剪")
        trainable = {name: parameter for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        # Pending backward work belongs to the update, not to the excluded copy time.
        if any(parameter.is_cuda for parameter in trainable.values()):
            torch.cuda.synchronize()
        started = time.monotonic()
        grads = {name: parameter.grad.detach().float().cpu().clone()
                 for name, parameter in trainable.items() if parameter.grad is not None}
        if set(grads) != set(trainable) or not any(torch.count_nonzero(g).item() for g in grads.values()):
            raise RuntimeError("固定32回答没有完整非零 LoRA 梯度")
        if not all(torch.isfinite(g).all().item() for g in grads.values()):
            raise RuntimeError("裁剪前 LoRA 梯度有 NaN/Inf")
        torch.save(grads, destination)
        state["gradient_snapshot_seconds"] = time.monotonic() - started
        state["preclip_calls"] += 1
        return original(*args, **kwargs)

    trainer._clip_grad_norm = capture

    def restore():
        trainer._clip_grad_norm = original

    return restore


def run(bundle: Path, config_path: Path, adapter: Path, variant: str, output: Path,
        *, model: str | None = None, precision: str | None = None) -> dict:
    import torch
    from transformers import TrainerCallback
    from tools.quant_grounding_config import load_config
    from tools.swift_quant_grounding_reward import SHORT_RESPONSE_PREFIX
    from tools import train_quant_grounding as training

    if variant not in VARIANTS:
        raise ValueError(f"未知候选 {variant}")
    if adapter.name != "checkpoint-50" or not adapter.is_dir():
        raise ValueError("固定基准必须从共享 warmup checkpoint-50 LoRA 初始化")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    baseline_dir = output / "baseline"
    run_dir = output / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    if list(run_dir.iterdir()):
        raise RuntimeError(f"输出目录非空，拒绝覆盖已有基准: {run_dir}")
    config = load_config(config_path, model=model, precision=precision)
    os.environ['MAX_PIXELS'] = str(config.get('max_pixels', 1204224))
    os.environ['MIN_PIXELS'] = str(config.get('min_pixels', 200704))
    if config["arm"] != "A":
        raise ValueError("固定答案基准只使用 A 奖励与 64-token 短回答")
    microbatch, gc_every = VARIANTS[variant]
    config.update({"output_dir": str(run_dir.resolve()), "adapters": str(adapter.resolve()),
                   "train_micro_batch_size": microbatch,
                   "gradient_checkpointing_every_n_layers": gc_every,
                   "runtime_optimization": variant not in {"baseline", "baseline_repeat"},
                   "benchmark_mode": False,
                   "max_gpu_hours": 2, "save_hours": [],
                   "save_steps": 1000000})
    expected = {row["id"]: row["responses"] for row in manifest["selected"]}
    expected_rewards = {row["id"]: row["rewards"] for row in manifest["selected"]}
    targets_path = baseline_dir / "fixed_targets.pt"
    targets = torch.load(targets_path, map_location="cpu", weights_only=True) if variant != "baseline" else None
    state = {"seen": {sample_id: 0 for sample_id in expected}, "generated": [], "rows": [], "losses": [],
             "native_logp_deviation": {"old_max_abs": 0.0, "ref_max_abs": 0.0},
             "timing": {"rollout_score_seconds": 0.0, "training_step_seconds": 0.0},
             "training_events": []}

    def trainer_hook(trainer):
        if trainer.state.global_step != 0:
            raise RuntimeError("固定基准必须从新优化器的第0步开始")
        if getattr(trainer.accelerator, "scaler", None) is not None:
            raise RuntimeError("固定基准要求 BF16 无 GradScaler，以比较真实裁剪前梯度")
        if variant.endswith("_conv"):
            selected_conv = training.audit_kernels()["causal_conv1d_fn"]
            if "causal_conv1d" not in selected_conv["module"]:
                raise RuntimeError(f"conv 候选没有调用优化因果卷积核: {selected_conv}")
        free_before, total_before = torch.cuda.mem_get_info()
        state["non_pytorch_device_bytes_at_start"] = (
            total_before - free_before - torch.cuda.memory_reserved())
        model = trainer.model
        initial = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
                   if p.requires_grad}
        if not initial:
            raise RuntimeError("未找到可训练 LoRA 参数")
        original_generate = trainer._generate_completions
        original_batch = trainer._generate_and_score_completions
        original_step = trainer.training_step
        original_loss = trainer.compute_loss
        restore_clip_snapshot = install_preclip_snapshot(
            trainer, model, run_dir / "gradients.pt", state, torch)

        def fixed_generate(samples):
            samples = trainer._preprocess_inputs(samples)
            if len(samples) % 8:
                raise RuntimeError("固定 rollout 必须由完整 G8 题组组成")
            for start in range(0, len(samples), 8):
                group = samples[start:start + 8]
                identities = {(sample.extra["id"], sample.extra.get("occurrence_id")) for sample in group}
                if len(identities) != 1:
                    raise RuntimeError(f"固定 G8 题组出现混合 id/occurrence_id: {identities}")
            for sample in samples:
                sample_id = sample.extra["id"]
                position = state["seen"][sample_id]
                if position >= 8:
                    raise RuntimeError(f"{sample_id}: 固定答案被重复消费")
                answer = expected[sample_id][position]
                token_ids = trainer.template.tokenizer.encode(answer["text"], add_special_tokens=False)
                decoded = trainer.template.decode_generate_ids(
                    token_ids, first_token=False,
                    template_inputs=SimpleNamespace(chat_template_kwargs=sample.extra.get("chat_template_kwargs") or {}))
                if decoded != answer["text"]:
                    raise RuntimeError(f"{sample_id}: tokenization 无法还原固定答案")
                sample.messages.append({"role": "assistant", "content": SHORT_RESPONSE_PREFIX + answer["text"]})
                sample.response_token_ids = token_ids
                sample.finish_reason = answer["finish_reason"]
                sample.add_eos = False
                state["generated"].append({"sample_id": sample_id, "token_ids": token_ids,
                                           "response_index": position,
                                           "expected_reward": expected_rewards[sample_id][position]})
                state["seen"][sample_id] += 1
            trainer.model.train()
            return samples

        def fixed_batch(inputs):
            started = time.monotonic()
            batches = original_batch(inputs)
            count = sum(batch["grpo_batch"].completion_mask.shape[0] for batch in batches)
            rewards = trainer._rewards_per_func[:, 0].detach().float().cpu().tolist()
            rows = _masked_rows(batches, rewards)
            offset = len(state["rows"])
            generated = state["generated"][offset:offset + count]
            if len(generated) != count:
                raise RuntimeError("生成记录与 Swift 编码批次不一致")
            for row, identity in zip(rows, generated):
                row.update(identity)
                if not math.isclose(row["reward"], row["expected_reward"], abs_tol=1e-6):
                    raise RuntimeError(f"{row['sample_id']}: Swift 奖励与正式标注核算不一致")
            if targets is not None:
                deviations = _replay_targets(batches, rows, targets[offset:offset + count])
                for key, value in deviations.items():
                    state["native_logp_deviation"][key] = max(state["native_logp_deviation"][key], value)
            state["rows"].extend(rows)
            state["timing"]["rollout_score_seconds"] += time.monotonic() - started
            return batches

        def measured_step(*args, **kwargs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = original_step(*args, **kwargs)
            end_event.record()
            state["training_events"].append((start_event, end_event))
            return result

        def observed_loss(*args, **kwargs):
            loss = original_loss(*args, **kwargs)
            state["losses"].append(loss.detach())
            return loss

        class Snapshot(TrainerCallback):
            def on_step_begin(self, args, trainer_state, control, **kwargs):
                state["step_started"] = time.monotonic()

            def on_pre_optimizer_step(self, args, trainer_state, control, **kwargs):
                if state["preclip_calls"] != 1:
                    raise RuntimeError("优化器执行前没有裁剪前梯度快照")
                postclip = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                norms = torch.stack(torch._foreach_norm(postclip, ord=2)).float()
                maxima = torch.stack(torch._foreach_norm(postclip, ord=float("inf"))).float()
                postclip_norm, finite = torch.stack((torch.linalg.vector_norm(norms),
                                                      torch.isfinite(maxima).all().float())).cpu().tolist()
                if not finite:
                    raise RuntimeError("裁剪后 LoRA 梯度有 NaN/Inf")
                state["postclip_grad_norm"] = postclip_norm

            def on_step_end(self, args, trainer_state, control, **kwargs):
                step_wall = time.monotonic() - state["step_started"]
                if trainer_state.global_step != 1 or len(state["rows"]) != 32:
                    raise RuntimeError("基准应只完成一轮32回答更新")
                deltas = {name: p.detach().float().cpu() - initial[name]
                          for name, p in model.named_parameters() if p.requires_grad}
                if not all(torch.isfinite(value).all().item() for value in deltas.values()):
                    raise RuntimeError("LoRA 更新有 NaN/Inf")
                if not any(torch.count_nonzero(value).item() for value in deltas.values()):
                    raise RuntimeError("固定基准无 LoRA 权重更新")
                torch.save(deltas, run_dir / "updates.pt")
                torch.save(state["rows"], run_dir / "fixed_targets.pt") if variant == "baseline" else None
                loss_values = torch.stack(state["losses"]).float().cpu().tolist()
                state["timing"]["training_step_seconds"] = sum(
                    start.elapsed_time(end) for start, end in state["training_events"]) / 1000
                elapsed = time.monotonic() - started_run
                free, total = torch.cuda.mem_get_info()
                resource = next(_jsonl(run_dir / "resource_steps.jsonl"))
                peak_allocated = resource["allocated_peak_bytes"]
                peak_reserved = resource["reserved_peak_bytes"]
                gradients = torch.load(run_dir / "gradients.pt", weights_only=True)
                grad_squared = sum(float((g.double() ** 2).sum()) for g in gradients.values())
                report = {"variant": variant, "precision": config["precision"],
                          "adapter": str(adapter.resolve()), "global_step": 1,
                          "answer_count": len(state["rows"]),
                          "loss_microbatch_values": loss_values, "loss_sum": sum(loss_values),
                          "loss_effective": sum(loss_values) / trainer.args.gradient_accumulation_steps,
                          "grad_norm": math.sqrt(grad_squared),
                          "gradient_snapshot_stage": "after_backward_before_clip_grad_norm",
                          "postclip_grad_norm": state["postclip_grad_norm"],
                          "timing": {**state["timing"], "step_wall_seconds_before_snapshot": step_wall,
                                     "step_wall_excluding_grad_snapshot": (
                                         step_wall - state["gradient_snapshot_seconds"]),
                                     "gradient_snapshot_seconds": state["gradient_snapshot_seconds"],
                                     "run_wall_seconds_including_model_load": elapsed},
                          "native_logp_deviation": state["native_logp_deviation"],
                          "memory": {"peak_allocated_bytes": peak_allocated,
                                     "peak_reserved_bytes": peak_reserved,
                                     "device_total_bytes": total, "device_free_after_step_bytes": free,
                                     "non_pytorch_device_bytes_at_start": (
                                         state["non_pytorch_device_bytes_at_start"]),
                                     "estimated_whole_gpu_peak_headroom_bytes": (
                                         total - peak_reserved
                                         - state["non_pytorch_device_bytes_at_start"])}}
                (run_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                control.should_save = False
                return control

        trainer._generate_completions = fixed_generate
        trainer._generate_and_score_completions = fixed_batch
        trainer.training_step = measured_step
        trainer.compute_loss = observed_loss
        callback = Snapshot()
        trainer.add_callback(callback)

        def cleanup():
            trainer._generate_completions = original_generate
            trainer._generate_and_score_completions = original_batch
            trainer.training_step = original_step
            trainer.compute_loss = original_loss
            restore_clip_snapshot()
        return cleanup

    started_run = time.monotonic()
    training.run_swift(config, Path(manifest["dataset"]), max_steps=1, save_steps=1000000,
                       trainer_hook=trainer_hook)
    if state["seen"] != {sample_id: 8 for sample_id in expected}:
        raise RuntimeError(f"固定答案消费不完整: {state['seen']}")
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def _vector_comparison(reference, candidate):
    import torch

    if reference.keys() != candidate.keys():
        raise RuntimeError("LoRA 参数集合不同")
    dot = ref_norm = cand_norm = diff_norm = 0.0
    for key in reference:
        a, b = reference[key].double(), candidate[key].double()
        dot += float((a * b).sum())
        ref_norm += float((a * a).sum())
        cand_norm += float((b * b).sum())
        diff_norm += float(((a - b) ** 2).sum())
    return {"cosine": dot / math.sqrt(ref_norm * cand_norm),
            "relative_l2": math.sqrt(diff_norm / ref_norm),
            "reference_l2": math.sqrt(ref_norm), "candidate_l2": math.sqrt(cand_norm)}


def compare(runs: Path) -> dict:
    import torch
    torch.set_num_threads(4)

    baseline = json.loads((runs / "baseline" / "result.json").read_text(encoding="utf-8"))
    result = {"baseline": baseline, "candidates": {}, "not_run": []}
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        path = runs / variant / "result.json"
        if not path.exists():
            result["not_run"].append(variant)
            continue
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate["adapter"] != baseline["adapter"] or candidate["precision"] != baseline["precision"]:
            raise RuntimeError("候选与基线的 LoRA 初始化或精度不一致")
        comparisons = {}
        for kind, filename in (("gradient", "gradients.pt"), ("update", "updates.pt")):
            reference = torch.load(runs / "baseline" / filename, map_location="cpu", weights_only=True)
            actual = torch.load(runs / variant / filename, map_location="cpu", weights_only=True)
            comparisons[kind] = _vector_comparison(reference, actual)
        loss_ref, loss_now = baseline["loss_effective"], candidate["loss_effective"]
        comparisons["loss_relative"] = abs(loss_ref - loss_now) / max(abs(loss_ref), 1e-12)
        comparisons["numeric_pass"] = (comparisons["gradient"]["cosine"] >= 0.999
                                       and comparisons["gradient"]["relative_l2"] <= 0.01
                                       and comparisons["update"]["cosine"] >= 0.999
                                       and comparisons["update"]["relative_l2"] <= 0.01)
        comparisons["gpu_training_speed_ratio"] = (
            baseline["timing"]["training_step_seconds"] / candidate["timing"]["training_step_seconds"])
        comparisons["step_wall_speed_ratio"] = (
            baseline["timing"]["step_wall_excluding_grad_snapshot"]
            / candidate["timing"]["step_wall_excluding_grad_snapshot"])
        result["candidates"][variant] = {"run": candidate, "comparison": comparisons}
    (runs / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    prepared.add_argument("--formal", required=True, type=Path)
    prepared.add_argument("--data-root", required=True, type=Path)
    evidence = Path(__file__).resolve().parents[1] / "docs/research/2026-10-26-quant-rl/evidence"
    prepared.add_argument("--raw", type=Path, default=evidence / "warmup50-diagnostic32-full-vocab-raw.jsonl")
    prepared.add_argument("--scored", type=Path, default=evidence / "warmup50-diagnostic32-full-vocab-scored.jsonl")
    prepared.add_argument("--output", required=True, type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--bundle", required=True, type=Path)
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--adapter", required=True, type=Path)
    run_parser.add_argument("--model", help="local BF16 base-model directory on the GPU host")
    run_parser.add_argument("--precision", choices=("nf4", "bf16"))
    run_parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    run_parser.add_argument("--output", required=True, type=Path)
    compared = commands.add_parser("compare")
    compared.add_argument("--runs", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.formal, args.data_root, args.raw, args.scored, args.output)
    elif args.command == "run":
        report = run(args.bundle, args.config, args.adapter, args.variant, args.output,
                     model=args.model, precision=args.precision)
    else:
        report = compare(args.runs)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
