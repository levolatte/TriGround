"""Paired, fixed-answer Qwen3.5 shared-prefix benchmark on one loaded PEFT model.

CPU preparation::
  python tools/benchmark_shared_prefix.py prepare --formal DATA/schedule.jsonl --data-root DATA --output BUNDLE

GPU comparison (one group is a correctness probe; four groups are a full update)::
  python tools/benchmark_shared_prefix.py run --bundle BUNDLE --config arm_a.json --adapter checkpoint-50 --groups 1 --output RESULT
  python tools/benchmark_shared_prefix.py run --bundle BUNDLE --config arm_a.json --adapter checkpoint-50 --groups 4 --output RESULT32

Both paths use their OWN old/reference logps and Swift's original loss. Fixed
answers replace sampling, so the measured update excludes online generation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(formal: Path, data_root: Path, output: Path, *, answers: Path | None = None,
            raw: Path | None = None, scored: Path | None = None) -> dict:
    """Reuse the audited four-group fixture, or accept explicit fixed-answer JSONL.

    Custom rows have ``id`` and ``responses`` (eight {text,finish_reason} items).
    Their ids select the original formal rows; rewards are computed, never trusted
    from the supplied answers.
    """
    from tools.quant_grounding_config import prepare_dataset
    from tools.quant_grounding_reward import score_completion

    output.mkdir(parents=True, exist_ok=True)
    if answers is None:
        from tools.benchmark_quant_grounding import prepare as prepare_original
        evidence = Path(__file__).resolve().parents[1] / "docs/research/2026-10-26-quant-rl/evidence"
        return prepare_original(
            formal, data_root,
            raw or evidence / "warmup50-diagnostic32-full-vocab-raw.jsonl",
            scored or evidence / "warmup50-diagnostic32-full-vocab-scored.jsonl", output)
    fixed = _jsonl(answers)
    if len({row["id"] for row in fixed}) != len(fixed):
        raise ValueError("固定回答 JSONL 的 id 必须唯一")
    wanted = {row["id"] for row in fixed}
    source = {}
    for row in _jsonl(formal):
        if row["id"] in wanted and row["id"] not in source:
            source[row["id"]] = row
    if set(source) != wanted:
        raise ValueError("固定回答中的 id 未全部出现在 formal 数据")
    selected = []
    source_path = output / "selected_source.jsonl"
    with source_path.open("w", encoding="utf-8") as stream:
        for item in fixed:
            row, responses = source[item["id"]], item["responses"]
            if len(row["images"]) != 3 or len(responses) != 8:
                raise ValueError("基准需要单题三图及完整 G8 固定回答")
            if any(not {"text", "finish_reason"} <= set(response) for response in responses):
                raise ValueError("每个回答须有 text 和 finish_reason")
            rewards = [score_completion(
                response["text"], row["solution"], row.get("negative_boxes"),
                reward_mode="contrast", answer_mode="short", finish_reason=response["finish_reason"],
                require_termination=True)["reward"] for response in responses]
            selected.append({"id": row["id"], "images": row["images"], "responses": responses,
                             "rewards": rewards, "mixed_reward": len(set(rewards)) > 1})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset = output / "swift_dataset.jsonl"
    prepare_dataset(source_path, dataset, data_root, "A")
    manifest = {"dataset": str(dataset.resolve()), "selected": selected,
                "group_count": len(selected), "answer_count": 8 * len(selected),
                "source_formal": str(formal.resolve()), "source_answers": str(answers.resolve())}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def answer_logps(logits, input_ids, completion_width: int, temperature: float = 1.0,
                 selective=None):
    """Exactly Swift's causal shift, including P-1 predicting answer token zero."""
    if selective is None:
        import torch
        def selective(values, targets):
            return torch.log_softmax(values.float(), -1).gather(
                -1, targets.unsqueeze(-1)).squeeze(-1)
    prediction = logits[:, -(completion_width + 1):-1, :] / temperature
    if prediction.shape[1] != completion_width:
        raise ValueError("logits 必须包含首个回答前的 P-1 预测行")
    return selective(prediction, input_ids[:, -completion_width:])


def collate_shared_inputs(batches: list[dict], positions: list, pad_token_id: int) -> dict:
    """Expand the compact plan from the original mb1 inputs, using left padding.

    A single question's pixels/grids must be exactly equal across all eight
    answers. Position ids come from the actual native language-model invocation,
    after the multimodal wrapper has calculated its three-dimensional positions.
    """
    import torch

    if len(batches) != len(positions) or not batches:
        raise ValueError("原始输入与实际位置编码数量不一致")
    sequences, coordinates, prefix_lengths = [], [], []
    for batch, position in zip(batches, positions):
        inputs, signal = batch["model_inputs"], batch["grpo_batch"]
        ids = inputs["input_ids"]
        if ids.shape[0] != 1 or position.ndim != 3 or position.shape[1:] != ids.shape:
            raise ValueError("基准原始路径要求 mb1 及实际三维位置编码")
        mask = inputs.get("attention_mask", torch.ones_like(ids)).bool()[0]
        active = torch.where(mask)[0]
        if not len(active) or not torch.equal(active, torch.arange(active[0], active[-1] + 1, device=ids.device)):
            raise ValueError("有效输入必须连续")
        if int(active[-1]) != ids.shape[1] - 1:
            raise ValueError("原始 mb1 输入不应含右填充")
        width = int(signal.logits_to_keep)
        if width <= 0 or width != signal.completion_mask.shape[1]:
            raise ValueError("Swift 回答掩码与 logits_to_keep 不一致")
        sequences.append(ids[0, active])
        coordinates.append(position[:, 0, active])
        prefix_lengths.append(len(active) - width)
    if len(set(prefix_lengths)) != 1 or prefix_lengths[0] <= 0:
        raise ValueError("同题固定回答的提示长度不同")
    prefix = prefix_lengths[0]
    length = max(len(ids) for ids in sequences)
    count = len(sequences)
    ids = sequences[0].new_full((count, length), pad_token_id)
    mask = ids.new_zeros((count, length))
    pos = coordinates[0].new_zeros((coordinates[0].shape[0], count, length))
    for index, (sequence, coordinate) in enumerate(zip(sequences, coordinates)):
        if not torch.equal(sequence[:prefix], sequences[0][:prefix]):
            raise ValueError("同题提示词元内容不同")
        if not torch.equal(coordinate[:, :prefix], coordinates[0][:, :prefix]):
            raise ValueError("同题提示的三维位置不同")
        ids[index, -len(sequence):], mask[index, -len(sequence):] = sequence, 1
        pos[:, index, -len(sequence):] = coordinate
    first = batches[0]["model_inputs"]
    for key in ("pixel_values", "image_grid_thw"):
        if key not in first:
            raise ValueError(f"真实三图基准缺少 {key}")
        for batch in batches[1:]:
            if not torch.equal(first[key], batch["model_inputs"][key]):
                raise ValueError(f"同题的 {key} 不同，不能共享")
    if first["image_grid_thw"].shape[0] != 3:
        raise ValueError("每题必须恰好有三张图的网格")
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
            "pixel_values": first["pixel_values"], "image_grid_thw": first["image_grid_thw"],
            "prefix_length": prefix, "completion_width": max(int(b["grpo_batch"].logits_to_keep) for b in batches)}


def compare_expanded_inputs(batches, positions, packed) -> dict:
    """Check every effective id/mask/position/pixel/grid, with no hashes."""
    import torch

    checked = defaultdict(int)
    for index, (batch, position) in enumerate(zip(batches, positions)):
        original = batch["model_inputs"]
        original_mask = original.get("attention_mask", torch.ones_like(original["input_ids"])).bool()[0]
        compact_mask = packed["attention_mask"][index].bool()
        pairs = {
            "input_ids": (original["input_ids"][0, original_mask], packed["input_ids"][index, compact_mask]),
            "attention_mask": (original_mask[original_mask], compact_mask[compact_mask]),
            "position_ids": (position[:, 0, original_mask], packed["position_ids"][:, index, compact_mask]),
            "pixel_values": (original["pixel_values"], packed["pixel_values"]),
            "image_grid_thw": (original["image_grid_thw"], packed["image_grid_thw"]),
        }
        for name, (left, right) in pairs.items():
            if left.dtype != right.dtype or not torch.equal(left, right):
                raise RuntimeError(f"紧凑输入展开后 {name} 与原始回答 {index} 不一致")
            checked[name] += left.numel()
        width = int(batch["grpo_batch"].logits_to_keep)
        first_prediction = int(torch.where(compact_mask)[0][0]) + packed["prefix_length"] - 1
        if first_prediction != packed["input_ids"].shape[1] - width - 1:
            raise RuntimeError("首个回答词元预测位置偏移")
    return {"all_equal": True, "checked_elements": dict(checked),
            "padding": "left; compare all effective tokens and explicit effective masks",
            "pixels_shape": list(packed["pixel_values"].shape),
            "image_grid_thw": packed["image_grid_thw"].cpu().tolist(),
            "prefix_length": packed["prefix_length"], "answer_count": len(batches)}


def normalized_group_loss(losses, gradient_accumulation_steps: int):
    """Keep the exact original per-answer DAPO denominator and /32 scaling."""
    return sum(losses) / gradient_accumulation_steps


class VectorStatistics:
    """Accumulate per-parameter CPU statistics without storing candidate vectors."""
    def __init__(self):
        self.reference_squared = self.candidate_squared = self.difference_squared = self.dot = 0.0
        self.max_abs = 0.0
        self.elements = 0
        self.sign_changes = 0
        self.finite = True

    def add(self, reference, candidate):
        import torch
        a, b = reference.detach().cpu().double(), candidate.detach().cpu().double()
        self.reference_squared += float(a.square().sum())
        self.candidate_squared += float(b.square().sum())
        self.difference_squared += float((a - b).square().sum())
        self.dot += float((a * b).sum())
        self.max_abs = max(self.max_abs, float((a - b).abs().max()))
        self.elements += a.numel()
        self.sign_changes += int(((a * b) < 0).sum())
        self.finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())

    def report(self):
        product = self.reference_squared * self.candidate_squared
        return {"cosine": self.dot / math.sqrt(product) if product else None,
                "relative_l2": math.sqrt(self.difference_squared / self.reference_squared)
                if self.reference_squared else (0.0 if not self.difference_squared else None),
                "reference_l2": math.sqrt(self.reference_squared),
                "candidate_l2": math.sqrt(self.candidate_squared), "max_abs": self.max_abs,
                "elements": self.elements, "sign_change_fraction": self.sign_changes / max(self.elements, 1),
                "finite": self.finite}


def diagnostic_token_positions(prefix_length: int, answer_length: int) -> list[int]:
    """Sample prompt/boundary rows and every answer row, using unpadded indices."""
    boundary = prefix_length // 64 * 64
    return sorted({0, prefix_length // 2, max(0, boundary - 1),
                   min(boundary, prefix_length - 1), max(0, prefix_length - 4), prefix_length - 1,
                   *range(prefix_length, prefix_length + answer_length)})


def sample_trace_rows(tensor, rows, *, seq_dim: int, batch_dim: int = 0, batch_index: int = 0):
    """Canonical CPU [sampled-token, all-other-features] for native/compact layouts."""
    import torch
    index = torch.tensor(rows, device=tensor.device, dtype=torch.long)
    selected = tensor.detach().index_select(seq_dim, index).select(batch_dim, batch_index)
    remaining_seq_dim = seq_dim - int(batch_dim < seq_dim)
    return selected.movedim(remaining_seq_dim, 0).reshape(len(rows), -1).cpu()


def compare_trace_records(reference: dict, candidate: dict) -> dict:
    """Compare aligned sampled rows; missing event pairs remain explicit."""
    comparisons = {}
    for name, original in reference.items():
        if name not in candidate:
            continue
        actual = candidate[name]
        if original["positions"] != actual["positions"] or original["values"].shape != actual["values"].shape:
            raise RuntimeError(f"诊断事件 {name} 的采样词元或特征形状未对齐")
        stats = VectorStatistics()
        stats.add(original["values"], actual["values"])
        comparisons[name] = {**stats.report(), "positions": original["positions"],
                             "native_shape": original["source_shape"], "candidate_shape": actual["source_shape"],
                             "native_dtype": original["source_dtype"], "candidate_dtype": actual["source_dtype"]}
    different = [name for name, stats in comparisons.items() if stats["max_abs"] > 0]
    return {"events": comparisons, "first_observed_nonzero_event": different[0] if different else None,
            "reference_only_events": sorted(reference.keys() - candidate.keys()),
            "candidate_only_events": sorted(candidate.keys() - reference.keys())}


def compare_operator_rows(reference, candidate, plan) -> dict:
    """Full compact-row statistics, including the first differing token and regions."""
    import torch

    if reference.shape != candidate.shape or reference.shape[:2] != (1, plan.compact_length):
        raise ValueError("算子诊断需要相同形状的全部紧凑词元行")
    left = reference.detach().reshape(plan.compact_length, -1).cpu().double()
    right = candidate.detach().reshape(plan.compact_length, -1).cpu().double()

    def summarize(start, end):
        a, b = left[start:end], right[start:end]
        stats = VectorStatistics()
        if a.numel():
            stats.add(a, b)
        changed = a != b
        rows = changed.any(-1).nonzero().flatten()
        first = start + int(rows[0]) if rows.numel() else None
        token = None
        if first is not None:
            token = {"compact_row": first, "branch": None, "unpadded_position": first}
            for index, region in enumerate(plan.suffix_rows):
                if region.start <= first < region.stop:
                    token.update(branch=index, unpadded_position=plan.prefix_length + first - region.start)
                    break
        return {**stats.report(), "nonzero_count": int(changed.sum()),
                "nonzero_token_count": int(rows.numel()), "first_nonzero_token": token}

    regions = {"prefix_before_boundary": (0, plan.gdn_boundary),
               "prefix_tail": (plan.gdn_boundary, plan.prefix_length),
               **{f"suffix.{i}": (rows.start, rows.stop) for i, rows in enumerate(plan.suffix_rows)}}
    return {"all_rows": summarize(0, plan.compact_length),
            "regions": {name: summarize(*bounds) for name, bounds in regions.items()}}


def diagnose_layer0_operators(mixer, normed, plan, *, kernel, native_conv,
                              native_sequence_kwargs: dict) -> dict:
    """Isolate projections, convolution, varlen metadata and GDN state forking."""
    import torch
    import torch.nn.functional as F

    if torch.is_grad_enabled():
        raise ValueError("算子阶梯诊断只允许 no_grad")
    native_cu = native_sequence_kwargs.get("cu_seq_lens_q")
    if native_cu is not None and native_cu.numel() != 2:
        raise ValueError("算子阶梯限定原生 mb1 单序列 cu_seq_lens_q")

    def cu(length):
        return torch.tensor([0, length], dtype=native_cu.dtype if native_cu is not None else torch.int32,
                            device=normed.device)

    def branchwise(function, *tensors):
        parts = []
        for i, rows in enumerate(plan.suffix_rows):
            branch = [torch.cat((t[:, :plan.prefix_length], t[:, rows]), dim=1) for t in tensors]
            out = function(*branch)
            if i == 0:
                parts.append(out[:, :plan.prefix_length])
            parts.append(out[:, plan.prefix_length:])
        return torch.cat(parts, dim=1)

    projections, projected = {}, {}
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        module = getattr(mixer, name)
        shared = module(normed)
        serial = branchwise(module, normed)
        projections[name] = compare_operator_rows(serial, shared, plan)
        projected[name] = shared

    qkv = projected["in_proj_qkv"]
    def conv(rows):
        sequence_kwargs = {name: cu(rows.shape[1]) if name in ("cu_seq_lens_q", "cu_seq_lens_k")
                           else rows.shape[1] for name in native_sequence_kwargs}
        return native_conv(rows.transpose(1, 2), mixer.conv1d.weight.squeeze(1), mixer.conv1d.bias,
                           activation=mixer.activation, **sequence_kwargs).transpose(1, 2)
    serial_conv = branchwise(conv, qkv)
    shared_conv = plan.causal_conv1d(qkv, mixer.conv1d, mixer.activation)
    convolution = compare_operator_rows(serial_conv, shared_conv, plan)
    # Every GDN variant below consumes exactly this canonical native-convolution
    # output; projection/conv discrepancies cannot leak into the GDN comparison.
    q, k, v = torch.split(serial_conv, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    q = q.reshape(1, plan.compact_length, -1, mixer.head_k_dim)
    k = k.reshape(1, plan.compact_length, -1, mixer.head_k_dim)
    v = v.reshape(1, plan.compact_length, -1, mixer.head_v_dim)
    groups = mixer.num_v_heads // mixer.num_k_heads
    q, k = [t.repeat_interleave(groups, dim=2) for t in (q, k)]
    beta = projected["in_proj_b"].sigmoid()
    g = -mixer.A_log.float().exp() * F.softplus(projected["in_proj_a"].float() + mixer.dt_bias)
    tensors = q, k, v, g, beta

    def full(*rows, varlen, final=False):
        return kernel(*rows, initial_state=None, output_final_state=final,
                      cu_seqlens=cu(rows[0].shape[1]) if varlen else None,
                      use_qk_l2norm_in_kernel=True)[0]

    native = branchwise(partial(full, varlen=native_cu is not None), *tensors)
    comparisons = {}
    def record(name, value, reference=native):
        comparisons[name] = compare_operator_rows(reference, value, plan)

    record("single_call_dense", branchwise(partial(full, varlen=False), *tensors))
    record("single_call_varlen", branchwise(partial(full, varlen=True), *tensors))
    record("single_call_native_export_state", branchwise(
        partial(full, varlen=native_cu is not None, final=True), *tensors))

    def prefix_state(explicit_cu):
        boundary = plan.gdn_boundary
        if boundary:
            return kernel(*(t[:, :boundary].contiguous() for t in tensors), initial_state=None,
                          output_final_state=True, cu_seqlens=cu(boundary) if explicit_cu else None,
                          use_qk_l2norm_in_kernel=True)
        return v[:, :0], torch.zeros((1, k.shape[2], k.shape[-1], v.shape[-1]),
                                     dtype=torch.float32, device=k.device)

    prefix, state = prefix_state(True)
    parts = [prefix]
    for i, rows in enumerate(plan.suffix_rows):
        branch = [torch.cat((t[:, plan.gdn_boundary:plan.prefix_length], t[:, rows]), dim=1).contiguous()
                  for t in tensors]
        tail, _ = kernel(*branch, initial_state=state, output_final_state=False,
                         cu_seqlens=cu(branch[0].shape[1]), use_qk_l2norm_in_kernel=True)
        if i == 0:
            parts.append(tail[:, :plan.prefix_length - plan.gdn_boundary])
        parts.append(tail[:, plan.prefix_length - plan.gdn_boundary:])
    split_serial = torch.cat(parts, dim=1)
    record("split_serial_explicit_cu", split_serial)
    current = plan.gated_delta_rule(kernel, *tensors)
    record("fork_current", current)
    initial = state.index_select(0, torch.zeros(len(plan.gdn_b_lengths), dtype=torch.long, device=state.device))
    tail, _ = kernel(*(t.index_select(1, plan.gdn_b_index).contiguous() for t in tensors),
                     initial_state=initial, output_final_state=False,
                     cu_seqlens=plan.gdn_b_cu, use_qk_l2norm_in_kernel=True)
    explicit = torch.cat((prefix, tail.index_select(1, plan.gdn_b_keep)), dim=1)
    record("fork_prefix_explicit_cu", explicit)
    record("fork_prefix_cu_vs_current", explicit, current)
    record("fork_packed_vs_split_serial", explicit, split_serial)
    return {"projections": projections, "convolution_same_qkv": convolution,
            "gdn_same_inputs": comparisons,
            "gdn_reference": "native single call per full branch; captured native cu presence",
            "gdn_input_source": "same shared projections; same native-convolution compact q/k/v/g/beta",
            "gdn_boundary": plan.gdn_boundary, "prefix_length": plan.prefix_length,
            "suffix_lengths": list(plan.suffix_lengths), "tail_cu_seqlens": plan.gdn_b_cu.cpu().tolist()}


def diagnose_operators(trainer, batches, *, gdn_backend="fla") -> dict:
    """One native forward captures real layer-zero inputs; all later work is local."""
    import torch
    from swift.trainers import disable_gradient_checkpointing
    from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn
    from tools.qwen35_shared_prefix import build_shared_prefix_plan, _resolve_gdn_kernel

    base = trainer.model.get_base_model()
    language = base.model.language_model
    layer = language.layers[0]
    if layer.block_type != "linear_attention":
        raise ValueError("算子阶梯限定 layer0 为线性注意力")
    positions = [batch["model_inputs"]["position_ids"] for batch in batches]
    packed = collate_shared_inputs(batches, positions, trainer.template.tokenizer.pad_token_id)
    plan = build_shared_prefix_plan(packed["input_ids"], packed["attention_mask"],
                                    packed["position_ids"], packed["prefix_length"])
    captured = {}
    def capture_kwargs(module, args, kwargs):
        captured["kwargs"] = {key: kwargs[key] for key in (
            "cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k") if key in kwargs}
    def capture_normed(module, args, output):
        captured["normed"] = output.detach()
    handles = [language.register_forward_pre_hook(capture_kwargs, with_kwargs=True),
               layer.input_layernorm.register_forward_hook(capture_normed)]
    with torch.no_grad(), disable_gradient_checkpointing(trainer.model, trainer.args.gradient_checkpointing_kwargs):
        try:
            with trainer.accelerator.autocast():
                first = batches[0]
                trainer._get_per_token_logps_and_entropies(trainer.model, first["model_inputs"], first["grpo_batch"])
        finally:
            for handle in handles:
                handle.remove()
        active = first["model_inputs"]["attention_mask"][0].bool()
        native_normed = captured["normed"][:, active]
        with trainer.accelerator.autocast():
            # At layer zero every answer token is its original text embedding.
            # Preserve native per-answer shapes while rebuilding the unobserved suffixes.
            suffixes = [layer.input_layernorm(language.embed_tokens(
                packed["input_ids"][i:i + 1, -n:])) for i, n in enumerate(plan.suffix_lengths)]
            reconstruction = VectorStatistics()
            reconstruction.add(native_normed[:, plan.prefix_length:], suffixes[0])
            if reconstruction.max_abs != 0:
                raise RuntimeError("首答 layer0 后缀嵌入恢复未逐值相等")
            normed = torch.cat((native_normed[:, :plan.prefix_length], *suffixes), dim=1)
            operators = diagnose_layer0_operators(layer.linear_attn, normed, plan,
                kernel=_resolve_gdn_kernel(gdn_backend), native_conv=causal_conv1d_fn,
                native_sequence_kwargs=captured["kwargs"])
    return {"scope": "fixed G8 layer0 no_grad operators; one native forward; no optimizer/update or acceptance",
            "input_comparison": compare_expanded_inputs(batches, positions, packed),
            "native_forward_calls": 1, "first_answer_suffix_reconstruction": reconstruction.report(),
            "native_sequence_metadata": {key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value
                                         for key, value in captured["kwargs"].items()},
            "branch_metadata": "captured native single-sequence metadata; lengths adapted per answer",
            "operators": operators}


def diagnose_forward(trainer, batches, *, gdn_backend: str = "fla", trace_layer: int | None = None) -> dict:
    """No-update probes: repeat noise, sampled layer boundaries, and native mb2.

    Timing includes trace copies and is diagnostic only. Native batch two is a
    shape control, not a proposed training configuration or an equivalence gate.
    """
    import torch
    from swift.trainers import disable_gradient_checkpointing
    from trl.trainer.utils import selective_log_softmax
    from tools.qwen35_shared_prefix import (
        build_shared_prefix_plan, native_sequence_mixer_forward, shared_prefix_forward,
    )

    model = trainer.model
    base = model.get_base_model()
    language = base.model.language_model
    if trace_layer is not None and not 0 <= trace_layer < len(language.layers):
        raise ValueError("trace-layer 超出实际语言层范围")
    positions = [batch["model_inputs"]["position_ids"] for batch in batches]
    packed = collate_shared_inputs(batches, positions, trainer.template.tokenizer.pad_token_id)
    input_report = compare_expanded_inputs(batches, positions, packed)
    plan = build_shared_prefix_plan(packed["input_ids"], packed["attention_mask"],
                                    packed["position_ids"], packed["prefix_length"])
    samples = diagnostic_token_positions(plan.prefix_length, plan.suffix_lengths[0])
    state = {"native": False, "branch": 0, "captures": {}, "native_length": 0,
             "native_left": 0, "native_batch": 1, "full_target": {},
             "capture_embeddings": False, "native_embeddings": {}, "probe_name": "",
             "native_call_metadata": {}, "native_sequence_kwargs": {}}
    handles = []

    def wanted(name):
        match = re.match(r"layers\.(\d+)\.(.+)", name)
        return (not match or match[2] in {"input", "output"}
                or trace_layer is not None and int(match[1]) == trace_layer)

    def store(name, tensor, original_positions, rows, *, seq_dim=1, batch_dim=0, batch_index=0):
        if not wanted(name):
            return
        state["captures"][name] = {
            "positions": original_positions,
            "values": sample_trace_rows(tensor, rows, seq_dim=seq_dim, batch_dim=batch_dim, batch_index=batch_index),
            "source_shape": list(tensor.shape), "source_dtype": str(tensor.dtype)}

    def native_store(name, tensor, *, seq_dim=1, batch_dim=0, tail=False):
        if not state["native"] or state["branch"] != 0 or not wanted(name):
            return
        offset = state["native_length"] - tensor.shape[seq_dim] if tail else 0
        selected = [(position, position + state["native_left"] - offset) for position in samples
                    if 0 <= position + state["native_left"] - offset < tensor.shape[seq_dim]]
        store(name, tensor, [p for p, _ in selected], [row for _, row in selected],
              seq_dim=seq_dim, batch_dim=batch_dim)
        if name == "layers.0.input":
            state["captures"]["compact.input_embeds"] = state["captures"][name]

    def before(name, module, args, kwargs):
        tensor = args[0] if args else kwargs["hidden_states"]
        if (name == "layers.0.input" and state["capture_embeddings"]
                and state["native"] and state["branch"] < 2):
            state["native_embeddings"][state["branch"]] = tensor.detach().clone()
        native_store(name, tensor)

    def after(name, module, args, output):
        native_store(name, output[0] if isinstance(output, tuple) else output)

    def language_before(module, args, kwargs):
        sequence_names = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")
        if state["native"] and state["capture_embeddings"]:
            state["native_sequence_kwargs"][state["branch"]] = {
                name: kwargs[name] for name in sequence_names if name in kwargs}
        if state["native"] and state["branch"] == 0:
            extra = {}
            for name in (*sequence_names, "use_cache"):
                if name in kwargs:
                    value = kwargs[name]
                    extra[name] = value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
            state["native_call_metadata"][state["probe_name"]] = {
                "argument_keys": sorted(kwargs), "sequence_metadata": extra}
        position_ids = kwargs["position_ids"]
        native_store("compact.position_ids", position_ids[1:] if position_ids.shape[0] == 4 else position_ids,
                     seq_dim=2, batch_dim=1)

    def rope_after(module, args, output):
        for name, tensor in zip(("rope.cos", "rope.sin"), output):
            native_store(name, tensor)

    def selected_before(module, args):
        native_store("final.selected", args[0], tail=True)

    def logits_after(module, args, output):
        native_store("final.logits", output, tail=True)

    def gdn_core_before(name, module, args):
        if state["native"]:
            native_store(name, args[0].reshape(state["native_batch"], state["native_length"], -1))

    def gdn_norm_after(name, module, args, output):
        if state["native"]:
            native_store(name, output.reshape(state["native_batch"], state["native_length"], -1))

    def gdn_projection_after(name, mixer, module, args, output):
        native_store(name, output)
        if name.endswith(".in_proj_b"):
            native_store(name.rsplit(".", 1)[0] + ".beta", output.sigmoid())
        elif name.endswith(".in_proj_a"):
            g = -mixer.A_log.float().exp() * torch.nn.functional.softplus(output.float() + mixer.dt_bias)
            native_store(name.rsplit(".", 1)[0] + ".g", g)

    handles.append(language.register_forward_pre_hook(language_before, with_kwargs=True))
    handles.append(language.rotary_emb.register_forward_hook(rope_after))
    handles.append(language.norm.register_forward_hook(partial(after, "final.norm")))
    handles.append(base.lm_head.register_forward_pre_hook(selected_before))
    handles.append(base.lm_head.register_forward_hook(logits_after))
    for index, layer in enumerate(language.layers):
        prefix = f"layers.{index}"
        handles.append(layer.input_layernorm.register_forward_pre_hook(partial(before, prefix + ".input"), with_kwargs=True))
        handles.append(layer.register_forward_hook(partial(after, prefix + ".output")))
        if trace_layer == index:
            handles.append(layer.input_layernorm.register_forward_hook(partial(after, prefix + ".input_layernorm")))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
                partial(before, prefix + ".after_mixer"), with_kwargs=True))
            handles.append(layer.post_attention_layernorm.register_forward_hook(
                partial(after, prefix + ".post_attention_layernorm")))
            handles.append(layer.mlp.register_forward_hook(partial(after, prefix + ".mlp")))
            mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
            handles.append(mixer.register_forward_hook(partial(after, prefix + ".mixer")))
            kind = ".linear_attn" if layer.block_type == "linear_attention" else ".self_attn"
            for name in (["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
                         if layer.block_type == "linear_attention" else ["q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"]):
                module = getattr(mixer, name)
                callback = (partial(gdn_projection_after, prefix + kind + "." + name, mixer)
                            if layer.block_type == "linear_attention" else partial(after, prefix + kind + "." + name))
                handles.append(module.register_forward_hook(callback))
            if layer.block_type == "linear_attention":
                handles.append(mixer.norm.register_forward_pre_hook(partial(gdn_core_before, prefix + kind + ".core")))
                handles.append(mixer.norm.register_forward_hook(partial(gdn_norm_after, prefix + kind + ".norm")))

    def shared_trace(name, tensor, metadata):
        if trace_layer is not None and (name in {"rope.cos", "rope.sin", f"layers.{trace_layer}.input_layernorm",
                                               f"layers.{trace_layer}.mixer"}):
            state["full_target"][name] = tensor.detach()
        if not wanted(name):
            return
        if metadata["layout"] == "expanded":
            offset = metadata["sequence_offset"]
            selected = [(position, plan.starts[0] + position - offset) for position in samples
                        if 0 <= plan.starts[0] + position - offset < tensor.shape[metadata["seq_dim"]]]
            rows = [row for _, row in selected]
            original = [position for position, _ in selected]
        else:
            rows = plan.expand_index[0, [plan.starts[0] + position for position in samples]].tolist()
            original = samples
        store(name, tensor, original, rows, seq_dim=metadata["seq_dim"],
              batch_dim=1 if name == "compact.position_ids" else 0)

    def native_serial():
        rows = []
        state["native"], state["native_batch"] = True, 1
        for index, batch in enumerate(batches):
            state["branch"] = index
            inputs = batch["model_inputs"]
            state["native_length"] = inputs["input_ids"].shape[1]
            state["native_left"] = int(inputs["attention_mask"][0].nonzero()[0])
            values = trainer._get_per_token_logps_and_entropies(model, inputs, batch["grpo_batch"])[0]
            rows.append(values[batch["grpo_batch"].completion_mask.bool()].float().cpu())
        state["native"] = False
        return rows

    def native_batch_two():
        state["native"], state["branch"], state["native_batch"] = True, 0, 2
        length = max(int(batch["model_inputs"]["attention_mask"].sum()) for batch in batches[:2])
        first = state["native_embeddings"][0]
        embeddings = first.new_zeros((2, length, first.shape[-1]))
        for index, batch in enumerate(batches[:2]):
            mask = batch["model_inputs"]["attention_mask"][0].bool()
            original = state["native_embeddings"][index][0, mask]
            embeddings[index, -len(original):] = original
        mask = packed["attention_mask"][:2, -length:]
        state["native_length"], state["native_left"] = length, int(mask[0].nonzero()[0])
        # Exact serial embeddings remove visual batching as a cause. Native mb1
        # may carry varlen metadata absent here: report this as an execution-path
        # control, rather than attributing all differences to GEMM batch shape.
        output = language(inputs_embeds=embeddings, attention_mask=mask,
                          position_ids=packed["position_ids"][:, :2, -length:], use_cache=False)
        width = max(int(batch["grpo_batch"].logits_to_keep) for batch in batches[:2])
        logits = base.lm_head(output.last_hidden_state[:, -(width + 1):]).float()
        values = answer_logps(logits, packed["input_ids"][:2, -length:], width,
                              trainer.temperature, selective_log_softmax)
        state["native"] = False
        return [values[index, -int(batch["grpo_batch"].logits_to_keep):][batch["grpo_batch"].completion_mask[0].bool()].float().cpu()
                for index, batch in enumerate(batches[:2])]

    def shared(attention):
        state["native"] = False
        output = shared_prefix_forward(
            model, plan=plan, input_ids=packed["input_ids"], attention_mask=packed["attention_mask"],
            position_ids=packed["position_ids"], pixel_values=packed["pixel_values"],
            image_grid_thw=packed["image_grid_thw"], logits_to_keep=packed["completion_width"] + 1,
            checkpoint_layers=False, gdn_backend=gdn_backend, trace=shared_trace,
            diagnostic_attention=attention)
        values = answer_logps(output.logits.float(), packed["input_ids"], packed["completion_width"],
                              trainer.temperature, selective_log_softmax)
        return [values[index, -int(batch["grpo_batch"].logits_to_keep):][batch["grpo_batch"].completion_mask[0].bool()].float().cpu()
                for index, batch in enumerate(batches)]

    probes, snapshots, logps = {}, {}, {}
    operations = [("native_serial_1", native_serial), ("native_serial_2", native_serial),
                  ("shared_1", partial(shared, "current")), ("shared_2", partial(shared, "current")),
                  ("native_batch2", native_batch_two), ("shared_native_gqa", partial(shared, "native_gqa"))]
    try:
        with torch.no_grad(), disable_gradient_checkpointing(model, trainer.args.gradient_checkpointing_kwargs):
            for name, operation in operations:
                state["captures"], state["full_target"] = {}, {}
                state["probe_name"] = name
                state["capture_embeddings"] = name == "native_serial_1"
                torch.cuda.synchronize()
                started = time.perf_counter()
                with trainer.accelerator.autocast():
                    rows = operation()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                captures = state["captures"]
                reference = name if name == "native_serial_1" else ("shared_1" if name == "shared_2" else "native_serial_1")
                if name in {"native_serial_1", "shared_1"}:
                    snapshots[name], logps[name] = captures, rows
                trace = compare_trace_records(snapshots[reference], captures)
                stats = VectorStatistics()
                for left, right in zip(logps[reference], rows):
                    stats.add(left, right)
                probes[name] = {"reference": reference, "diagnostic_wall_seconds_including_trace_copies": elapsed,
                                "logp_comparison": stats.report(), "trace_comparison": trace,
                                "effective_logps": [row.tolist() for row in rows]}
                if name == "native_batch2":
                    probes[name]["input_embeddings"] = "exact native serial visual/text embeddings"
                    probes[name]["comparison_scope"] = "two-sample language/head execution path; batch shape and possible varlen changes"
                if trace_layer is not None and name == "shared_1":
                    full = state["full_target"]
                    with trainer.accelerator.autocast():
                        replay = native_sequence_mixer_forward(
                            language.layers[trace_layer], plan=plan,
                            hidden_states=full[f"layers.{trace_layer}.input_layernorm"],
                            position_embeddings=(full["rope.cos"], full["rope.sin"]),
                            native_sequence_kwargs=[state["native_sequence_kwargs"][index] for index in range(len(batches))])
                    mixer_stats = VectorStatistics()
                    mixer_stats.add(replay, full[f"layers.{trace_layer}.mixer"])
                    probes[name]["same_input_native_mixer_replay"] = {
                        "layer": trace_layer, "block_type": language.layers[trace_layer].block_type,
                        "comparison": mixer_stats.report(), "scope": "all compact rows; native mb1 sequence kernels",
                        "sequence_metadata": "each answer's captured native cu_seq_lens/max_length kwargs"}
    finally:
        for handle in handles:
            handle.remove()
    noise = probes["native_serial_2"]["trace_comparison"]["events"]
    for name in ("shared_1", "native_batch2", "shared_native_gqa"):
        observed = probes[name]["trace_comparison"]["events"]
        above_noise = [event for event, metric in observed.items()
                       if metric["max_abs"] > noise[event]["max_abs"]]
        probes[name]["trace_comparison"]["first_above_native_repeat_noise_event"] = above_noise[0] if above_noise else None
    return {"scope": "fixed G8, no_grad forward diagnosis; no optimizer/update or equivalence acceptance",
            "trace_branch": 0, "sampled_unpadded_token_positions": samples,
            "trace_limit": "only sampled rows; first observed divergence does not prove all earlier rows equal",
            "trace_layer": trace_layer, "input_comparison": input_report,
            "prefix_length": plan.prefix_length, "suffix_lengths": list(plan.suffix_lengths),
            "native_call_metadata": state["native_call_metadata"],
            "batch_shape_control_note": (
                "native_batch2 reuses exact serial embeddings; original mb1 may also carry cu_seq_lens/max_length "
                "metadata. Inspect native_call_metadata before attributing differences only to matrix batch size."),
            "probes": probes}


@contextmanager
def _replace(instance, name, value):
    original = getattr(instance, name)
    setattr(instance, name, value)
    try:
        yield original
    finally:
        setattr(instance, name, original)


class _BenchmarkComplete(Exception):
    pass


def run(bundle: Path, config_path: Path, adapter: Path, output: Path, *, groups: int = 1,
        model: str | None = None, precision: str | None = None, repeats: int = 1,
        deterministic: bool = True, gdn_backend: str = "fla", forward_diagnostics: bool = False,
        trace_layer: int | None = None, operator_diagnostics: bool = False) -> dict:
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from swift.trainers import disable_gradient_checkpointing
    from trl.trainer.utils import selective_log_softmax
    from tools.quant_grounding_config import load_config
    from tools.swift_quant_grounding_reward import SHORT_RESPONSE_PREFIX
    from tools import train_quant_grounding as training
    from tools.qwen35_shared_prefix import build_shared_prefix_plan, shared_prefix_forward

    if groups not in (1, 4) or repeats < 1:
        raise ValueError("groups 只允许 1/4，repeats 必须为正")
    diagnostics = forward_diagnostics or operator_diagnostics
    if diagnostics and groups != 1:
        raise ValueError("前向诊断只运行一个固定 G8 题组")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest["selected"][:groups]
    if len(selected) != groups:
        raise ValueError("固定回答 fixture 不足指定题组数")
    if not any(row["mixed_reward"] for row in selected):
        raise ValueError("所选题组奖励全部相同，没有有效策略梯度对照")
    output.mkdir(parents=True, exist_ok=True)
    if list(output.iterdir()):
        raise ValueError("结果目录必须为空，避免覆盖已有实验")
    config = load_config(config_path, model=model, precision=precision)
    if config["arm"] != "A":
        raise ValueError("该固定回答工具限定当前 A 臂短回答/原奖励")
    config.update({"output_dir": str((output / "swift_load").resolve()), "adapters": str(adapter.resolve()),
                   "train_micro_batch_size": 1, "gradient_checkpointing_every_n_layers": 1,
                   "runtime_optimization": False, "benchmark_mode": True,
                   "deterministic_training": deterministic, "save_steps": 1000000, "save_hours": []})
    for key in ("matched_query_exposures", "max_gpu_hours", "resume_from_checkpoint"):
        config.pop(key, None)
    # This benchmark owns both implementations, even when a production config enables the new integration.
    config["shared_prefix_training"] = False
    config["shared_prefix"] = False
    config["num_generations"] = 8
    config["geometry_reward"] = "capped_iou"
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["QUANT_REWARD_METRICS_JSONL"] = str(output / "reward_metrics.jsonl")
    rows_by_id = {row["id"]: row for row in _jsonl(Path(manifest["dataset"]))}
    dataset = output / "selected_dataset.jsonl"
    dataset.write_text("".join(json.dumps(rows_by_id[row["id"]], ensure_ascii=False) + "\n"
                               for row in selected), encoding="utf-8")
    report = {"groups": groups, "answer_count": groups * 8, "repeats": repeats,
              "adapter": str(adapter.resolve()), "precision": config["precision"],
              "deterministic": deterministic, "gdn_backend": gdn_backend,
              "targets": ("fixed existing answers; no loss targets" if diagnostics else
                          "each path's native old/reference; no baseline target replay"),
              "scope": ("fixed-answer no_grad forward diagnosis; no optimizer/update" if diagnostics else
                        "fixed-answer GRPO update; online sampling excluded"),
              "full_original_update": groups == 4, "runs": []}

    def benchmark(trainer):
        model_object = trainer.model
        if trainer.template.padding_free or trainer.template.sequence_parallel_size != 1:
            raise ValueError("基准需要普通填充及单卡非序列并行")
        if trainer.loss_type != "dapo" or trainer.kl_in_reward or trainer.compute_entropy:
            raise ValueError("基准需要原 DAPO、KL 在 loss 内且不计算熵")
        if trainer.args.gradient_accumulation_steps != 32:
            raise ValueError("原始 mb1 必须保持 32 次梯度累积归一化")
        trainable = {name: parameter for name, parameter in model_object.named_parameters() if parameter.requires_grad}
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("基准要求仅 LoRA 参数可训练")
        initial = ({} if diagnostics else
                   {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()})
        language = [(name, module) for name, module in model_object.named_modules()
                    if name.endswith(".language_model")]
        if len(language) != 1:
            raise ValueError(f"无法唯一找到 Qwen3.5 language_model: {[name for name, _ in language]}")
        language = language[0][1]
        # The pipeline hook runs before SwiftMixin.train. Install exactly the
        # multimodal preprocessing and checkpoint preparation that train would do.
        trainer.template.register_post_encode_hook([trainer.accelerator.unwrap_model(model_object)])
        trainer._prepare_gradient_checkpointing(
            trainer.template.get_base_model(trainer.accelerator.unwrap_model(model_object)))
        if not all(layer.gradient_checkpointing for layer in language.layers):
            raise RuntimeError("原始路径必须启用全部语言层激活重计算")
        prepared_model = trainer.accelerator.prepare_model(model_object)
        if prepared_model is not model_object or getattr(trainer.accelerator, "scaler", None) is not None:
            raise RuntimeError("单卡 BF16 基准需要同一 PEFT 模型且无 GradScaler")
        parameters_after_prepare = dict(prepared_model.named_parameters())
        if any(parameters_after_prepare[name] is not parameter for name, parameter in trainable.items()):
            raise RuntimeError("Accelerate 准备前后 LoRA 参数对象发生变化")
        trainer.model = trainer.model_wrapped = prepared_model
        report["model_preparation"] = {"same_peft_object": True, "same_trainable_parameter_objects": True,
                                       "trainable_parameter_count": len(trainable),
                                       "reference_adapter": trainer.ref_adapter_name}
        current_phase, phase_counts = ["idle"], defaultdict(lambda: defaultdict(int))
        output_dtypes = defaultdict(set)
        handles = []

        def observe_output_dtype(module, args, output):
            if hasattr(output, "logits"):
                output_dtypes[current_phase[0]].add(str(output.logits.dtype))

        handles.append(model_object.register_forward_hook(observe_output_dtype))

        def count_forward(name):
            def hook(module, args, kwargs):
                values = args[0] if args else kwargs.get("hidden_states", kwargs.get("pixel_values"))
                counts = phase_counts[current_phase[0]]
                counts[f"{name}.calls"] += 1
                if values is not None:
                    counts[f"{name}.input_rows"] += math.prod(values.shape[:-1])
            return hook

        for name, module in model_object.named_modules():
            # Hooks count real projection work, including checkpoint recomputation.
            if (name.endswith(".visual") or name.endswith(".mlp.gate_proj")
                    or name.endswith(".self_attn.q_proj") or name.endswith(".linear_attn.in_proj_qkv")
                    or name.endswith("lm_head")):
                kind = "visual" if name.endswith(".visual") else ".".join(name.split(".")[-2:])
                handles.append(module.register_forward_pre_hook(count_forward(kind), with_kwargs=True))

        fixed_index = [0]

        def fixed_generate(samples):
            samples = trainer._preprocess_inputs(samples)
            fixed = selected[fixed_index[0]]
            if len(samples) != 8 or {sample.extra["id"] for sample in samples} != {fixed["id"]}:
                raise ValueError("固定 rollout 不是指定单题 G8")
            for sample, answer in zip(samples, fixed["responses"]):
                ids = trainer.template.tokenizer.encode(answer["text"], add_special_tokens=False)
                decoded = trainer.template.decode_generate_ids(
                    ids, first_token=False, template_inputs=SimpleNamespace(
                        chat_template_kwargs=sample.extra.get("chat_template_kwargs") or {}))
                if decoded != answer["text"]:
                    raise ValueError("固定回答 tokenization 无法还原原文")
                sample.messages.append({"role": "assistant", "content": SHORT_RESPONSE_PREFIX + answer["text"]})
                sample.response_token_ids = ids
                sample.finish_reason = answer["finish_reason"]
                sample.add_eos = False
            model_object.train()
            return samples

        def placeholders(model, inputs, signal, *args, **kwargs):
            return torch.zeros_like(signal.completion_mask, dtype=torch.float32), None

        def timed(phase, function):
            current_phase[0] = phase
            torch.cuda.synchronize()
            started = time.perf_counter()
            with trainer.accelerator.autocast():
                result = function()
            torch.cuda.synchronize()
            return result, time.perf_counter() - started

        # Encoding and reward calculation run through Swift. No model work is hidden in this stage.
        def encode_groups():
            encoded = []
            with _replace(trainer, "_generate_completions", fixed_generate), _replace(
                    trainer, "_get_per_token_logps_and_entropies", placeholders):
                for index, fixed in enumerate(selected):
                    fixed_index[0] = index
                    batches = trainer._generate_and_score_completions(
                        [copy.deepcopy(rows_by_id[fixed["id"]]) for _ in range(8)])
                    rewards = trainer._rewards_per_func[:, 0].float().cpu().tolist()
                    if len(batches) != 8 or any(not math.isclose(a, b, abs_tol=1e-6)
                                               for a, b in zip(rewards, fixed["rewards"])):
                        raise RuntimeError("Swift 编码/奖励与固定 G8 不一致")
                    encoded.append(batches)
            return encoded

        try:
            if diagnostics:
                encoded, _ = timed("diagnostic_encode", encode_groups)
                if operator_diagnostics:
                    report["operator_diagnostics"] = diagnose_operators(trainer, encoded[0], gdn_backend=gdn_backend)
                else:
                    report["forward_diagnostics"] = diagnose_forward(
                        trainer, encoded[0], gdn_backend=gdn_backend, trace_layer=trace_layer)
                (output / ("operator_diagnostic.json" if operator_diagnostics else "diagnostic.json")).write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                raise _BenchmarkComplete()
            for repeat in range(repeats):
                reference_gradients, reference_updates, baseline_signals = {}, {}, []
                packed_groups, position_groups, input_reports, plan_reports = [], [], [], {}
                for variant in ("baseline", "shared"):
                    model_object.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for name, parameter in trainable.items():
                            parameter.copy_(initial[name].to(parameter.device))
                    trainer.optimizer = None
                    trainer.create_optimizer()
                    optimizer = trainer.optimizer
                    torch.manual_seed(int(config.get("seed", 2026)))
                    torch.cuda.manual_seed_all(int(config.get("seed", 2026)))
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    free_memory, total_memory = torch.cuda.mem_get_info()
                    non_pytorch_bytes = total_memory - free_memory - torch.cuda.memory_reserved()
                    phase_counts.clear()
                    output_dtypes.clear()
                    measurements = {}
                    variant_started = time.perf_counter()
                    groups_encoded, measurements["fixed_rollout_reward_encode_seconds"] = timed(
                        "encode", encode_groups)
                    signal_rows, losses_cpu, ratios = [], [], []
                    native_forward = trainer._get_per_token_logps_and_entropies

                    for group_index, batches in enumerate(groups_encoded):
                        positions = []
                        if variant == "baseline":
                            def capture_position(module, args, kwargs):
                                positions.append(kwargs["position_ids"].detach().clone())
                            capture = language.register_forward_pre_hook(capture_position, with_kwargs=True)
                        else:
                            capture = None
                            candidate_positions = [batch["model_inputs"]["position_ids"] for batch in batches]
                            packed, seconds = timed("shared_input_assembly", lambda: collate_shared_inputs(
                                batches, candidate_positions, trainer.template.tokenizer.pad_token_id))
                            measurements["shared_input_assembly_seconds"] = measurements.get(
                                "shared_input_assembly_seconds", 0) + seconds
                            input_reports[group_index]["candidate_encoded_inputs_equal"] = compare_expanded_inputs(
                                batches, candidate_positions, packed_groups[group_index])["all_equal"]
                            packed_groups[group_index] = packed

                        def shared_logps(grad, packed_groups=packed_groups):
                            packed = packed_groups[group_index]
                            plan = build_shared_prefix_plan(
                                packed["input_ids"], packed["attention_mask"], packed["position_ids"],
                                packed["prefix_length"])
                            plan_reports[group_index] = {
                                "compact_length": plan.compact_length, "total_tokens": plan.total_tokens,
                                "prefix_length": plan.prefix_length, "suffix_lengths": list(plan.suffix_lengths),
                                "gdn_boundary": plan.gdn_boundary, "gdn_replay_tokens": plan.gdn_replay_tokens}
                            output_logits = shared_prefix_forward(
                                model_object, plan=plan, input_ids=packed["input_ids"],
                                attention_mask=packed["attention_mask"], position_ids=packed["position_ids"],
                                pixel_values=packed["pixel_values"], image_grid_thw=packed["image_grid_thw"],
                                logits_to_keep=packed["completion_width"] + 1,
                                checkpoint_layers=grad, gdn_backend=gdn_backend).logits
                            # Accelerate wraps native model.forward with convert_outputs_to_fp32;
                            # the functional shared call bypasses that wrapper.
                            output_logits = output_logits.float()
                            output_dtypes[current_phase[0]].add(str(output_logits.dtype))
                            return answer_logps(output_logits, packed["input_ids"], packed["completion_width"],
                                                trainer.temperature, selective_log_softmax)

                        def fill_native_targets(reference=False):
                            active_before = copy.deepcopy(model_object.active_adapter)
                            with torch.no_grad(), disable_gradient_checkpointing(
                                    model_object, trainer.args.gradient_checkpointing_kwargs):
                                context = trainer.null_ref_context() if reference else nullcontext()
                                with context:
                                    if reference and trainer.ref_adapter_name not in model_object.active_adapters:
                                        raise RuntimeError("参考前向没有激活原 ref_adapter")
                                    if variant == "baseline":
                                        result = [native_forward(model_object, batch["model_inputs"],
                                                                 batch["grpo_batch"])[0] for batch in batches]
                                    else:
                                        values = shared_logps(False)
                                        result = [values[index:index + 1, -int(batch["grpo_batch"].logits_to_keep):]
                                                  for index, batch in enumerate(batches)]
                            if model_object.active_adapter != active_before or any(
                                    not parameter.requires_grad for parameter in trainable.values()):
                                raise RuntimeError("参考适配器上下文没有恢复原训练 LoRA")
                            key = "ref_per_token_logps" if reference else "old_per_token_logps"
                            for batch, values in zip(batches, result):
                                setattr(batch["grpo_batch"], key, values.detach())

                        _, seconds = timed("old", lambda: fill_native_targets(False))
                        measurements["old_forward_seconds"] = measurements.get("old_forward_seconds", 0) + seconds
                        if capture is not None:
                            capture.remove()
                            if len(positions) != 8:
                                raise RuntimeError("没有捕获完整八答实际位置编码")
                            position_groups.append(positions)
                            packed = collate_shared_inputs(batches, positions, trainer.template.tokenizer.pad_token_id)
                            input_reports.append(compare_expanded_inputs(batches, positions, packed))
                            packed_groups.append(packed)
                        _, seconds = timed("reference", lambda: fill_native_targets(True))
                        measurements["reference_forward_seconds"] = measurements.get("reference_forward_seconds", 0) + seconds

                        def loss_for(batch, logps):
                            def injected(*args, **kwargs):
                                return logps, None
                            with _replace(trainer, "_get_per_token_logps_and_entropies", injected):
                                return trainer.compute_loss(model_object, batch)

                        def current_forward_backward():
                            current_rows, losses = [], []
                            if variant == "baseline":
                                for batch in batches:
                                    values = native_forward(model_object, batch["model_inputs"], batch["grpo_batch"])[0]
                                    current_rows.append(values.detach())
                                    loss = loss_for(batch, values)
                                    losses_cpu.append(float(loss.detach()))
                                    (loss / trainer.args.gradient_accumulation_steps).backward()
                            else:
                                values = shared_logps(True)
                                for index, batch in enumerate(batches):
                                    row = values[index:index + 1, -int(batch["grpo_batch"].logits_to_keep):]
                                    current_rows.append(row.detach())
                                    losses.append(loss_for(batch, row))
                                losses_cpu.extend(float(loss.detach()) for loss in losses)
                                normalized_group_loss(losses, trainer.args.gradient_accumulation_steps).backward()
                            return current_rows

                        current_rows, seconds = timed("current_forward_backward", current_forward_backward)
                        measurements["current_forward_backward_seconds"] = measurements.get(
                            "current_forward_backward_seconds", 0) + seconds
                        for index, (batch, current) in enumerate(zip(batches, current_rows)):
                            signal = batch["grpo_batch"]
                            mask = signal.completion_mask.bool()
                            old, ref = signal.old_per_token_logps[mask], signal.ref_per_token_logps[mask]
                            current = current[mask]
                            ratio = (current - old).exp()
                            ratios.extend(ratio.float().cpu().tolist())
                            signal_rows.append({"group": group_index, "response": index,
                                "current": current.float().cpu().tolist(), "old": old.float().cpu().tolist(),
                                "reference": ref.float().cpu().tolist(),
                                "advantage": signal.advantages[mask].float().cpu().tolist(),
                                "completion_ids": batch["model_inputs"]["input_ids"][:, -int(signal.logits_to_keep):][mask].cpu().tolist(),
                                "completion_mask": signal.completion_mask.cpu().tolist(),
                                "num_items_in_batch": float(signal.num_items_in_batch),
                                "truncated": signal.truncated_mask.cpu().tolist()})
                    torch.cuda.synchronize()
                    # CPU copies and pairwise statistics are deliberately outside training timings.
                    measurements["fixed_update_wall_before_diagnostics_seconds"] = time.perf_counter() - variant_started
                    diagnostic_started = time.perf_counter()
                    gradient_stats = VectorStatistics()
                    for name, parameter in trainable.items():
                        if parameter.grad is None:
                            raise RuntimeError(f"LoRA 参数缺少梯度: {name}")
                        gradient = parameter.grad.detach().cpu()
                        if variant == "baseline":
                            reference_gradients[name] = gradient.clone()
                        gradient_stats.add(reference_gradients[name], gradient)
                    if not gradient_stats.finite or gradient_stats.candidate_squared == 0:
                        raise RuntimeError("该次累计 LoRA 梯度非有限或全部为零")
                    measurements["gradient_cpu_diagnostic_seconds"] = time.perf_counter() - diagnostic_started

                    def clip_and_update(optimizer=optimizer):
                        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), trainer.args.max_grad_norm)
                        optimizer.step()
                        return norm
                    preclip, measurements["clip_and_optimizer_seconds"] = timed("optimizer", clip_and_update)
                    update_stats = VectorStatistics()
                    diagnostic_started = time.perf_counter()
                    for name, parameter in trainable.items():
                        delta = parameter.detach().cpu().float() - initial[name].float()
                        if variant == "baseline":
                            reference_updates[name] = delta.clone()
                        update_stats.add(reference_updates[name], delta)
                    if not update_stats.finite or update_stats.candidate_squared == 0:
                        raise RuntimeError("该次 LoRA 参数更新非有限或全部为零")
                    measurements["update_cpu_diagnostic_seconds"] = time.perf_counter() - diagnostic_started
                    if variant == "baseline":
                        baseline_signals = signal_rows
                    logp_stats = {}
                    for key in ("current", "old", "reference", "advantage"):
                        stat = VectorStatistics()
                        for left, right in zip(baseline_signals, signal_rows):
                            if (left["completion_ids"] != right["completion_ids"]
                                    or left["num_items_in_batch"] != right["num_items_in_batch"]
                                    or left["completion_mask"] != right["completion_mask"]
                                    or left["truncated"] != right["truncated"]):
                                raise RuntimeError("路径之间回答词元、掩码、截断或 DAPO 分母不同")
                            stat.add(torch.tensor(left[key]), torch.tensor(right[key]))
                        logp_stats[key] = stat.report()
                    measured_total = sum(measurements[key] for key in (
                        "fixed_rollout_reward_encode_seconds", "old_forward_seconds", "reference_forward_seconds",
                        "current_forward_backward_seconds", "clip_and_optimizer_seconds")) + measurements.get(
                            "shared_input_assembly_seconds", 0)
                    if any(output_dtypes[phase] != {"torch.float32"}
                           for phase in ("old", "reference", "current_forward_backward")):
                        raise RuntimeError(f"概率计算前 logits 精度未对齐 FP32: {dict(output_dtypes)}")
                    result = {"repeat": repeat, "variant": variant, "timing": measurements,
                        "fixed_update_measured_seconds": measured_total,
                        "loss_effective": sum(losses_cpu) / trainer.args.gradient_accumulation_steps,
                        "loss_microbatch_values": losses_cpu, "gradient_accumulation_steps": 32,
                        "preclip_gradient_norm": float(preclip), "gradient_comparison": gradient_stats.report(),
                        "update_comparison": update_stats.report(), "logp_comparison": logp_stats,
                        "ratio": {"minimum": min(ratios), "maximum": max(ratios),
                                  "outside_clip_fraction": sum(x < 1 - trainer.epsilon_low or x > 1 + trainer.epsilon_high
                                                               for x in ratios) / len(ratios)},
                        "memory": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                   "device_total_bytes": total_memory,
                                   "non_pytorch_device_bytes_at_start": non_pytorch_bytes},
                        "structure_counts": {phase: dict(counts) for phase, counts in phase_counts.items()},
                        "logits_dtype_before_probability": {phase: sorted(values)
                                                            for phase, values in output_dtypes.items()},
                        "signals": signal_rows, "input_comparison": copy.deepcopy(input_reports),
                        "shared_plan": copy.deepcopy(plan_reports)}
                    if variant == "shared":
                        baseline_result = report["runs"][-1]
                        result["speed_ratio_fixed_update"] = baseline_result["fixed_update_measured_seconds"] / measured_total
                        result["speed_ratio_current_forward_backward"] = baseline_result["timing"][
                            "current_forward_backward_seconds"] / measurements["current_forward_backward_seconds"]
                    report["runs"].append(result)
                    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    optimizer.state.clear()
                    trainer.optimizer = None
                    del optimizer, groups_encoded
                del reference_gradients, reference_updates, packed_groups, position_groups
            raise _BenchmarkComplete()
        finally:
            for handle in handles:
                handle.remove()
            trainer.template.remove_post_encode_hook()

    try:
        training.run_swift(config, dataset, max_steps=1, save_steps=1000000, trainer_hook=benchmark)
    except _BenchmarkComplete:
        pass
    if diagnostics:
        return report
    import statistics
    report["median_seconds"] = {
        variant: statistics.median(row["fixed_update_measured_seconds"] for row in report["runs"]
                                   if row["variant"] == variant) for variant in ("baseline", "shared")}
    report["median_speed_ratio_fixed_update"] = report["median_seconds"]["baseline"] / report["median_seconds"]["shared"]
    report["timing_note"] = (
        "Synchronized CUDA wall time, sum of encoding/reward, input assembly, old, reference, "
        "current forward/backward and clip/optimizer; excludes input-equality checks, CPU gradient/update "
        "diagnostics, model load, and fixed-answer sampling replacement. Repeats reset weights and optimizer.")
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--formal", required=True, type=Path)
    prepare_parser.add_argument("--data-root", required=True, type=Path)
    prepare_parser.add_argument("--answers", type=Path)
    prepare_parser.add_argument("--raw", type=Path)
    prepare_parser.add_argument("--scored", type=Path)
    prepare_parser.add_argument("--output", required=True, type=Path)
    for command in ("run", "diagnose", "diagnose-operators"):
        run_parser = commands.add_parser(command)
        run_parser.add_argument("--bundle", required=True, type=Path)
        run_parser.add_argument("--config", required=True, type=Path)
        run_parser.add_argument("--adapter", required=True, type=Path)
        run_parser.add_argument("--output", required=True, type=Path)
        run_parser.add_argument("--model")
        run_parser.add_argument("--precision", choices=("nf4", "bf16"))
        run_parser.add_argument("--groups", type=int, choices=(1, 4) if command == "run" else (1,), default=1)
        run_parser.add_argument("--repeats", type=int, default=1)
        run_parser.add_argument("--allow-nondeterminism", action="store_true")
        run_parser.add_argument("--gdn-backend", choices=("fla", "torch_reference"), default="fla")
        if command == "diagnose":
            run_parser.add_argument("--trace-layer", type=int, help="增加指定层的算子记录及同输入原生混合器重放")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.formal, args.data_root, args.output,
                         answers=args.answers, raw=args.raw, scored=args.scored)
    else:
        result = run(args.bundle, args.config, args.adapter, args.output, groups=args.groups,
                     model=args.model, precision=args.precision, repeats=args.repeats,
                     deterministic=not args.allow_nondeterminism, gdn_backend=args.gdn_backend,
                     forward_diagnostics=args.command == "diagnose", trace_layer=getattr(args, "trace_layer", None),
                     operator_diagnostics=args.command == "diagnose-operators")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
