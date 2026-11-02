"""独立 GPU 冒烟：真实 G2、四分流、视觉缓存等价性和原生检查点恢复。

synthetic fixture（合成夹具）只覆盖分流，不代表自然采样分布，也不计入正式 4400 题。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import traceback
import weakref
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from itertools import count
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_correction import (Completion, QuestionSampler, classify,
    random_state_from_json, serialize_gt)
from tools.quant_grounding_config import load_config, read_train_index
from tools.train_quant_grounding import (append_jsonl, process_question, restore_rng,
    rng_state, save_checkpoint, seed_all, synchronize, write_json)


def _progress(output, stage, **values):
    path = output / "progress.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    write_json(path, previous | values | {"stage": stage, "updated_unix_seconds": time.time()})


def _real_timing(records):
    return [{"id": record["id"], "image_count": record["image_count"], "seconds": record["seconds"],
             "generated_tokens": [len(item["token_ids"]) for item in record["completions"]]}
            for record in records]


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return deepcopy(value)


def _assert_exact(actual, expected, path="state"):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, msg=path)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_exact(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _assert_exact(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _parameter_snapshot(runtime, *, reference=False):
    result = {name: _cpu(parameter) for name, parameter in runtime.model.named_parameters()
              if (".reference." in name if reference else parameter.requires_grad)}
    assert result, "未找到需要检查的 adapter 参数"
    return result


def _assert_reference_unchanged(runtime, expected):
    named = dict(runtime.model.named_parameters())
    for name, value in expected.items():
        assert not named[name].requires_grad and named[name].grad is None, name
        _assert_exact(named[name].detach(), value, name)


def _parameter_changes(runtime, before):
    named = dict(runtime.model.named_parameters())
    maximum, changed = 0.0, 0
    for name, value in before.items():
        current = named[name].detach().float().cpu()
        assert torch.isfinite(current).all(), name
        delta = current - value.float()
        maximum = max(maximum, float(delta.abs().max()))
        changed += int(delta.count_nonzero())
    assert maximum > 0 and changed > 0, "完整更新后语言 LoRA 没有发生非零变化"
    return {"max_abs_change": maximum, "changed_values": changed}


def synthetic_completions(runtime, row, kind):
    """明确的正确/错误 JSON 词元夹具；不修改生成提示或训练损失。"""
    correct_text = serialize_gt(row)

    def completion(text):
        return Completion(text, [*runtime.tokenizer.encode(text, add_special_tokens=False),
                                 runtime.answer_eos_id], "stop")

    correct = completion(correct_text)
    # 四个不相交的小角框中必有一个与合法 GT 不命中；判分仍调用正式 classify。
    for box in ([0, 0, 1, 1], [999, 999, 1000, 1000],
                [0, 999, 1, 1000], [999, 0, 1000, 1]):
        wrong = completion(json.dumps({"bbox_2d": box}, separators=(",", ":")))
        if classify(row, [correct, wrong])["hits"] == [True, False]:
            break
    else:
        raise AssertionError(f"{row['id']}: 不能构造一中一错夹具，检查 GT 量化后是否有效")
    if kind == "mixed":
        return [correct, wrong]
    if kind == "gt":
        return [wrong, wrong]
    if kind in {"retain", "skip"}:
        return [correct, correct]
    raise ValueError(kind)


def _retention_rng(seed, probability, *, all_skip=False):
    assert 0 < probability < 1, "smoke 四分流检查要求 0 < keep_probability < 1"
    # 选取可重放的 RNG 初态，分流仍执行原始 keep_probability 判定。
    for candidate in count(seed):
        probe = random.Random(candidate)
        if all_skip:
            matches = all(probe.random() >= probability for _ in range(4))
        else:
            matches = probe.random() < probability
            if matches:
                probe.randrange(2)
                matches = probe.random() >= probability
        if matches:
            return random.Random(candidate), candidate


class _FixtureQuestion:
    def __init__(self, question, owner, row, kind):
        self.question, self.owner, self.row, self.kind = question, owner, row, kind

    def generate(self):
        if self.owner.generate_actual:
            _actual_generation(self.question, self.row, self.owner.output, self.owner.real_records)
        return synthetic_completions(self.owner.runtime, self.row, self.kind)

    def logps(self, token_ids, reference=False):
        return self.question.logps(token_ids, reference=reference)

    def supervised_logps(self, text):
        return self.question.supervised_logps(text)


def _actual_generation(question, row, output, records):
    synchronize()
    started = time.perf_counter()
    actual = question.generate()
    synchronize()
    assert len(actual) == 2
    record = {"id": row["id"], "image_count": len(row["images"]),
              "source": "actual_hf_g2", "synthetic": False,
              "seconds": time.perf_counter() - started,
              "completions": [asdict(item) for item in actual],
              "classification": classify(row, actual)}
    # 原始真实采样在任何合成分流返回之前独立落盘。
    append_jsonl(output / "real_generations.jsonl", [record])
    records.append(record)
    _progress(output, "actual_g2_generated", real_g2_timings=_real_timing(records))
    return record


@contextmanager
def _audited_question(runtime, row, audits):
    """直接观察模板边界和真实 visual.forward，不依赖 runtime 自报计数。"""
    original = runtime.template.encode
    had_override = "encode" in runtime.template.__dict__
    encoded = []
    visuals = []

    def encode(inputs, *args, **kwargs):
        assert set(inputs) == {"messages", "images", "chat_template_kwargs"}, "监督字段进入模板"
        assert inputs["messages"] == row["messages"], "生成 messages 被监督答案改写"
        assert not any(message["role"] == "assistant" for message in inputs["messages"])
        assert inputs["chat_template_kwargs"] == {"enable_thinking": False}
        encoded.append(deepcopy(inputs))
        return original(inputs, *args, **kwargs)

    runtime.template.encode = encode
    hook = runtime.multimodal.visual.register_forward_hook(lambda *args: visuals.append(1))
    try:
        with runtime.question(row) as question:
            yield question
        assert len(encoded) == 1, "每题模板编码次数不为一"
        assert len(visuals) == 1, f"{row['id']}: 实际视觉前向 {len(visuals)} 次"
        audits.append({"id": row["id"], "image_count": len(row["images"]),
                       "prompt_fields": sorted(encoded[0]), "messages_equal_source": True,
                       "assistant_in_prompt": False, "visual_forwards": len(visuals)})
    finally:
        hook.remove()
        if had_override:
            runtime.template.encode = original
        else:
            del runtime.template.encode


class _FixtureRuntime:
    def __init__(self, runtime, output, kinds, audits, real_records, *, generate_actual):
        self.runtime, self.output, self.kinds = runtime, output, kinds
        self.audits, self.real_records = audits, real_records
        self.generate_actual = generate_actual

    @contextmanager
    def question(self, row):
        with _audited_question(self.runtime, row, self.audits) as question:
            yield _FixtureQuestion(question, self, row, self.kinds[row["id"]])


def _batch(runtime, optimizer, rows, kinds, retention_rng, config, output, audits, actual, *, generate_actual):
    proxy = _FixtureRuntime(runtime, output, dict(zip((row["id"] for row in rows), kinds)),
                            audits, actual, generate_actual=generate_actual)
    parameters = runtime.trainable_parameters()
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    started = time.perf_counter()
    records = [process_question(proxy, row, "correction", retention_rng, config) for row in rows]
    assert [record["route"] for record in records] == kinds
    active = sum(record["route"] != "skip" for record in records)
    grad_norm = None
    if active:
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        grad_norm = float(norm)
        assert np.isfinite(grad_norm) and grad_norm > 0
        optimizer.step()
    else:
        assert all(parameter.grad is None for parameter in parameters), "all-skip 批次产生了梯度"
    assert all(parameter.grad is None for parameter in runtime.model.parameters() if not parameter.requires_grad)
    synchronize()
    seconds = time.perf_counter() - started
    for record, kind in zip(records, kinds):
        record.update(synthetic=True, fixture=kind, actual_g2_generated=generate_actual)
    append_jsonl(output / "synthetic_branch_records.jsonl", records)
    return records, {"seconds": seconds, "routes": kinds, "active_questions": active,
                     "optimizer_updates": int(bool(active)), "grad_norm": grad_norm,
                     "includes_actual_g2": generate_actual, "synthetic_branch_fixtures": True}


def _restore_policy(runtime, values):
    named = dict(runtime.model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            named[name].copy_(value)


def _sft_benchmark(runtime, initial_policy, rows, retention_rng, config, output, audits):
    """驻留模型临时回到 C550，用独立 fresh AdamW 测完整四题 SFT，再精确恢复。"""
    trained_policy = _parameter_snapshot(runtime)
    original_rng = rng_state()
    _restore_policy(runtime, initial_policy)
    parameters = runtime.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    proxy = _FixtureRuntime(runtime, output, {row["id"]: "gt" for row in rows}, audits, [], generate_actual=False)
    optimizer.zero_grad(set_to_none=True)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    synchronize()
    started = time.perf_counter()
    try:
        records = [process_question(proxy, row, "sft", retention_rng, config) for row in rows]
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        assert torch.isfinite(norm) and norm > 0
        optimizer.step()
        synchronize()
        metric = {"seconds": time.perf_counter() - started, "questions": 4, "optimizer_updates": 1,
                  "grad_norm": float(norm), "image_counts": [len(row["images"]) for row in rows],
                  "initialization": "C550 policy and fresh AdamW", "smoke_only": True}
        if runtime.device.type == "cuda":
            metric["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(runtime.device)
            metric["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(runtime.device)
        _progress(output, "sft_four_question_step_complete", sft_four_question_step=metric)
        metric["parameter_change"] = _parameter_changes(runtime, initial_policy)
        append_jsonl(output / "sft_probe_records.jsonl", records)
        return metric
    finally:
        runtime.model.zero_grad(set_to_none=True)
        del optimizer, parameters, proxy
        _restore_policy(runtime, trained_policy)
        _assert_exact(_parameter_snapshot(runtime), trained_policy, "sft_probe.policy_reset")
        restore_rng(original_rng)


def _difference(actual, expected):
    a = actual.detach().float().cpu().numpy()
    b = expected.detach().float().cpu().numpy()
    delta = a - b
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    squared_error = float(np.square(delta, dtype=np.float64).sum())
    squared_reference = float(np.square(b, dtype=np.float64).sum())
    return {"max_abs": float(np.abs(delta).max()) if finite else None,
            "global_relative_l2": float(np.sqrt(squared_error / max(squared_reference, np.finfo(float).tiny))) if finite else None,
            "nonzero_differences": int(np.count_nonzero(delta)), "finite": finite,
            "squared_error": squared_error, "squared_reference": squared_reference}


def _gradient_difference(actual, expected):
    assert actual.keys() == expected.keys()
    total_error = total_reference = maximum = 0.0
    different = 0
    first = largest = None
    missing = []
    finite = True
    for name in expected:
        if actual[name] is None or expected[name] is None:
            if actual[name] is not expected[name]:
                missing.append(name)
            continue
        metric = _difference(actual[name], expected[name])
        finite = finite and metric["finite"]
        total_error += metric["squared_error"]
        total_reference += metric["squared_reference"]
        different += metric["nonzero_differences"]
        if metric["nonzero_differences"] and first is None:
            first = name
        if metric["max_abs"] is not None and metric["max_abs"] > maximum:
            maximum, largest = metric["max_abs"], name
    return {"max_abs": maximum if finite else None,
            "global_relative_l2": float(np.sqrt(total_error / max(total_reference, np.finfo(float).tiny))) if finite else None,
            "nonzero_differences": different, "first_module": first, "max_abs_module": largest,
            "gradient_presence_mismatch": missing, "finite": finite}


def _cache_equivalence(runtime, row, tokens, output=None):
    """cached/native 各重复两次记录自身噪声，最后仍执行零误差门槛。"""
    assert tokens, "真实生成没有词元，无法验证缓存等价性"
    original = runtime.multimodal.get_image_features
    visuals = []
    hook = runtime.multimodal.visual.register_forward_hook(lambda *args: visuals.append(1))
    synchronize()
    started = time.perf_counter()
    snapshots, runs = {}, []
    try:
        with runtime.question(row) as question:
            random_state = rng_state()
            cached_method = runtime.multimodal.get_image_features
            try:
                for label in ("cached_0", "cached_1", "native_0", "native_1"):
                    runtime.model.zero_grad(set_to_none=True)
                    restore_rng(random_state)
                    runtime.multimodal.get_image_features = cached_method if label.startswith("cached") else original
                    synchronize()
                    run_started = time.perf_counter()
                    values = question.logps(tokens)
                    assert torch.isfinite(values).all()
                    (-values.mean()).backward()
                    snapshot = {"logps": _cpu(values), "gradients": {
                        name: _cpu(parameter.grad) for name, parameter in runtime.model.named_parameters()
                        if parameter.requires_grad}}
                    assert any(value is not None and value.count_nonzero() > 0 for value in snapshot["gradients"].values())
                    snapshots[label] = snapshot
                    del values
                    synchronize()
                    runs.append({"label": label, "seconds": time.perf_counter() - run_started,
                                 "logps": snapshot["logps"].tolist(), "visual_forwards_so_far": len(visuals)})
                    if output:
                        _progress(output, f"cache_{label}_complete", cache_runs=runs)
                assert len(visuals) == 3
            finally:
                runtime.multimodal.get_image_features = cached_method
            runtime.model.zero_grad(set_to_none=True)
    finally:
        hook.remove()
    synchronize()
    comparisons = {}
    for name, left, right in (
        ("cached_repeat", "cached_0", "cached_1"), ("native_repeat", "native_0", "native_1"),
        ("cached_vs_native_0", "cached_0", "native_0"), ("cached_vs_native_1", "cached_1", "native_1")):
        comparisons[name] = {"logps": _difference(snapshots[left]["logps"], snapshots[right]["logps"]),
                             "gradients": _gradient_difference(snapshots[left]["gradients"], snapshots[right]["gradients"])}
    diagnostics = {"id": row["id"], "tokens": len(tokens), "rtol": 0, "atol": 0,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "runs": runs, "comparisons": comparisons,
                   "visual_forwards_cached": 1, "visual_forwards_uncached_extra": 2,
                   "seconds": time.perf_counter() - started}
    if output:
        write_json(output / "cache_diagnostics.json", diagnostics)
        _progress(output, "cache_diagnostics_complete", cache_equivalence=diagnostics)
    for name, metrics in comparisons.items():
        for item in ("logps", "gradients"):
            value = metrics[item]
            assert value["finite"] and not value["nonzero_differences"] and not value.get("gradient_presence_mismatch"), (
                f"cache exact check failed: {name}.{item}: {json.dumps(value, ensure_ascii=False)}")
    return diagnostics | {"logps_exact": True, "all_trainable_gradients_exact": True}


def _rng_probe(device):
    result = {"python": random.random(), "numpy": np.random.random(4), "torch": torch.rand(4)}
    if device.type == "cuda":
        result["cuda"] = torch.rand(4, device=device).cpu()
    return result


def _protect_output(config, output):
    initial = Path(config["init_adapter"]).resolve()
    target = output.resolve()
    if target == initial or target.is_relative_to(initial):
        raise ValueError("smoke 输出目录不得位于 C550 初始化 adapter 内")


def run_smoke(config, output: Path, *, runtime_factory=None, require_cuda=True):
    _protect_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("真实 Qwen3.6-27B smoke 必须在 CUDA GPU 上执行")
    _progress(output, "loading_model", status="running", model=config["model"],
              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
              cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    if runtime_factory is None:
        from tools.quant_correction_runtime import CorrectionRuntime
        runtime_factory = CorrectionRuntime
    rows = read_train_index(Path(config["train_index"]))
    representatives = {size: next((row for row in rows if len(row["images"]) == size), None) for size in (2, 3)}
    assert all(representatives.values()), "训练索引必须同时提供真实两图和三图输入"
    assert config["gradient_checkpointing"] is True, "首轮 smoke 必须启用原生语言 checkpoint"
    seed_all(config["seed"])
    load_started = time.perf_counter()
    runtime = runtime_factory(config, "correction")
    _progress(output, "model_loaded", model_load_seconds=time.perf_counter() - load_started)
    device = runtime.device
    parameters = runtime.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
    sampler = QuestionSampler(rows, config["seed"])
    retention_rng, fixture_seed = _retention_rng(config["seed"] + 1, config["keep_probability"])
    reference_before = _parameter_snapshot(runtime, reference=True)
    audits, actual = [], []
    first_rows = sampler.next_batch()
    # 前三题均有真实反向；若另一种图像数量只落在 skip 或不在本批，先补齐其 GT 反向。
    backward_sizes = {len(row["images"]) for row in first_rows[:3]}
    preflight_backward = []
    for size in sorted({2, 3} - backward_sizes):
        row = representatives[size]
        runtime.model.zero_grad(set_to_none=True)
        with _audited_question(runtime, row, audits) as question:
            _actual_generation(question, row, output, actual)
            loss = -question.supervised_logps(serialize_gt(row)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            grads = [parameter.grad for parameter in runtime.trainable_parameters()]
            assert any(gradient is not None and gradient.count_nonzero() > 0 for gradient in grads)
            assert all(torch.isfinite(gradient).all() for gradient in grads if gradient is not None)
            preflight_backward.append({"id": row["id"], "image_count": size,
                                       "finite_nonzero_gradients": True, "optimizer_updates": 0})
        runtime.model.zero_grad(set_to_none=True)
        del question, loss, grads
        _progress(output, "preflight_gt_backward_complete", preflight_gt_backward=preflight_backward)
    policy_before = _parameter_snapshot(runtime)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records, step = _batch(runtime, optimizer, first_rows, ["mixed", "gt", "retain", "skip"],
                           retention_rng, config, output, audits, actual, generate_actual=True)
    if device.type == "cuda":
        step["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        step["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    _progress(output, "correction_four_question_step_complete", four_question_step=step,
              real_g2_timings=_real_timing(actual))
    for row, record in zip(first_rows, records):
        sampler.observe(row["id"], record["classification"])
    step["parameter_change"] = _parameter_changes(runtime, policy_before)
    _assert_reference_unchanged(runtime, reference_before)
    sft_step = _sft_benchmark(runtime, policy_before, first_rows, retention_rng, config, output, audits)
    del policy_before
    _assert_reference_unchanged(runtime, reference_before)
    assert {record["image_count"] for record in actual} == {2, 3}
    real_first = next(record for record in actual if record["id"] == first_rows[0]["id"])
    cache = _cache_equivalence(runtime, first_rows[0], real_first["completions"][0]["token_ids"], output)
    policy_before_skip = _parameter_snapshot(runtime)
    optimizer_before_skip = _cpu(optimizer.state_dict())
    skip_rng, skip_seed = _retention_rng(config["seed"] + 1, config["keep_probability"], all_skip=True)
    # 独立 all-skip 负例不推进采样器；检查点保留首批实际建立的非空 mixed 池。
    _, skip_step = _batch(runtime, optimizer, first_rows, ["skip"] * 4,
        skip_rng, config, output, audits, actual, generate_actual=False)
    skip_step["affects_checkpoint_sampler"] = False
    _assert_exact(_parameter_snapshot(runtime), policy_before_skip, "all_skip.parameters")
    _assert_exact(_cpu(optimizer.state_dict()), optimizer_before_skip, "all_skip.optimizer")
    _assert_reference_unchanged(runtime, reference_before)
    del policy_before_skip, optimizer_before_skip
    _progress(output, "all_skip_complete", all_skip_step=skip_step)
    expected_policy = _parameter_snapshot(runtime)
    expected_optimizer = _cpu(optimizer.state_dict())
    expected_rng = rng_state()
    assert sampler.mixed_pool, "恢复检查必须实际覆盖非空 mixed 池"
    state = {"mode": "correction", "presentations": 4, "updates": 1, "batches": 1, "smoke_only": True}
    checkpoint = save_checkpoint(runtime, optimizer, output, state, sampler, retention_rng, config)
    _progress(output, "checkpoint_saved", checkpoint=str(checkpoint))
    saved = json.loads((checkpoint / "loop_state.json").read_text(encoding="utf-8"))
    saved_rng = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False)
    _assert_exact(saved_rng, expected_rng, "saved.rng")
    expected_draws = _rng_probe(device)
    expected_orders = [[row["id"] for row in sampler.next_batch()] for _ in range(3)]
    expected_pool = list(sampler.mixed_pool)
    expected_retention = [retention_rng.random() for _ in range(8)]
    audit_before_release = dict(runtime.audit)
    # 先销毁整套 NF4 模型和优化器，再从保存的 adapter 建立新实例。
    released_model = weakref.ref(runtime.model)
    del runtime, optimizer, parameters
    gc.collect()
    assert released_model() is None, "旧 smoke 模型仍被引用，不能同时重载 27B 模型"
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _progress(output, "reloading_checkpoint", previous_model_released=True)
    reloaded = runtime_factory(config, "correction", adapter_path=str(checkpoint))
    restored_optimizer = torch.optim.AdamW(reloaded.trainable_parameters(), lr=config["learning_rate"], weight_decay=0)
    restored_optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
    restored_sampler = QuestionSampler(rows, config["seed"])
    restored_sampler.load_state_dict(saved["sampler"])
    restored_retention = random.Random()
    restored_retention.setstate(random_state_from_json(saved["retention_rng"]))
    restore_rng(saved_rng)
    _assert_exact(_parameter_snapshot(reloaded), expected_policy, "reload.adapter")
    _assert_reference_unchanged(reloaded, reference_before)
    _assert_exact(_cpu(restored_optimizer.state_dict()), expected_optimizer, "reload.optimizer")
    _assert_exact(rng_state(), saved_rng, "reload.rng")
    _assert_exact(_rng_probe(device), expected_draws, "reload.random_draws")
    actual_orders = [[row["id"] for row in restored_sampler.next_batch()] for _ in range(3)]
    assert actual_orders == expected_orders
    assert list(restored_sampler.mixed_pool) == expected_pool
    assert [restored_retention.random() for _ in range(8)] == expected_retention
    report = {"status": "passed", "smoke_only": True, "counts_toward_formal_4400": False,
              "device": str(device), "model": config["model"], "init_adapter": config["init_adapter"],
              "real_g2_questions": len(actual), "real_image_counts": sorted({r["image_count"] for r in actual}),
              "real_backward_image_counts": sorted(backward_sizes | {r["image_count"] for r in preflight_backward}),
              "preflight_gt_backward": preflight_backward,
              "fixture_retention_seed": fixture_seed, "all_skip_retention_seed": skip_seed,
              "synthetic_routes_are_natural_sampling_evidence": False,
              "four_question_step": step, "all_skip_step": skip_step, "cache_equivalence": cache,
              "sft_four_question_step": sft_step,
              "prompt_and_visual_audits": audits, "runtime_audit": audit_before_release,
              "checkpoint": str(checkpoint),
              "restore": {"adapter_exact": True, "reference_exact": True, "optimizer_exact": True,
                          "previous_model_released": True,
                          "rng_exact": True, "random_draws_exact": True, "retention_rng_exact": True,
                          "next_three_batches": actual_orders, "mixed_pool": expected_pool}}
    write_json(output / "report.json", report)
    _progress(output, "passed", status="passed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true",
                        help="在 CUDA 初始化前启用 PyTorch 确定性算法与 CUBLAS 工作区配置")
    args = parser.parse_args(argv)
    writable = False
    try:
        if args.deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
        config = load_config(args.config)
        _protect_output(config, args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        writable = True
        report = run_smoke(config, args.output_dir)
    except Exception as error:
        report = {"status": "failed", "smoke_only": True, "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc()}
        if writable:
            _progress(args.output_dir, "failed", status="failed", error_type=type(error).__name__, error=str(error))
            report["progress"] = json.loads((args.output_dir / "progress.json").read_text(encoding="utf-8"))
            write_json(args.output_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
