"""在一个原生纠错模型上配对检验 G2 提示缓存复用及完整四题训练批耗时。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_config import load_config, read_train_index
from tools.train_quant_grounding import process_question, restore_rng, rng_state, seed_all, synchronize


def write_report(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return copy.deepcopy(value)


def assert_equal(first, second, path="value") -> None:
    """逐元素精确比较；不把相同采样词元误当成相同完整分布。"""
    if isinstance(first, torch.Tensor):
        if not isinstance(second, torch.Tensor) or first.shape != second.shape or first.dtype != second.dtype:
            raise AssertionError(f"{path}: tensor shape/dtype changed")
        if not torch.equal(first, second):
            raise AssertionError(f"{path}: tensor values changed")
    elif isinstance(first, np.ndarray):
        if not isinstance(second, np.ndarray) or not np.array_equal(first, second):
            raise AssertionError(f"{path}: NumPy values changed")
    elif isinstance(first, dict):
        if not isinstance(second, dict) or first.keys() != second.keys():
            raise AssertionError(f"{path}: dict keys changed")
        for key in first:
            assert_equal(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise AssertionError(f"{path}: sequence length/type changed")
        for index, (left, right) in enumerate(zip(first, second)):
            assert_equal(left, right, f"{path}[{index}]")
    elif first != second:
        raise AssertionError(f"{path}: {first!r} != {second!r}")


def select_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for image_count in (2, 3):
        group = [row for row in rows if len(row["images"]) == image_count]
        if len(group) < 2:
            raise ValueError(f"正式训练索引需要至少两道{image_count}图题")
        selected.extend(rng.sample(group, 2))
    return selected


def timing_decision(pairs: list[dict], *, minimum_gain: float = 0.03) -> dict:
    """所有预定重复都计入；拒绝负收益、冷启动和局部生成收益替代全步收益。"""
    if len(pairs) < 3:
        raise ValueError("稳态计时至少需要三组配对")
    full_gains, generation_gains = [], []
    for index, pair in enumerate(pairs):
        for arm in ("baseline", "reuse"):
            for metric in ("full_batch_seconds", "generation_seconds"):
                value = pair[arm][metric]
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"pair {index}: 无效计时 {arm}.{metric}")
        full_gains.append(1 - pair["reuse"]["full_batch_seconds"] / pair["baseline"]["full_batch_seconds"])
        generation_gains.append(1 - pair["reuse"]["generation_seconds"] / pair["baseline"]["generation_seconds"])
    baseline_full = statistics.median(p["baseline"]["full_batch_seconds"] for p in pairs)
    reuse_full = statistics.median(p["reuse"]["full_batch_seconds"] for p in pairs)
    baseline_generation = statistics.median(p["baseline"]["generation_seconds"] for p in pairs)
    reuse_generation = statistics.median(p["reuse"]["generation_seconds"] for p in pairs)
    full_gain = 1 - reuse_full / baseline_full
    generation_gain = 1 - reuse_generation / baseline_generation
    checks = {
        "at_least_three_steady_pairs": len(pairs) >= 3,
        "full_batch_median_gain_at_least_3pct": full_gain >= minimum_gain,
        "paired_full_batch_median_gain_at_least_3pct": statistics.median(full_gains) >= minimum_gain,
        "every_full_batch_pair_faster": all(gain > 0 for gain in full_gains),
        "generation_median_faster": generation_gain > 0,
        "paired_generation_median_faster": statistics.median(generation_gains) > 0,
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "minimum_full_batch_gain": minimum_gain,
            "full_batch_paired_gains": full_gains, "generation_paired_gains": generation_gains,
            "full_batch_median_gain": full_gain, "generation_median_gain": generation_gain,
            "baseline_full_batch_median_seconds": baseline_full, "reuse_full_batch_median_seconds": reuse_full,
            "baseline_generation_median_seconds": baseline_generation,
            "reuse_generation_median_seconds": reuse_generation}


def native_state(runtime) -> dict:
    return {"rope_deltas": to_cpu(getattr(runtime.multimodal, "rope_deltas", None))}


def restore_native_state(runtime, state: dict) -> None:
    value = state["rope_deltas"]
    runtime.multimodal.rope_deltas = value.to(runtime.device) if isinstance(value, torch.Tensor) else value


def trace_forward(_module, _args, kwargs) -> dict:
    cache = kwargs.get("past_key_values")
    return {key: to_cpu(kwargs.get(key)) for key in (
        "input_ids", "position_ids", "attention_mask", "mm_token_type_ids")} | {
            "past_length": cache.get_seq_length() if cache is not None else 0}


def capture_generation(runtime, row: dict, seed: int, reuse: bool, greedy: bool, initial_native: dict) -> dict:
    runtime.config["generation_prefill_reuse"] = reuse
    restore_native_state(runtime, initial_native)
    seed_all(seed)
    traces = []
    with runtime.question(row) as question:
        original_inputs = to_cpu(question.inputs)
        hook = runtime.base.register_forward_pre_hook(
            lambda module, args, kwargs: traces.append(trace_forward(module, args, kwargs)), with_kwargs=True)
        try:
            completions = question.generate(greedy=greedy, record_generation_diagnostics=True)
        finally:
            hook.remove()
        assert_equal(original_inputs, to_cpu(question.inputs), "unmutated_prompt")
        diagnostics = question.generation_diagnostics
        audit = copy.deepcopy(question.prefill_reuse_audit)
        prefix_length = question.prefix_length
    return {"completions": [asdict(completion) for completion in completions],
            "diagnostics": diagnostics, "audit": audit, "traces": traces,
            "inputs": original_inputs, "prefix_length": prefix_length,
            "vocab_size": getattr(runtime.base.config, "text_config", runtime.base.config).vocab_size,
            "rng_after": rng_state(), "native_after": native_state(runtime)}


def compare_generation(first: dict, second: dict, *, greedy: bool) -> dict:
    for key in ("completions", "inputs", "prefix_length", "vocab_size", "rng_after", "native_after"):
        assert_equal(first[key], second[key], key)
    if len(first["diagnostics"]) != 2 or len(second["diagnostics"]) != 2:
        raise AssertionError("每题诊断必须包含两个原生独立回答")
    token_counts = []
    cache_lengths = []
    for index, (left, right) in enumerate(zip(first["diagnostics"], second["diagnostics"])):
        for state in (left["rng_before"], left["rng_after"], right["rng_before"], right["rng_after"]):
            if not {"python", "numpy", "torch", "cuda"}.issubset(state):
                raise AssertionError("每答需要完整 Python/NumPy/Torch/CUDA 随机状态")
        for key in ("sequences", "token_ids", "text", "generation_config", "rng_before", "rng_after", "logits", "scores", "cache", "rope_deltas"):
            assert_equal(left[key], right[key], f"answer[{index}].{key}")
        token_ids = first["completions"][index]["token_ids"]
        assert_equal(left["token_ids"], token_ids, f"answer[{index}].decoded_tokens")
        if len(left["logits"]) != len(token_ids) or len(left["scores"]) != len(token_ids):
            raise AssertionError("没有覆盖每个生成词元的完整 logits/scores")
        if any(tensor.shape != (1, first["vocab_size"])
               for tensor in [*left["logits"], *left["scores"]]):
            raise AssertionError("诊断必须保留每个词元的完整词表")
        prefix = first["prefix_length"]
        assert_equal(left["sequences"][0, :prefix], first["inputs"]["input_ids"][0], "native_prompt_prefix")
        assert_equal(left["sequences"][0, prefix:].tolist(), token_ids, "native_generated_tokens")
        length = left["cache"]["seq_length"]
        if length != prefix + len(token_ids) - 1:
            raise AssertionError(f"原生最终缓存长度异常: {length}, prompt={prefix}, answer={len(token_ids)}")
        if not left["cache"]["layers"] or not left["cache"]["tensors"]:
            raise AssertionError("缺少原生 KV/卷积/循环缓存快照")
        for field in ("keys", "values", "conv_states", "recurrent_states"):
            if not any(f".{field}" in name for name in left["cache"]["tensors"]):
                raise AssertionError(f"缺少 Qwen3.5 原生缓存字段 {field}")
        if len(left["prefill"]) != 1 or len(right["prefill"]) != 1:
            raise AssertionError("每答必须恰好调用一次原生 _prefill 接口")
        if left["prefill"][0]["cache_length"] != prefix or right["prefill"][0]["cache_length"] != prefix:
            raise AssertionError("prefill 缓存不是完整原生提示长度")
        for event, replayed in ((left["prefill"][0], False), (right["prefill"][0], index == 1 and not greedy)):
            if event["prompt_length"] != prefix or event["native_prefills"] != int(not replayed) or event["prefill_replays"] != int(replayed):
                raise AssertionError("原生 prefill/第二答复用事件不符合预期")
        token_counts.append(len(token_ids))
        cache_lengths.append(length)
    # 复用只移除第二次完整提示 forward，其余解码输入、位置轴和缓存长度逐项不变。
    expected = first["traces"]
    if not greedy:
        expected = expected[:token_counts[0]] + expected[token_counts[0] + 1:]
    assert_equal(expected, second["traces"], "native_decode_trace")
    expected_prefills = 2 if greedy else 1
    if first["audit"]["enabled"] or second["audit"]["enabled"] != (not greedy):
        raise AssertionError("复用开关与原生采样/greedy 分支不符")
    if second["audit"].get("native_prefills") != expected_prefills:
        raise AssertionError("复用分支实际原生 prefill 次数异常")
    if second["audit"].get("prefill_replays", 0) != (0 if greedy else 1):
        raise AssertionError("复用分支没有恰好复用第二答提示，或错误复用 greedy")
    if first["audit"].get("native_prefills") != 2 or first["audit"].get("prefill_replays", 0):
        raise AssertionError("baseline 不是两次独立原生 prefill")
    return {"passed": True, "answer_tokens": token_counts, "prefix_tokens": first["prefix_length"],
            "completions": first["completions"], "vocab_size": first["vocab_size"],
            "generation_config": first["diagnostics"][0]["generation_config"],
            "cache_lengths": cache_lengths, "baseline_native_forwards": len(first["traces"]),
            "reuse_native_forwards": len(second["traces"]), "reuse_audit": second["audit"],
            "checks": ["tokens/text/finish reason", "every full-vocabulary token logits and scores",
                       "per-answer and final Python/NumPy/Torch/CUDA RNG", "generation config unchanged",
                       "prompt unchanged", "native decoding ids/positions/masks/mm types",
                       "native final KV/conv/recurrent caches and rope state"]}


class TimedRuntime:
    """仅计量原有 generate 调用；训练及其自然路由仍由 process_question 执行。"""

    def __init__(self, runtime):
        self.runtime = runtime
        self.generation_seconds = []

    @contextmanager
    def question(self, row):
        with self.runtime.question(row) as question:
            owner = self

            class TimedQuestion:
                def generate(self):
                    start = time.perf_counter()
                    result = question.generate()
                    # 原生 generate 的 .tolist()/文本解码已经等待返回词元。
                    owner.generation_seconds.append(time.perf_counter() - start)
                    return result

                def __getattr__(self, name):
                    return getattr(question, name)

            yield TimedQuestion()


def parameter_snapshot(parameters) -> list:
    return [to_cpu(parameter) for parameter in parameters]


def restore_training(runtime, optimizer, parameters, initial: dict, seed: int, reuse: bool) -> None:
    with torch.no_grad():
        for parameter, saved in zip(parameters, initial["parameters"]):
            parameter.copy_(saved)
    optimizer.load_state_dict(copy.deepcopy(initial["optimizer"]))
    optimizer.zero_grad(set_to_none=True)
    restore_native_state(runtime, initial["native"])
    seed_all(seed)
    runtime.config["generation_prefill_reuse"] = reuse


def run_batch(runtime, optimizer, parameters, rows: list, config: dict, seed: int, *, capture: bool) -> dict:
    observed = TimedRuntime(runtime)
    retention_rng = random.Random(seed + 1)
    optimizer.zero_grad(set_to_none=True)
    synchronize()
    started = time.perf_counter()
    records = [process_question(observed, row, "correction", retention_rng, config) for row in rows]
    active = sum(record["route"] != "skip" for record in records)
    gradients = None
    if capture:
        gradients = [to_cpu(parameter.grad) for parameter in parameters]
    norm = None
    if active:
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        optimizer.step()
    synchronize()
    elapsed = time.perf_counter() - started
    result = {"full_batch_seconds": elapsed, "generation_seconds": sum(observed.generation_seconds),
              "g2_question_seconds": observed.generation_seconds,
              "records": records, "active_questions": active, "grad_norm": norm,
              "rng_after": rng_state(), "retention_rng_after": retention_rng.getstate()}
    if capture:
        result.update(gradients=gradients, parameters=parameter_snapshot(parameters),
                      optimizer=to_cpu(optimizer.state_dict()), native_after=native_state(runtime))
    return result


def batch_summary(result: dict) -> dict:
    return {key: result[key] for key in (
        "full_batch_seconds", "generation_seconds", "g2_question_seconds", "records", "active_questions", "grad_norm")}


def compare_batches(first: dict, second: dict, *, capture: bool) -> None:
    keys = ["records", "active_questions", "rng_after", "retention_rng_after"]
    if capture:
        keys.extend(["grad_norm", "gradients", "parameters", "optimizer", "native_after"])
    for key in keys:
        assert_equal(first[key], second[key], f"full_batch.{key}")


def benchmark(config: dict, adapter: Path, output: Path, *, seeds: list[int], repeats: int = 3,
              max_seconds: float = 1500, runtime_factory=None) -> dict:
    report = {"schema": "native-prefill-reuse-benchmark-v1", "status": "running", "accepted": False,
              "reason": "benchmark not completed", "config": config, "adapter": str(adapter),
              "seeds": seeds, "timing_repeats": repeats, "max_seconds": max_seconds,
              "correctness": {"passed": False, "generation_cases": []}, "timing": {"pairs": []}}
    started = time.perf_counter()

    def phase(name):
        if time.perf_counter() - started > max_seconds:
            raise TimeoutError(f"基准超过 {max_seconds:g} 秒预算")
        report.update(phase=name, elapsed_seconds=time.perf_counter() - started)
        write_report(output, report)
        print(json.dumps({"phase": name, "elapsed_seconds": report["elapsed_seconds"]}), flush=True)

    try:
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError("正确性诊断至少需要两个不同 seed")
        if repeats < 3:
            raise ValueError("稳态计时至少需要三组配对")
        rows = select_rows(read_train_index(Path(config["train_index"])), config["seed"])
        report["questions"] = [{"id": row["id"], "image_count": len(row["images"])} for row in rows]
        optimizer_file = adapter / "optimizer.pt"
        loop_file = adapter / "loop_state.json"
        if not optimizer_file.is_file() or not loop_file.is_file():
            raise ValueError("需要含 loop_state.json/optimizer.pt 的完整原生纠错检查点")
        saved_loop = json.loads(loop_file.read_text(encoding="utf-8"))
        if saved_loop["schema"] != "acc-correction-v1" or saved_loop["mode"] != "correction":
            raise ValueError("只接受原生 correction 模式检查点")
        report["checkpoint"] = {key: saved_loop[key] for key in ("schema", "mode", "presentations", "updates")}
        phase("loading_one_runtime")
        if runtime_factory is None:
            from tools.quant_correction_runtime import CorrectionRuntime
            runtime_factory = CorrectionRuntime
        runtime = runtime_factory(config, "correction", adapter_path=str(adapter))
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(runtime.device)}
        parameters = list(runtime.trainable_parameters())
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0)
        optimizer.load_state_dict(torch.load(optimizer_file, map_location="cpu", weights_only=False))
        report["optimizer"] = {"class": "AdamW", "restored_from_checkpoint": True,
                               "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        initial = {"parameters": parameter_snapshot(parameters), "optimizer": to_cpu(optimizer.state_dict()),
                   "native": native_state(runtime)}
        for seed in seeds:
            for row in rows:
                phase(f"generation_sampling:{row['id']}:{seed}")
                baseline = capture_generation(runtime, row, seed, False, False, initial["native"])
                reused = capture_generation(runtime, row, seed, True, False, initial["native"])
                case = compare_generation(baseline, reused, greedy=False)
                report["correctness"]["generation_cases"].append(
                    {"id": row["id"], "image_count": len(row["images"]), "seed": seed, "greedy": False, **case})
                del baseline, reused
        for row in (rows[0], rows[2]):
            phase(f"generation_greedy:{row['id']}")
            baseline = capture_generation(runtime, row, seeds[0], False, True, initial["native"])
            reused = capture_generation(runtime, row, seeds[0], True, True, initial["native"])
            case = compare_generation(baseline, reused, greedy=True)
            report["correctness"]["generation_cases"].append(
                {"id": row["id"], "image_count": len(row["images"]), "seed": seeds[0], "greedy": True, **case})
            del baseline, reused
        phase("full_batch_correctness")
        was_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            restore_training(runtime, optimizer, parameters, initial, seeds[0], False)
            baseline = run_batch(runtime, optimizer, parameters, rows, config, seeds[0], capture=True)
            restore_training(runtime, optimizer, parameters, initial, seeds[0], True)
            reused = run_batch(runtime, optimizer, parameters, rows, config, seeds[0], capture=True)
            compare_batches(baseline, reused, capture=True)
            if not baseline["active_questions"] or not any(gradient is not None for gradient in baseline["gradients"]):
                raise AssertionError("自然路由整批均 skip，未检验反传和参数更新")
            if all(torch.equal(before, after) for before, after in zip(initial["parameters"], baseline["parameters"])):
                raise AssertionError("自然路由整批没有真实参数更新")
            report["correctness"]["full_batch"] = {
                "passed": True, "natural_routes": [record["route"] for record in baseline["records"]],
                "records": baseline["records"], "trainable_tensors_compared": len(parameters),
                "gradient_tensors_compared": sum(gradient is not None for gradient in baseline["gradients"]),
                "checks": ["unchanged process_question generation/routing/scoring/backward/optimizer",
                           "all gradients exact", "all trainable parameters exact after update",
                           "optimizer state exact", "loss/routes/completions exact",
                           "final global and retention RNG exact"], "deterministic_algorithms": True}
            del baseline, reused
        finally:
            torch.use_deterministic_algorithms(was_deterministic)
        report["correctness"]["passed"] = True
        # 每个模式各完成一个无诊断完整训练批次，排除模型加载/首次分配/内核编译。
        report["timing"]["warmup"] = []
        for reuse in (False, True):
            phase(f"steady_warmup:{reuse}")
            restore_training(runtime, optimizer, parameters, initial, seeds[0], reuse)
            result = run_batch(runtime, optimizer, parameters, rows, config, seeds[0], capture=False)
            report["timing"]["warmup"].append({"reuse": reuse, **batch_summary(result)})
        for index in range(repeats):
            phase(f"steady_pair:{index + 1}")
            order = (False, True) if index % 2 == 0 else (True, False)
            results = {}
            for reuse in order:
                restore_training(runtime, optimizer, parameters, initial, seeds[index % len(seeds)], reuse)
                results["reuse" if reuse else "baseline"] = run_batch(
                    runtime, optimizer, parameters, rows, config, seeds[index % len(seeds)], capture=False)
            compare_batches(results["baseline"], results["reuse"], capture=False)
            report["timing"]["pairs"].append({"index": index + 1, "order": ["reuse" if arm else "baseline" for arm in order],
                "seed": seeds[index % len(seeds)], **{key: batch_summary(value) for key, value in results.items()}})
        phase("timing_decision")
        report["timing"].update(timing_decision(report["timing"]["pairs"]))
        report["timing"]["measurement"] = {
            "diagnostics": False, "same_four_formal_training_questions": True,
            "full_batch_includes": ["template encode/collate", "visual encode", "two native answers per question",
                                    "natural classification/routing", "policy/reference scoring", "backward", "gradient clip", "AdamW step"],
            "excluded": ["model load", "checkpoint load", "parameter/optimizer/RNG restore", "warmup"],
            "generation_seconds": "sum of four G2 generate wall times, including native token transfer and text decode"}
        report["accepted"] = report["correctness"]["passed"] and report["timing"]["accepted"]
        report["status"] = "passed" if report["accepted"] else "rejected"
        report["reason"] = ("all correctness checks passed and steady full batch improved at least 3%"
                            if report["accepted"] else "correctness passed; conservative steady timing gate rejected reuse")
    except Exception as error:
        report.update(status="failed", accepted=False, reason=f"{type(error).__name__}: {error}",
                      error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(phase="complete", elapsed_seconds=time.perf_counter() - started)
    write_report(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=1500)
    args = parser.parse_args(argv)
    # 必须在 CUDA 初始化前声明，供完整梯度的确定性对照使用。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        config = load_config(args.config)
        report = benchmark(config, args.adapter, args.output_dir,
                           seeds=args.seeds or [config["seed"], config["seed"] + 1],
                           repeats=args.repeats, max_seconds=args.max_seconds)
    except Exception as error:
        report = {"schema": "native-prefill-reuse-benchmark-v1", "status": "failed", "accepted": False,
                  "reason": f"{type(error).__name__}: {error}", "config_path": str(args.config),
                  "adapter": str(args.adapter), "correctness": {"passed": False}, "timing": {},
                  "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}}
        write_report(args.output_dir, report)
    print(json.dumps({"status": report["status"], "accepted": report["accepted"],
                      "reason": report["reason"], "report": str(args.output_dir / "report.json")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "passed" and report["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
