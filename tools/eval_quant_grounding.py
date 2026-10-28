"""独立量化 grounding 评价与 G8 采样诊断。

生成：python tools/eval_quant_grounding.py --dataset DEV.jsonl --data-root DATA \
    --model MODEL --engine hf --precision nf4 --samples-per-query 8 --limit 128 \
    --output-dir OUT
复算：python tools/eval_quant_grounding.py --dataset DEV.jsonl \
    --predictions OUT/predictions.jsonl --output-dir RESCORED
greedy 开发集使用默认 --samples-per-query 1；不使用提前停止、重试或修框。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_reward import parse_bbox, score_completion, validate_normalized_box  # noqa: E402
from tools.quant_grounding_config import configure_unrestricted_hf_sampling  # noqa: E402
from tools.quant_grounding_postprocess import normalize_bbox_completion  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def independent_iou(a: list[float], b: list[float]) -> float:
    """独立于奖励实现的面积计算，最终 ACC 只使用本函数。"""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def score_sample(sample: dict, row: dict, answer_mode: str, reward_mode: str,
                 format_policy: str = "strict") -> dict:
    text = sample["text"]
    finish_reason = sample["finish_reason"]
    if not isinstance(text, str) or not isinstance(finish_reason, str):
        raise ValueError(f"{row['id']}: text/finish_reason must be strings")
    normalized_text = normalize_bbox_completion(text, answer_mode, format_policy)
    reward = score_completion(
        normalized_text, row["solution"], row.get("negative_boxes", []),
        answer_mode=answer_mode, reward_mode=reward_mode,
        finish_reason=finish_reason, require_termination=True,
    )
    box = parse_bbox(normalized_text, answer_mode)
    iou = independent_iou(box, row["solution"]) if box is not None else None
    if iou is not None and not math.isclose(iou, reward["iou"], abs_tol=1e-12):
        raise AssertionError(f"{row['id']}: independent IoU differs from reward IoU")
    result = {"text": text, "finish_reason": finish_reason,
              "tokens": sample.get("tokens"), "seconds": sample.get("seconds"),
              "bbox": box, "iou": iou, "hit": bool(reward["valid"] and iou is not None and iou >= 0.5),
              "score": reward}
    if result["hit"] != reward["hit"]:
        raise AssertionError(f"{row['id']}: independent ACC differs from reward hit")
    for key in ("token_ids", "thinking_trace"):
        if key in sample:
            result[key] = sample[key]
    if format_policy != "strict":
        result["normalized_text"] = normalized_text
        result["format_repaired"] = normalized_text != text
    return result


def score_rows(dataset: list[dict], predictions: list[dict], answer_mode: str,
               reward_mode: str, format_policy: str = "strict") -> list[dict]:
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate prediction id")
    expected = {row["id"] for row in dataset}
    if set(by_id) != expected:
        raise ValueError(f"prediction coverage mismatch: missing={sorted(expected - set(by_id))[:5]}, "
                         f"extra={sorted(set(by_id) - expected)[:5]}")
    sample_counts = {len(prediction["samples"]) for prediction in predictions}
    if len(sample_counts) != 1 or 0 in sample_counts:
        raise ValueError(f"inconsistent samples per query: {sample_counts}")
    scored = []
    for row in dataset:
        if any(message["role"] == "assistant" for message in row["messages"]):
            raise ValueError(f"{row['id']}: evaluation messages contain an assistant answer")
        validate_normalized_box(row["solution"], f"{row['id']}.solution")
        for bad in row.get("negative_boxes", []):
            validate_normalized_box(bad, f"{row['id']}.negative_boxes")
        samples = by_id[row["id"]]["samples"]
        scored.append({
            "id": row["id"], "source": row["source"], "group_id": row["group_id"],
            "negative_kind": row.get("negative_kind", []), "solution": row["solution"],
            "samples": [score_sample(sample, row, answer_mode, reward_mode, format_policy)
                        for sample in samples],
        })
    return scored


def summarize(rows: list[dict]) -> dict:
    def stats(subset: list[dict]) -> dict:
        samples = [sample for row in subset for sample in row["samples"]]
        n = len(subset)
        valid = sum(s["score"]["valid"] for s in samples)
        hits = sum(s["hit"] for s in samples)
        first_hits = sum(row["samples"][0]["hit"] for row in subset)
        all_wrong = mixed = all_hit = all_same_reward = zero_advantage = 0
        for row in subset:
            group = row["samples"]
            group_hits = sum(sample["hit"] for sample in group)
            all_wrong += group_hits == 0
            all_hit += group_hits == len(group)
            mixed += 0 < group_hits < len(group)
            rewards = [sample["score"]["reward"] for sample in group]
            same = max(rewards) - min(rewards) <= 1e-12
            all_same_reward += same
            zero_advantage += same  # scale_rewards=none: centered group advantage is zero
        tokens = [s["tokens"] for s in samples if s["tokens"] is not None]
        seconds = [s["seconds"] for s in samples if s["seconds"] is not None]
        return {
            "queries": n, "samples": len(samples),
            "acc05_first": first_hits / n if n else None,
            "acc05_all_samples": hits / len(samples) if samples else None,
            "format_valid": sum(s["score"]["format_valid"] for s in samples),
            "format_repaired": sum(s.get("format_repaired", False) for s in samples),
            "valid": valid, "invalid": len(samples) - valid,
            "truncated": sum(s["score"]["truncated"] for s in samples),
            "all_wrong_groups": all_wrong, "mixed_groups": mixed,
            "all_hit_groups": all_hit, "all_same_reward_groups": all_same_reward,
            "zero_advantage_groups": zero_advantage,
            "generated_tokens": sum(tokens), "timed_seconds": sum(seconds),
            "mean_tokens": sum(tokens) / len(tokens) if tokens else None,
            "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
        }

    sources: dict[str, list[dict]] = defaultdict(list)
    negatives: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        sources[row["source"]].append(row)
        for kind in set(row["negative_kind"]) or {"none"}:
            negatives[kind].append(row)
    return {"overall": stats(rows),
            "by_source": {key: stats(value) for key, value in sorted(sources.items())},
            "by_negative_kind": {key: stats(value) for key, value in sorted(negatives.items())}}


def compare_baseline(rows: list[dict], baseline: list[dict]) -> dict:
    by_id = {row["id"]: row for row in baseline}
    if len(by_id) != len(baseline) or set(by_id) != {row["id"] for row in rows}:
        raise ValueError("baseline id coverage must exactly match evaluation dataset")
    flips = []
    counts = Counter()
    for row in rows:
        prior = by_id[row["id"]]
        box = prior["prediction"] if "prediction" in prior else prior["bbox"]
        if box is not None:
            try:
                box = validate_normalized_box(box, f"baseline {row['id']}")
            except ValueError:
                box = None  # existing baseline's explicitly invalid prediction counts as wrong
        baseline_iou = independent_iou(box, row["solution"]) if box is not None else None
        before = baseline_iou is not None and baseline_iou >= 0.5
        after = row["samples"][0]["hit"]
        kind = "rescue" if after and not before else "regression" if before and not after else "unchanged"
        counts[kind] += 1
        flips.append({"id": row["id"], "source": row["source"], "negative_kind": row["negative_kind"],
                      "baseline_iou": baseline_iou, "baseline_hit": before,
                      "model_iou": row["samples"][0]["iou"], "model_hit": after, "change": kind})
    return {"counts": dict(counts), "baseline_acc05": sum(r["baseline_hit"] for r in flips) / len(flips),
            "model_acc05": sum(r["model_hit"] for r in flips) / len(flips), "per_query": flips}


def audit_adapter(adapter: Path, *, vllm: bool) -> dict:
    """加载前核对权重与 PEFT 配置；vLLM 不接受超出语言线性层的权重。"""
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config["peft_type"] != "LORA" or config.get("modules_to_save"):
        raise ValueError("only pure LoRA adapter without modules_to_save is supported")
    if config.get("r", 0) > 32 and vllm:
        raise ValueError("vLLM max_lora_rank=32 is smaller than adapter rank")
    from safetensors import safe_open
    files = sorted(adapter.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{adapter}: adapter safetensors missing")
    keys = []
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as tensors:
            keys.extend(tensors.keys())
    lora_keys = [key for key in keys if ".lora_A." in key or ".lora_B." in key]
    if not lora_keys or len(lora_keys) != len(keys):
        raise ValueError("adapter contains non-LoRA weights or no LoRA weights")
    paired: dict[str, set[str]] = defaultdict(set)
    for key in lora_keys:
        module, remainder = key.split(".lora_", 1)
        paired[module.removeprefix("base_model.model.")].add(remainder.split(".", 1)[0])
    if any(sides != {"A", "B"} for sides in paired.values()):
        raise ValueError("adapter has unpaired LoRA A/B weights")
    modules = set(paired)
    targets = config["target_modules"]
    if isinstance(targets, str):
        unmatched = [module for module in modules if re.fullmatch(targets, module) is None]
    else:
        unmatched = [module for module in modules
                     if not any(module == target or module.endswith("." + target) for target in targets)]
    if unmatched:
        raise ValueError(f"adapter tensors are outside target_modules: {unmatched[:3]}")
    if any(not module.startswith("model.language_model.") for module in modules):
        raise ValueError("adapter includes visual/aligner or unknown modules")
    if vllm:
        # Unsupported modules must fail at vLLM load, not be silently removed here.
        if any(module.endswith(("embed_tokens", "lm_head")) for module in modules):
            raise ValueError("vLLM adapter contains embedding or lm_head modules")
    return {"rank": config["r"], "module_count": len(modules), "tensor_count": len(keys),
            "target_leaf_names": sorted({module.rsplit(".", 1)[-1] for module in modules})}


def validate_vllm_targets(adapter_audit: dict) -> None:
    """按当前安装 vLLM 的 Qwen3.5/3.6 打包映射拒绝可能被忽略的 LoRA 模块。"""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    packed = Qwen3_5ForConditionalGeneration.packed_modules_mapping
    supported = {leaf for leaves in packed.values() for leaf in leaves}
    supported.update({"down_proj", "o_proj", "out_proj"})
    missing = set(adapter_audit["target_leaf_names"]) - supported
    if missing:
        raise RuntimeError(f"installed vLLM lacks LoRA mapping for adapter modules: {sorted(missing)}")


def model_quantization(model: str, precision: str) -> None:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model, trust_remote_code=False).to_dict()
    metadata = json.dumps(config.get("quantization_config", {})).lower()
    is_fp8 = "fp8" in metadata or "float8" in metadata
    if precision == "fp8" and not is_fp8:
        raise ValueError("FP8 evaluation requires an official prequantized checkpoint")
    if precision in {"bf16", "nf4"} and config.get("quantization_config"):
        raise ValueError(f"{precision} evaluation requires an unquantized base checkpoint")


def enforce_max_length(engine, max_length: int, max_tokens: int) -> None:
    """Swift 默认缩短生成预算；评价必须在完整预算放不下时直接报错。"""
    engine.template.max_length = max_length
    engine.template.truncation_strategy = "raise"
    engine.max_model_len = max_length
    original = engine.set_default_max_tokens

    def checked(request_config, inputs):
        prompt_tokens = engine._get_num_tokens(inputs)
        if prompt_tokens + max_tokens > max_length:
            raise ValueError(f"input {prompt_tokens} + generation {max_tokens} exceeds max_length {max_length}")
        return original(request_config, inputs)

    engine.set_default_max_tokens = checked


def create_engine(args):
    import torch
    if args.engine == "hf":
        from swift.infer_engine import TransformersEngine
        model_quantization(args.model, args.precision)
        quantization_config = None
        if args.precision == "nf4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=["model.visual", "lm_head", "linear_attn.in_proj_a", "linear_attn.in_proj_b"],
            )
        engine = TransformersEngine(
            args.model, adapters=[str(args.adapter)] if args.adapter else None,
            torch_dtype=torch.bfloat16, quantization_config=quantization_config,
            attn_impl="sdpa", device_map="auto", max_batch_size=args.hf_max_batch_size,
            model_type="qwen3_5", template_type="qwen3_5",
        )
        modules = dict(engine.model.named_modules())
        fourbit = [name for name, module in modules.items() if type(module).__name__ == "Linear4bit"]
        gates = [name for name in modules if name.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b"))]
        visual_linear = [name for name, module in modules.items()
                         if ".visual." in name and isinstance(module, torch.nn.Linear)]
        gate_dtypes = {str(modules[name].get_base_layer().weight.dtype if hasattr(modules[name], "get_base_layer")
                           else modules[name].weight.dtype) for name in gates}
        heads = [module for name, module in modules.items()
                 if name.endswith("lm_head") and hasattr(module, "weight")]
        embeddings = [module for name, module in modules.items()
                      if name.endswith("embed_tokens") and hasattr(module, "weight")]
        if args.precision == "nf4" and (len(fourbit) < 100 or any(".visual." in name for name in fourbit)
                                          or any(name == gate or name.startswith(gate + ".")
                                                 for name in fourbit for gate in gates)
                                          or not gates):
            raise RuntimeError("NF4 quantization coverage differs from training")
        if args.precision == "bf16" and fourbit:
            raise RuntimeError("BF16 model unexpectedly includes 4-bit layers")
        if not visual_linear or gate_dtypes != {"torch.bfloat16"} or not heads or not embeddings:
            raise RuntimeError("sensitive module coverage or dtype differs from training")
        if any(module.weight.dtype != torch.bfloat16 for module in heads + embeddings):
            raise RuntimeError("lm_head/embed_tokens are not BF16")
        if args.adapter:
            lora_modules = [name for name, module in modules.items()
                            if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
            if not lora_modules or any("language_model" not in name for name in lora_modules):
                raise RuntimeError("adapter LoRA was not loaded entirely on language side")
        enforce_max_length(engine, args.max_length, args.max_tokens)
        return engine
    from swift.infer_engine import VllmEngine
    model_quantization(args.model, args.precision)
    engine = VllmEngine(
        args.model, adapters=[str(args.adapter)] if args.adapter else None,
        torch_dtype=torch.bfloat16, max_model_len=args.max_length,
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
        max_lora_rank=32, seed=args.seed,
        model_type="qwen3_5", template_type="qwen3_5",
    )
    enforce_max_length(engine, args.max_length, args.max_tokens)
    return engine


def generate_rows(args, dataset: list[dict]) -> tuple[list[dict], dict]:
    import torch
    from swift.infer_engine import InferRequest, RequestConfig
    if not torch.cuda.is_available():
        raise RuntimeError("generation requires CUDA; use --predictions for CPU rescoring")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.adapter:
        adapter_audit = audit_adapter(args.adapter, vllm=args.engine == "vllm")
        if args.engine == "vllm":
            validate_vllm_targets(adapter_audit)
    else:
        adapter_audit = None
    os.environ["MIN_PIXELS"] = str(args.min_pixels)
    os.environ["MAX_PIXELS"] = str(args.max_pixels)
    engine = create_engine(args)
    thinking_hook = None
    if getattr(args, "bounded_thinking", False):
        from tools.quant_grounding_thinking import BoundedHFGenerationHook
        thinking_hook = BoundedHFGenerationHook(
            engine, thought_limit=args.thought_limit, answer_limit=args.answer_limit,
            total_limit=args.max_tokens,
            language_autocast_factory=lambda: torch.autocast(
                device_type="cuda", dtype=torch.bfloat16)).install()
    if args.engine == "hf" and adapter_audit:
        loaded_lora = sum(hasattr(module, "lora_A") and hasattr(module, "lora_B")
                          for module in engine.model.modules())
        if loaded_lora != adapter_audit["module_count"]:
            raise RuntimeError(f"HF adapter module coverage mismatch: saved={adapter_audit['module_count']}, "
                               f"loaded={loaded_lora}")
    request_config = RequestConfig(
        max_tokens=args.max_tokens, n=1 if args.engine == "hf" else args.samples_per_query,
        temperature=0 if args.samples_per_query == 1 else 1,
        top_p=1, top_k=-1, seed=args.seed, stop=[], return_details=True,
    )
    if args.engine == "hf":
        effective_sampling = configure_unrestricted_hf_sampling(engine, request_config)
    else:
        effective = engine._prepare_generation_config(request_config)
        effective_sampling = {"temperature": effective.temperature, "top_p": effective.top_p,
                              "top_k": effective.top_k, "do_sample": effective.temperature > 0}
    generated = []
    wall_start = time.monotonic()
    max_observed_device_used = 0
    raw_path = args.output_dir / "predictions_raw.jsonl"
    # Only greedy HF evaluation shares an infer call across different Queries.
    # G8 keeps eight n=1 requests for one Query; vLLM keeps its existing path.
    batch_size = args.hf_max_batch_size if args.engine == "hf" and args.samples_per_query == 1 else 1
    with raw_path.open("x", encoding="utf-8") as raw_out:
      for offset in range(0, len(dataset), batch_size):
        batch = dataset[offset:offset + batch_size]
        requests = []
        for row in batch:
            messages = deepcopy(row["messages"])
            if thinking_hook is not None:
                from tools.quant_grounding_thinking import apply_bounded_thinking_prompt
                messages = apply_bounded_thinking_prompt(messages)
            images = [str((args.data_root / image).resolve()) for image in row["images"]]
            missing = [path for path in images if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"{row['id']}: {missing}")
            if sum(message["content"].count("<image>") for message in row["messages"]) != len(images):
                raise ValueError(f"{row['id']}: image markers do not match images")
            # HF n=8 expands one request to eight simultaneous generate sequences;
            # repeat requests with n=1 so the measured cap bounds the caches.
            request_count = args.samples_per_query if args.engine == "hf" else 1
            requests.extend(InferRequest(
                messages=deepcopy(messages), images=images.copy(),
                chat_template_kwargs={"enable_thinking": args.answer_mode == "thinking"},
            ) for _ in range(request_count))
        started = time.monotonic()
        responses = engine.infer(requests, request_config, use_tqdm=False)
        traces = thinking_hook.pop_traces() if thinking_hook is not None else None
        if traces is not None and len(traces) != len(responses):
            raise RuntimeError("bounded thinking traces do not match HF responses")
        elapsed = time.monotonic() - started
        if len(responses) != len(requests):
            raise RuntimeError(f"{batch[0]['id']}: generation response count differs from requests")
        # Swift infer returns responses in request order. Each HF request has n=1.
        for index, row in enumerate(batch):
            if args.engine == "hf":
                group = responses[index * args.samples_per_query:(index + 1) * args.samples_per_query]
                if any(len(response.choices) != 1 for response in group):
                    raise RuntimeError(f"{row['id']}: HF request must return one choice")
                choices = [response.choices[0] for response in group]
            else:
                choices = responses[index].choices
            if len(choices) != args.samples_per_query:
                raise RuntimeError(f"{row['id']}: generation count differs from requested G")
            samples = []
            for sample_index, choice in enumerate(choices):
                if choice.message.reasoning_content is not None:
                    raise RuntimeError("reasoning parser separated raw thinking text; disable parser for strict scoring")
                if choice.token_ids is None or choice.finish_reason is None:
                    raise RuntimeError("generation omitted token IDs or finish_reason")
                # Decode only generated token IDs, excluding Swift's synthetic prompt prefix.
                text = engine.template.decode_generate_ids(choice.token_ids, first_token=False)
                sample = {"text": text, "finish_reason": choice.finish_reason,
                                "tokens": len(choice.token_ids),
                                "seconds": elapsed / (len(batch) * args.samples_per_query)}
                if traces is not None:
                    trace = traces[index * args.samples_per_query + sample_index]
                    if tuple(choice.token_ids) != trace.token_ids:
                        raise RuntimeError("Swift response changed raw bounded generation tokens")
                    sample["thinking_trace"] = asdict(trace)
                    sample["token_ids"] = list(trace.token_ids)
                    if trace.answer_truncated or not trace.complete:
                        sample["finish_reason"] = "length"
                samples.append(sample)
            raw_record = {"id": row["id"], "samples": samples}
            generated.append(raw_record)
            raw_out.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            raw_out.flush()
        free, total = torch.cuda.mem_get_info()
        max_observed_device_used = max(max_observed_device_used, total - free)
    if thinking_hook is not None:
        thinking_hook.restore()
    return generated, {"generation_seconds": time.monotonic() - wall_start,
                       "effective_sampling": effective_sampling,
                       "sample_seconds_semantics": "infer call wall time divided equally among its samples",
                       "hf_max_batch_size": engine.max_batch_size if args.engine == "hf" else None,
                       "max_observed_device_used_bytes": max_observed_device_used,
                       "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "bounded_thinking_amp": thinking_hook.amp_audit if thinking_hook is not None else None,
                       "adapter_audit": adapter_audit}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--predictions", type=Path, help="只复算保存的原始生成，不加载 GPU")
    parser.add_argument("--baseline", type=Path, help="逐 Query 比较已有 normalized prediction/bbox JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--precision", choices=["nf4", "bf16", "fp8"], default="nf4")
    parser.add_argument("--answer-mode", choices=["short", "thinking"], default="short")
    parser.add_argument("--reward-mode", choices=["contrast", "acc05"], default="contrast")
    parser.add_argument("--format-policy", choices=["strict", "punctuation", "single_box"], default="strict")
    parser.add_argument("--bounded-thinking", action="store_true",
                        help="D 修复：分配独立思考与答案词元预算，共用训练控制逻辑")
    parser.add_argument("--thought-limit", type=int, default=256)
    parser.add_argument("--answer-limit", type=int, default=64)
    parser.add_argument("--samples-per-query", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1204224)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--hf-max-batch-size", type=int, choices=[1, 2], default=1,
                        help="HF concurrent requests; 2 is for measured memory/throughput comparison")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.bounded_thinking and (args.engine != "hf" or args.answer_mode != "thinking"):
        raise ValueError("bounded-thinking requires HF and thinking answer mode")
    if args.samples_per_query < 1 or args.limit < 0:
        raise ValueError("samples-per-query must be positive and limit must be nonnegative")
    if args.engine == "hf" and args.precision == "fp8":
        raise ValueError("FP8 deployment comparison uses vLLM only")
    if args.engine == "vllm" and args.precision == "nf4":
        raise ValueError("NF4 training comparison uses HF only")
    if not args.predictions and (not args.model or not args.data_root):
        raise ValueError("generation requires --model and --data-root")
    if args.max_tokens == 0:
        args.max_tokens = (args.thought_limit + args.answer_limit + 2 if args.bounded_thinking
                           else 384 if args.answer_mode == "thinking" else 64)
    if args.bounded_thinking and (min(args.thought_limit, args.answer_limit) <= 0 or
                                 args.max_tokens < args.thought_limit + args.answer_limit + 2):
        raise ValueError("bounded thinking budgets must be positive and fit max-tokens")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if not 0 < args.min_pixels <= args.max_pixels or args.max_length <= args.max_tokens:
        raise ValueError("pixel limits or max_length/max_tokens are inconsistent")
    dataset = read_jsonl(args.dataset)
    if args.limit:
        dataset = dataset[:args.limit]
    if not dataset or len({row["id"] for row in dataset}) != len(dataset):
        raise ValueError("dataset empty or duplicate id")
    if args.predictions:
        predictions = read_jsonl(args.predictions)
        resource = {"rescore_only": True}
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions, resource = generate_rows(args, dataset)
    scored = score_rows(dataset, predictions, args.answer_mode, args.reward_mode, args.format_policy)
    summary = summarize(scored)
    summary.update({"answer_mode": args.answer_mode, "reward_mode": args.reward_mode,
                    "format_policy": args.format_policy,
                    "samples_per_query": len(scored[0]["samples"]), "resource": resource})
    if args.bounded_thinking:
        summary["bounded_thinking"] = {"thought_limit": args.thought_limit,
                                       "answer_limit": args.answer_limit, "max_tokens": args.max_tokens}
    if args.baseline:
        summary["baseline_comparison"] = compare_baseline(scored, read_jsonl(args.baseline))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "predictions.jsonl", scored)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
