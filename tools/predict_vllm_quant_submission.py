"""Run prepared grounding queries with vLLM NF4 and LoRA, preserving raw answers.

与 `predict_submission_vllm.py`（复赛 BF16 提交工具）同源分叉。本文件供半决赛 NF4 使用。

2026-10-30 审计后收紧了两处：
  * `--max-pixels` / `--max-length` / `--upscale` **没有默认值**。此前它们默认成开发集
    档位（1204224 / 8192 / 1.0），漏传即静默降配——10-29 的半决赛提交正是这样从
    0.8267 掉到 0.8014 的。
  * 放大倍数必须与像素上限配套，否则处理器会把放大后的图缩回上限，放大成为空操作，
    而日志、耗时、显存全程不报错。

改这里的护栏时请同时看复赛那侧。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import completed_prediction_prefix, positive_int, read_jsonl

# 赛题的图像原生分辨率上限，用于校验放大倍数与像素上限是否配套。
NATIVE_PIXELS = 1920 * 1080


def build_requests(rows: list[dict], data_root: Path, processor, upscale: float = 1.0) -> list[dict]:
    from PIL import Image

    requests = []
    for row in rows:
        if len(row["messages"]) != 1 or row["messages"][0]["role"] != "user":
            raise ValueError(f"{row['id']}: expected one native user message")
        images = []
        for path in row["images"]:
            with Image.open(data_root / path) as image:
                rgb = image.convert("RGB")
                if upscale != 1.0:
                    rgb = rgb.resize((round(rgb.width * upscale), round(rgb.height * upscale)), Image.Resampling.LANCZOS)
                images.append(rgb.copy())
        chunks = row["messages"][0]["content"].split("<image>")
        if len(chunks) != len(images) + 1:
            raise ValueError(f"{row['id']}: image placeholders differ from image count")
        content = []
        for index, chunk in enumerate(chunks):
            if chunk:
                content.append({"type": "text", "text": chunk})
            if index < len(images):
                content.append({"type": "image", "image": images[index]})
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
    return requests


def build_llm_options(config: dict, max_num_seqs: int = 8) -> dict:
    # 后面三项曾是一次 15 分钟探针的临时设置，被当成规范硬编码了下来。现在改为可覆盖，
    # 缺省值保持原样，避免改变既有运行与历史配置的复现结果。
    return dict(
        model=config["model"], dtype="bfloat16", max_model_len=config["max_length"],
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=config.get("max_num_batched_tokens", 8192),
        gpu_memory_utilization=0.90, enable_lora=True, max_lora_rank=32,
        enable_prefix_caching=config.get("enable_prefix_caching", False),
        mamba_cache_mode=config.get("mamba_cache_mode", "none"),
        enforce_eager=config.get("enforce_eager", True),
        limit_mm_per_prompt={"image": 3}, seed=config["seed"],
        mm_processor_kwargs={"min_pixels": config["min_pixels"], "max_pixels": config["max_pixels"]},
        quantization="bitsandbytes", load_format="bitsandbytes",
        hf_overrides={"quantization_config": {
            "quant_method": "bitsandbytes", "load_in_4bit": True, "load_in_8bit": False,
            "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "llm_int8_skip_modules": ["visual", "lm_head", "in_proj_ba"],
        }},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-size", type=positive_int, default=64)
    parser.add_argument("--max-num-seqs", type=positive_int, default=8)
    parser.add_argument("--min-pixels", type=positive_int, default=200704)
    # 分辨率三参数不给默认值。此前它们默认成开发集档位（1204224 / 8192 / 1.0），
    # 漏传即静默降配，10-29 的半决赛提交就是这样掉到 0.8014 的。必须显式声明。
    parser.add_argument("--max-pixels", type=positive_int, required=True)
    parser.add_argument("--max-length", type=positive_int, required=True)
    parser.add_argument("--max-tokens", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--upscale", type=float, required=True)
    # 自一致性采样投票：k>1 时 vLLM 用一次预填充产 k 个样本（官方题输出只有 22 token，
    # 增量成本约 k×解码）。k=1 且未指定温度时行为与旧版逐字一致。
    parser.add_argument("--samples", type=positive_int, default=1,
                        help="每题采样数；>1 需配合 --temperature > 0")
    parser.add_argument("--temperature", type=float, default=None,
                        help="生成温度；不给则 k=1 用 0、k>1 用 1（与评测工具同口径）")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_chunks(args, dataset: list[dict], completed: list[dict], llm, processor, sampling, lora) -> dict:
    raw_path = args.output_dir / "predictions_raw.jsonl"
    generation_seconds = 0.0
    prepared_seconds = 0.0
    generated_tokens = 0
    count = len(completed)
    samples_per_query = getattr(args, "samples", 1)
    started = time.monotonic()
    with raw_path.open("a" if args.resume else "x", encoding="utf-8") as target:
        for offset in range(count, len(dataset), args.chunk_size):
            chunk = dataset[offset:offset + args.chunk_size]
            before = time.monotonic()
            requests = build_requests(chunk, args.data_root, processor, getattr(args, "upscale", 1.0))
            prepared_seconds += time.monotonic() - before
            before = time.monotonic()
            outputs = llm.generate(requests, sampling, lora_request=lora, use_tqdm=False)
            elapsed = time.monotonic() - before
            generation_seconds += elapsed
            if len(outputs) != len(chunk):
                raise RuntimeError("vLLM response count differs from submitted queries")
            for row, output in zip(chunk, outputs):
                # n>1 时 vLLM 用一次预填充产 k 个样本——这正是自一致性采样在成本上可行的原因：
                # 官方题的输入中位 24,236 token、输出中位 22 token，增量只有 k×解码。
                if len(output.outputs) != samples_per_query:
                    raise RuntimeError(f"{row['id']}: expected {samples_per_query} completions, "
                                       f"got {len(output.outputs)}")
                samples = []
                for answer in output.outputs:
                    if answer.finish_reason is None:
                        raise RuntimeError(f"{row['id']}: missing finish reason")
                    samples.append({"text": answer.text, "tokens": len(answer.token_ids),
                                    "finish_reason": answer.finish_reason,
                                    "seconds": elapsed / len(chunk) / samples_per_query})
                    generated_tokens += len(answer.token_ids)
                record = {"id": row["id"], "prompt_token_ids": list(output.prompt_token_ids),
                          "samples": samples}
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                count += 1
            print(json.dumps({"completed_queries": count, "expected_queries": len(dataset),
                              "generation_seconds": generation_seconds}, ensure_ascii=False), flush=True)
            del requests, outputs
    return {"completed_queries": count, "resumed_queries": len(completed), "new_queries": count - len(completed),
            "generation_seconds": generation_seconds, "preparation_seconds": prepared_seconds,
            "wall_seconds": time.monotonic() - started, "new_generated_tokens": generated_tokens,
            "samples_per_query": samples_per_query,
            "sample_seconds_semantics": "generate call wall time divided equally among its queries and samples"}


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.upscale <= 0 or args.limit < 0 or args.min_pixels > args.max_pixels or args.max_tokens >= args.max_length:
        raise ValueError("inconsistent query/pixel/token limits")
    # 温度与采样数的耦合：温度 0 采 k 个样本只会得到 k 份相同结果，聚合等于没做。
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        raise ValueError("temperature must be within [0, 2]")
    effective_temperature = (0.0 if args.samples == 1 else 1.0) \
        if args.temperature is None else args.temperature
    if args.samples > 1 and effective_temperature == 0:
        raise ValueError("--samples > 1 requires --temperature > 0; "
                         "identical samples make consensus aggregation meaningless")
    if args.samples == 1 and effective_temperature > 0:
        raise ValueError("--temperature > 0 with --samples 1 is not reproducible submission output")
    if args.samples > 1 and args.resume:
        raise ValueError("--resume only supports --samples 1; sample counts are not recoverable "
                         "from a partially written raw file")
    # 放大后的图像若超过 max_pixels，处理器会把它缩回该上限，放大就静默失效了：
    # 付了 LANCZOS 的 CPU 代价，模型收到的仍是低分辨率图，且日志、耗时、显存都不报错。
    if args.upscale > 1.0:
        required = args.upscale ** 2 * NATIVE_PIXELS * 0.95
        if args.max_pixels < required:
            raise ValueError(
                f"upscale {args.upscale} needs --max-pixels >= {round(required)} "
                f"(got {args.max_pixels}); the upscaled images would be downscaled back, "
                f"making the upscale a no-op"
            )
    args.output_dir = args.output_dir.resolve()
    args.dataset = args.dataset.resolve()
    args.data_root = args.data_root.resolve()
    args.adapter = args.adapter.resolve()
    dataset = read_jsonl(args.dataset)
    if args.limit:
        dataset = dataset[:args.limit]
    if not dataset or len({row["id"] for row in dataset}) != len(dataset):
        raise ValueError("empty dataset or duplicate query IDs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "predictions_raw.jsonl"
    if raw_path.exists() and not args.resume:
        raise FileExistsError("raw predictions already exist; use --resume or a fresh directory")
    completed = completed_prediction_prefix(raw_path, dataset) if args.resume else []
    config = json.loads(json.dumps(vars(args), default=str))
    # samples_per_query 记实际生效的采样数；temperature 记生效温度而不是 None，
    # 否则事后无法从 run_config 复现到底用了什么温度。
    config.update(engine="vllm", precision="nf4", answer_mode="short",
                  samples_per_query=args.samples, temperature=effective_temperature)
    config["llm_options"] = build_llm_options(config, args.max_num_seqs)
    config_path = args.output_dir / "run_config.json"
    if args.resume:
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("dataset", "data_root", "model", "adapter", "min_pixels", "max_pixels", "max_length",
                    "max_tokens", "seed", "llm_options", "upscale"):
            if config[key] != (saved.get(key, 1.0) if key == "upscale" else saved[key]):
                raise ValueError(f"resume configuration differs: {key}")
        saved.setdefault("resume_history", []).append({"completed_queries": len(completed), "chunk_size": args.chunk_size})
        config = saved
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path = args.output_dir / "summary.json"
    summary = {"status": "loading", "expected_queries": len(dataset), "resumed_queries": len(completed)}
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["previous_runs"] = [*previous.pop("previous_runs", []), previous]

    def save(**values):
        summary.update(values)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if len(completed) == len(dataset):
        save(status="completed", completed_queries=len(completed), new_queries=0, generation_seconds=0.0)
        return summary
    save()
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch

    try:
        processor = AutoProcessor.from_pretrained(args.model)
        before = time.monotonic()
        llm = LLM(**config["llm_options"])
        save(status="generating", loading_seconds=time.monotonic() - before)
        sampling = SamplingParams(temperature=effective_temperature, n=args.samples,
                                  max_tokens=args.max_tokens, seed=args.seed)
        lora = LoRARequest("grounding", 1, str(args.adapter))
        resource = run_chunks(args, dataset, completed, llm, processor, sampling, lora)
        save(status="completed", **resource, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        save(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2), flush=True)
