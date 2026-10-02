"""用 vLLM 离线批处理跑官方提交，输出与 `predict_native_submission.py` 完全同构。

为什么需要它：HF 逐条 `generate` 在 27B + 原生分辨率下约 6.3 秒/条（预填充 3.67 s +
解码 2.59 s），5690 条要 10 小时。解码是显存带宽受限的，vLLM 的连续批处理能让一次权重
读取服务多条序列，同时它的 Gated DeltaNet kernel 比 transformers 的兜底实现快得多。

**安全设计（两阶段）**：零样本 27B 会在 bbox 之后重复输出查询文本作为 label，而 vLLM
的离线接口不支持自定义 StoppingCriteria。因此：
  阶段 1：`stop=["]"]` —— 在第一个 `]` 处停下。对正常输出而言，那正是 `bbox_2d` 数组的
          收尾括号，省掉后面整段冗余 label；
  阶段 2：**凡是阶段 1 结果不满足"含 bbox_2d 键且能解析出合法框"的，一律禁用 stop、
          用完整预算重跑**。
这样"提前停止"只可能发生在已经拿到合法答案之后，不可能引入静默错误；两阶段合起来
等价于一次性用完整预算跑，但绝大多数样本省掉了冗余解码。

判据与 `BboxCompleteCriteria` 完全一致（同一个 `parse_generated_bbox`），因此 HF 与 vLLM
两条路径的"可接受输出"定义相同。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import (  # noqa: E402
    BBOX_KEYED_PATTERN,
    build_user_content,
    load_prediction_rows,
    parse_generated_bbox,
    verify_run_config,
)
from tools.predict_native_submission import (  # noqa: E402
    FALLBACK_BOX,
    IMAGE_FIELDS,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    PROMPTS,
    build_run_config,
    image_paths,
    load_modal_images,
    load_queries,
    package_submission,
    summarize,
    validate_rows,
)

STAGE1_MAX_TOKENS = 64


def accept(text: str) -> list[float] | None:
    """与 HF 侧 `BboxCompleteCriteria` 相同的接受判据：含键且能解析出合法框。"""
    if BBOX_KEYED_PATTERN.search(text) is None:
        return None
    return parse_generated_bbox(text)


def build_requests(processor_tokenizer, items, data_root: Path, queries: Path,
                   modality: str) -> list[dict[str, Any]]:
    # 注意：对 **tokenizer** 的 apply_chat_template，模板变量要作为**直接关键字参数**传入。
    # 曾写成 chat_template_kwargs={"enable_thinking": False}（那是 processor / vLLM API 的形式），
    # 结果 enable_thinking 被静默忽略、thinking 保持开启：200 条全部生成思维链并在 token 上限
    # 处被截断，一个 bbox 都解析不出来。闸门 2 抓到了它——否则会交出一份全是兜底框的废提交。
    # 这里的写法与 HF 路径 (predict_native_submission.py 的 **chat_template_kwargs) 完全一致。
    requests = []
    for index, sample_id, record in items:
        paths = image_paths(record, data_root, queries)
        images = load_modal_images(paths, modality)
        prompt = PROMPTS[modality](record["query"])
        messages = [{"role": "user", "content": build_user_content(
            prompt, images, prompt_has_image_placeholders=True)}]
        text = processor_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not text.rstrip().endswith("</think>"):
            raise ValueError(
                f"{sample_id}: 生成的提示词未以 </think> 结尾，thinking 可能没有被关掉；"
                "这会导致模型输出思维链而不是 bbox")
        requests.append({
            "index": index, "id": sample_id, "query": record["query"],
            "image_paths": paths, "prompt": prompt,
            "request": {"prompt": text, "multi_modal_data": {"image": images}},
        })
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, help="可选：vLLM LoRA 适配器；本流程默认用合并后的基座")
    parser.add_argument("--modality", choices=PROMPTS, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=5690)
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--model-max-length", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="并发序列上限。Qwen3.6 的线性注意力层每个解码序列要占一个 Mamba "
                             "cache block，本机可用约 397 块；vLLM 默认 1024 会直接报 "
                             "'max_num_seqs exceeds available Mamba cache blocks' 而启动失败。")
    parser.add_argument("--mamba-cache-mode", default="none",
                        help="混合架构（Qwen3.6 有 48/64 层线性注意力）的前缀缓存需要设为 align 才生效；"
                             "vLLM 默认是 none，等于把 KV 前缀缓存关掉了。设为 align 后必须做"
                             "开/关的逐题等价性比对——缓存只能改变速度，不能改变结果。")
    parser.add_argument("--no-prefix-caching", action="store_true",
                        help="显式关闭 KV 前缀缓存（vLLM 该版本默认是开启的）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条；不打包")
    parser.add_argument("--adapter", type=Path,
                        help="PEFT LoRA 适配器目录。vLLM 离线接口原生支持 LoRA，因此**不需要**"
                             "先 merge_and_unload 合并——省掉一次 54GB 写盘，也避免合并带来的数值变化。"
                             "但必须验证 vLLM 确实把适配器应用到了全部目标模块（我们用的是 "
                             "all-linear，含 48 层线性注意力），否则会静默产出只适配了一部分的结果。")
    parser.add_argument("--max-lora-rank", type=int, default=32,
                        help="必须 >= 适配器的 r。默认 r=32；vLLM 的 LoRAConfig 默认只有 16。")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="每块处理的查询数。分块是必需的：一次性物化全部请求会把"
                             "数万张图读进内存并在生成前耗掉大量时间。")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()

    queries = args.queries.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    source = load_queries(queries, args.expected_query_count)
    config_path = output_dir / "run_config.json"
    rows_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    config = build_run_config(args.model, args.adapter, queries, data_root, args.modality,
                              args.min_pixels, args.max_pixels, args.model_max_length, True)
    config["engine"] = "vllm"
    config["stage1_max_tokens"] = STAGE1_MAX_TOKENS
    config["max_new_tokens"] = MAX_NEW_TOKENS
    config["max_num_seqs"] = args.max_num_seqs
    config["mamba_cache_mode"] = args.mamba_cache_mode
    config["enable_prefix_caching"] = not args.no_prefix_caching
    config["lora_adapter"] = str(adapter_path) if adapter_path else None
    config["max_lora_rank"] = args.max_lora_rank
    config["chunk_size"] = args.chunk_size

    if args.package_only:
        saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
        verify_run_config(saved, config)
        rows_by_id = validate_rows(load_prediction_rows(rows_path), source, data_root,
                                   queries, args.modality)
        json_path, zip_path = package_submission(source, rows_by_id, output_dir)
        ordered = [rows_by_id[sample_id] for sample_id in source]
        previous = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        summary = {**config, **summarize(ordered, len(source),
                                        int(previous.get("gpu_peak_allocated_bytes", 0)),
                                        int(previous.get("gpu_peak_reserved_bytes", 0))),
                   "predictions_jsonl": str(rows_path), "predictions_json": str(json_path),
                   "submission_zip": str(zip_path)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = list(source.items())[:args.limit] if args.limit else list(source.items())
    items = [(index, sample_id, record) for index, (sample_id, record) in enumerate(selected)]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_path = args.adapter.resolve() if args.adapter else None
    if adapter_path is not None and not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter_path}")

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.model_max_length,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=not args.no_prefix_caching,
              mamba_cache_mode=args.mamba_cache_mode,
              enable_lora=adapter_path is not None,
              max_lora_rank=args.max_lora_rank,
              limit_mm_per_prompt={"image": len(IMAGE_FIELDS) if args.modality == "trimodal" else 1},
              mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})
    tokenizer = llm.get_tokenizer()
    lora_request = (LoRARequest("trained", 1, str(adapter_path))
                    if adapter_path is not None else None)
    if lora_request is not None:
        print(f"启用 LoRA 适配器: {adapter_path}（max_lora_rank={args.max_lora_rank}）", flush=True)

    # 分块处理：**必须分块**。曾把 5690 条请求（含 17070 张图）一次性物化再调一次
    # llm.generate()，结果引擎初始化完成后 GPU 长时间 0% 占用、系统内存持续上涨——
    # 全部时间花在生成开始前把图像读进内存并预处理。分块后：
    #   * 峰值内存有界（每块 chunk_size × 3 张图）；
    #   * 每块结束就落盘并打印进度，出问题能立刻看见；
    #   * 块与块之间释放图像对象，不再累积。
    import gc

    stage1 = SamplingParams(temperature=0.0, max_tokens=STAGE1_MAX_TOKENS,
                            stop=["]"], include_stop_str_in_output=True,
                            skip_special_tokens=True)
    stage2 = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                            skip_special_tokens=True)

    rows_by_id: dict[str, dict[str, Any]] = {}
    retry: list[dict[str, Any]] = []
    chunks = [items[start:start + args.chunk_size]
              for start in range(0, len(items), args.chunk_size)]
    started = time.perf_counter()

    # 逐块**追加落盘**：块是 items 的连续切片、块内按序处理，因此追加写入天然保持
    # 官方要求的 QueryID 顺序。这样即使中途挂掉，已完成的部分仍在磁盘上，
    # 不至于"跑到第 80 分钟全丢"。
    handle = rows_path.open("w", encoding="utf-8")

    def emit(row: dict[str, Any]) -> None:
        rows_by_id[row["id"]] = row
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for chunk_index, chunk in enumerate(chunks, 1):
        requests = build_requests(tokenizer, chunk, data_root, queries, args.modality)
        outputs = llm.generate([item["request"] for item in requests], stage1,
                               use_tqdm=False, lora_request=lora_request)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text.strip()
            prediction = accept(text)
            if prediction is None:
                retry.append(item)
                continue
            emit({
                "index": item["index"], "id": item["id"], "query": item["query"],
                "image_paths": item["image_paths"], "prompt": item["prompt"],
                "bbox": prediction, "parsed": True, "fallback": False,
                "generated_tokens": len(output.outputs[0].token_ids),
                "input_tokens": len(output.prompt_token_ids),
                "image_grid_thw": None,
                "generation_cap_hit": output.outputs[0].finish_reason == "length",
                "latency_seconds": 0.0, "raw_text": text, "stage": 1,
            })
        handle.flush()
        done = min(chunk_index * args.chunk_size, len(items))
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0
        remaining = (len(items) - done) / rate / 60 if rate else 0
        print(f"[阶段1 {chunk_index}/{len(chunks)}] 已完成 {done}/{len(items)} "
              f"({len(rows_by_id)} 成功)  {elapsed / 60:.1f} 分钟已用  "
              f"剩余约 {remaining:.0f} 分钟", flush=True)
        del requests, outputs
        gc.collect()

    print(f"阶段1 直接成功 {len(rows_by_id)}，需重跑 {len(retry)}", flush=True)
    for chunk_index, chunk in enumerate(
            [retry[start:start + args.chunk_size]
             for start in range(0, len(retry), args.chunk_size)], 1):
        requests = build_requests(tokenizer, chunk, data_root, queries, args.modality)
        outputs = llm.generate([item["request"] for item in requests], stage2,
                               use_tqdm=False, lora_request=lora_request)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text.strip()
            prediction = parse_generated_bbox(text)
            fallback = prediction is None
            emit({
                "index": item["index"], "id": item["id"], "query": item["query"],
                "image_paths": item["image_paths"], "prompt": item["prompt"],
                "bbox": FALLBACK_BOX.copy() if fallback else prediction,
                "parsed": not fallback, "fallback": fallback,
                "generated_tokens": len(output.outputs[0].token_ids),
                "input_tokens": len(output.prompt_token_ids),
                "image_grid_thw": None,
                "generation_cap_hit": output.outputs[0].finish_reason == "length",
                "latency_seconds": 0.0, "raw_text": text, "stage": 2,
            })
        handle.flush()
        print(f"[阶段2 {chunk_index}] 已完成 {len(rows_by_id)}/{len(items)}", flush=True)
        del requests, outputs
        gc.collect()
    handle.close()
    print(f"已写出 {rows_path}（逐块追加，无需重写）", flush=True)

    ordered = [rows_by_id[sample_id] for _, sample_id, _ in items]
    summary = {**config, **summarize(ordered, len(source), 0, 0),
               "limit": args.limit, "predictions_jsonl": str(rows_path),
               "stage1_success": sum(1 for row in ordered if row["stage"] == 1),
               "stage2_retries": sum(1 for row in ordered if row["stage"] == 2)}
    if not args.limit and len(rows_by_id) == len(source):
        json_path, zip_path = package_submission(source, rows_by_id, output_dir)
        summary.update(predictions_json=str(json_path), submission_zip=str(zip_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
