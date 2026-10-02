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
    requests = []
    for index, sample_id, record in items:
        paths = image_paths(record, data_root, queries)
        images = load_modal_images(paths, modality)
        prompt = PROMPTS[modality](record["query"])
        messages = [{"role": "user", "content": build_user_content(
            prompt, images, prompt_has_image_placeholders=True)}]
        text = processor_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False})
        requests.append({
            "index": index, "id": sample_id, "query": record["query"],
            "image_paths": paths, "prompt": prompt,
            "request": {"prompt": text, "multi_modal_data": {"image": images}},
        })
    return requests


def run_stage(llm, sampling_params, requests, label: str) -> list[Any]:
    started = time.perf_counter()
    outputs = llm.generate([item["request"] for item in requests], sampling_params,
                           use_tqdm=True)
    print(f"[{label}] {len(requests)} 条，用时 {time.perf_counter() - started:.1f} 秒", flush=True)
    return outputs


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
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条；不打包")
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

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.model_max_length,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": len(IMAGE_FIELDS) if args.modality == "trimodal" else 1},
              mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})
    tokenizer = llm.get_tokenizer()

    requests = build_requests(tokenizer, items, data_root, queries, args.modality)
    stage1 = SamplingParams(temperature=0.0, max_tokens=STAGE1_MAX_TOKENS,
                            stop=["]"], include_stop_str_in_output=True,
                            skip_special_tokens=True)
    outputs = run_stage(llm, stage1, requests, "阶段1 首括号即停")

    rows_by_id: dict[str, dict[str, Any]] = {}
    retry: list[dict[str, Any]] = []
    for item, output in zip(requests, outputs):
        text = output.outputs[0].text.strip()
        prediction = accept(text)
        if prediction is None:
            retry.append(item)
            continue
        rows_by_id[item["id"]] = {
            "index": item["index"], "id": item["id"], "query": item["query"],
            "image_paths": item["image_paths"], "prompt": item["prompt"],
            "bbox": prediction, "parsed": True, "fallback": False,
            "generated_tokens": len(output.outputs[0].token_ids),
            "input_tokens": len(output.prompt_token_ids),
            "image_grid_thw": None,
            "generation_cap_hit": output.outputs[0].finish_reason == "length",
            "latency_seconds": 0.0, "raw_text": text, "stage": 1,
        }

    print(f"阶段1 直接成功 {len(rows_by_id)}，需重跑 {len(retry)}", flush=True)
    if retry:
        stage2 = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                                skip_special_tokens=True)
        outputs2 = run_stage(llm, stage2, retry, "阶段2 完整预算重跑")
        for item, output in zip(retry, outputs2):
            text = output.outputs[0].text.strip()
            prediction = parse_generated_bbox(text)
            fallback = prediction is None
            rows_by_id[item["id"]] = {
                "index": item["index"], "id": item["id"], "query": item["query"],
                "image_paths": item["image_paths"], "prompt": item["prompt"],
                "bbox": FALLBACK_BOX.copy() if fallback else prediction,
                "parsed": not fallback, "fallback": fallback,
                "generated_tokens": len(output.outputs[0].token_ids),
                "input_tokens": len(output.prompt_token_ids),
                "image_grid_thw": None,
                "generation_cap_hit": output.outputs[0].finish_reason == "length",
                "latency_seconds": 0.0, "raw_text": text, "stage": 2,
            }

    with rows_path.open("w", encoding="utf-8") as handle:
        for index, sample_id, _ in items:
            handle.write(json.dumps(rows_by_id[sample_id], ensure_ascii=False) + "\n")
    print(f"已写出 {rows_path}", flush=True)

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
