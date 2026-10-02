"""多分辨率 / 多基座本地定位探针（云端 GPU 运行）。

背景（2026-10-02 实测）：保留基线 A 在 max_pixels=602112 时，每图只有 576 个合并
视觉 token；116 个失败题的目标面积中位数折合 **0.88 个视觉 token**，30 个"框住同图
另一个被标注实例"的案例中，本题 GT 与邻居 GT 的中心距离中位数只有 **2.31 个 token**。
本工具用**同一份权重、同一份 prompt、同一份解析与评分**，只改每图像素预算（和可选
的基座），测这件事到底值多少分。

它复用 evaluate_pretrained_grounder 的 prompt 构造、框解析、IoU 评分与汇总函数，
不修改已验证的基线评估路径；输出 predictions.jsonl 与基线同 schema，可直接交给
tools/verify_acc05_dev.py 复算。

用法（云端 GPU）：
    python tools/probe_native_resolution.py \
        --model /root/rematch_models/Qwen3.5-9B \
        --manifest <city_val.json> --target-manifest <city_gt.json> \
        --data-root <.../data/city/train> \
        --output-dir <out_dir> \
        --pixels 602112 1204224 2073600 \
        --model-max-length 16384 --no-thinking

带 LoRA 适配器（例如把已验证的 A 权重直接放到高分辨率下测）：
    ... --adapter <A/main/checkpoint-600> --pixels 602112 1204224
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import (  # noqa: E402
    apply_target_manifest,
    build_prompt,
    build_user_content,
    configure_torch_precision,
    effective_prompt_style,
    load_model_images,
    load_records,
    parse_generated_bbox,
    resolve_image_path,
    score_prediction,
    summarize_rows,
)
from tools.native_model_loading import resolve_model_class  # noqa: E402

# 与项目现有云端配置一致；只在这里改像素与上下文，其余保持基线口径。
MIN_PIXELS = 200704
DEFAULT_MAX_NEW_TOKENS = 128


def load_model(model_name: str, adapter: Path | None):
    configure_torch_precision()
    model_class, class_name = resolve_model_class(model_name)
    model = model_class.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, adapter, is_trainable=False, autocast_adapter_dtype=False
        )
    model = model.eval().to(device="cuda", dtype=torch.bfloat16)
    return model, class_name


def build_processor(model_name: str, max_pixels: int, model_max_length: int):
    """必须走 AutoProcessor 的公开参数入口，否则像素上限不会真正生效（项目已踩过这个坑）。"""
    processor = AutoProcessor.from_pretrained(
        model_name, min_pixels=MIN_PIXELS, max_pixels=max_pixels
    )
    processor.tokenizer.model_max_length = model_max_length
    return processor


def generate_one(model, processor, prompt: str, images, max_new_tokens: int,
                 chat_template_kwargs: dict) -> dict:
    messages = [{"role": "user", "content": build_user_content(
        prompt, images, prompt_has_image_placeholders="<image>" in prompt)}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", **chat_template_kwargs,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw_text = processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    return {
        "raw_text": raw_text,
        "generated_tokens": int(len(new_tokens)),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
        "generation_cap_hit": len(new_tokens) >= max_new_tokens and not any(
            int(token) in eos_ids for token in new_tokens
        ),
        "latency_seconds": latency,
    }


def run_setting(model, model_name: str, prepared: list, max_pixels: int,
                model_max_length: int, max_new_tokens: int, chat_template_kwargs: dict,
                output_dir: Path) -> dict:
    processor = build_processor(model_name, max_pixels, model_max_length)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index, (record, prompt, image_paths) in enumerate(prepared, 1):
        images = load_model_images(image_paths)
        result = generate_one(model, processor, prompt, images, max_new_tokens, chat_template_kwargs)
        prediction = parse_generated_bbox(result["raw_text"])
        iou, hit = score_prediction(prediction, record["bbox"])
        rows.append({
            "index": index - 1,
            "id": record["id"],
            "image": [str(path) for path in image_paths],
            "query": record.get("query"),
            "prompt": prompt,
            "target": record["bbox"],
            "target_source": record.get("target_source"),
            "target_source_kind": record.get("target_source_kind"),
            "prediction": prediction,
            "parsed": prediction is not None,
            "iou": iou,
            "acc_0.5": hit,
            "hit": hit,
            **result,
        })
        if index % 25 == 0 or index == len(prepared):
            hits = sum(1 for row in rows if row["hit"])
            print(f"  [{index}/{len(prepared)}] hits={hits} acc={hits / index:.4f}", flush=True)
    summary = {
        "model": model_name,
        "min_pixels": MIN_PIXELS,
        "max_pixels": max_pixels,
        "model_max_length": model_max_length,
        "max_new_tokens": max_new_tokens,
        "chat_template_kwargs": chat_template_kwargs,
        "manifest_samples": len(prepared),
        **summarize_rows(rows),
        "mean_input_tokens": sum(row["input_tokens"] for row in rows) / len(rows),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    setting_dir = output_dir / f"pixels_{max_pixels}"
    setting_dir.mkdir(parents=True, exist_ok=True)
    with (setting_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (setting_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    del processor
    torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多分辨率 / 多基座本地定位探针")
    parser.add_argument("--model", required=True, help="本地权重目录或 HF 名称")
    parser.add_argument("--adapter", type=Path, help="可选的 PEFT LoRA 目录")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels", type=int, nargs="+", default=[602112],
                        help="要测的每图 max_pixels 列表，例如 602112 1204224 2073600")
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt-style", choices=("egm", "native"), default="native")
    parser.add_argument("--limit", type=int, default=0, help="0 表示全量")
    parser.add_argument("--no-thinking", action="store_true",
                        help="对新版基座传 enable_thinking=False，避免输出超长思维链")
    parser.add_argument("--dry-run", action="store_true",
                        help="只准备 manifest/prompt 并报告规模，不加载模型（CPU 可跑）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for value in args.pixels:
        if value < MIN_PIXELS:
            raise ValueError(f"--pixels {value} 小于 min_pixels {MIN_PIXELS}")

    manifest = args.manifest.resolve()
    data_root = args.data_root.resolve()
    adapter = args.adapter.resolve() if args.adapter else None
    records = load_records(manifest)
    if args.limit:
        records = records[: args.limit]
    if args.target_manifest is not None:
        apply_target_manifest(records, args.target_manifest.resolve())
    target_source = str(args.target_manifest or manifest)
    for record in records:
        record.setdefault("target_source", target_source)
        record.setdefault("target_source_kind",
                          "target_manifest_bbox" if args.target_manifest else "manifest_bbox")
    prompt_style = effective_prompt_style(records, args.prompt_style)
    prepared = []
    for record in records:
        prompt = record.get("prompt") or build_prompt(record["query"], args.prompt_style)
        image_paths = [
            resolve_image_path(value, data_root, manifest,
                               from_data_root=record["images_from_data_root"])
            for value in record["images"]
        ]
        prepared.append((record, prompt, image_paths))

    if args.dry_run:
        images_per_record = {len(paths) for _, _, paths in prepared}
        prompt_chars = [len(prompt) for _, prompt, _ in prepared]
        print(json.dumps({
            "manifest": str(manifest),
            "records": len(prepared),
            "unique_ids": len({record["id"] for record, _, _ in prepared}),
            "images_per_record": sorted(images_per_record),
            "prompt_style": prompt_style,
            "prompt_chars_min_max": [min(prompt_chars), max(prompt_chars)],
            "pixels_to_probe": args.pixels,
            "planned_visual_tokens_per_image": [value // 1024 for value in args.pixels],
            "planned_visual_tokens_per_record": [3 * (value // 1024) for value in args.pixels],
            "model_max_length": args.model_max_length,
            "first_image_paths": [str(path) for path in prepared[0][2]],
        }, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("本探针需要 CUDA；先加 --dry-run 可在 CPU 上做预检")

    chat_template_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_name = load_model(args.model, adapter)
    print(f"model class = {class_name}; samples = {len(prepared)}; pixels = {args.pixels}", flush=True)

    summaries = []
    for max_pixels in args.pixels:
        print(f"=== max_pixels={max_pixels} ===", flush=True)
        summaries.append(run_setting(
            model, args.model, prepared, max_pixels, args.model_max_length,
            args.max_new_tokens, chat_template_kwargs, output_dir,
        ))

    print(f"\n{'max_pixels':>12s} {'hits':>6s} {'ACC@0.5':>9s} {'mIoU':>8s} "
          f"{'parse_fail':>11s} {'in_tokens':>10s} {'peak_GiB':>9s}")
    for summary in summaries:
        print(f"{summary['max_pixels']:12d} {summary['hits']:6d} {summary['acc_0.5']:9.4f} "
              f"{summary['mean_iou']:8.4f} {summary['parse_failures']:11d} "
              f"{summary['mean_input_tokens']:10.1f} "
              f"{summary['gpu_peak_allocated_bytes'] / 2**30:9.2f}")
    (output_dir / "probe_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
