"""局部放大（zoom-in）细化：对第一阶段给出的极小目标框，在放大后的局部窗口上重新定位。

## 为什么做这个

失败结构里 77% 的目标面积 < 0.2% 全图（约 4 像素见方）；已知「极小目标桶」对分辨率
最敏感（k1.0 命中 75 → k2.0 命中 90，同一适配器、严格配对）。全图 2.0× 已经是分辨率
最优档（k2.5 相对 k2.0：−12 题，极小目标桶 −14，是真实反转），所以**不能靠继续放大
全图**提分，只能对小目标给**相对更高**的有效分辨率。

## 做法

以第一阶段预测框的中心为心，取一个与原图同宽高比（16:9）的窗口：

    win_h = clip(max(bw, bh) × crop_factor, min_frac × H, H)
    win_w = win_h × W / H        （超宽时按 W 反夹）

三路图（RGB/红外/深度）裁同一区域，再等比 resize 到 `W×upscale, H×upscale`
——**与第一阶段全图 2.0× 完全相同的输入尺寸**，只是视野变窄。这样模型的输入尺寸
分布不变，唯一变量是视野。输出框是窗口归一化坐标，线性映射回全图。

## 提示词

直接复用第一阶段记录在 `predictions.jsonl` 里的 `prompt` 字段：它已含该题的兄弟
查询（sibling context），且「These are aligned views of the same scene」在裁剪窗口上
仍然成立。这样 zoom 阶段与第一阶段只差视野一个变量。

## 安全阀

zoom 阶段解析失败、或映射后框不合法时，`full_bbox` 留空——下游合并时保留第一阶段框，
不会把好框换成兜底框。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import build_user_content  # noqa: E402
from tools.predict_native_submission import load_modal_images, valid_box  # noqa: E402
from tools.predict_submission_vllm import accept  # noqa: E402

STAGE_TOKENS = 64


def plan_window(box, width, height, crop_factor, min_frac):
    """按预测框算裁剪窗口（浮点像素边界，全在画面内，宽高比等于原图）。"""
    x1, y1, x2, y2 = box
    bw = (x2 - x1) * width
    bh = (y2 - y1) * height
    side = max(bw, bh) * crop_factor
    side = max(side, min_frac * min(width, height))
    side = min(side, height)
    win_h = side
    win_w = win_h * width / height
    if win_w > width:
        win_w = width
        win_h = win_w * height / width
    cx = (x1 + x2) / 2 * width
    cy = (y1 + y2) / 2 * height
    left = min(max(cx - win_w / 2, 0.0), width - win_w)
    top = min(max(cy - win_h / 2, 0.0), height - win_h)
    return [left, top, win_w, win_h]


def crop_images(images, window):
    left, top, win_w, win_h = window
    box = (round(left), round(top), round(left + win_w), round(top + win_h))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"degenerate crop box {box}")
    return [image.crop(box) for image in images], box


def to_full(box, crop_box, width, height):
    """窗口归一化坐标 → 全图归一化坐标。"""
    cx1, cy1, cx2, cy2 = crop_box
    cw = cx2 - cx1
    ch = cy2 - cy1
    return [
        (cx1 + box[0] * cw) / width,
        (cy1 + box[1] * ch) / height,
        (cx1 + box[2] * cw) / width,
        (cy1 + box[3] * ch) / height,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--modality", default="trimodal")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="第一阶段 predictions.jsonl，含 bbox 与 prompt")
    parser.add_argument("--ids", type=Path,
                        help="只细化这些 id（json 对象键 / 列表 / jsonl）")
    parser.add_argument("--area-threshold", type=float, default=0.002,
                        help="预测框面积占全图比例上限，默认 0.2%")
    parser.add_argument("--crop-factor", type=float, default=3.0)
    parser.add_argument("--min-frac", type=float, default=0.25,
                        help="窗口短边至少占图高比例")
    parser.add_argument("--upscale", type=float, default=2.0,
                        help="裁剪窗口 resize 到 原尺寸 × upscale")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=8294400)
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=12)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            args.predictions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows_by_id = {row["id"]: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise SystemExit("predictions.jsonl has duplicate ids")

    wanted = set(rows_by_id)
    if args.ids is not None:
        raw = args.ids.read_text(encoding="utf-8-sig")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        wanted = set(parsed) if isinstance(parsed, (dict, list)) else {
            json.loads(line)["id"] for line in raw.splitlines() if line.strip()}
        wanted &= set(rows_by_id)

    selected = []
    for sample_id, row in rows_by_id.items():
        if sample_id not in wanted:
            continue
        b = row["bbox"]
        if (b[2] - b[0]) * (b[3] - b[1]) >= args.area_threshold:
            continue
        selected.append(sample_id)
    if args.limit:
        selected = selected[:args.limit]
    print(f"选中 {len(selected)} 题（面积 < {args.area_threshold:.4%}）", flush=True)
    if not selected:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "zoom_predictions.jsonl"
    (args.output_dir / "config.json").write_text(json.dumps({
        "predictions": str(args.predictions), "area_threshold": args.area_threshold,
        "crop_factor": args.crop_factor, "min_frac": args.min_frac,
        "upscale": args.upscale, "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels, "model_max_length": args.model_max_length,
        "selected": len(selected), "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.model_max_length,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, enable_lora=args.adapter is not None,
              max_lora_rank=args.max_lora_rank, limit_mm_per_prompt={"image": 3},
              enable_prefix_caching=True,
              mm_processor_kwargs={"min_pixels": args.min_pixels,
                                   "max_pixels": args.max_pixels})
    tokenizer = llm.get_tokenizer()
    lora_request = LoRARequest("trained", 1, str(args.adapter)) if args.adapter else None
    sampling = SamplingParams(temperature=0.0, max_tokens=STAGE_TOKENS, stop=["]"],
                              include_stop_str_in_output=True, skip_special_tokens=True)

    handle = out_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    done = 0
    chunks = [selected[i:i + args.chunk_size]
              for i in range(0, len(selected), args.chunk_size)]
    for chunk_index, chunk in enumerate(chunks, 1):
        requests = []
        for sample_id in chunk:
            row = rows_by_id[sample_id]
            images = load_modal_images(row["image_paths"], args.modality)
            width, height = images[0].size
            window = plan_window(row["bbox"], width, height,
                                 args.crop_factor, args.min_frac)
            crops, crop_box = crop_images(images, window)
            target = (round(width * args.upscale), round(height * args.upscale))
            crops = [image.resize(target, Image.LANCZOS) for image in crops]
            messages = [{"role": "user", "content": build_user_content(
                row["prompt"], crops, prompt_has_image_placeholders=True)}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            requests.append({
                "id": sample_id, "crop_box": crop_box, "width": width, "height": height,
                "orig_bbox": list(row["bbox"]),
                "request": {"prompt": text, "multi_modal_data": {"image": crops}},
            })
        outputs = llm.generate([item["request"] for item in requests], sampling,
                              use_tqdm=False, lora_request=lora_request)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text.strip()
            crop_box = item["crop_box"]
            box = accept(text)
            full_box = None
            if box is not None:
                mapped = to_full(box, crop_box, item["width"], item["height"])
                if valid_box(mapped):
                    full_box = mapped
            handle.write(json.dumps({
                "id": item["id"], "crop_box": list(crop_box),
                "orig_bbox": item["orig_bbox"], "raw_text": text,
                "crop_bbox": box, "full_bbox": full_box,
                "generated_tokens": len(output.outputs[0].token_ids),
                "input_tokens": len(output.prompt_token_ids),
            }, ensure_ascii=False) + "\n")
        handle.flush()
        done += len(chunk)
        elapsed = time.perf_counter() - started
        print(f"[{chunk_index}/{len(chunks)}] {done}/{len(selected)}  "
              f"{elapsed / 60:.1f} 分钟", flush=True)
        del requests, outputs
        gc.collect()
    handle.close()
    print(f"完成 {done} 题 → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
