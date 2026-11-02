"""裁剪-复核：按第一次的预测框裁出局部、放大、用同一个提示词再问一次，映射回原图。

## 为什么它值得一试（而其他后处理不值得）

开发集 43 道错题里有 **16 道是"位置对、框太小"**：中心偏移只有 0.14 倍对角，而尺寸比
0.61~0.73——模型框住了物体的一部分。事后统一缩放救不了它们（实测命中的那 335 题尺寸比
中位已经是 0.996，放大只会牺牲它们换这 16 题，净负）。

**但"换一个更近的视角重新看"是另一回事。** 本方案唯一被证实有效的杠杆就是放大
（2 倍超原生放大带来官方 +0.0142），它的机制是让小目标占更多视觉 token。裁剪本质上
是在局部做更激进的放大：一个占全图 0.2% 的框，裁出来后可以占满整个像素预算。

## 为什么它可能失败（必须一起量）

裁掉周围会破坏"同图多实例消歧"所依赖的上下文。21 道"彻底找错物体"的题有可能变得更糟。
**所以验证时必须分别报告这两类的净变化，不能只看总分。**

## 坐标映射

裁剪区是原图上的 `[cx0, cy0, cx1, cy1]`（像素）。模型在裁剪图上输出的仍是 0-1 归一化框，
映射回原图：

    full_x = cx0 + box_x * (cx1 - cx0)
    full_y = cy0 + box_y * (cy1 - cy0)

再除以原图宽高。**因为模型输出的是归一化坐标，裁剪区的放大倍数不影响这一步。**

## 用法

    python tools/refine_by_crop.py \
        --queries DEV/city400_dev_queries.json --data-root DATA \
        --base-predictions OUT/base_k1/predictions.jsonl \
        --output-dir OUT/refine_m25 --margin 2.5 --model ... --adapter ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from tools.predict_native_submission import (  # noqa: E402
    FALLBACK_BOX,
    image_paths,
    load_modal_images,
    load_queries,
)
from tools.predict_submission_vllm import (  # noqa: E402
    accept,
    build_user_content,
    check_upscale_consistency,
    prompt_for_query,
)
from tools.predict_native_submission import IMAGE_FIELDS  # noqa: E402


def crop_region(box: list[float], size: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    """以预测框中心为中心、按 margin 倍框尺寸取正方形裁剪区，并夹到图像内。

    正方形而不是跟随框的长宽比：避免送进模型前还要额外拉伸，模型见到的几何关系保持不变。
    """
    width, height = size
    center_x = (box[0] + box[2]) / 2 * width
    center_y = (box[1] + box[3]) / 2 * height
    box_w = max(1.0, (box[2] - box[0]) * width)
    box_h = max(1.0, (box[3] - box[1]) * height)
    side = max(box_w, box_h) * margin
    # 裁剪区不能小于 64 像素：太小的区域即使放大也几乎没有内容，且会放大噪声。
    side = max(64.0, side)
    # 也不能超过原图短边，否则裁剪没有意义。
    side = min(side, float(min(width, height)))
    half = side / 2
    left = int(round(center_x - half))
    top = int(round(center_y - half))
    left = max(0, min(left, width - int(side)))
    top = max(0, min(top, height - int(side)))
    return left, top, left + int(side), top + int(side)


def crop_and_zoom(images: list[Image.Image], region: tuple[int, int, int, int],
                   target_pixels: int) -> list[Image.Image]:
    """裁出 region 后放大，使总像素尽量接近 target_pixels。

    放大用 LANCZOS：极小目标放大后要靠插值保持边缘可辨（与 `upscale_images` 同理）。
    """
    left, top, right, bottom = region
    out = []
    for image in images:
        crop = image.crop((left, top, right, bottom))
        pixels = crop.width * crop.height
        factor = (target_pixels / pixels) ** 0.5 if pixels > 0 else 1.0
        # 只放大不缩小：缩小会丢掉我们正想保住的高频细节。
        factor = max(1.0, factor)
        if factor > 1.0:
            crop = crop.resize((round(crop.width * factor), round(crop.height * factor)),
                               Image.LANCZOS)
        out.append(crop)
    return out


def map_back(box: list[float], region: tuple[int, int, int, int],
             size: tuple[int, int]) -> list[float]:
    """把裁剪图上的 0-1 框映射回原图的 0-1 框。"""
    left, top, right, bottom = region
    width, height = size
    span_x = (right - left) / width
    span_y = (bottom - top) / height
    origin_x = left / width
    origin_y = top / height
    return [origin_x + box[0] * span_x, origin_y + box[1] * span_y,
            origin_x + box[2] * span_x, origin_y + box[3] * span_y]


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True,
                        help="第一遍的 predictions.jsonl；用它的 bbox 决定裁剪区")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "trimodal"), default="trimodal")
    parser.add_argument("--margin", type=float, default=2.5,
                        help="裁剪边长 = margin × 框最长边。越大保留越多上下文、放大倍数越小。")
    parser.add_argument("--target-pixels", type=int, default=8294400,
                        help="裁剪图放大后的目标总像素，与提交配置的 max_pixels 对齐")
    parser.add_argument("--max-pixels", type=int, default=8294400)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--depth-policy", default=None)
    args = parser.parse_args()
    if args.margin < 1.05:
        raise ValueError("margin 必须 > 1，否则裁剪区比目标框还小")
    check_upscale_consistency(1.0, args.max_pixels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = args.queries.resolve()
    data_root = args.data_root.resolve()
    source = load_queries(queries, None)
    base = {}
    for line in args.base_predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            base[row["id"]] = row
    items = list(source.items())
    if args.limit:
        items = items[:args.limit]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    config = {key: value for key, value in vars(args).items() if key not in ("adapter", "model")}
    config.update(model=args.model, adapter=str(args.adapter), engine="vllm", margin=args.margin)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    processor = AutoProcessor.from_pretrained(args.model)
    # 引擎参数与 `predict_submission_vllm` 逐项对齐——复核这一遍必须与第一遍同引擎同配置，
    # 否则测的是引擎差异而不是裁剪的效果。
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.model_max_length,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=True, mamba_cache_mode="align",
              enable_lora=True, max_lora_rank=args.max_lora_rank,
              limit_mm_per_prompt={"image": len(IMAGE_FIELDS)
                                   if args.modality == "trimodal" else 1},
              mm_processor_kwargs={"min_pixels": args.min_pixels,
                                   "max_pixels": args.max_pixels})
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True)
    lora = LoRARequest("grounding", 1, str(args.adapter.resolve()))

    rows_path = args.output_dir / "predictions.jsonl"
    started = time.monotonic()
    written = 0
    with rows_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(items), args.chunk_size):
            chunk = items[start:start + args.chunk_size]
            requests = []
            meta = []
            for sample_id, record in chunk:
                base_row = base.get(sample_id)
                if base_row is None or not base_row.get("parsed"):
                    # 第一遍就没有合法框，没有裁剪依据；原样保留它的兜底结果。
                    meta.append((sample_id, record, None, None))
                    requests.append(None)
                    continue
                paths = image_paths(record, data_root, queries)
                images = load_modal_images(paths, args.modality)
                size = images[0].size
                region = crop_region(base_row["bbox"], size, args.margin)
                zoomed = crop_and_zoom(images, region, args.target_pixels)
                prompt = prompt_for_query(record["query"], args.modality, args.depth_policy)
                messages = [{"role": "user", "content": build_user_content(
                    prompt, zoomed, prompt_has_image_placeholders=True)}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                meta.append((sample_id, record, region, size))
                requests.append({"prompt": text, "multi_modal_data": {"image": zoomed}})

            live = [item for item in requests if item is not None]
            outputs = llm.generate(live, sampling, use_tqdm=False, lora_request=lora) if live else []
            it = iter(outputs)
            for sample_id, record, region, size in meta:
                if region is None:
                    base_row = base[sample_id]
                    handle.write(json.dumps({
                        "id": sample_id, "bbox": base_row["bbox"], "parsed": base_row["parsed"],
                        "fallback": base_row["fallback"], "stage": "no_base_box",
                        "raw_text": base_row.get("raw_text", ""), "query": record["query"],
                        "image_paths": image_paths(record, data_root, queries),
                        "crop": None,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    continue
                output = next(it)
                text = output.outputs[0].text.strip()
                box = accept(text)
                fallback = box is None
                mapped = FALLBACK_BOX.copy() if fallback else map_back(box, region, size)
                handle.write(json.dumps({
                    "id": sample_id, "bbox": mapped, "parsed": not fallback, "fallback": fallback,
                    "stage": "crop", "raw_text": text, "query": record["query"],
                    "image_paths": image_paths(record, data_root, queries),
                    "crop": list(region), "crop_box": box,
                }, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            done = min(start + args.chunk_size, len(items))
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0
            print(f"[{done}/{len(items)}] {elapsed/60:.1f} 分钟  剩余约 "
                  f"{(len(items)-done)/rate/60 if rate else 0:.0f} 分钟", flush=True)

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    summary = {"completed": len(rows), "margin": args.margin,
               "target_pixels": args.target_pixels,
               "fallbacks": sum(1 for row in rows if row["fallback"]),
               "seconds": time.monotonic() - started}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
