"""逐框真假核验：把候选框画进图里，让模型回答"这个框里是不是查询描述的物体"。

## 为什么这条路和之前六条都不同

前面失败的方向都在**猜**：

* 自一致性投票 = 几何共识。两个候选都像时，共识无解。
* 裁剪-复核 = 重新定位。它会改动已答对的题。
* 事后缩放 / SAM2 = 改框的尺寸。与模型对"物体在哪"的判断无关。

核验换的是**任务形式**：不再是"给我一个框"，而是"这个框里是不是 X？是/否"。
这在文献里被认为显著优于"从多个候选里挑一个"，因为它把 N 选 1 分解成 N 个
互相独立的判断，去掉了跨框干扰（2509.09958、Set-of-Mark 2310.11441）。

## 必须先验证的事：判别力

裁剪-复核死在"同一个模型判不准自己的框以 为代价"。所以本工具**分成两个阶段**：

1. `--probe`：只跑一批已知对错的题，直接报**判别力**（对框的 yes 率 vs 错框的 yes 率）。
   判别力不行（两条 yes 率接近）就立刻停，不要浪费 GPU 去跑全量。
2. 不带 `--probe`：对真实的候选集打分，输出每题每个候选的 yes 概率。

## 为什么必须用模型自己的框去测判别力

如果只测"模型对正确答案说 yes、对随机框说 no"，那任何模型都能通过——随机框太容易否掉。
真正的考验是**在模型自己产出的、看起来合理的错框上**能不能说 no。
所以 probe 模式用真实的预测框配真实的 GT 来分组，而不是合成随机框。

## 用法

    # 阶段1：判别力
    python tools/verify_by_prompt.py --probe --queries ... --data-root ... \
        --candidates ... --gt ... --limit 200 --out-dir ...

    # 阶段2：给候选打分
    python tools/verify_by_prompt.py --queries ... --data-root ... \
        --candidates cands.jsonl --out-dir ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw  # noqa: E402

from tools.predict_native_submission import load_modal_images  # noqa: E402

# 提示词刻意只问 identity，不问位置/大小/IOU。
# ReCLIP 的受控实验证明 CLIP/VLM 开箱做不了空间关系推理（2204.05991），
# 所以任何"左边那个对吗""框够大吗"都会引入模型不擅长的子任务。
VERIFY_PROMPT = (
    "The attached image has a red rectangle drawn on it. "
    "Look carefully at what is inside the red rectangle.\n"
    "Question to check: {query}\n"
    "Is the object inside the red rectangle the object described by the question? "
    "Answer only YES or NO.")
YES_WORDS = ("yes", "是")
NO_WORDS = ("no", "否")


def draw_box(image: Image.Image, box: list[float], width: int = 6) -> Image.Image:
    """在图上画红框。line width 按短边比例取，避免小目标上框线盖住物体本身。"""
    out = image.convert("RGB").copy()
    painter = ImageDraw.Draw(out)
    span = min(out.width, out.height)
    stroke = max(2, int(round(span * 0.004)))
    x1, y1 = box[0] * out.width, box[1] * out.height
    x2, y2 = box[2] * out.width, box[3] * out.height
    painter.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(width, stroke))
    return out


def answer_to_probability(text: str) -> float:
    """把 YES/NO 映射成概率。两种都不出现时给 0.5（不确定），**不静默当成 NO**。

    静默当 NO 会让"解析失败"和"模型真的否定"混在一起，判别力会被系统性低估。
    """
    lowered = text.strip().lower()
    head = lowered[:16]
    has_yes = any(word in head for word in YES_WORDS)
    has_no = any(word in head for word in NO_WORDS)
    if has_yes and not has_no:
        return 1.0
    if has_no and not has_yes:
        return 0.0
    return 0.5


def build_requests(processor, items, data_root: Path, queries: Path, modality: str,
                   min_pixels: int, max_pixels: int) -> list[dict]:
    requests = []
    for entry in items:
        index, sample_id, record, box = entry
        paths = {key: str((data_root / record[key]).resolve())
                 for key in ("visible", "infrared", "depth")}
        images = load_modal_images(paths, modality)
        images[0] = draw_box(images[0], box)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": VERIFY_PROMPT.format(query=record["query"])})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        requests.append({"index": index, "id": sample_id, "box": box,
                         "request": {"prompt": text, "multi_modal_data": {"image": images}}})
    return requests


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="每行 {id, candidates:[{bbox}]}；probe 模式下用它取真实预测框")
    parser.add_argument("--gt", type=Path, help="probe 模式必需：{id: bbox}")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe", action="store_true",
                        help="判别力模式：按 GT 把候选框分成对/错两组，报两组 yes 率")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "trimodal"), default="trimodal")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=2073600,
                        help="核验不需要提交级分辨率：判别「是不是这个物体」用中等分辨率即可，"
                             "能省 4 倍预填充成本。默认取原图 1920×1080 的像素数。")
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.probe and args.gt is None:
        raise ValueError("--probe 需要 --gt")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    queries_path = args.queries.resolve()
    data_root = args.data_root.resolve()
    source = json.loads(queries_path.read_text(encoding="utf-8-sig"))
    candidate_rows = [json.loads(line) for line in
                      args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    gt = ({key: value["bbox"] if isinstance(value, dict) else value
           for key, value in json.loads(args.gt.read_text(encoding="utf-8")).items()}
          if args.gt else None)

    items = []
    for row in candidate_rows:
        sample_id = row["id"]
        if sample_id not in source:
            continue
        record = source[sample_id]
        for position, candidate in enumerate(row.get("candidates") or []):
            items.append((len(items), sample_id, record, candidate["bbox"]))
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError("没有可核验的候选框")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.model_max_length,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=True,
              mamba_cache_mode="align", enable_lora=True, max_lora_rank=args.max_lora_rank,
              limit_mm_per_prompt={"image": 3 if args.modality == "trimodal" else 1},
              mm_processor_kwargs={"min_pixels": args.min_pixels,
                                   "max_pixels": args.max_pixels})
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                              skip_special_tokens=True)
    lora = LoRARequest("grounding", 1, str(args.adapter.resolve()))

    started = time.monotonic()
    results = []
    for start in range(0, len(items), args.chunk_size):
        chunk = items[start:start + args.chunk_size]
        requests = build_requests(processor, chunk, data_root, queries_path,
                                  args.modality, args.min_pixels, args.max_pixels)
        outputs = llm.generate([item["request"] for item in requests], sampling,
                               use_tqdm=False, lora_request=lora)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text
            results.append({"id": item["id"], "bbox": item["box"], "text": text,
                            "prob": answer_to_probability(text)})
        done = min(start + args.chunk_size, len(items))
        elapsed = time.monotonic() - started
        print(f"[{done}/{len(items)}] {elapsed/60:.1f} 分钟", flush=True)

    out_path = args.out_dir / ("probe.jsonl" if args.probe else "scores.jsonl")
    out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
                        encoding="utf-8")
    summary: dict = {"questions": len({row["id"] for row in results}),
                     "boxes": len(results), "seconds": time.monotonic() - started,
                     "output": str(out_path)}
    if args.probe:
        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            if x2 <= x1 or y2 <= y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / union if union > 0 else 0.0
        right = [row["prob"] for row in results if iou(row["bbox"], gt[row["id"]]) >= 0.5]
        wrong = [row["prob"] for row in results if iou(row["bbox"], gt[row["id"]]) < 0.5]
        summary["right_boxes"] = len(right)
        summary["wrong_boxes"] = len(wrong)
        summary["yes_rate_right"] = sum(right) / len(right) if right else None
        summary["yes_rate_wrong"] = sum(wrong) / len(wrong) if wrong else None
        # 判别力 = 两组 yes 率之差。差值接近 0 说明模型分不清自己的对错框。
        summary["discrimination"] = (summary["yes_rate_right"] - summary["yes_rate_wrong"]
                                    if right and wrong else None)
        # 只用"在对框上 yes、在错框上 no"能选对多少题，直接对应可用增益。
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
