"""锚点式互斥细化（exclusive refine）：把「兄弟题占用的区域」直接写进提示，逼模型换目标。

## 为什么需要它

`sibling` 提示只给了兄弟查询的**文字**，没有给**位置**。全量实测：银标准配对净差只有
+11 题（z=0.61，不显著），且低密度图子集是 −4。机制核查显示「同图两题框 IoU≥0.7 且
真值不同」的抢框对在 halfA 有 143 对，其中 79% 是一对一错，而 sibling 只拆开了 50%。

这些对是**可定位的错误**：两题指向同一块区域。所以这里把对方那题的预测框当锚点写进
提示，明确告诉模型「那块已经被另一条查询占了，你的查询指向别的物体」。

## 提示词形状

在原提示的 `Locate the object...` 之前插入：

    The query "<other>" refers to the object in the region [x1,y1,x2,y2]
    (normalized 0-1000 coordinates). Your query refers to a DIFFERENT object,
    so do not output that region. [最多 3 条锚点]

## 安全阀

解析失败或框不合法 → 保留原框，`replaced: false`。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import build_user_content  # noqa: E402
from tools.predict_native_submission import load_modal_images, valid_box  # noqa: E402
from tools.predict_submission_vllm import accept  # noqa: E402

STAGE_TOKENS = 64


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def build_prompt(base_prompt, anchors):
    """把锚点句插在 `Locate the object...` 之前。"""
    anchor = "Locate the object described by this query: "
    head, sep, tail = base_prompt.partition(anchor)
    if not sep:
        raise ValueError("base prompt lost its 'Locate the object...' anchor")
    # 原提示里 sibling 版已经在 head 末尾留了 "…of a DIFFERENT object. "，这里顺接
    pieces = []
    for query, box in anchors:
        coords = ",".join(str(round(v * 1000)) for v in box)
        pieces.append(
            f'The query "{query}" refers to the object in the region [{coords}] '
            f"(normalized 0-1000 coordinates). Your query refers to a DIFFERENT object, "
            f"so do not output that region. ")
    return head + "".join(pieces) + sep + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--modality", default="trimodal")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ids", type=Path, help="只细化这些 id")
    parser.add_argument("--conflict-iou", type=float, default=0.7)
    parser.add_argument("--max-anchors", type=int, default=3)
    parser.add_argument("--upscale", type=float, default=2.0)
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

    by_image = defaultdict(list)
    for sample_id, row in rows_by_id.items():
        by_image[row["image_paths"]["visible"]].append(sample_id)

    anchors = defaultdict(list)
    for image, ids in by_image.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if iou(rows_by_id[a]["bbox"], rows_by_id[b]["bbox"]) < args.conflict_iou:
                    continue
                if a in wanted:
                    anchors[a].append((rows_by_id[b]["query"], rows_by_id[b]["bbox"]))
                if b in wanted:
                    anchors[b].append((rows_by_id[a]["query"], rows_by_id[a]["bbox"]))

    selected = sorted(q for q in anchors if q in wanted)
    selected = [(q, anchors[q][:args.max_anchors]) for q in selected]
    if args.limit:
        selected = selected[:args.limit]
    print(f"冲突题 {len(selected)} 题（IoU≥{args.conflict_iou} 的同图对）", flush=True)
    if not selected:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "exclusive_predictions.jsonl"
    (args.output_dir / "config.json").write_text(json.dumps({
        "predictions": str(args.predictions), "conflict_iou": args.conflict_iou,
        "max_anchors": args.max_anchors, "upscale": args.upscale,
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
        for sample_id, anchor_list in chunk:
            row = rows_by_id[sample_id]
            images = load_modal_images(row["image_paths"], args.modality)
            if args.upscale != 1.0:
                target = (round(images[0].width * args.upscale),
                          round(images[0].height * args.upscale))
                images = [image.resize(target, Image.LANCZOS) for image in images]
            prompt = build_prompt(row["prompt"], anchor_list)
            messages = [{"role": "user", "content": build_user_content(
                prompt, images, prompt_has_image_placeholders=True)}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            requests.append({
                "id": sample_id, "prompt": prompt,
                "anchors": [[q, b] for q, b in anchor_list],
                "orig_bbox": list(row["bbox"]),
                "request": {"prompt": text, "multi_modal_data": {"image": images}},
            })
        outputs = llm.generate([item["request"] for item in requests], sampling,
                              use_tqdm=False, lora_request=lora_request)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text.strip()
            box = accept(text)
            handle.write(json.dumps({
                "id": item["id"], "orig_bbox": item["orig_bbox"],
                "new_bbox": box if (box is not None and valid_box(box)) else None,
                "replaced": box is not None and valid_box(box),
                "raw_text": text, "prompt": item["prompt"],
                "anchors": item["anchors"],
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
