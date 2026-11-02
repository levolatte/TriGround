"""比较式核验：把两个候选框分别画成红框和蓝框，强制模型二选一。

## 为什么要换提问形式

逐框的是/否核验已经测过：全量 5,576 题上**净 +6 题（+0.11pp）**，等于噪声。
根因是**多候选的 751 题里有 622 题（82.8%）的所有候选同分**——模型对 84% 的对框和 77% 的
错框都回答 YES。

**"是/否"给了模型一个一致的逃逸出口：永远答 YES。** 这不是阈值能修的，是提问形式的问题。
比较式移除了这个出口：必须选一个，答"两个都是"没有位置可写。

## 与既有方案的关系

文献普遍认为逐框核验优于选择式提示（2509.09958），**但那个结论的前提是逐框还能分辨**。
我们的逐框已经饱和，所以在这里比较式是严格更强的一个测试，而不是重复。

## 为什么成对比较而不是一次给三个编号框

编号三选一的输出空间更大（要生成"3"），而且模型可能复述编号列表而不做选择。
成对比较把输出压到单字符，解析最稳。三个候选时做两次比较（首选 vs 次选、首选 vs 三选），
只有挑战者赢过首选才替换——**这是保守规则，避免两次比较互相矛盾时乱换。**

## 用法

    python tools/verify_pairwise.py --queries ... --data-root ... \
        --candidates cands3_full.jsonl --gt gt.json --out-dir ... --model ... --adapter ...
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

PAIR_PROMPT = (
    "The attached image has two rectangles drawn on it: box 1 is RED and box 2 is BLUE.\n"
    "Question: {query}\n"
    "Which box contains the object described by the question? "
    "Answer with only the digit 1 or the digit 2.")
# 只认开头几个字符，避免模型在解释里顺带提到另一个数字造成误判。
FIRST_MARKERS = {"1": "1", "2": "2"}


def draw_pair(image: Image.Image, box_a: list[float], box_b: list[float]) -> Image.Image:
    """红框标 1、蓝框标 2。线宽按短边比例取，避免小目标上框线盖住物体本身。"""
    out = image.convert("RGB").copy()
    painter = ImageDraw.Draw(out)
    stroke = max(3, int(round(min(out.width, out.height) * 0.006)))
    for box, colour, label in ((box_a, (255, 0, 0), "1"), (box_b, (0, 64, 255), "2")):
        x1, y1 = box[0] * out.width, box[1] * out.height
        x2, y2 = box[2] * out.width, box[3] * out.height
        painter.rectangle([x1, y1, x2, y2], outline=colour, width=stroke)
        painter.text((x1 + stroke, max(0.0, y1 - stroke * 3)), label, fill=colour)
    return out


def parse_choice(text: str) -> int | None:
    """返回 1 或 2；无法判定返回 None。**不默认成 1**——那会把解析失败算成"首选被支持"。"""
    head = text.strip()[:8]
    for character in head:
        if character in FIRST_MARKERS:
            return int(FIRST_MARKERS[character])
    return None


def build_requests(processor, items, data_root: Path, modality: str) -> list[dict]:
    requests = []
    for entry in items:
        index, sample_id, record, box_a, box_b = entry
        paths = {key: str((data_root / record[key]).resolve())
                 for key in ("visible", "infrared", "depth")}
        images = load_modal_images(paths, modality)
        images[0] = draw_pair(images[0], box_a, box_b)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": PAIR_PROMPT.format(query=record["query"])})
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        requests.append({"index": index, "id": sample_id,
                         "request": {"prompt": text, "multi_modal_data": {"image": images}}})
    return requests


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "trimodal"), default="trimodal")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=2073600)
    parser.add_argument("--model-max-length", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道分歧题，用于快速探路")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.queries.read_text(encoding="utf-8-sig"))
    gt = {key: value["bbox"] if isinstance(value, dict) else value
          for key, value in json.loads(args.gt.read_text(encoding="utf-8")).items()}
    rows = [json.loads(line) for line in
            args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["id"] in source and row["id"] in gt]
    if args.limit:
        rows = rows[:args.limit]

    # 每题生成 (首选, 挑战者) 的对。候选按原顺序，第一个是 arm2 BF16 2x 的答案。
    items = []
    for row in rows:
        boxes = [candidate["bbox"] for candidate in row["candidates"]]
        for other in range(1, len(boxes)):
            items.append((len(items), row["id"], source[row["id"]], boxes[0], boxes[other]))
    if not items:
        raise ValueError("没有可比较的候选对")

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
        requests = build_requests(processor, chunk, args.data_root.resolve(), args.modality)
        outputs = llm.generate([item["request"] for item in requests], sampling,
                               use_tqdm=False, lora_request=lora)
        for item, output in zip(requests, outputs):
            text = output.outputs[0].text
            results.append({"id": item["id"], "text": text, "choice": parse_choice(text)})
        done = min(start + args.chunk_size, len(items))
        print(f"[{done}/{len(items)}] {(time.monotonic()-started)/60:.1f} 分钟", flush=True)

    (args.out_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n", encoding="utf-8")

    def iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    # 保守规则：只有挑战者赢过首选才替换；多个挑战者都赢时取第一个（不叠加替换）。
    challenged = {}
    for result in results:
        if result["choice"] == 2:
            challenged.setdefault(result["id"], 0)
            challenged[result["id"]] += 1
    choice_map = {}
    for row in rows:
        boxes = [candidate["bbox"] for candidate in row["candidates"]]
        wins = challenged.get(row["id"], 0)
        # 半数以上的挑战者都赢过首选才换到第一个挑战者
        if wins * 2 > (len(boxes) - 1):
            choice_map[row["id"]] = boxes[1]

    base_hit = sum(1 for row in rows if iou(row["candidates"][0]["bbox"], gt[row["id"]]) >= 0.5)
    new_hit = sum(1 for row in rows
                  if iou(choice_map.get(row["id"], row["candidates"][0]["bbox"]),
                         gt[row["id"]]) >= 0.5)
    unparsed = sum(1 for result in results if result["choice"] is None)
    summary = {"questions": len(rows), "pairs": len(results), "switched": len(choice_map),
               "unparsed": unparsed,
               "base_hits": base_hit, "new_hits": new_hit, "net": new_hit - base_hit,
               "seconds": time.monotonic() - started}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
