from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def parse_box(text: str) -> list[float] | None:
    candidates = re.findall(r'"?bbox_2d"?\s*:\s*\[([^\]]+)\]', text, flags=re.I)
    if not candidates:
        candidates = re.findall(r'\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3})\s*\]', text)
    for candidate in candidates:
        values = re.findall(r"-?\d+(?:\.\d+)?", candidate)
        if len(values) != 4:
            continue
        box = [float(value) for value in values]
        scale = 1000.0 if max(box) > 1.5 else 1.0
        box = [value / scale for value in box]
        if all(0 <= value <= 1 for value in box) and box[0] < box[2] and box[1] < box[3]:
            return box
    return None


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def balanced_sample(records: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        prefix = str(record.get("id", "unknown")).split("_", 1)[0]
        groups.setdefault(prefix, []).append(record)
    rng = random.Random(seed)
    selected = []
    per_group = count // len(groups)
    remainder = count % len(groups)
    for index, name in enumerate(sorted(groups)):
        selected.extend(rng.sample(groups[name], min(len(groups[name]), per_group + (index < remainder))))
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-pixels", type=int, default=802816)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = balanced_sample(records, args.samples, args.seed)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=200704, max_pixels=args.max_pixels, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype="auto", local_files_only=True).eval().cuda()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed = 0
    ious: list[float] = []
    with output_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, record in enumerate(records, 1):
            image = Image.open(record["rgb"]).convert("RGB")
            query = str(record["query"])
            prompt = (
                f'Locate the object described by this referring expression: "{query}". '
                'Return exactly one JSON object: {"bbox_2d":[x1,y1,x2,y2]}. '
                'Use integer coordinates normalized to the range 0 to 1000.'
            )
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            answer = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            prediction = parse_box(answer)
            score = iou(prediction, record["bbox"]) if prediction else 0.0
            if prediction:
                parsed += 1
            ious.append(score)
            result = {"id": record.get("id"), "source": str(record.get("id", "")).split("_", 1)[0], "query": query, "gt": record["bbox"], "prediction": prediction, "iou": score, "answer": answer}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 8 == 0:
                print(json.dumps({"done": index, "parse_rate": parsed / index, "mean_iou": sum(ious) / index, "acc_0.5": sum(v >= .5 for v in ious) / index}), flush=True)
    report = {
        "samples": len(records),
        "parse_rate": parsed / len(records),
        "mean_iou": sum(ious) / len(records),
        "acc_0.5": sum(value >= .5 for value in ious) / len(records),
        "acc_0.7": sum(value >= .7 for value in ious) / len(records),
        "median_iou": sorted(ious)[len(ious) // 2],
        "output": str(output_path),
    }
    print("FINAL " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
