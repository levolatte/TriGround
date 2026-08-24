from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from mm_grounding.data import NativeGroundingCollator
from mm_grounding.engine import parse_bbox
from mm_grounding.metrics import grounding_metrics


def load_records(manifest: Path) -> list[dict]:
    text = manifest.read_text(encoding="utf-8-sig")
    if manifest.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            return [{"id": sample_id, **record} for sample_id, record in payload.items()]
        if isinstance(payload, list):
            return payload
        raise ValueError("JSON manifest must be an object or array")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def stratified_subset(records: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(records):
        return records
    groups: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (str(record.get("class_name", "")), str(record.get("scale_bin", "")))
        groups.setdefault(key, []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[i] for i in order]
    exact = {key: len(indices) * limit / len(records) for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key])),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    selected: list[dict] = []
    for key in sorted(groups):
        selected.extend(records[index] for index in groups[key][: quotas[key]])
    return selected


def image_path(manifest: Path, record: dict) -> Path:
    value = record.get("rgb") or record.get("visible")
    if not value:
        raise KeyError("record has neither 'rgb' nor 'visible'")
    path = Path(str(value))
    return path if path.is_absolute() else manifest.parent / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=512, help="0 means all records")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    records = stratified_subset(load_records(manifest), args.samples, args.seed)
    if not records:
        raise ValueError("evaluation set is empty")

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype="auto",
        local_files_only=True,
    ).eval().cuda()
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    parsed = 0
    cap_hits = 0
    cap_parse_failures = 0
    with output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for index, record in enumerate(records, 1):
            with Image.open(image_path(manifest, record)) as source:
                image = source.convert("RGB").copy()
            prompt = NativeGroundingCollator._prompt(str(record["query"]))
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            capped = len(new_tokens) >= args.max_new_tokens and not any(
                int(token) in eos_ids for token in new_tokens
            )
            answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
            prediction = parse_bbox(answer)
            parsed += prediction is not None
            cap_hits += capped
            cap_parse_failures += capped and prediction is None
            row = {
                "id": str(record.get("id", index - 1)),
                "query": str(record["query"]),
                "prediction": prediction,
                "answer": answer,
                "generation_cap_hit": capped,
            }
            if "bbox" in record:
                target = [float(value) for value in record["bbox"]]
                row["target"] = target
                predictions.append(prediction or [0.0, 0.0, 0.0, 0.0])
                targets.append(target)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 16 == 0 or index == len(records):
                print(json.dumps({
                    "done": index,
                    "total": len(records),
                    "parse_rate": parsed / index,
                    "cap_hit_rate": cap_hits / index,
                }), flush=True)

    report: dict[str, object] = {
        "model": args.model,
        "manifest": str(manifest),
        "samples": len(records),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "parse_rate": parsed / len(records),
        "generation_cap_hit_rate": cap_hits / len(records),
        "cap_parse_failure_rate": cap_parse_failures / len(records),
        "predictions": str(output),
    }
    if targets:
        report.update(grounding_metrics(
            torch.tensor(predictions, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        ))
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
