"""Evaluate IR localization and depth relation reading probes with one model load."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import (
    _image_values,
    _modalities,
    _read_jsonl,
    _validated_target,
    generate_text,
    load_model_images,
    load_model_and_processor,
    parse_ir_reading,
    resolve_image_path,
    score_prediction,
)

DEPTH_ANSWERS = ("A_nearer", "B_nearer", "unknown")


def load_reading_records(manifest: Path) -> list[dict[str, Any]]:
    text = manifest.read_text(encoding="utf-8-sig")
    payload = _read_jsonl(manifest, text) if manifest.suffix.lower() == ".jsonl" else json.loads(text)
    items = payload.values() if isinstance(payload, dict) else payload
    records = []
    seen: set[str] = set()
    for record in items:
        sample_id = str(record["id"])
        if sample_id in seen:
            raise ValueError(f"duplicate reading ID: {sample_id}")
        seen.add(sample_id)
        task_type = record["task_type"]
        if task_type not in {"ir_bbox", "depth_relation"}:
            raise ValueError(f"unsupported reading task: {task_type}")
        images = _image_values(record)
        modalities = _modalities(record, images)
        if modalities is None:
            raise ValueError(f"reading record {sample_id} needs explicit modalities")
        human = [item for item in record["conversations"] if item["from"] == "human"]
        if len(human) != 1:
            raise ValueError(f"reading record {sample_id} needs one human prompt")
        prompt = str(human[0]["value"])
        if prompt.count("<image>") != len(images):
            raise ValueError(f"reading record {sample_id} image placeholders mismatch")
        expected = record["expected_answer"]
        if task_type == "ir_bbox":
            if "ir" not in modalities or record["coordinate_system"] != "qwen_0_1000":
                raise ValueError(f"reading record {sample_id} requires IR 0-1000 coordinates")
            status, target_box = parse_ir_reading(json.dumps(expected))
            if status == "reading_parse_failure":
                raise ValueError(f"invalid expected IR answer: {sample_id}")
            ir_gt_bbox = record.get("ir_gt_bbox")
            if "ir_gt_bbox" in record:
                if (status == "unknown") != (ir_gt_bbox is None):
                    raise ValueError(f"IR answer and raw GT disagree: {sample_id}")
                if ir_gt_bbox is not None:
                    ir_gt_bbox = _validated_target(ir_gt_bbox)
        else:
            if "depth" not in modalities or record["coordinate_system"] != "categorical":
                raise ValueError(f"reading record {sample_id} requires depth relation")
            if expected not in DEPTH_ANSWERS:
                raise ValueError(f"invalid expected depth answer: {sample_id}")
            status, target_box = "", None
            ir_gt_bbox = None
        records.append({
            "id": sample_id, "task_type": task_type, "images": images,
            "modalities": modalities, "prompt": prompt,
            "expected_answer": expected, "expected_status": status,
            "expected_box_ir": target_box,
            "ir_gt_bbox": ir_gt_bbox,
            "source_task_id": record.get("source_task_id"),
            "pair_id": record.get("pair_id"),
            "pair_orientation": record.get("pair_orientation"),
            "uses_oracle_regions": bool(record.get("uses_oracle_regions", task_type == "depth_relation")),
        })
    return records


def parse_depth_answer(text: str) -> str | None:
    answer = text.strip()
    if answer.startswith('"'):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError:
            return None
    return answer if answer in DEPTH_ANSWERS else None


def score_reading(record: dict[str, Any], raw_text: str) -> dict[str, Any]:
    if record["task_type"] == "depth_relation":
        prediction = parse_depth_answer(raw_text)
        return {
            "prediction": prediction,
            "parsed": prediction is not None,
            "correct": prediction == record["expected_answer"],
        }
    status, box = parse_ir_reading(raw_text)
    expected_status = record["expected_status"]
    expected_box = record["expected_box_ir"]
    iou = 0.0
    hit = False
    if expected_status == "box" and status == "box":
        target = record.get("ir_gt_bbox") or [coordinate / 1000.0 for coordinate in expected_box]
        iou, hit = score_prediction(
            [coordinate / 1000.0 for coordinate in box],
            target,
        )
    return {
        "prediction": {"bbox_2d": box} if status == "box" else ({"bbox_2d": None} if status == "unknown" else None),
        "parsed": status != "reading_parse_failure",
        "status": status,
        "valid_bbox": status == "box",
        "iou": iou,
        "acc_0.5": hit,
        "unknown_correct": expected_status == "unknown" and status == "unknown",
        "correct": hit if expected_status == "box" else status == "unknown",
    }


def summarize_reading(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ir = [row for row in rows if row["task_type"] == "ir_bbox"]
    known = [row for row in ir if row["expected_status"] == "box"]
    known_valid = [row for row in known if row["valid_bbox"]]
    unknown = [row for row in ir if row["expected_status"] == "unknown"]
    depth = [row for row in rows if row["task_type"] == "depth_relation"]
    pairs = {}
    for row in depth:
        if row.get("pair_id"):
            group=pairs.setdefault(row["pair_id"], {})
            orientation=row["pair_orientation"]
            if orientation not in {"forward","reverse"} or orientation in group:
                raise ValueError("depth pair must have at most one prediction per orientation")
            group[orientation]=row
    both_correct=sum(len(pair)==2 and all(row["correct"] for row in pair.values()) for pair in pairs.values())
    order_consistent=sum(len(pair)==2 and {row["prediction"] for row in pair.values()}=={"A_nearer","B_nearer"} for pair in pairs.values())
    return {
        "samples": len(rows),
        "ir_bbox": {
            "samples": len(ir), "known_targets": len(known), "unknown_targets": len(unknown),
            "valid_bbox": sum(row["valid_bbox"] for row in ir),
            "known_valid_bbox": sum(row["valid_bbox"] for row in known),
            "known_valid_bbox_rate": len(known_valid) / len(known) if known else None,
            "known_mean_iou": sum(row["iou"] for row in known) / len(known) if known else None,
            "known_acc_0.5": sum(row["acc_0.5"] for row in known) / len(known) if known else None,
            "valid_bbox_mean_iou": sum(row["iou"] for row in known_valid) / len(known_valid) if known_valid else None,
            "valid_bbox_acc_0.5": sum(row["acc_0.5"] for row in known_valid) / len(known_valid) if known_valid else None,
            "unknown_correct": sum(row["unknown_correct"] for row in unknown),
            "unknown_recognition": sum(row["unknown_correct"] for row in unknown) / len(unknown) if unknown else None,
            "parse_failures": sum(not row["parsed"] for row in ir),
        },
        "depth_relation": {
            "samples": len(depth),
            "scope": "oracle_region_reading_only",
            "paired_order_check": {
                "pairs": len(pairs), "complete_pairs": sum(len(pair)==2 for pair in pairs.values()),
                "both_correct": both_correct, "order_consistent": order_consistent,
                "both_correct_rate": both_correct/len(pairs) if pairs else None,
            },
            "accuracy": sum(row["correct"] for row in depth) / len(depth) if depth else None,
            "by_answer": {
                answer: {
                    "samples": sum(row["expected_answer"] == answer for row in depth),
                    "correct": sum(row["expected_answer"] == answer and row["correct"] for row in depth),
                    "accuracy": (
                        sum(row["expected_answer"] == answer and row["correct"] for row in depth)
                        / sum(row["expected_answer"] == answer for row in depth)
                        if any(row["expected_answer"] == answer for row in depth) else None
                    ),
                    "predictions": {
                        predicted: sum(row["expected_answer"] == answer and row["prediction"] == (None if predicted == "parse_failure" else predicted) for row in depth)
                        for predicted in (*DEPTH_ANSWERS, "parse_failure")
                    },
                }
                for answer in DEPTH_ANSWERS
            },
            "parse_failures": sum(not row["parsed"] for row in depth),
        },
        "total_calls": len(rows),
        "total_latency_seconds": sum(row["latency_seconds"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mixed IR/depth reading probes.")
    parser.add_argument("--model", default="nvidia/EGM-8B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 0 or args.max_new_tokens <= 0:
        raise ValueError("limit must be non-negative and token cap positive")
    manifest = args.manifest.resolve()
    records = load_reading_records(manifest)
    if args.limit:
        records = records[:args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if rows_path.exists() or summary_path.exists():
        raise FileExistsError(f"reading output already exists: {output_dir}")
    adapter = args.adapter.resolve() if args.adapter else None
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter}")
    model, processor = (load_model_and_processor(args.model, adapter, args.min_pixels, args.max_pixels)
                        if records else (None, None))
    rows = []
    with rows_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for record in records:
            paths = [resolve_image_path(value, args.data_root.resolve(), manifest, from_data_root=True)
                     for value in record["images"]]
            images = load_model_images(paths)
            output = generate_text(model, processor, record["prompt"], images, True, args.max_new_tokens)
            score = score_reading(record, output["raw_text"])
            row = {
                "id": record["id"], "source_task_id": record["source_task_id"],
                "task_type": record["task_type"], "modalities": record["modalities"],
                "images": [str(path) for path in paths], "prompt": record["prompt"],
                "expected_answer": record["expected_answer"],
                "ir_gt_bbox": record["ir_gt_bbox"],
                "expected_status": record["expected_status"],
                "pair_id": record["pair_id"], "pair_orientation": record["pair_orientation"],
                "uses_oracle_regions": record["uses_oracle_regions"],
                "raw_text": output["raw_text"], **score, **{key: output[key] for key in (
                    "generated_tokens", "input_tokens", "generation_cap_hit", "latency_seconds",
                )},
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"done": len(rows), "total": len(records), "id": record["id"],
                              "correct": row["correct"]}, ensure_ascii=False), flush=True)
    summary = {
        "model": args.model, "adapter": str(adapter) if adapter else None,
        "manifest": str(manifest), "predictions": str(rows_path),
        "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens, **summarize_reading(rows),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if records else 0,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
