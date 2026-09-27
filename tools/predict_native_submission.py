"""Run a validated native Qwen3-VL adapter on unlabeled competition queries."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import (  # noqa: E402
    build_user_content,
    configure_torch_precision,
    ensure_jsonl_append_separator,
    load_prediction_rows,
    parse_generated_bbox,
    resolve_image_path,
    verify_run_config,
)
from tools.prepare_qwen3vl_native_sft import (  # noqa: E402
    _rgb_prompt,
    _trimodal_prompt,
    depth_mm_to_grayscale_rgb,
)


MIN_PIXELS = 200704
MAX_PIXELS = 602112
MAX_NEW_TOKENS = 128
FALLBACK_BOX = [0.0, 0.0, 1.0, 1.0]
IMAGE_FIELDS = ("visible", "infrared", "depth")
PROMPTS = {"rgb": _rgb_prompt, "trimodal": _trimodal_prompt}
DEPTH_POLICY = "uint16_mm_absolute_grayscale_or_uint8_rgb_visualization_v2"


def valid_box(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
               for item in value):
        return False
    x1, y1, x2, y2 = value
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def load_queries(path: Path, expected_count: int | None) -> dict[str, dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict) or not source:
        raise ValueError("queries must be a non-empty object keyed by Query ID")
    if expected_count is not None and len(source) != expected_count:
        raise ValueError(f"expected {expected_count} Query IDs, found {len(source)}")
    for sample_id, record in source.items():
        if not isinstance(record, dict) or "bbox" in record:
            raise ValueError(f"{sample_id}: expected an unlabeled query object")
        if any(not isinstance(record.get(key), str) or not record[key]
               for key in (*IMAGE_FIELDS, "query")):
            raise ValueError(f"{sample_id}: visible, infrared, depth, query must be nonempty strings")
    return source


def image_paths(record: dict[str, Any], data_root: Path, queries: Path) -> dict[str, str]:
    return {
        key: str(resolve_image_path(record[key], data_root, queries, from_data_root=True))
        for key in IMAGE_FIELDS
    }


def build_run_config(model: str, adapter: Path, queries: Path, data_root: Path,
                     modality: str) -> dict[str, Any]:
    return {
        "model": model,
        "adapter": str(adapter),
        "queries": str(queries),
        "data_root": str(data_root),
        "modality": modality,
        "prompt_source": f"prepare_qwen3vl_native_sft._{modality}_prompt",
        "prompt_template": PROMPTS[modality]("{query}"),
        "modalities": ["visible"] if modality == "rgb" else list(IMAGE_FIELDS),
        "depth_policy": DEPTH_POLICY if modality == "trimodal" else None,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "peft_autocast_adapter_dtype": False,
        "do_sample": False,
        "tf32": False,
        "matmul_precision": "highest",
        "fallback_box": FALLBACK_BOX,
    }


def validate_rows(
    rows: list[dict[str, Any]], source: dict[str, dict[str, Any]], data_root: Path,
    queries: Path, modality: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        sample_id = str(row.get("id"))
        if sample_id not in source:
            raise ValueError(f"predictions.jsonl line {line_number}: unknown ID {sample_id}")
        if sample_id in indexed:
            raise ValueError(f"predictions.jsonl contains duplicate ID {sample_id}")
        record = source[sample_id]
        expected = {
            "query": record["query"],
            "image_paths": image_paths(record, data_root, queries),
            "prompt": PROMPTS[modality](record["query"]),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"resume {key} mismatch for Query ID {sample_id}")
        if not valid_box(row.get("bbox")):
            raise ValueError(f"predictions.jsonl line {line_number}: invalid bbox")
        if not isinstance(row.get("parsed"), bool) or not isinstance(row.get("fallback"), bool):
            raise ValueError(f"predictions.jsonl line {line_number}: missing parse status")
        if row["fallback"] == row["parsed"]:
            raise ValueError(f"predictions.jsonl line {line_number}: inconsistent parse status")
        if row["fallback"] and row["bbox"] != FALLBACK_BOX:
            raise ValueError(f"predictions.jsonl line {line_number}: inconsistent fallback box")
        parsed_box = parse_generated_bbox(row.get("raw_text", ""))
        if row["parsed"] != (parsed_box is not None):
            raise ValueError(f"predictions.jsonl line {line_number}: raw text and parse status differ")
        if parsed_box is not None and row["bbox"] != parsed_box:
            raise ValueError(f"predictions.jsonl line {line_number}: bbox differs from raw text")
        indexed[sample_id] = row
    return indexed


def load_modal_images(paths: dict[str, str], modality: str) -> list[Image.Image]:
    with Image.open(paths["visible"]) as source_image:
        images = [source_image.convert("RGB").copy()]
    if modality == "trimodal":
        with Image.open(paths["infrared"]) as infrared_image:
            images.append(infrared_image.convert("RGB").copy())
        with Image.open(paths["depth"]) as depth_image:
            depth = np.asarray(depth_image)
        # Official rematch data mixes raw millimetres with already rendered JPEGs.
        # Preserve supplied visualizations: their metric scale cannot be recovered.
        if depth.dtype == np.uint8 and depth.ndim == 3 and depth.shape[2] == 3:
            images.append(Image.fromarray(depth))
        elif depth.dtype == np.uint16 and depth.ndim == 2:
            images.append(Image.fromarray(depth_mm_to_grayscale_rgb(depth)))
        else:
            raise ValueError(f"unsupported official depth encoding: {depth.shape} {depth.dtype}")
    return images


def summarize(rows: list[dict[str, Any]], total: int, peak_allocated: int,
              peak_reserved: int) -> dict[str, Any]:
    latencies = [float(row["latency_seconds"]) for row in rows]
    return {
        "total_queries": total,
        "completed_queries": len(rows),
        "parse_failures": sum(not row["parsed"] for row in rows),
        "fallback_predictions": sum(row["fallback"] for row in rows),
        "generation_cap_hits": sum(row["generation_cap_hit"] for row in rows),
        "total_generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "total_generation_seconds": sum(latencies),
        "median_generation_seconds": median(latencies) if latencies else 0.0,
        "gpu_peak_allocated_bytes": peak_allocated,
        "gpu_peak_reserved_bytes": peak_reserved,
    }


def validate_submission(source: dict[str, dict[str, Any]],
                        submission: dict[str, dict[str, Any]]) -> None:
    if list(source) != list(submission):
        raise ValueError("submission Query IDs or order differ from source")
    for sample_id, original in source.items():
        predicted = submission[sample_id]
        if set(predicted) != set(original) | {"bbox"}:
            raise ValueError(f"{sample_id}: fields differ from source plus bbox")
        if any(predicted[field] != value for field, value in original.items()):
            raise ValueError(f"{sample_id}: a source field was modified")
        if not valid_box(predicted["bbox"]):
            raise ValueError(f"{sample_id}: invalid normalized xyxy bbox")


def package_submission(source: dict[str, dict[str, Any]], rows_by_id: dict[str, dict[str, Any]],
                       output_dir: Path) -> tuple[Path, Path]:
    missing = set(source) - set(rows_by_id)
    if missing:
        raise ValueError(f"cannot package incomplete predictions: {len(missing)} Query IDs missing")
    submission = {
        sample_id: {**record, "bbox": rows_by_id[sample_id]["bbox"]}
        for sample_id, record in source.items()
    }
    validate_submission(source, submission)
    json_path = output_dir / "predictions.json"
    zip_path = output_dir / "submission.zip"
    json_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="predictions.json")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["predictions.json"]:
            raise ValueError("ZIP must contain predictions.json at its root")
        packed = json.loads(archive.read("predictions.json"))
    validate_submission(source, packed)
    return json_path, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local Qwen3-VL-8B base model")
    parser.add_argument("--adapter", type=Path, help="R2 or M2 checkpoint-928 PEFT adapter")
    parser.add_argument("--modality", choices=PROMPTS, help="RGB R2 or three-image M2")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Infer only the first N IDs; never package")
    parser.add_argument("--package-only", action="store_true", help="Verify and package complete JSONL without CUDA")
    parser.add_argument("--expected-query-count", type=int, default=5690)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.expected_query_count <= 0:
        raise ValueError("--limit must be nonnegative and --expected-query-count positive")
    if args.package_only and args.limit:
        raise ValueError("--package-only cannot be combined with --limit")
    queries = args.queries.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    source = load_queries(queries, args.expected_query_count)
    config_path = output_dir / "run_config.json"
    rows_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    if args.package_only:
        saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
        model = args.model or saved["model"]
        adapter = args.adapter.resolve() if args.adapter else Path(saved["adapter"])
        modality = args.modality or saved["modality"]
    else:
        if args.model is None or args.adapter is None or args.modality is None:
            raise ValueError("inference requires --model, --adapter, and --modality")
        model = args.model
        adapter = args.adapter.resolve()
        modality = args.modality
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter}")
    config = build_run_config(model, adapter, queries, data_root, modality)

    if args.package_only or args.resume:
        if rows_path.exists() and not config_path.exists():
            raise FileNotFoundError(f"cannot resume {rows_path}: run_config.json missing")
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
            verify_run_config(saved, config)
        elif args.package_only:
            raise FileNotFoundError(f"run_config.json missing: {config_path}")
        rows_by_id = validate_rows(load_prediction_rows(rows_path), source, data_root,
                                   queries, modality) if rows_path.exists() else {}
    else:
        if any(path.exists() for path in (config_path, rows_path, summary_path,
                                           output_dir / "predictions.json", output_dir / "submission.zip")):
            raise FileExistsError(f"output already contains results: {output_dir}; use --resume")
        rows_by_id = {}

    if args.package_only:
        json_path, zip_path = package_submission(source, rows_by_id, output_dir)
        previous = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        ordered = [rows_by_id[sample_id] for sample_id in source]
        summary = {**config, **summarize(ordered, len(source),
                                        int(previous.get("gpu_peak_allocated_bytes", 0)),
                                        int(previous.get("gpu_peak_reserved_bytes", 0))),
                   "predictions_jsonl": str(rows_path), "predictions_json": str(json_path),
                   "submission_zip": str(zip_path)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = list(source.items())[:args.limit] if args.limit else list(source.items())
    remaining = [(index, sample_id, record) for index, (sample_id, record) in enumerate(selected)
                 if sample_id not in rows_by_id]
    peak_allocated = peak_reserved = 0
    if remaining:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for native Qwen3-VL inference")
        configure_torch_precision()
        processor = AutoProcessor.from_pretrained(model, min_pixels=MIN_PIXELS,
                                                  max_pixels=MAX_PIXELS, local_files_only=True)
        grounder = Qwen3VLForConditionalGeneration.from_pretrained(
            model, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        from peft import PeftModel

        grounder = PeftModel.from_pretrained(
            grounder, adapter, is_trainable=False, autocast_adapter_dtype=False,
            local_files_only=True).eval().to(device="cuda", dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats()
        eos = processor.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        mode = "a" if rows_path.exists() else "w"
        with rows_path.open(mode, encoding="utf-8") as handle, torch.inference_mode():
            if mode == "a":
                ensure_jsonl_append_separator(rows_path, handle)
            for index, sample_id, record in remaining:
                paths = image_paths(record, data_root, queries)
                images = load_modal_images(paths, modality)
                prompt = PROMPTS[modality](record["query"])
                messages = [{"role": "user", "content": build_user_content(
                    prompt, images, prompt_has_image_placeholders=True)}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                inputs = inputs.to("cuda")
                torch.cuda.synchronize()
                started = time.perf_counter()
                generated = grounder.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                              do_sample=False)
                torch.cuda.synchronize()
                latency = time.perf_counter() - started
                new_tokens = generated[0, inputs["input_ids"].shape[1]:]
                raw_text = processor.decode(new_tokens, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip()
                prediction = parse_generated_bbox(raw_text)
                fallback = prediction is None
                row = {
                    "index": index, "id": sample_id, "query": record["query"],
                    "image_paths": paths, "prompt": prompt,
                    "bbox": FALLBACK_BOX.copy() if fallback else prediction,
                    "parsed": not fallback, "fallback": fallback,
                    "generated_tokens": int(len(new_tokens)),
                    "input_tokens": int(inputs["input_ids"].shape[1]),
                    "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
                    "generation_cap_hit": len(new_tokens) >= MAX_NEW_TOKENS and not any(
                        int(token) in eos_ids for token in new_tokens),
                    "latency_seconds": latency, "raw_text": raw_text,
                }
                rows_by_id[sample_id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"done": len(rows_by_id), "total": len(source), "id": sample_id,
                                  "parsed": row["parsed"], "tokens": row["generated_tokens"],
                                  "seconds": latency}, ensure_ascii=False), flush=True)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())

    previous = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
    ordered = [rows_by_id[sample_id] for sample_id, _ in selected]
    summary = {**config, **summarize(ordered, len(source),
                                    max(peak_allocated, int(previous.get("gpu_peak_allocated_bytes", 0))),
                                    max(peak_reserved, int(previous.get("gpu_peak_reserved_bytes", 0)))),
               "limit": args.limit, "predictions_jsonl": str(rows_path)}
    if not args.limit and len(rows_by_id) == len(source):
        json_path, zip_path = package_submission(source, rows_by_id, output_dir)
        summary.update(predictions_json=str(json_path), submission_zip=str(zip_path))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
