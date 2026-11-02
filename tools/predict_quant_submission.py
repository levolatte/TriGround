"""Run unlabeled official queries with the ACC-correction HF/NF4 protocol.

The generation implementation is shared with eval_quant_grounding. Only the
official data conversion and final submission packaging are added here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import (  # noqa: E402
    completed_prediction_prefix, generate_rows, positive_int, read_jsonl, write_jsonl,
)
from tools.export_quant_submission import make_submission  # noqa: E402
from tools.prepare_qwen3vl_native_sft import native_prompt, render_depth  # noqa: E402


IMAGE_FIELDS = ("visible", "infrared", "depth")


def load_queries(path: Path, expected_count: int) -> dict:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(source, dict) or not source or len(source) != expected_count:
        raise ValueError(f"expected {expected_count} official queries")
    for sample_id, row in source.items():
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key]
            for key in (*IMAGE_FIELDS, "query")
        ):
            raise ValueError(f"{sample_id}: image paths and query must be nonempty strings")
    return source


def prepare_dataset(source: dict, data_root: Path, output_dir: Path) -> list[dict]:
    """Keep the City prompt; adapt raw millimetres and supplied RGB depth views."""
    rendered: dict[Path, tuple[Path, str]] = {}
    dataset = []
    for sample_id, row in source.items():
        paths = [(data_root / row[key]).resolve() for key in IMAGE_FIELDS]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"{sample_id}: {path}")
        depth = paths[2]
        if depth not in rendered:
            with Image.open(depth) as image:
                pixels = np.asarray(image)
            if pixels.dtype == np.uint16 and pixels.ndim == 2:
                destination = (output_dir / "depth_rgb" / f"{len(rendered):06d}.png").resolve()
                render_depth(depth, destination)
                rendered[depth] = (destination, "millimeter")
            elif pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3:
                # Already-rendered official depth views have no recoverable metric scale.
                # Pass their original pixels through, as predict_native_submission does.
                rendered[depth] = (depth, "visual")
            else:
                raise ValueError(f"unsupported official depth encoding: {pixels.shape} {pixels.dtype}")
            if len(rendered) % 100 == 0:
                print(json.dumps({"prepared_depth_images": len(rendered)}), flush=True)
        paths[2], depth_policy = rendered[depth]
        dataset.append({
            "id": sample_id,
            "images": [str(path) for path in paths],
            "modalities": ["rgb", "infrared", "depth"],
            "depth_policy": depth_policy,
            "messages": [{"role": "user", "content": native_prompt(
                row["query"], ("rgb", "infrared", "depth"), "millimeter")}],
        })
    return dataset


def package_predictions(source: dict, predictions: list[dict], output_dir: Path,
                        format_policy: str) -> dict:
    result = make_submission(source, predictions, "short", format_policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "predictions.json"
    zip_path = output_dir / "submission.zip"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, "predictions.json")
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["predictions.json"]:
            raise AssertionError("ZIP must contain predictions.json at its root")
        packed = json.loads(archive.read("predictions.json"))
    if list(packed) != list(source) or packed != result:
        raise AssertionError("packaged JSON differs from source-preserving predictions")
    return {"queries": len(result), "json": str(json_path), "zip": str(zip_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--predictions", type=Path, help="Package saved raw JSONL without CUDA")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Append remaining queries after the saved raw ID prefix")
    parser.add_argument("--expected-query-count", type=int, default=5691)
    parser.add_argument("--limit", type=int, default=0, help="Infer first N queries; never package")
    parser.add_argument("--format-policy", choices=["strict", "punctuation", "single_box"], default="punctuation")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=1204224)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--hf-max-batch-size", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.limit < 0 or args.expected_query_count <= 0:
        raise ValueError("limit must be nonnegative and expected-query-count positive")
    if not 0 < args.min_pixels <= args.max_pixels or not 0 < args.max_tokens < args.max_length:
        raise ValueError("inconsistent pixel/token limits")
    if args.resume and (args.prepare_only or args.predictions):
        raise ValueError("resume cannot be combined with prepare-only or predictions")
    if args.predictions and (args.prepare_only or args.limit):
        raise ValueError("packaging cannot be combined with prepare-only or limit")
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    source = load_queries(args.queries, args.expected_query_count)
    if args.predictions:
        return package_predictions(source, read_jsonl(args.predictions), args.output_dir, args.format_policy)
    if not args.prepare_only and not args.model:
        raise ValueError("generation requires --model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only and not args.resume and (args.output_dir / "predictions_raw.jsonl").exists():
        raise FileExistsError("raw predictions already exist; choose a fresh output directory")
    selected = dict(list(source.items())[:args.limit]) if args.limit else source
    if args.resume:
        dataset = read_jsonl(args.output_dir / "unlabeled.jsonl")
        if [row["id"] for row in dataset] != list(selected):
            raise ValueError("saved dataset IDs differ from the selected official query order")
        completed = completed_prediction_prefix(args.output_dir / "predictions_raw.jsonl", dataset)
    else:
        dataset = prepare_dataset(selected, args.data_root, args.output_dir)
        write_jsonl(args.output_dir / "unlabeled.jsonl", dataset)
    config = vars(deepcopy(args))
    config.update(engine="hf", precision="nf4", answer_mode="short", samples_per_query=1,
                  bounded_thinking=False, prompt_source="prepare_qwen3vl_native_sft.native_prompt",
                  depth_policy="uint16_mm_absolute_grayscale_or_uint8_rgb_visualization_v2",
                  prompt_depth_policy="millimeter", expected_queries=len(source), selected_queries=len(dataset))
    if args.resume:
        saved = json.loads((args.output_dir / "run_config.json").read_text(encoding="utf-8"))
        for key in ("model", "adapter", "queries", "data_root", "min_pixels", "max_pixels", "max_length",
                    "max_tokens", "seed", "format_policy", "engine", "precision", "answer_mode"):
            value = str(config[key]) if isinstance(config[key], Path) else config[key]
            if saved[key] != value:
                raise ValueError(f"resume configuration differs: {key}")
        saved.setdefault("resume_history", []).append({
            "completed_queries": len(completed), "hf_max_batch_size": args.hf_max_batch_size,
        })
        config = saved
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if args.prepare_only:
        return {"prepared_queries": len(dataset), "dataset": str(args.output_dir / "unlabeled.jsonl")}
    args.engine, args.precision, args.answer_mode, args.samples_per_query = "hf", "nf4", "short", 1
    predictions, resource = generate_rows(args, dataset)
    summary = {"completed_queries": len(predictions), "expected_queries": len(source),
               "format_policy": args.format_policy, "resource": resource}
    summary_path = args.output_dir / "summary.json"
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["previous_resources"] = [*previous.get("previous_resources", []), previous["resource"]]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.limit:
        summary.update(package_predictions(source, predictions, args.output_dir, args.format_policy))
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
