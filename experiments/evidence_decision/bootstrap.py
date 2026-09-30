"""Create C-compatible starting assets for train32/T250 without reading GT online.

Run from the code root as ``python -m experiments.evidence_decision.bootstrap``.
The prompts stage only writes the original C prompt map and planned Qwen CLI
arguments; it does not start GPU work. The candidates stage loads DINO on CPU.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from PIL import Image

from tools.aux_selection_evidence import (
    BOX_THRESHOLD, TEXT_THRESHOLD, DEDUP_IOU, ID_SEED,
    MAX_TARGET_CANDIDATES, MAX_REFERENCE_CANDIDATES,
    build_candidate_row, detect_image_proposals, resolve_data_path,
)
from tools.predict_aux_selection import load_manifest, parse_query_json, require_box
from tools.prepare_qwen3vl_native_sft import _trimodal_prompt


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def by_id(rows: list[dict], label: str) -> dict[str, dict]:
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {label} IDs")
    return result


def input_stat(path: Path) -> dict:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prompts(args) -> dict:
    rows = load_manifest(args.manifest, require_images=True)
    mapping = {str(row["id"]): _trimodal_prompt(row["query"]) for row in rows}
    command_inputs = (args.model, args.adapter, args.baseline_output, args.query_output_dir)
    if any(command_inputs) and not all(command_inputs):
        raise ValueError("Qwen command planning needs --model, --adapter, --baseline-output and --query-output-dir together")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8-sig"))
        if existing != mapping:
            raise ValueError(f"existing C prompt map differs: {args.output}")
    else:
        args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"queries": len(rows), "prompts": str(args.output),
              "prompt_source": "tools.prepare_qwen3vl_native_sft._trimodal_prompt"}
    if all(command_inputs):
        common = ["--manifest", str(args.manifest), "--model", args.model,
                  "--adapter", str(args.adapter)]
        result["commands"] = [
            ["python", "-m", "tools.predict_aux_selection", "baseline", *common,
             "--prompts", str(args.output), "--output", str(args.baseline_output)],
            ["python", "-m", "tools.predict_aux_selection", "parse-query", *common,
             "--output-dir", str(args.query_output_dir)],
        ]
        result["query_info"] = str(args.query_output_dir / "query_info.jsonl")
    return result


def cpu_dino(model_name: str, threads: int):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_name, local_files_only=True,
    ).eval().to(device="cpu", dtype=torch.float32)
    return processor, model


def load_baseline(path: Path, ids: set[str]) -> dict[str, dict]:
    rows = by_id(read_jsonl(path), "C baseline")
    if set(rows) != ids:
        raise ValueError(f"baseline ID set differs from manifest: missing={len(ids-set(rows))}, extra={len(set(rows)-ids)}")
    return {sample_id: {"id": sample_id, "bbox": require_box(
        row.get("bbox", row.get("prediction")), field=f"C prediction {sample_id}")}
            for sample_id, row in rows.items()}


def load_query_info(path: Path, ids: set[str]) -> dict[str, dict]:
    rows = by_id(read_jsonl(path), "query-info")
    if set(rows) != ids:
        raise ValueError(f"query-info ID set differs from manifest: missing={len(ids-set(rows))}, extra={len(set(rows)-ids)}")
    result = {}
    for sample_id, row in rows.items():
        info = {key: row.get(key) for key in
                ("target_category", "reference_categories", "relation_type", "scope")}
        parsed = parse_query_json(json.dumps(info, ensure_ascii=False))
        if row.get("parsed") is not True or parsed is None:
            raise ValueError(f"invalid C query parse for {sample_id}")
        result[sample_id] = parsed
    return result


def completed_rows(output: Path, config: dict, ids: set[str], resume: bool) -> dict[str, dict]:
    config_path = output.with_suffix(output.suffix + ".config.json")
    if resume:
        if not config_path.exists():
            raise FileNotFoundError(f"resume config missing: {config_path}")
        if json.loads(config_path.read_text(encoding="utf-8-sig")) != config:
            raise ValueError("candidate resume config differs")
        rows = by_id(read_jsonl(output), "existing candidates") if output.exists() else {}
        if not set(rows) <= ids:
            raise ValueError("existing candidate output has IDs outside manifest")
        return rows
    if output.exists() or config_path.exists():
        raise FileExistsError(f"candidate output already exists; pass --resume: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {}


def candidates(args) -> dict:
    rows = load_manifest(args.manifest, require_images=True)
    ids = {str(row["id"]) for row in rows}
    baseline = load_baseline(args.baseline, ids)
    query_info = load_query_info(args.query_info, ids)
    config = {
        "stage": "visual_agent_initial_candidates", "manifest": input_stat(args.manifest),
        "baseline": input_stat(args.baseline), "query_info": input_stat(args.query_info),
        "data_root": str(args.data_root.resolve()), "dino_model": args.model,
        "device": "cpu", "threads": args.threads, "id_seed": ID_SEED,
        "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD,
        "dedup_iou": DEDUP_IOU, "max_target_candidates": MAX_TARGET_CANDIDATES,
        "max_reference_candidates": MAX_REFERENCE_CANDIDATES,
    }
    existing = completed_rows(args.output, config, ids, args.resume)
    pending = [row for row in rows if str(row["id"]) not in existing]
    processor = model = None
    load_seconds = 0.0
    if any(query_info[str(row["id"])]["scope"] == "single" for row in pending):
        started = time.perf_counter()
        processor, model = cpu_dino(args.model, args.threads)
        load_seconds = time.perf_counter() - started
    run_started = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            sample_id = str(row["id"])
            if sample_id in existing:
                continue
            if args.max_run_seconds and time.perf_counter() - run_started >= args.max_run_seconds:
                break
            started = time.perf_counter()
            detections = []
            if query_info[sample_id]["scope"] == "single":
                for modality in ("rgb", "ir"):
                    image_path = resolve_data_path(row["images"][modality], args.data_root)
                    with Image.open(image_path) as source:
                        image = source.convert("RGB").copy()
                    detections.extend(detect_image_proposals(
                        image, modality, query_info[sample_id], processor, model, "cpu"))
            result = build_candidate_row(row, baseline[sample_id], query_info[sample_id],
                                         detections, random.Random(ID_SEED + index))
            result["initial_candidate_seconds"] = time.perf_counter() - started
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            existing[sample_id] = result
            print(json.dumps({"id": sample_id, "completed": len(existing), "total": len(rows),
                              "seconds": result["initial_candidate_seconds"]}), flush=True)
    summary = {"expected": len(rows), "completed": len(existing),
               "complete": len(existing) == len(rows), "cpu_model_load_seconds": load_seconds,
               "candidate_generation_seconds": sum(row.get("initial_candidate_seconds", 0.0)
                                                   for row in existing.values()),
               "output": str(args.output), "device": "cpu"}
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def training_gt(args) -> dict:
    """Map original raw floating GT into a separate offline scoring manifest."""
    if "scoring" not in args.output.parts:
        raise ValueError("training GT output must be inside a scoring directory")
    raw = json.loads(args.train_source.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("original qwen_generation_train_100 must be keyed by ID")
    manifest = load_manifest(args.manifest, require_images=True)
    result = {}
    for row in manifest:
        sample_id = str(row["id"])
        source = raw[sample_id]
        if source["query"] != row["query"]:
            raise ValueError(f"raw train GT Query differs for {sample_id}")
        bbox = require_box(source["bbox"], field=f"raw train GT {sample_id}")
        result[sample_id] = {
            "visible": row["images"]["rgb"], "infrared": row["images"]["ir"],
            "depth": row["images"]["depth_raw"], "query": row["query"],
            "bbox": bbox,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text(encoding="utf-8-sig")) != result:
            raise ValueError(f"existing training GT differs: {args.output}")
    else:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"queries": len(result), "output": str(args.output),
            "source": str(args.train_source), "online_manifest_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prompts", "candidates", "training-gt"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--query-info", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--adapter", help="Adapter path for printed Qwen commands; preserved verbatim")
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--query-output-dir", type=Path)
    parser.add_argument("--train-source", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prompts":
        result = prompts(args)
    elif args.stage == "candidates":
        for key in ("baseline", "query_info", "data_root", "model"):
            if getattr(args, key) is None:
                parser.error(f"--{key.replace('_', '-')} is required for candidates")
        result = candidates(args)
    else:
        if args.train_source is None:
            parser.error("--train-source is required for training-gt")
        result = training_gt(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
