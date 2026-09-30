"""Run the object-centered TriGround controller on a frozen GT-free cohort."""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from tools.predict_aux_selection import load_manifest, read_jsonl, index_rows_by_id
from .model import QwenBackend
from .object_controller import OBJECT_PROFILE, PROMPT_VERSION, build_initial_messages, run_episode
from .object_tools import ObjectTools
from .vision_tools import CandidatePool


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def path_map(value, mappings, relative_to):
    if value is None:
        return None
    normalized = str(value).replace("\\", "/")
    for source, destination in mappings:
        if normalized == source or normalized.startswith(source.rstrip("/") + "/"):
            return str(Path(destination) / normalized[len(source):].lstrip("/"))
    path = Path(value)
    return str(path if path.is_absolute() else relative_to / path)


def map_row_paths(row, mappings, relative_to):
    result = copy.deepcopy(row)
    result["images"] = {key: path_map(value, mappings, relative_to)
                        for key, value in row["images"].items()}
    for candidate in result.get("candidates", []):
        if candidate.get("mask_path"):
            candidate["mask_path"] = path_map(candidate["mask_path"], mappings, relative_to)
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--candidate-cache", type=Path, required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--adapter", required=True,
                       help="Frozen C for the untrained reference, or the trained controller")
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--dino-model")
    value.add_argument("--sam-model")
    value.add_argument("--limit", type=int)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--max-run-seconds", type=float)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--context-tokens", type=int, default=8192,
                       help="Context limit passed to the native Qwen processor profile")
    value.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    return value


def run_config(args, profile):
    return {
        "schema_version": "visual-agent-evidence-decision",
        "prompt_version": PROMPT_VERSION,
        "model": args.model, "adapter": args.adapter,
        "profile": profile.__dict__.copy(),
        "manifest": str(args.manifest.resolve()),
        "candidate_cache": str(args.candidate_cache.resolve()),
        "path_map": args.path_map, "dino_model": args.dino_model, "sam_model": args.sam_model,
        "device": args.device, "limit": args.limit,
        "decode": {"do_sample": False, "dtype": "bfloat16", "attention": "sdpa"},
    }


def main(argv=None):
    args = parser().parse_args(argv)
    if args.context_tokens < 1:
        raise ValueError("--context-tokens must be positive")
    profile = replace(OBJECT_PROFILE, context_tokens=args.context_tokens)
    rows = load_manifest(args.manifest, require_images=True)
    source_rows = index_rows_by_id(read_jsonl(args.manifest), source=str(args.manifest))
    if args.limit is not None:
        rows = rows[:args.limit]
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [map_row_paths(row, mappings, args.manifest.parent) for row in rows]
    candidates = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    out = args.output_dir.resolve()
    config = run_config(args, profile)
    config_path = out / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"run exists: {out}; use another output directory or --resume")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume config differs from frozen run")
    elif args.resume:
        raise FileNotFoundError("cannot resume without run_config.json")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)

    predictions_path = out / "predictions.jsonl"
    completed = (index_rows_by_id(read_jsonl(predictions_path), source=str(predictions_path))
                 if predictions_path.exists() else {})
    expected = {str(row["id"]) for row in rows}
    if not set(completed) <= expected:
        raise ValueError("resume predictions contain IDs outside the frozen manifest")
    remaining = [row for row in rows if str(row["id"]) not in completed]
    if not remaining:
        print(json.dumps({"status": "complete", "count": len(completed)}))
        return

    invocation_started = time.perf_counter()
    backend = QwenBackend(args.model, args.adapter, device=args.device)
    dino, sam = None, None
    with predictions_path.open("a", encoding="utf-8") as prediction_handle:
        for row in remaining:
            if args.max_run_seconds is not None and time.perf_counter() - invocation_started >= args.max_run_seconds:
                break
            sample_id = str(row["id"])
            for field in ("ir_rgb_registration", "registration_source", "depth_visual_encoding",
                          "source", "image_group", "scene_id"):
                if field in source_rows[sample_id]:
                    row[field] = source_rows[sample_id][field]
            preparation_started = time.perf_counter()
            if args.dino_model:
                row["dino_model_path"] = args.dino_model
            if args.sam_model:
                row["sam_model_path"] = args.sam_model
            candidate_row = map_row_paths(candidates[sample_id], mappings, args.candidate_cache.parent)
            with Image.open(row["images"]["rgb"]) as rgb:
                pool = CandidatePool(candidate_row, rgb.size)
            sample_dir = out / "traces" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            tools = ObjectTools(row, pool, sample_dir / "observations",
                                tool_pixels=profile.tool_pixels,
                                dino_model=dino, sam_model=sam, seed=2026)
            atlas = tools.atlas(modalities=["rgb"])
            initial = build_initial_messages(row, tools.public_candidates(), atlas, profile)
            preparation_seconds = time.perf_counter() - preparation_started

            events_path = sample_dir / "events.jsonl"
            with events_path.open("a", encoding="utf-8") as event_handle:
                def record_event(event):
                    event_handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    event_handle.flush()

                try:
                    trace = run_episode(row, tools, backend, initial, profile=profile,
                                        on_event=record_event)
                    trace["initial_atlas"] = atlas
                except Exception as error:
                    record_event({"phase": "fatal_error", "type": type(error).__name__, "message": str(error)})
                    raise
            trace["metrics"]["preparation_seconds"] = preparation_seconds
            trace["metrics"]["elapsed_seconds"] += preparation_seconds
            dino, sam = tools.visual.dino_model, tools.visual.sam_model
            trace_path = sample_dir / "trace.json"
            write_json(trace_path, trace)
            prediction = {key: value for key, value in trace.items()
                          if key not in {"events", "initial_messages", "initial_atlas", "query"}}
            prediction["trace_path"] = str(trace_path)
            prediction_handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            prediction_handle.flush()
            completed[sample_id] = prediction
            print(json.dumps({"id": sample_id, "status": trace["final_status"],
                              "selected_id": trace["selected_id"],
                              "selected_public_id": trace["selected_public_id"],
                              "completed": len(completed)}, ensure_ascii=False), flush=True)

    elapsed = time.perf_counter() - invocation_started
    write_json(out / "execution.json", {
        "expected": len(rows), "completed": len(completed), "complete": len(completed) == len(rows),
        "model_load_seconds": backend.load_seconds, "invocation_wall_seconds": elapsed,
        "gpu_hours_this_invocation": elapsed / 3600, "scored": False,
    })


if __name__ == "__main__":
    main()
