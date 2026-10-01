"""Run a frozen GT-free visual-agent cohort using a local Qwen controller."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.predict_aux_selection import load_manifest, read_jsonl, index_rows_by_id
from .controller import SYSTEM_PROMPT, run_episode
from .model import QwenBackend, resize_for_budget
from .profiles import PROFILES
from .vision_tools import CandidatePool, VisualTools
from .presentation import PROMPT_VERSION, model_candidates, compact_json


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    result["images"] = {key: path_map(value, mappings, relative_to) for key, value in row["images"].items()}
    for candidate in result.get("candidates", []):
        if candidate.get("mask_path"):
            candidate["mask_path"] = path_map(candidate["mask_path"], mappings, relative_to)
    return result


def build_initial_messages(row, candidates, directory, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keep_box = next(candidate["bbox"] for candidate in candidates if str(candidate["id"]) == "KEEP")
    content = [{"type": "text", "text":
                f"Full query: {row['query']}\nInitial candidates (RGB normalized xyxy): " +
                compact_json(model_candidates(candidates)) +
                "\nKEEP is the exact initial prediction. Boxes denote hypotheses, including possibly overlapping boxes of the same object.",
                "latest_memory_text":
                f"Full query: {row['query']}\nKEEP is the exact initial prediction (RGB normalized xyxy): " +
                compact_json(keep_box) +
                "\nUse the current candidate pool for other IDs and boxes. Boxes denote hypotheses, including possibly overlapping boxes of the same object."}]
    for modality, key in (("rgb", "rgb"), ("ir", "ir"), ("depth", "depth_visual")):
        if not row["images"].get(key):
            content.append({"type": "text", "text": f"Global {modality}: unavailable."})
            continue
        with Image.open(row["images"][key]) as source:
            original_size = list(source.size)
            image = resize_for_budget(source.convert("RGB"), profile.global_pixels)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=16)
        for candidate in candidates:
            box = candidate["bbox"]
            coords = (box[0] * image.width, box[1] * image.height,
                      box[2] * image.width, box[3] * image.height)
            color = "#ff3030" if candidate["role"] == "target" else "#00a6ff"
            draw.rectangle(coords, outline=color, width=max(2, image.width // 600))
            draw.text((coords[0], max(0, coords[1] - 18)), str(candidate["id"]), fill="white",
                      font=font, stroke_width=2, stroke_fill="black")
        image_path = directory / f"global_{modality}.png"
        image.save(image_path)
        guide = {
            "rgb": "Visible appearance, color and global spatial/ordinal reference.",
            "ir": "Aligned infrared appearance: examine silhouettes and relative contrast at the RGB candidate locations, especially where RGB identity is uncertain. Brightness alone is not a target label.",
            "depth": "Depth visualization: examine scene layers. It is not a color photograph; verify valid camera-distance ordering with measure_depth.",
        }[modality]
        content += [{"type": "text", "text": f"Global {modality}. {guide} Red boxes are target hypotheses; blue boxes are reference objects. All scene positions refer to this original frame."},
                    {"type": "image", "image": str(image_path.resolve()), "max_pixels": profile.global_pixels,
                     "modality": modality, "view": "global", "candidate_ids": [c["id"] for c in candidates],
                     "original_size": original_size,
                     "target_boxes": [{"id": str(c["id"]), "bbox": c["bbox"]} for c in candidates]}]
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mechanism", choices=["A", "B", "S", "C", "D"], required=True)
    p.add_argument("--controller", choices=["native", "c-lora", "t-lora"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--profile", choices=PROFILES, default="fast")
    p.add_argument("--memory", choices=["full", "latest"], default="full")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--evidence-cache", type=Path)
    p.add_argument("--dino-model")
    p.add_argument("--sam-model")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-run-seconds", type=float)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.controller == "native") != (args.adapter is None):
        raise ValueError("native requires no adapter; c-lora/t-lora require an explicit adapter")
    if args.controller == "t-lora" and (args.profile != "sft" or args.memory != "latest"):
        raise ValueError("t-lora must use --profile sft --memory latest to match training")
    if (args.profile == "sft") != (args.memory == "latest"):
        raise ValueError("the sft profile and latest memory must be selected together, including its untrained control")
    profile = PROFILES[args.profile]
    rows = load_manifest(args.manifest, require_images=True)
    if args.limit is not None:
        rows = rows[:args.limit]
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [map_row_paths(row, mappings, args.manifest.parent) for row in rows]
    candidates = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    evidence = (index_rows_by_id(read_jsonl(args.evidence_cache), source=str(args.evidence_cache))
                if args.evidence_cache else {})
    out = args.output_dir.resolve()
    config = {"schema_version": "visual-agent-v1", "mechanism": args.mechanism,
              "prompt_version": PROMPT_VERSION,
              "controller": args.controller, "model": args.model, "adapter": args.adapter,
              "profile": profile.to_dict(), "memory": args.memory,
              "manifest": str(args.manifest.resolve()), "candidate_cache": str(args.candidate_cache.resolve()),
              "evidence_cache": str(args.evidence_cache.resolve()) if args.evidence_cache else None,
              "path_map": args.path_map, "dino_model": args.dino_model, "sam_model": args.sam_model,
              "device": args.device, "limit": args.limit,
              "decode": {"do_sample": False, "dtype": "bfloat16", "attention": "sdpa"}}
    config_path = out / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"run exists: {out}; use a new directory or --resume")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume config differs from frozen run")
    elif args.resume:
        raise FileNotFoundError("cannot resume without run_config.json")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    predictions_path = out / "predictions.jsonl"
    completed = index_rows_by_id(read_jsonl(predictions_path), source=str(predictions_path)) if predictions_path.exists() else {}
    expected = {str(row["id"]) for row in rows}
    if not set(completed) <= expected:
        raise ValueError("resume predictions contain IDs outside the frozen manifest")
    remaining = [row for row in rows if str(row["id"]) not in completed]
    if not remaining:
        print(json.dumps({"status": "complete", "count": len(completed)}))
        return
    started = time.perf_counter()
    backend = QwenBackend(args.model, args.adapter, device=args.device)
    # CPU detector/segmenter instances are shared between episodes after lazy loading.
    dino, sam = None, None
    with predictions_path.open("a", encoding="utf-8") as handle:
        for row in remaining:
            if args.max_run_seconds is not None and time.perf_counter() - started >= args.max_run_seconds:
                break
            sample_id = str(row["id"])
            preparation_start = time.perf_counter()
            if args.dino_model:
                row["dino_model_path"] = args.dino_model
            if args.sam_model:
                row["sam_model_path"] = args.sam_model
            candidate_row = map_row_paths(candidates[sample_id], mappings, args.candidate_cache.parent)
            evidence_row = (map_row_paths(evidence[sample_id], mappings, args.evidence_cache.parent)
                            if sample_id in evidence else None)
            with Image.open(row["images"]["rgb"]) as image:
                pool = CandidatePool(candidate_row, image.size)
            sample_dir = out / "traces" / sample_id
            initial = build_initial_messages(row, pool.public_candidates(), sample_dir, profile)
            tools = VisualTools(row, pool, sample_dir / "observations", profile.tool_pixels,
                dino_model=dino, sam_model=sam, evidence_row=evidence_row,
                max_searches=profile.search_calls, allow_search=args.mechanism == "D")
            preparation_seconds = time.perf_counter() - preparation_start
            # Preserve model decisions even if a real tool/backend error stops this run.
            with (sample_dir / "events.jsonl").open("a", encoding="utf-8") as event_handle:
                def record_event(event):
                    event_handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                    event_handle.flush()
                try:
                    trace = run_episode(row, pool, tools, backend, initial, mechanism=args.mechanism,
                                        profile=profile, memory=args.memory, on_event=record_event)
                except Exception as error:
                    record_event({"phase": "fatal_error", "type": type(error).__name__, "message": str(error)})
                    raise
            trace["metrics"]["preparation_seconds"] = preparation_seconds
            trace["metrics"]["elapsed_seconds"] += preparation_seconds
            dino, sam = tools.dino_model, tools.sam_model
            trace_path = sample_dir / "trace.json"
            write_json(trace_path, trace)
            prediction = {key: value for key, value in trace.items() if key not in
                          {"events", "initial_messages", "manifest_row", "query", "final_pool_snapshot", "initial_pool_snapshot"}}
            prediction["trace_path"] = str(trace_path)
            handle.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            completed[sample_id] = prediction
            print(json.dumps({"id": sample_id, "status": trace["final_status"],
                              "selected_id": trace["selected_id"], "completed": len(completed)}, ensure_ascii=False), flush=True)
    write_json(out / "execution.json", {"expected": len(rows), "completed": len(completed),
               "complete": len(completed) == len(rows), "model_load_seconds": backend.load_seconds,
               "invocation_wall_seconds": time.perf_counter() - started,
               "gpu_hours_this_invocation": (time.perf_counter() - started) / 3600,
               "scored": False})


if __name__ == "__main__":
    main()
