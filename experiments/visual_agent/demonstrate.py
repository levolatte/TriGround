"""Run GT-free, scripted demonstrations through the real CPU visual tools.

This entry point records observations only. A separate offline exporter may
label a final decision against GT, using the candidates visible at that time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image

from tools.predict_aux_selection import index_rows_by_id, load_manifest, read_jsonl
from .controller import text_message
from .export_trajectories import load_frozen_manifest, verify_frozen_splits
from .presentation import PROMPT_VERSION
from .profiles import PROFILES
from .run import build_initial_messages, map_row_paths, write_json
from .state import build_state_messages
from .vision_tools import CandidatePool, VisualTools


DECISION_INSTRUCTION = (
    "Identify the unresolved query condition in evidence_note and choose one observation that can "
    "distinguish the competing IDs, or finish using observed evidence. Consider IR for uncertain "
    "object identity/visibility and measured Depth for camera-distance layers; RGB-only zoom is "
    "useful only if it can resolve the remaining condition. search_candidates is enabled."
)
FINAL_INSTRUCTION = (
    "Final decision now. Only action=finish with a current target candidate_id or KEEP is legal. "
    "No tools or coordinates."
)
TOOL_ACTIONS = {"inspect_regions", "measure_depth", "search_candidates"}


def frozen_train_rows(manifest: Path, holdout_manifest: Path, debug_manifest: Path) -> list[dict]:
    """Accept only the frozen T train partition and verify group isolation."""
    manifest = Path(manifest)
    if manifest.name != "t_train200.jsonl":
        raise ValueError("demonstrations require the frozen t_train200.jsonl manifest")
    train = load_frozen_manifest(manifest)
    holdout = load_frozen_manifest(holdout_manifest)
    debug = load_frozen_manifest(debug_manifest)
    verify_frozen_splits(train, holdout, debug)
    rows = load_manifest(manifest, require_images=True)
    if set(train) != {str(row["id"]) for row in rows}:
        raise ValueError("manifest differs from the frozen train partition")
    return rows


def demonstrate_one(
    row: dict, candidate_row: dict, sample_dir: Path, *,
    dino_model=None, sam_model=None, evidence_row=None, policy=None,
    max_steps: int = 6, on_event=None, source_row=None,
) -> tuple[dict, VisualTools]:
    """Execute up to six policy actions; never select or label a final ID."""
    if policy is None:
        from .demonstration_policy import next_action
        policy = next_action
    profile = PROFILES["sft"]
    if not 1 <= max_steps <= profile.evidence_calls:
        raise ValueError("max_steps must fit the frozen sft evidence-call budget")
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(row["images"]["rgb"]) as image:
        pool = CandidatePool(candidate_row, image.size)
    initial_candidates = pool.public_candidates()
    initial_snapshot = pool.snapshot()
    initial = build_initial_messages(row, initial_candidates, sample_dir, profile)
    tools = VisualTools(
        row, pool, sample_dir / "observations", profile.tool_pixels,
        dino_model=dino_model, sam_model=sam_model, evidence_row=evidence_row,
        max_searches=profile.search_calls, allow_search=True,
    )
    semantic = candidate_row.get("query_info", {})
    events = []
    started = time.perf_counter()
    stop_reason = "evidence_limit"
    for step in range(max_steps):
        before = pool.public_candidates()
        action = policy(row, before, events, semantic)
        if action is None:
            stop_reason = "policy_stop"
            break
        if not isinstance(action, dict) or action.get("action") not in TOOL_ACTIONS:
            raise ValueError(f"scripted policy returned a non-tool action: {action}")
        messages = build_state_messages(initial, events, before)
        messages.append(text_message(DECISION_INSTRUCTION))
        tool_start = time.perf_counter()
        observation = tools.execute(action)
        seconds = time.perf_counter() - tool_start
        event = {
            "step": step, "origin": "scripted", "messages": messages,
            "candidates_before": before, "action": action, "executed_action": action,
            "observation": observation, "candidates_after": pool.public_candidates(),
            "tool_seconds": seconds,
        }
        events.append(event)
        if on_event:
            on_event(event)
        if observation["status"] in {"ERROR", "LIMIT"}:
            raise ValueError(f"scripted tool action failed at {row['id']} step {step}: {observation}")
    terminal_messages = build_state_messages(initial, events, pool.public_candidates())
    terminal_messages.append(text_message(
        DECISION_INSTRUCTION if stop_reason == "policy_stop" else FINAL_INSTRUCTION))
    trace = {
        "schema_version": "visual-agent-v1", "prompt_version": PROMPT_VERSION,
        "id": str(row["id"]), "query": row["query"],
        "manifest_row": source_row if source_row is not None else row,
        "initial_messages": initial, "initial_candidates": initial_candidates,
        "initial_bbox": pool.finish("KEEP"), "image_group": row["images"]["rgb"],
        "events": events, "terminal_messages": terminal_messages,
        "stop_reason": stop_reason,
        "final_status": "OBSERVATION_COMPLETE", "selected_id": None, "bbox": None,
        "initial_pool_snapshot": initial_snapshot, "final_pool_snapshot": pool.snapshot(),
        "final_candidates": pool.public_candidates(),
        "mechanism": "D", "profile": profile.to_dict(), "memory": "latest",
        "trajectory_origin": "scripted", "label_source": "pending_offline_teacher",
        "metrics": {
            "tool_calls": len(events), "search_calls": sum(event["action"]["action"] == "search_candidates" for event in events),
            "tool_status_counts": dict(Counter(event["observation"]["status"] for event in events)),
            "tool_seconds": sum(event["tool_seconds"] for event in events),
            "elapsed_seconds": time.perf_counter() - started,
            **tools.cost_counts,
        },
    }
    return trace, tools


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="Frozen t_train200.jsonl; siblings holdout/debug are checked")
    p.add_argument("--holdout-manifest", type=Path, help="Frozen t_holdout50.jsonl; defaults to manifest sibling")
    p.add_argument("--debug-manifest", type=Path, help="Frozen train32.jsonl; defaults to manifest sibling")
    p.add_argument("--candidate-cache", type=Path, required=True)
    p.add_argument("--evidence-cache", type=Path)
    p.add_argument("--dino-model")
    p.add_argument("--sam-model")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    p.add_argument("--max-steps", type=int, choices=range(1, 7), default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    holdout_manifest = args.holdout_manifest or args.manifest.with_name("t_holdout50.jsonl")
    debug_manifest = args.debug_manifest or args.manifest.with_name("train32.jsonl")
    frozen_rows = frozen_train_rows(args.manifest, holdout_manifest, debug_manifest)
    if args.limit is not None:
        frozen_rows = frozen_rows[:args.limit]
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    rows = [map_row_paths(row, mappings, args.manifest.parent) for row in frozen_rows]
    candidates = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    evidence = (index_rows_by_id(read_jsonl(args.evidence_cache), source=str(args.evidence_cache))
                if args.evidence_cache else {})
    out = args.output_dir.resolve()
    config = {
        "schema_version": "visual-agent-demonstration-v1", "trajectory_origin": "scripted",
        "prompt_version": PROMPT_VERSION, "profile": PROFILES["sft"].to_dict(), "memory": "latest",
        "manifest": str(args.manifest.resolve()), "candidate_cache": str(args.candidate_cache.resolve()),
        "holdout_manifest": str(holdout_manifest.resolve()), "debug_manifest": str(debug_manifest.resolve()),
        "evidence_cache": str(args.evidence_cache.resolve()) if args.evidence_cache else None,
        "dino_model": args.dino_model, "sam_model": args.sam_model,
        "path_map": args.path_map, "max_steps": args.max_steps, "limit": args.limit,
        "device": "cpu", "final_label": "offline_only",
    }
    config_path = out / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"demonstration run exists: {out}")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume config differs from the existing demonstration")
    elif args.resume:
        raise FileNotFoundError("cannot resume without run_config.json")
    out.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    dino = sam = None
    completed = 0
    started = time.perf_counter()
    for source_row, row in zip(frozen_rows, rows, strict=True):
        sample_id = str(row["id"])
        sample_dir = out / "traces" / sample_id
        trace_path = sample_dir / "trace.json"
        if trace_path.exists():
            if not args.resume:
                raise FileExistsError(trace_path)
            completed += 1
            continue
        if args.dino_model:
            row["dino_model_path"] = args.dino_model
        if args.sam_model:
            row["sam_model_path"] = args.sam_model
        candidate_row = map_row_paths(candidates[sample_id], mappings, args.candidate_cache.parent)
        evidence_row = (map_row_paths(evidence[sample_id], mappings, args.evidence_cache.parent)
                        if sample_id in evidence else None)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with (sample_dir / "events.jsonl").open("w", encoding="utf-8") as event_handle:
            def record_event(event):
                event_handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                event_handle.flush()
            try:
                trace, tools = demonstrate_one(
                    row, candidate_row, sample_dir, dino_model=dino, sam_model=sam,
                    evidence_row=evidence_row, max_steps=args.max_steps, on_event=record_event,
                    source_row=source_row,
                )
            except Exception as error:
                write_json(out / "failure.json", {"id": sample_id, "type": type(error).__name__,
                                                  "message": str(error)})
                raise
        dino, sam = tools.dino_model, tools.sam_model
        write_json(trace_path, trace)
        completed += 1
        print(json.dumps({"id": sample_id, "status": trace["final_status"],
                          "tool_calls": trace["metrics"]["tool_calls"], "completed": completed}), flush=True)
    write_json(out / "execution.json", {
        "expected": len(rows), "completed": completed, "complete": completed == len(rows),
        "device": "cpu", "scored": False, "final_labels": False,
        "invocation_wall_seconds": time.perf_counter() - started,
    })


if __name__ == "__main__":
    main()
