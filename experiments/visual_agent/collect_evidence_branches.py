"""Collect GT-free, real tool observations for every visible target branch.

Branches are scripted evidence probes, not autonomous trajectories or scored
predictions. GT-based branch selection belongs only to a later offline export.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image

from .controller import text_message
from .demonstrate import DECISION_INSTRUCTION, FINAL_INSTRUCTION
from .export_trajectories import load_frozen_manifest, verify_frozen_splits
from .profiles import PROFILES
from .run import map_row_paths, write_json
from .state import build_state_messages
from .vision_tools import CandidatePool, VisualTools


PREFIX_TOOLS = {"search_candidates", "measure_depth"}


def _prefix(events):
    fields = ("step", "origin", "action", "executed_action", "observation",
              "candidates_before", "candidates_after", "tool_seconds")
    return [{key: event[key] for key in fields if key in event} for event in events
            if (event.get("executed_action") or event.get("action") or {}).get("action") in PREFIX_TOOLS]


def _pool(trace, row, rgb_size):
    snapshot = trace["final_pool_snapshot"]
    candidate_row = {"id": trace["id"], "c_bbox": snapshot["c_bbox"],
                     "candidates": snapshot["source_candidates"],
                     "images": row["images"], "depth_encoding": row.get("depth_encoding")}
    pool = CandidatePool(candidate_row, rgb_size)
    if pool.public_candidates() != trace["final_candidates"]:
        raise ValueError(f"{trace['id']}: final candidate snapshot disagrees with visible pool")
    return pool


def collect_one(trace, row, sample_dir, parent_trace_path):
    """Return independent branch records; never read a target label or GT."""
    if trace.get("trajectory_origin") != "scripted" or trace.get("final_status") != "OBSERVATION_COMPLETE":
        raise ValueError(f"{trace['id']}: expected a completed scripted observation trace")
    events = trace["events"]
    prefix = _prefix(events)
    if len(events) >= PROFILES["sft"].evidence_calls or len(prefix) + 1 > PROFILES["sft"].evidence_calls:
        return [], "evidence_budget"
    with Image.open(row["images"]["rgb"]) as image:
        rgb_size = image.size
    candidates = trace["final_candidates"]
    targets = [candidate for candidate in candidates if candidate["role"] == "target"]
    other = next((candidate for candidate in targets if str(candidate["id"]) != "KEEP"), None)
    modalities = [modality for modality in ("rgb", "ir") if row["images"].get(modality)]
    if not modalities:
        raise ValueError(f"{trace['id']}: no RGB/IR observation modality")
    sample_dir = Path(sample_dir)
    branches = []
    for candidate in targets:
        candidate_id = str(candidate["id"])
        ids = (["KEEP", str(other["id"])] if candidate_id == "KEEP" and other else
               ["KEEP", candidate_id] if candidate_id != "KEEP" else ["KEEP"])
        view = "cross" if len(modalities) == 2 else "pair" if len(ids) == 2 else "single"
        action = {"action": "inspect_regions", "candidate_ids": ids,
                  "modalities": modalities, "view": view,
                  "evidence_note": "Compare current target hypotheses across available RGB/IR views."}
        pool = _pool(trace, row, rgb_size)
        branch_dir = sample_dir / "images" / candidate_id
        tools = VisualTools(row, pool, branch_dir, PROFILES["sft"].tool_pixels,
                            max_searches=0, allow_search=False)
        before = pool.public_candidates()
        messages = build_state_messages(trace["initial_messages"], prefix, before)
        messages.append(text_message(DECISION_INSTRUCTION))
        started = time.perf_counter()
        observation = tools.execute(action)
        if observation["status"] != "OK" or not observation["images"]:
            raise ValueError(f"{trace['id']} branch {candidate_id}: inspect did not return real images")
        event = {"step": len(prefix), "origin": "scripted_branch", "messages": messages,
                 "candidates_before": before, "action": action, "executed_action": action,
                 "observation": observation, "candidates_after": pool.public_candidates(),
                 "tool_seconds": time.perf_counter() - started}
        terminal = build_state_messages(trace["initial_messages"], [*prefix, event], pool.public_candidates())
        terminal.append(text_message(FINAL_INSTRUCTION if len(prefix) + 1 ==
                                     PROFILES['sft'].evidence_calls else DECISION_INSTRUCTION))
        branches.append({"id": str(trace["id"]), "candidate_id": candidate_id,
                         "trajectory_origin": "scripted_branch", "parent_trace_path": str(parent_trace_path),
                         "parent_event_count": len(events), "prefix_events": prefix,
                         "event": event, "terminal_messages": terminal,
                         "candidates": pool.public_candidates()})
    return branches, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--debug-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    train = load_frozen_manifest(args.train_manifest)
    holdout = load_frozen_manifest(args.holdout_manifest)
    debug = load_frozen_manifest(args.debug_manifest)
    verify_frozen_splits(train, holdout, debug)
    traces = {}
    for path in sorted(args.traces.rglob("trace.json")):
        trace = json.loads(path.read_text(encoding="utf-8-sig"))
        sample_id = str(trace["id"])
        if sample_id in traces:
            raise ValueError(f"duplicate trace {sample_id}")
        traces[sample_id] = (trace, path)
    if set(traces) != set(train):
        raise ValueError(f"branch input must contain frozen train starts: missing={len(set(train)-set(traces))}, extra={len(set(traces)-set(train))}")
    config = {"traces": str(args.traces.resolve()), "train_manifest": str(args.train_manifest.resolve()),
              "holdout_manifest": str(args.holdout_manifest.resolve()),
              "debug_manifest": str(args.debug_manifest.resolve()), "path_map": args.path_map,
              "branch_origin": "scripted_branch", "gt_used": False}
    config_path = args.output_dir / "run_config.json"
    if args.output_dir.exists():
        if not args.resume or not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"branch output exists or resume config differs: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
        write_json(config_path, config)
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    counts = Counter()
    started = time.perf_counter()
    for sample_id, (trace, path) in traces.items():
        branch_file = args.output_dir / "traces" / sample_id / "branches.jsonl"
        if branch_file.exists():
            counts["resumed_samples"] += 1
            continue
        frozen = train[sample_id]
        if (trace["query"] != frozen["query"] or
                trace["manifest_row"]["images"]["rgb"] != frozen["images"]["rgb"]):
            raise ValueError(f"{sample_id}: trace differs from frozen Query/scene")
        row = map_row_paths(trace["manifest_row"], mappings, args.train_manifest.parent)
        branches, skipped = collect_one(trace, row, branch_file.parent, path)
        branch_file.parent.mkdir(parents=True, exist_ok=True)
        branch_file.write_text("".join(json.dumps(branch, ensure_ascii=False) + "\n" for branch in branches),
                               encoding="utf-8")
        counts["processed_samples"] += 1
        counts["branches"] += len(branches)
        if skipped:
            counts[skipped + "_samples"] += 1
    inventory = {"origin": "scripted_branch", "gt_used": False,
                 "frozen_train_starts": len(train), **dict(counts)}
    write_json(args.output_dir / "inventory.json", inventory)
    write_json(args.output_dir / "execution.json", {
        "complete": counts['processed_samples'] + counts['resumed_samples'] == len(train),
        "expected": len(train), "completed": counts['processed_samples'] + counts['resumed_samples'],
        "device": "cpu", "invocation_wall_seconds": time.perf_counter() - started,
    })
    print(json.dumps(inventory, ensure_ascii=False))


if __name__ == "__main__":
    main()
