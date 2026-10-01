"""Record a real, deliberately invalid three-ID cross-modal probe for offline recovery."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image

from .collect_evidence_branches import _prefix
from .controller import text_message
from .demonstrate import DECISION_INSTRUCTION
from .export_trajectories import load_frozen_manifest, verify_frozen_splits
from .profiles import PROFILES
from .run import map_row_paths
from .state import build_state_messages
from .vision_tools import CandidatePool, VisualTools


def _restore_pool(trace, row):
    snapshot = trace["final_pool_snapshot"]
    candidate_row = {
        "id": trace["id"], "c_bbox": snapshot["c_bbox"],
        "candidates": snapshot["source_candidates"], "images": row["images"],
        "depth_encoding": row.get("depth_encoding"),
    }
    with Image.open(row["images"]["rgb"]) as image:
        pool = CandidatePool(candidate_row, image.size)
    if pool.public_candidates() != trace["final_candidates"]:
        raise ValueError(f"{trace['id']}: final candidate snapshot disagrees with visible pool")
    return pool


def collect_one(trace, row, tool_output_dir, trace_path):
    """Run the protocol probe once, preserving the actual tool response verbatim."""
    if trace.get("trajectory_origin") != "scripted" or trace.get("final_status") != "OBSERVATION_COMPLETE":
        raise ValueError(f"{trace['id']}: expected a completed scripted observation trace")
    pool = _restore_pool(trace, row)
    pool_before = pool.public_candidates()
    targets = [candidate for candidate in pool_before if candidate["role"] == "target"]
    common = {
        "id": str(trace["id"]), "origin": "scripted_protocol_probe",
        "trace_path": str(Path(trace_path).resolve()),
        "pool_before": pool_before, "pool_after": pool_before,
        "candidate_count": len(targets), "gt_used": False, "supervised": False,
    }
    if len(targets) < 3:
        return {**common, "status": "SKIP", "skip_reason": "fewer_than_three_target_candidates",
                "intentional_invalid_action": False, "input_action": None,
                "observation": None, "event": None, "prefix_events": _prefix(trace["events"])}

    candidate_ids = [str(candidate["id"]) for candidate in targets[:3]]
    action = {"action": "inspect_regions", "candidate_ids": candidate_ids,
              "modalities": ["rgb", "ir"], "view": "cross"}
    contextual_prefix = _prefix(trace["events"])
    messages = build_state_messages(trace["initial_messages"], contextual_prefix, pool_before)
    messages.append(text_message(DECISION_INSTRUCTION))

    tool_output_dir = Path(tool_output_dir)
    tools = VisualTools(row, pool, tool_output_dir, PROFILES["sft"].tool_pixels,
                        max_searches=0, allow_search=False)
    snapshot_before = pool.snapshot()
    started = time.perf_counter()
    observation = tools.execute(action)
    tool_seconds = time.perf_counter() - started
    pool_after = pool.public_candidates()
    if pool.snapshot() != snapshot_before or pool_after != pool_before:
        raise ValueError(f"{trace['id']}: invalid protocol probe changed the candidate pool")
    if observation.get("status") != "ERROR" or observation.get("data", {}).get("kind") != "protocol":
        raise ValueError(f"{trace['id']}: expected VisualTools to return a protocol ERROR")
    if observation.get("images"):
        raise ValueError(f"{trace['id']}: rejected protocol probe must not create image observations")
    created_files = [path for path in tool_output_dir.rglob("*") if path.is_file()]
    if created_files:
        raise ValueError(f"{trace['id']}: rejected protocol probe created files: {created_files}")

    event = {
        "step": len(contextual_prefix), "origin": "scripted_protocol_probe",
        "messages": messages, "candidates_before": pool_before,
        "action": action, "executed_action": action, "observation": observation,
        "candidates_after": pool_after, "tool_seconds": tool_seconds,
    }
    recovery_prefix = [*contextual_prefix, event]
    return {
        **common, "pool_after": pool_after, "candidate_ids": candidate_ids,
        "status": observation["status"], "intentional_invalid_action": True,
        "input_action": action, "observation": observation, "event": event,
        "prefix_events": recovery_prefix,
    }


def _read_traces(root):
    traces = {}
    paths = sorted(Path(root).rglob("trace.json"))
    for path in paths:
        trace = json.loads(path.read_text(encoding="utf-8-sig"))
        sample_id = str(trace["id"])
        if sample_id in traces:
            raise ValueError(f"duplicate observation trace {sample_id}")
        traces[sample_id] = (trace, path)
    return traces


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True,
                        help="Completed observations_v2/traces directory")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--debug-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args(argv)

    train = load_frozen_manifest(args.train_manifest)
    holdout = load_frozen_manifest(args.holdout_manifest)
    debug = load_frozen_manifest(args.debug_manifest)
    verify_frozen_splits(train, holdout, debug)
    traces = _read_traces(args.traces)
    if set(traces) != set(train):
        raise ValueError(f"protocol probes require frozen train starts: missing={len(set(train)-set(traces))}, "
                         f"extra={len(set(traces)-set(train))}")

    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    counts = Counter()
    started = time.perf_counter()
    for sample_id, (trace, trace_path) in traces.items():
        frozen = train[sample_id]
        source_row = trace["manifest_row"]
        if (trace["query"] != frozen["query"] or
                source_row["images"]["rgb"] != frozen["images"]["rgb"]):
            raise ValueError(f"{sample_id}: trace differs from frozen Query/scene")
        row = map_row_paths(source_row, mappings, args.train_manifest.parent)
        record = collect_one(trace, row, output_dir / "tool_output" / sample_id, trace_path)
        records.append(record)
        counts[record["status"].lower()] += 1

    (output_dir / "protocol_feedback.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    inventory = {
        "origin": "scripted_protocol_probe", "gt_used": False,
        "input_traces": len(records), "protocol_errors": counts["error"],
        "skipped_too_few_targets": counts["skip"],
        "tool_seconds": sum((record.get("event") or {}).get("tool_seconds", 0) for record in records),
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False))
    return inventory


if __name__ == "__main__":
    main()
