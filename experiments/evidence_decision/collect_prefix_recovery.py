"""Collect one real CPU inspect recovery after a native protocol error.

The prefix is copied from the autonomous trace. No GT, model, final decision,
or training label is used here; the recovered observation is a new tool call.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from PIL import Image

from tools.predict_aux_selection import index_rows_by_id, read_jsonl

from .controller import text_message
from .demonstrate import DECISION_INSTRUCTION
from .profiles import PROFILES
from .run import map_row_paths
from .state import build_state_messages
from .vision_tools import CandidatePool, VisualTools


TOOL_ACTIONS = {"inspect_regions", "measure_depth", "search_candidates"}


def _action(event):
    return event.get("executed_action") or event.get("action") or {}


def _read_traces(root):
    traces = {}
    for path in sorted(Path(root).rglob("trace.json")):
        trace = json.loads(path.read_text(encoding="utf-8-sig"))
        sample_id = str(trace["id"])
        if sample_id in traces:
            raise ValueError(f"duplicate native trace: {sample_id}")
        traces[sample_id] = (trace, path)
    return traces


def _choose_prefix(trace):
    """Select the first multi-ID inspect ERROR while respecting the real budget."""
    for index, event in enumerate(trace["events"]):
        action = _action(event)
        observation = event.get("observation") or {}
        if (action.get("action") != "inspect_regions" or
                not isinstance(action.get("candidate_ids"), list) or
                len(action["candidate_ids"]) <= 2 or
                observation.get("status") != "ERROR" or
                observation.get("data", {}).get("kind") != "protocol"):
            continue
        prefix = trace["events"][:index + 1]
        errors = sum((prior.get("observation") or {}).get("status") == "ERROR" for prior in prefix)
        if errors >= 2:
            return None, "prior_protocol_error"
        actual_calls = sum("tool_seconds" in prior and _action(prior).get("action") in TOOL_ACTIONS
                           for prior in prefix)
        if actual_calls + 1 > PROFILES["sft"].evidence_calls:
            return None, "evidence_budget"
        if any(_action(prior).get("action") == "search_candidates" and
               ((prior.get("observation") or {}).get("data", {}).get("appended_ids") or
                prior["candidates_after"] != prior["candidates_before"])
               for prior in prefix):
            return None, "search_changed_pool"
        before = event["candidates_before"]
        valid_ids = {str(candidate["id"]) for candidate in before}
        baseline_alias = set(trace["initial_pool_snapshot"]["aliases"])
        ids = []
        for value in action["candidate_ids"]:
            candidate_id = str(value)
            if candidate_id in valid_ids and candidate_id not in baseline_alias and candidate_id not in ids:
                ids.append(candidate_id)
            if len(ids) == 2:
                break
        if len(ids) < 2:
            return None, "fewer_than_two_valid_ids"
        modalities = action.get("modalities")
        if (not isinstance(modalities, list) or not 1 <= len(modalities) <= 3 or
                len(set(modalities)) != len(modalities) or
                any(modality not in {"rgb", "ir", "depth"} for modality in modalities)):
            return None, "invalid_requested_modalities"
        recovered = {"action": "inspect_regions", "candidate_ids": ids,
                     "modalities": modalities, "view": "pair" if len(modalities) == 1 else "cross"}
        return (prefix, recovered), None
    return None, "no_first_multi_id_inspect_error"


def collect_one(trace, row, candidate_row, output_dir, parent_trace_path):
    sample_id = str(trace["id"])
    base = {"id": sample_id, "origin": "scripted_recovery_from_native_prefix",
            "parent_trace_path": str(Path(parent_trace_path).resolve()),
            "gt_used": False, "autonomous_success": False, "training_label": False}
    choice, reason = _choose_prefix(trace)
    if choice is None:
        return {**base, "status": "SKIP", "skip_reason": reason}
    prefix, action = choice
    snapshot = trace["initial_pool_snapshot"]
    restored_row = {"id": sample_id, "c_bbox": snapshot["c_bbox"],
                    "candidates": snapshot["source_candidates"],
                    "images": row["images"], "depth_encoding": row.get("depth_encoding")}
    with Image.open(row["images"]["rgb"]) as image:
        pool = CandidatePool(restored_row, image.size)
        cached_pool = CandidatePool(candidate_row, image.size)
    initial = pool.public_candidates()
    if initial != trace["initial_candidates"] or cached_pool.public_candidates() != initial:
        raise ValueError(f"{sample_id}: initial snapshot/cache differs from native trace")
    for index, event in enumerate(prefix):
        if event.get("candidates_before", initial) != initial or event["candidates_after"] != initial:
            raise ValueError(f"{sample_id}: candidate pool changed in prefix at step {index}")
    before = pool.public_candidates()
    messages = build_state_messages(trace["initial_messages"], prefix, before)
    messages.append(text_message(DECISION_INSTRUCTION))
    tools = VisualTools(row, pool, Path(output_dir) / "tool_images", PROFILES["sft"].tool_pixels,
                        max_searches=0, allow_search=False)
    started = time.perf_counter()
    observation = tools.execute(action)
    tool_seconds = time.perf_counter() - started
    if observation["status"] != "OK" or not observation["images"]:
        raise ValueError(f"{sample_id}: legal recovery produced no real images: {observation}")
    after = pool.public_candidates()
    if after != before:
        raise ValueError(f"{sample_id}: inspect recovery changed candidate pool")
    event = {"step": len(prefix), "origin": base["origin"], "messages": messages,
             "candidates_before": before, "action": action, "executed_action": action,
             "observation": observation, "candidates_after": after, "tool_seconds": tool_seconds}
    terminal = build_state_messages(trace["initial_messages"], [*prefix, event], after)
    terminal.append(text_message(DECISION_INSTRUCTION))
    visible_paths = {block["image"] for message in terminal if isinstance(message.get("content"), list)
                     for block in message["content"]
                     if block.get("type") == "image"}
    if not {image["path"] for image in observation["images"]} <= visible_paths:
        raise ValueError(f"{sample_id}: recovery images absent from next decision messages")
    return {**base, "status": "OK", "prefix_events": prefix, "recovery_action": action,
            "event": event, "terminal_messages": terminal,
            "candidate_count": len(before), "tool_seconds": tool_seconds,
            "processor_consumed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    manifest = index_rows_by_id(read_jsonl(args.manifest), source=str(args.manifest))
    cache = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    traces = _read_traces(args.traces)
    if set(traces) != set(manifest) or set(cache) != set(manifest):
        raise ValueError("manifest, candidate cache, and native trace IDs must match exactly")
    for sample_id, (trace, _) in traces.items():
        frozen = manifest[sample_id]
        source = trace["manifest_row"]
        if (trace["query"] != frozen["query"] or source["query"] != frozen["query"] or
                source["images"]["rgb"] != frozen["images"]["rgb"] or
                cache[sample_id]["images"]["rgb"] != frozen["images"]["rgb"]):
            raise ValueError(f"{sample_id}: Query or RGB scene differs from manifest")
        if trace.get("memory") != "latest" or trace.get("profile", {}).get("name") != "sft":
            raise ValueError(f"{sample_id}: native trace does not use sft/latest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    started = time.perf_counter()
    selected = list(manifest)
    if args.limit is not None:
        selected = selected[:args.limit]
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in selected:
            trace, trace_path = traces[sample_id]
            row = map_row_paths(trace["manifest_row"], [], args.manifest.parent)
            candidate_row = map_row_paths(cache[sample_id], [], args.candidate_cache.parent)
            record = collect_one(trace, row, candidate_row, output / "traces" / sample_id, trace_path)
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            counts[record["status"]] += 1
            if record["status"] == "SKIP":
                counts["skip_" + record["skip_reason"]] += 1
            else:
                counts["tool_seconds"] += record["tool_seconds"]
    inventory = {"origin": "scripted_recovery_from_native_prefix", "gt_used": False,
                 "autonomous_success": False, "input_traces": len(traces),
                 "manifest_starts": len(manifest), "processed_starts": len(selected),
                 "selected_recoveries": counts["OK"], "skipped": counts["SKIP"],
                 "skip_reasons": {key[5:]: value for key, value in counts.items() if key.startswith("skip_")},
                 "tool_seconds": counts["tool_seconds"],
                 "wall_seconds": time.perf_counter() - started}
    (output / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False))
    return inventory


if __name__ == "__main__":
    main()
