"""Paired GT-free finish probe after real native Depth UNKNOWN prefixes.

Direct, scripted IR and scripted RGB branches share one frozen prefix and pool.
This is an evidence intervention, not an autonomous rollout or SFT export.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time

from PIL import Image
from tools.predict_aux_selection import index_rows_by_id, read_jsonl

from .controller import parse_action, text_message
from .model import InputBudgetExceeded, QwenBackend
from .profiles import PROFILES
from .run import map_row_paths
from .state import build_state_messages
from .vision_tools import CandidatePool, ProtocolError, VisualTools


FINAL_INSTRUCTION = (
    "Final decision now. Only action=finish with a current target candidate_id or KEEP is legal. "
    "No tools or coordinates."
)
BRANCHES = ("direct", "ir", "rgb")


def _action(event):
    return event.get("executed_action") or event.get("action") or {}


def _traces(root):
    found = {}
    for path in sorted(root.rglob("trace.json")):
        trace = json.loads(path.read_text(encoding="utf-8-sig"))
        sample_id = str(trace["id"])
        if sample_id in found:
            raise ValueError(f"duplicate trace ID: {sample_id}")
        found[sample_id] = (trace, path)
    return found


def prefix_choice(trace, kind="depth_unknown"):
    """First actual Depth UNKNOWN; reject changed pools and illegal 1–2 IDs."""
    if kind == "first_inspect_error":
        from .collect_prefix_recovery import _choose_prefix
        choice, reason = _choose_prefix(trace)
        if reason:
            return None, reason
        prefix, recovery = choice
        if any((event.get("observation") or {}).get("images") for event in prefix):
            return None, "prior_local_visual_observation"
        return (prefix, recovery["candidate_ids"]), None
    for index, event in enumerate(trace["events"]):
        if (_action(event).get("action") != "measure_depth" or
                (event.get("observation") or {}).get("status") != "UNKNOWN"):
            continue
        prefix = trace["events"][:index + 1]
        initial = trace["initial_candidates"]
        if any(prior.get("candidates_before") != initial or prior.get("candidates_after") != initial
               for prior in prefix):
            return None, "candidate_pool_changed_before_unknown"
        ids = _action(event).get("candidate_ids")
        valid = {str(candidate["id"]) for candidate in initial}
        aliases = set(trace["initial_pool_snapshot"]["aliases"])
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 2 or
                len({str(value) for value in ids}) != len(ids) or
                any(str(value) not in valid or str(value) in aliases for value in ids)):
            return None, "invalid_depth_candidate_ids"
        return (prefix, [str(value) for value in ids]), None
    return None, "no_measure_depth_unknown"


def freeze_selection(traces, manifest, limit, seed, kind="depth_unknown"):
    by_group = defaultdict(list)
    reasons = {}
    for sample_id, (trace, _) in sorted(traces.items()):
        frozen = manifest[sample_id]
        if (trace["query"] != frozen["query"] or
                trace["manifest_row"]["images"]["rgb"] != frozen["images"]["rgb"] or
                trace.get("memory") != "latest" or trace.get("profile", {}).get("name") != "sft"):
            raise ValueError(f"{sample_id}: trace differs from frozen sft/latest manifest")
        choice, reason = prefix_choice(trace, kind)
        if reason:
            reasons[sample_id] = reason
        else:
            group = Path(frozen["images"]["rgb"]).name
            by_group[group].append(sample_id)
    rng = random.Random(seed)
    groups = sorted(by_group)
    rng.shuffle(groups)
    selected = []
    for group in groups[:limit]:
        selected.append(rng.choice(sorted(by_group[group])))
    if len(selected) < limit:
        raise ValueError(f"only {len(selected)} eligible independent image groups for {limit} probes")
    selected_set = set(selected)
    selected_groups = set(groups[:limit])
    for group, ids in by_group.items():
        for sample_id in ids:
            if sample_id not in selected_set:
                reasons[sample_id] = "same_image_group" if group in selected_groups else "seed_limit"
    decisions = [{"id": sample_id, "image_group": Path(manifest[sample_id]["images"]["rgb"]).name,
                  "selected": sample_id in selected_set, "exclusion_reason": reasons.get(sample_id)}
                 for sample_id in sorted(traces)]
    return {"seed": seed, "prefix_kind": kind, "requested_independent_groups": limit, "input_traces": len(traces),
            "eligible_independent_groups": len(groups), "selected_ids": selected,
            "selected_independent_groups": len(selected), "exclusion_counts": dict(Counter(reasons.values())),
            "decisions": decisions, "gt_used": False}


def prefix_cost(prefix):
    return {"output_tokens": sum((event.get("usage") or {}).get("output_tokens", 0) for event in prefix),
            "visual_tokens": sum((event.get("usage") or {}).get("visual_tokens", 0) for event in prefix),
            "tool_calls": sum("tool_seconds" in event for event in prefix),
            "model_calls": sum(bool(event.get("usage")) for event in prefix)}


def probe_branch(trace, row, pool, prefix, ids, branch, output_dir, backend, profile=None):
    started = time.perf_counter()
    profile = profile or PROFILES["sft"]
    cost = prefix_cost(prefix)
    candidates = pool.public_candidates()
    prior_images = next((event["observation"]["images"] for event in reversed(prefix)
                         if (event.get("observation") or {}).get("images")), [])
    prior_pairs = {(str(candidate_id), image.get("modality"))
                   for event in prefix for image in (event.get("observation") or {}).get("images", [])
                   for candidate_id in image.get("candidate_ids", [])}
    planned_action = (None if branch == "direct" else
                      {"action": "inspect_regions", "candidate_ids": ids,
                       "modalities": [branch], "view": "pair" if len(ids) == 2 else "single"})
    repeated_ids = [candidate_id for candidate_id in ids if (candidate_id, branch) in prior_pairs]
    result = {"id": str(trace["id"]), "scene": Path(row["images"]["rgb"]).name,
              "query": row["query"], "branch": branch,
              "origin": "scripted_paired_evidence_probe", "autonomous_success": False,
              "gt_used": False, "training_label": False, "candidate_ids": ids,
              "prefix_events": prefix, "prefix_cost": cost, "action": planned_action,
              "observation": None, "messages": None, "usage": None,
              "raw_output": None, "final_action": None, "status": "LIMIT", "bbox": None,
              "prior_latest_tool_images": [{"modality": image.get("modality"),
                                            "candidate_ids": image.get("candidate_ids", [])}
                                           for image in prior_images],
              "latest_window_replaces_prior_tool_images": bool(prior_images and branch != "direct"),
              "repeated_observation_ids": repeated_ids,
              "repeated_observation": bool(repeated_ids),
              "model_generation_executed": False, "new_images_processed": False,
              "peak_memory_bytes": 0, "tool_seconds": 0.0}

    def done():
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

    if len(prefix) + 1 + int(branch != "direct") > profile.evidence_calls + 1:
        result["limit_reason"] = "decision_slots"
        return done()
    if cost["output_tokens"] + profile.finish_tokens > profile.output_tokens:
        result["limit_reason"] = "finish_output_reservation"
        return done()
    if cost["visual_tokens"] >= profile.cumulative_visual_tokens:
        result["limit_reason"] = "cumulative_visual_tokens"
        return done()
    if branch != "direct" and cost["tool_calls"] + 1 > profile.evidence_calls:
        result["limit_reason"] = "evidence_calls"
        return done()
    events = prefix
    if branch != "direct":
        action = planned_action
        tools = VisualTools(row, pool, output_dir / "tool_images", profile.tool_pixels,
                            max_searches=0, allow_search=False)
        tool_started = time.perf_counter()
        observation = tools.execute(action)
        tool_seconds = time.perf_counter() - tool_started
        result.update(action=action, observation=observation, tool_seconds=tool_seconds)
        event = {"step": len(prefix), "origin": result["origin"], "action": action,
                 "executed_action": action, "observation": observation,
                 "candidates_before": candidates, "candidates_after": pool.public_candidates(),
                 "tool_seconds": tool_seconds}
        if observation["status"] != "OK" or not observation["images"]:
            result.update(status="TOOL_" + observation["status"], event=event)
            return done()
        events = [*prefix, event]
        result["event"] = event
    messages = build_state_messages(trace["initial_messages"], events, candidates)
    messages.append(text_message(FINAL_INSTRUCTION))
    result["messages"] = messages
    if branch != "direct":
        shown = {block["image"] for message in messages
                 for block in (message.get("content") if isinstance(message.get("content"), list) else [])
                 if block.get("type") == "image"}
        if not {image["path"] for image in result["observation"]["images"]} <= shown:
            raise ValueError(f"{trace['id']} {branch}: new observation image absent from model messages")
    remaining_visual = profile.cumulative_visual_tokens - cost["visual_tokens"]

    def confirm_new_images(usage):
        if branch == "direct":
            return
        processed = {image.get("path") for image in usage.get("image_geometry", [])}
        new = {image["path"] for image in result["observation"]["images"]}
        if not new <= processed:
            raise ValueError(f"{trace['id']} {branch}: new observation image absent from Processor usage")
        result["new_images_processed"] = True

    backend.begin_sample()
    try:
        generated = backend.generate(messages, profile.finish_tokens, profile, remaining_visual)
    except InputBudgetExceeded as error:
        result.update(limit_reason=str(error), usage=error.usage)
        confirm_new_images(error.usage)
        return done()
    result.update(raw_output=generated["raw_output"], usage=generated["usage"],
                  model_generation_executed=True, peak_memory_bytes=backend.peak_memory())
    confirm_new_images(generated["usage"])
    try:
        action = parse_action(generated["raw_output"])
        result["final_action"] = action
        if action["action"] != "finish":
            raise ProtocolError("final action must be finish")
        bbox = pool.finish(action["candidate_id"])
    except (ProtocolError, KeyError) as error:
        result.update(status="INVALID_FINAL_ACTION", final_error=str(error))
        return done()
    result.update(status="FINISHED", final_action=action, bbox=bbox)
    return done()


def prediction_row(record, trace):
    """One legal or null result per frozen start for evaluate.load_run."""
    usage = record.get("usage") or {}
    generated = record.get("model_generation_executed", False)
    observation = record.get("observation")
    selected = (str(record["final_action"]["candidate_id"])
                if record["status"] == "FINISHED" else None)
    metrics = {key: usage.get(key, 0) if generated else 0
               for key in ("input_tokens", "input_text_tokens", "visual_tokens", "output_tokens",
                           "model_seconds")}
    metrics.update(preprocess_seconds=usage.get("preprocess_seconds", 0),
                   rejected_input_tokens=usage.get("input_tokens", 0) if not generated else 0,
                   rejected_visual_tokens=usage.get("visual_tokens", 0) if not generated else 0,
                   model_calls=int(generated),
                   tool_calls=int(observation is not None and "event" in record),
                   tool_seconds=record.get("tool_seconds", 0),
                   tool_status_counts={observation["status"]: 1} if observation is not None else {},
                   legal_finish=record["status"] == "FINISHED",
                   peak_memory_bytes=record.get("peak_memory_bytes", 0),
                   elapsed_seconds=record.get("elapsed_seconds", 0))
    return {"id": record["id"], "bbox": record["bbox"], "parsed": record.get("final_action") is not None,
            "final_status": record["status"], "selected_id": selected,
            "initial_bbox": trace["initial_pool_snapshot"]["c_bbox"],
            "initial_candidates": trace["initial_candidates"],
            "final_candidates": trace["initial_candidates"], "metrics": metrics,
            "origin": record["origin"], "autonomous_success": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("traces", "manifest", "candidate-cache", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--prefix-kind", choices=["depth_unknown", "first_inspect_error"],
                        default="depth_unknown")
    parser.add_argument("--max-run-seconds", type=float, required=True)
    parser.add_argument("--finish-tokens", type=int, default=PROFILES["sft"].finish_tokens,
                        help="One frozen cap for every branch; never adjusted per sample")
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_run_seconds <= 0 or not 1 <= args.finish_tokens <= PROFILES["sft"].finish_tokens:
        parser.error("positive limits required; --finish-tokens must be within 1..512")
    profile = replace(PROFILES["sft"], finish_tokens=args.finish_tokens)
    traces = _traces(args.traces)
    manifest = index_rows_by_id(read_jsonl(args.manifest), source=str(args.manifest))
    cache = index_rows_by_id(read_jsonl(args.candidate_cache), source=str(args.candidate_cache))
    if set(traces) != set(manifest) or set(cache) != set(manifest):
        raise ValueError("native traces, manifest, and candidate cache IDs must match exactly")
    selection = freeze_selection(traces, manifest, args.limit, args.seed, args.prefix_kind)
    selection["profile"] = profile.to_dict()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    started = time.perf_counter()
    backend = QwenBackend(args.model)
    statuses = Counter()
    predictions = {branch: [] for branch in BRANCHES}
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for sample_id in selection["selected_ids"]:
            trace, _ = traces[sample_id]
            row = map_row_paths(manifest[sample_id], [], args.manifest.parent)
            candidate_row = map_row_paths(cache[sample_id], [], args.candidate_cache.parent)
            with Image.open(row["images"]["rgb"]) as image:
                pool = CandidatePool(candidate_row, image.size)
            if pool.public_candidates() != trace["initial_candidates"]:
                raise ValueError(f"{sample_id}: candidate cache differs from frozen native pool")
            prefix, ids = prefix_choice(trace, args.prefix_kind)[0]
            for branch in BRANCHES:
                if time.perf_counter() - started >= args.max_run_seconds:
                    record = {"id": sample_id, "scene": Path(row["images"]["rgb"]).name,
                              "query": row["query"], "branch": branch, "status": "MISSING",
                              "origin": "scripted_paired_evidence_probe", "autonomous_success": False,
                              "gt_used": False, "training_label": False, "bbox": None,
                              "prefix_events": prefix, "prefix_cost": prefix_cost(prefix),
                              "action": None, "observation": None, "messages": None,
                              "usage": None, "raw_output": None, "final_action": None,
                              "reason": "max_run_seconds"}
                else:
                    record = probe_branch(trace, row, pool, prefix, ids, branch,
                                          output / "traces" / sample_id / branch, backend, profile=profile)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                statuses[record["status"]] += 1
                predictions[branch].append(prediction_row(record, trace))
    for branch, rows in predictions.items():
        branch_dir = output / branch
        branch_dir.mkdir(exist_ok=True)
        with (branch_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    inventory = {"selected_starts": len(selection["selected_ids"]), "expected_branches": 3 * args.limit,
                 "recorded_branches": sum(statuses.values()), "statuses": dict(statuses),
                 "model": args.model, "model_load_seconds": backend.load_seconds,
                 "profile": profile.to_dict(),
                 "elapsed_seconds": time.perf_counter() - started,
                 "gt_used": False, "autonomous_success": False, "training_label": False}
    (output / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False))
    return inventory


if __name__ == "__main__":
    main()
