"""One-shot Qwen probes of exported SFT decisions; no tools or GT scoring.

This diagnoses action imitation on training inputs. It is not an autonomous
Agent rollout or a generalization metric.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import replace
import json
import random
from pathlib import Path
import time

from .controller import parse_action
from .model import InputBudgetExceeded, QwenBackend
from .object_controller import OBJECT_PROFILE, parse_tool_call
from .profiles import PROFILES
from .train import read_decisions


def _canonical(action):
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _arguments(action):
    return {key: value for key, value in action.items() if key != "evidence_note"} if isinstance(
        action.get("evidence_note"), str) else action


def probe_rows(rows, backend, *, started=None, max_run_seconds=None):
    """Yield one record per input row, including time- and budget-limited rows."""
    started = time.perf_counter() if started is None else started
    profile = PROFILES["sft"]
    for index, row in enumerate(rows):
        target = json.loads(row["target_action"])
        if not isinstance(target, dict) or not isinstance(target.get("action"), str):
            raise ValueError(f"decision {index}: invalid target_action")
        record = {"row_index": index, "id": str(row["id"]),
                  "decision_index": row.get("decision_index"),
                  "origin": row.get("origin", row.get("trajectory_origin", "unknown")),
                  "label_source": row.get("label_source", "unknown"),
                  "target_action": row["target_action"], "target_action_name": target["action"],
                  "status": "NOT_RUN_TIME_LIMIT", "raw_output": None, "usage": None,
                  "predicted_action": None, "exact_canonical_match": False,
                  "action_arguments_match": False,
                  "action_name_match": False,
                  "finish_id_match": False if target["action"] == "finish" else None}
        if max_run_seconds is not None and time.perf_counter() - started >= max_run_seconds:
            yield record
            continue
        backend.begin_sample()
        try:
            generated = backend.generate(row["messages"], 128, profile,
                                         profile.cumulative_visual_tokens)
        except InputBudgetExceeded as error:
            record.update(status="INPUT_BUDGET_EXCEEDED", usage=error.usage,
                          error=str(error))
            yield record
            continue
        record.update(status="GENERATED", raw_output=generated["raw_output"],
                      usage=generated["usage"])
        try:
            predicted = parse_action(generated["raw_output"])
        except ValueError as error:
            record.update(status="PARSE_ERROR", error=str(error))
            yield record
            continue
        record["predicted_action"] = predicted
        record["exact_canonical_match"] = _canonical(predicted) == _canonical(target)
        record["action_arguments_match"] = _canonical(_arguments(predicted)) == _canonical(_arguments(target))
        record["action_name_match"] = predicted["action"] == target["action"]
        if target["action"] == "finish":
            record["finish_id_match"] = (predicted["action"] == "finish" and
                                         str(predicted.get("candidate_id")) == str(target.get("candidate_id")))
        yield record


def summarize(records):
    """Use every requested row as denominator, including unrun and failed rows."""
    def counts(items):
        n = len(items)
        finishes = [item for item in items if item["target_action_name"] == "finish"]
        exact = sum(item["exact_canonical_match"] for item in items)
        arguments = sum(item["action_arguments_match"] for item in items)
        name = sum(item["action_name_match"] for item in items)
        finish_id = sum(item["finish_id_match"] is True for item in finishes)
        return {"rows": n, "exact_canonical_matches": exact,
                "exact_canonical_rate": exact / n if n else None,
                "action_arguments_matches": arguments,
                "action_arguments_rate": arguments / n if n else None,
                "action_name_matches": name, "action_name_rate": name / n if n else None,
                "finish_targets": len(finishes), "finish_id_matches": finish_id,
                "finish_id_rate": finish_id / len(finishes) if finishes else None,
                "input_budget_exceeded": sum(item["status"] == "INPUT_BUDGET_EXCEEDED" for item in items),
                "parse_errors": sum(item["status"] == "PARSE_ERROR" for item in items),
                "not_run_time_limit": sum(item["status"] == "NOT_RUN_TIME_LIMIT" for item in items)}

    result = {"all": counts(records)}
    for field, title in (("target_action_name", "by_target_action"),
                         ("origin", "by_origin"), ("label_source", "by_label_source")):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        result[title] = {key: counts(items) for key, items in sorted(groups.items())}
    return result


def _load_traces(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.rglob("trace.json"))
    elif path.name == "trace.json":
        files = [path]
    elif path.suffix == ".jsonl":
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                       if line.strip()]
        files = [Path(row["trace_path"]) for row in predictions]
    else:
        raise ValueError("--traces requires trace.json, a trace directory, or predictions.jsonl")
    traces = [json.loads(file.read_text(encoding="utf-8-sig")) for file in files]
    ids = [str(trace["id"]) for trace in traces]
    if not traces or len(ids) != len(set(ids)):
        raise ValueError("trace input must be nonempty and contain unique IDs")
    return traces


def _message_images(messages):
    result = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image":
                    result.append((message_index, block_index, block))
    return result


def _scene_group(trace, development_row):
    row = development_row or trace.get("manifest_row") or trace
    for field in ("image_group", "scene_id", "group"):
        if row.get(field):
            return str(row[field])
    rgb = (row.get("images") or {}).get("rgb") or trace.get("image_group") or trace.get("id")
    return Path(str(rgb).replace("\\", "/")).stem


def _state_after_tool_image(trace, group):
    image_origin = {}
    events = trace.get("events") or []
    for index, event in enumerate(events):
        observation = event.get("observation") or {}
        action = event.get("action") or {}
        if action.get("name") not in {"inspect", "search"}:
            continue
        for picture in observation.get("images") or []:
            path = picture.get("path")
            if path:
                image_origin[str(path)] = {"event_index": index, "action": action,
                                           "status": observation.get("status"), "picture": picture}
    choices = []
    for decision_index, event in enumerate(events):
        messages = event.get("messages")
        if not isinstance(messages, list):
            continue
        for message_index, block_index, block in _message_images(messages):
            image_path = str(block.get("image", ""))
            origin = image_origin.get(image_path)
            if not origin or block.get("view") != "tool":
                continue
            if origin["event_index"] >= decision_index:
                continue
            choices.append((origin["event_index"], decision_index, message_index, block_index, block, origin, event))
    if not choices:
        return None
    _, decision_index, message_index, block_index, block, origin, event = max(
        choices, key=lambda item: (item[0], item[1], item[2], item[3]))
    messages_full = copy.deepcopy(event["messages"])
    messages_masked = copy.deepcopy(event["messages"])
    key_path = str(block["image"])
    removed = 0
    mask_meta = {key: block.get(key) for key in ("modality", "view", "candidate_ids", "image")}
    marker = ("[Probe mask: pixel content of this one returned tool image is withheld; "
              "the original query, global frames, RGB atlas, tool-role text/facts, and all other images remain unchanged. "
              f"Masked modality={mask_meta.get('modality')}; candidate IDs={mask_meta.get('candidate_ids', [])}.]")
    for message in messages_masked:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and str(item.get("image")) == key_path:
                new_content.append({"type": "text", "text": marker})
                removed += 1
            else:
                new_content.append(item)
        message["content"] = new_content
    if removed == 0:
        raise ValueError(f"{trace['id']}: selected tool image is absent from the decision messages")
    origin_action = origin["action"]
    tools = event.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{trace['id']}: decision state has no captured tool schema")
    return {
        "id": str(trace["id"]), "image_group": group, "query": trace.get("query"),
        "decision_index": event.get("step", decision_index),
        "after_tool": {"name": origin_action.get("name"),
                       "arguments": origin_action.get("arguments", {}),
                       "status": origin["status"]},
        "key_tool_image": mask_meta, "masked_occurrences": removed,
        "mask_scope": "selected tool image pixels only; all tool text and other conversation content retained",
        "tool_text_preserved": True, "available_tools": event.get("available_tools"),
        "tools": tools, "messages_full": messages_full, "messages_masked": messages_masked,
        "gt_used": False, "formal_score_row": False,
    }


def prepare_evidence_pairs(traces: list[dict], development_rows: list[dict], count=32, seed=2030) -> dict:
    development = {str(row["id"]): row for row in development_rows}
    if len(development) != len(development_rows):
        raise ValueError("development manifest has duplicate IDs")
    eligible = defaultdict(list)
    exclusions = Counter()
    for trace in traces:
        sample_id = str(trace["id"])
        if development_rows and sample_id not in development:
            exclusions["not_in_development_manifest"] += 1
            continue
        row = development.get(sample_id)
        group = _scene_group(trace, row)
        pair = _state_after_tool_image(trace, group)
        if pair is None:
            exclusions["no_retained_real_tool_image_before_decision"] += 1
            continue
        eligible[group].append(pair)
    groups = sorted(eligible)
    if len(groups) < count:
        raise ValueError(f"only {len(groups)} eligible independent development groups for {count} probes")
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected = [rng.choice(eligible[group]) for group in groups[:count]]
    return {"diagnostic_only": True, "gt_used": False, "formal_scoring": False,
            "seed": seed, "requested_independent_groups": count,
            "eligible_independent_groups": len(eligible), "selected_independent_groups": len(selected),
            "exclusion_counts": dict(exclusions), "selected_ids": [row["id"] for row in selected],
            "pairs": selected}


def _run_paired_condition(row, condition, backend, profile):
    messages = row["messages_full"] if condition == "full" else row["messages_masked"]
    tools = row["tools"]
    available = row.get("available_tools") or []
    max_new_tokens = profile.finish_tokens if available == ["finish"] else profile.action_tokens
    record = {"status": "NOT_RUN", "raw_output": None, "predicted_action": None,
              "model_note": None, "usage": None}
    backend.begin_sample()
    try:
        generated = backend.generate(messages, max_new_tokens, profile, profile.visual_tokens, tools=tools)
    except InputBudgetExceeded as error:
        record.update(status="INPUT_BUDGET_EXCEEDED", usage=error.usage, error=str(error))
        return record
    record.update(status="GENERATED", raw_output=generated["raw_output"], usage=generated["usage"])
    try:
        call, note = parse_tool_call(generated["raw_output"])
    except ValueError as error:
        record.update(status="PARSE_ERROR", error=str(error))
        return record
    record["predicted_action"] = call
    record["model_note"] = note
    return record


def probe_evidence_pairs(rows, backend, *, context_tokens=8192, started=None, max_run_seconds=None):
    profile = replace(OBJECT_PROFILE, context_tokens=context_tokens)
    started = time.perf_counter() if started is None else started
    records = []
    for row in rows:
        record = {key: row[key] for key in ("id", "image_group", "decision_index", "after_tool",
                                            "key_tool_image", "mask_scope", "tool_text_preserved",
                                            "gt_used", "formal_score_row")}
        if max_run_seconds is not None and time.perf_counter() - started >= max_run_seconds:
            record["full"] = {"status": "NOT_RUN_TIME_LIMIT", "raw_output": None,
                               "predicted_action": None, "model_note": None, "usage": None}
        else:
            record["full"] = _run_paired_condition(row, "full", backend, profile)
        if max_run_seconds is not None and time.perf_counter() - started >= max_run_seconds:
            record["masked"] = {"status": "NOT_RUN_TIME_LIMIT", "raw_output": None,
                                 "predicted_action": None, "model_note": None, "usage": None}
        else:
            record["masked"] = _run_paired_condition(row, "masked", backend, profile)
        full, masked = record["full"]["predicted_action"], record["masked"]["predicted_action"]
        record["action_name_changed"] = bool(full and masked and full["name"] != masked["name"])
        record["action_arguments_changed"] = bool(full and masked and full["arguments"] != masked["arguments"])
        records.append(record)
    paired = [record for record in records if record["full"]["predicted_action"] and
              record["masked"]["predicted_action"]]
    return {"diagnostic_only": True, "autonomous_rollout": False, "formal_scoring": False,
            "gt_used": False, "pairs": len(records), "both_generated": len(paired),
            "action_name_changed": sum(record["action_name_changed"] for record in records),
            "action_arguments_changed": sum(record["action_arguments_changed"] for record in records),
            "results": records,
            "interpretation": "This paired image intervention measures output sensitivity, not task success or autonomous capability."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--decisions", type=Path,
                        help="legacy one-shot action decisions; requires --model")
    inputs.add_argument("--traces", type=Path,
                        help="prepare paired tool-image masks from complete real traces; CPU only without --model")
    inputs.add_argument("--paired-decisions", type=Path,
                        help="run actual Qwen full/masked inference over prepared paired_decisions.jsonl")
    parser.add_argument("--development-manifest", type=Path,
                        help="GT-free manifest limiting --traces preparation to development IDs")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-run-seconds", type=float)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"probe output already exists: {args.output_dir}")

    if args.traces:
        if not args.development_manifest:
            raise ValueError("CPU preparation requires --development-manifest")
        if args.model or args.adapter:
            raise ValueError("omit --model/--adapter for the CPU-only preparation mode")
        if args.count < 1:
            raise ValueError("--count must be positive")
        manifest = [json.loads(line) for line in args.development_manifest.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()]
        prepared = prepare_evidence_pairs(_load_traces(args.traces), manifest, args.count, args.seed)
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "paired_decisions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared["pairs"]), encoding="utf-8")
        (args.output_dir / "selection.json").write_text(
            json.dumps({key: value for key, value in prepared.items() if key != "pairs"},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"prepared": prepared["selected_independent_groups"],
                          "gt_used": False, "model_loaded": False}, ensure_ascii=False))
        return

    if not args.model:
        raise ValueError("--model is required for an actual Qwen inference mode")
    if args.context_tokens < 1:
        raise ValueError("--context-tokens must be positive")
    if args.paired_decisions:
        rows = [json.loads(line) for line in args.paired_decisions.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
        if not rows:
            raise ValueError("paired decisions file is empty")
    else:
        if args.development_manifest:
            raise ValueError("--development-manifest is only used with --traces")
        rows = read_decisions(args.decisions)
        if not rows:
            raise ValueError("decisions file is empty")
    started = time.perf_counter()
    backend = QwenBackend(args.model, args.adapter, device=args.device)
    args.output_dir.mkdir(parents=True)
    if args.paired_decisions:
        result = probe_evidence_pairs(rows, backend, context_tokens=args.context_tokens,
                                      started=started, max_run_seconds=args.max_run_seconds)
        result.update({"input_pairs": str(args.paired_decisions.resolve()), "model": args.model,
                       "adapter": args.adapter, "device": args.device,
                       "context_tokens": args.context_tokens,
                       "model_load_seconds": backend.load_seconds,
                       "elapsed_seconds": time.perf_counter() - started})
        (args.output_dir / "paired_results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.pop("results")),
            encoding="utf-8")
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: result[key] for key in
                          ("pairs", "both_generated", "action_name_changed", "action_arguments_changed")},
                         ensure_ascii=False))
        return

    records = []
    with (args.output_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for record in probe_rows(rows, backend, started=started,
                                 max_run_seconds=args.max_run_seconds):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records.append(record)
    report = {"diagnostic_only": True, "autonomous_agent_result": False,
              "generalization_metric": False, "input_decisions": str(args.decisions.resolve()),
              "model": args.model, "adapter": args.adapter,
              "model_load_seconds": backend.load_seconds,
              "elapsed_seconds": time.perf_counter() - started,
              "summary": summarize(records)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]["all"], ensure_ascii=False))


if __name__ == "__main__":
    main()
