"""Summarize actual visual-agent actions and observations without ground truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .report import load_traces


TOOLS = {"inspect_regions", "measure_depth", "search_candidates"}


def _path(value):
    return str(value).replace("\\", "/")


def _modalities(action):
    kind = action["action"]
    if kind == "inspect_regions":
        modalities = action.get("modalities") or []
        return [part.strip() for part in modalities.split(",")] if isinstance(modalities, str) else list(modalities)
    if kind == "search_candidates":
        return [action.get("modality", "<missing>")]
    if kind == "measure_depth":
        return ["depth_numeric"]
    return []


def _signature(action):
    parameters = {key: value for key, value in action.items() if key != "evidence_note"}
    return json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _next_processor_has_image(events, index, image):
    if index + 1 >= len(events):
        return False
    next_event = events[index + 1]
    if (next_event.get("usage") or {}).get("model_seconds", 0) <= 0:
        return False
    return any(
        geometry.get("view") == "tool"
        and geometry.get("modality") == image.get("modality")
        and geometry.get("path") is not None
        and _path(geometry["path"]) == _path(image["path"])
        for geometry in (next_event.get("usage") or {}).get("image_geometry", [])
    )


def _trace_record(trace):
    events = trace["events"]
    initial_ids = {str(candidate["id"]) for candidate in trace["initial_candidates"]}
    repeated = Counter()
    actions = []
    returned_images = Counter()
    consumed_images = Counter()
    appended = Counter()
    depth_measured = False
    ir_search_calls = 0
    repeated_calls = 0

    for index, event in enumerate(events):
        model_action = event.get("action")
        actual_action = event.get("executed_action")
        observation = event.get("observation") or {}
        status = observation.get("status")
        invoked = "tool_seconds" in event and (actual_action or {}).get("action") in TOOLS
        modalities = _modalities(actual_action) if invoked else []

        if invoked:
            signature = _signature(actual_action)
            repeat = repeated[signature] > 0
            repeated_calls += repeat
            repeated[signature] += 1
            if actual_action["action"] == "measure_depth" and (observation.get("data") or {}).get("measurements"):
                depth_measured = True
            if actual_action["action"] == "search_candidates":
                ir_search_calls += actual_action["modality"] == "ir"
                after = {str(candidate["id"]): candidate for candidate in event["candidates_after"]}
                for candidate_id in (observation.get("data") or {}).get("appended_ids", []):
                    candidate_id = str(candidate_id)
                    if candidate_id in initial_ids:
                        raise ValueError(f"{trace['id']}: appended candidate was already initial: {candidate_id}")
                    appended[(actual_action["modality"], after[candidate_id]["role"])] += 1

        images = []
        for image in observation.get("images") or []:
            if not image.get("path"):
                raise ValueError(f"{trace['id']}: returned tool image has no path")
            modality = image["modality"]
            consumed = _next_processor_has_image(events, index, image)
            returned_images[modality] += 1
            consumed_images[modality] += consumed
            images.append({"modality": modality, "path": image["path"],
                           "next_processor_consumed": consumed})

        actions.append({
            "step": event.get("step", index),
            "raw_output": event.get("raw_output"),
            "model_action": model_action,
            "actual_action": actual_action if invoked else None,
            "tool_invoked": invoked,
            "repeated_same_parameters": repeat if invoked else False,
            "modalities": modalities,
            "evidence_note": (model_action or {}).get("evidence_note"),
            "observation_status": status,
            "observation_text": observation.get("text"),
            "returned_images": images,
        })

    return {
        "id": str(trace["id"]), "query": trace["query"],
        "mechanism": trace["mechanism"], "prompt_version": trace.get("prompt_version") or "<missing>",
        "controller_type": "fixed" if trace["mechanism"] == "S" else
                           "autonomous" if trace["mechanism"] in {"C", "D"} else "other",
        "profile": trace["profile"]["name"],
        "actions": actions,
        "final_status": trace["final_status"], "final_id": trace.get("selected_id"),
        "returned_images": dict(returned_images), "next_processor_consumed_images": dict(consumed_images),
        "depth_actually_measured": depth_measured,
        "ir_search_calls": ir_search_calls,
        "search_appended_by_modality_role": {
            f"{modality}/{role}": count for (modality, role), count in sorted(appended.items())
        },
        "repeated_same_parameter_tool_calls": repeated_calls,
    }


def _summary(records):
    model_actions, tool_calls, observations = Counter(), Counter(), Counter()
    tool_status = defaultdict(Counter)
    modality_status = defaultdict(Counter)
    returned, consumed, appended = Counter(), Counter(), Counter()
    ir_search_status = Counter()
    depth_queries = ir_return_queries = ir_consumed_queries = repeated = 0
    for record in records:
        depth_queries += record["depth_actually_measured"]
        ir_return_queries += record["returned_images"].get("ir", 0) > 0
        ir_consumed_queries += record["next_processor_consumed_images"].get("ir", 0) > 0
        repeated += record["repeated_same_parameter_tool_calls"]
        returned.update(record["returned_images"])
        consumed.update(record["next_processor_consumed_images"])
        appended.update(record["search_appended_by_modality_role"])
        for event in record["actions"]:
            action = event["model_action"] or {}
            if action.get("action"):
                model_actions[action["action"]] += 1
            status = event["observation_status"]
            if status:
                observations[status] += 1
            actual = event["actual_action"]
            if not actual:
                continue
            kind = actual["action"]
            tool_calls[kind] += 1
            tool_status[kind][status] += 1
            for modality in event["modalities"]:
                modality_status[f"{kind}/{modality}"][status] += 1
            if kind == "search_candidates" and actual["modality"] == "ir":
                ir_search_status[status] += 1
    return {
        "queries": len(records),
        "model_action_counts": dict(model_actions),
        "actual_tool_call_counts": dict(tool_calls),
        "observation_status_counts": dict(observations),
        "tool_status_counts": {kind: dict(counts) for kind, counts in sorted(tool_status.items())},
        "tool_modality_status_counts": {name: dict(counts) for name, counts in sorted(modality_status.items())},
        "returned_tool_images_by_modality": dict(returned),
        "next_processor_consumed_images_by_modality": dict(consumed),
        "ir_returned_queries": ir_return_queries,
        "ir_next_processor_consumed_queries": ir_consumed_queries,
        "depth_actually_measured_queries": depth_queries,
        "ir_search": {"calls": sum(ir_search_status.values()), "statuses": dict(ir_search_status),
                      "new_target_candidates": appended["ir/target"],
                      "new_reference_candidates": appended["ir/reference"]},
        "search_appended_by_modality_role": dict(appended),
        "repeated_same_parameter_tool_calls": repeated,
    }


def audit(traces):
    records = [_trace_record(trace) for trace in traces]
    strata = defaultdict(list)
    for record in records:
        strata[(record["mechanism"], record["prompt_version"])].append(record)
    return {
        "schema_version": "visual-agent-behavior-v1",
        "summary": _summary(records),
        "strata": [
            {"mechanism": mechanism, "prompt_version": prompt_version,
             "controller_type": group[0]["controller_type"], "summary": _summary(group)}
            for (mechanism, prompt_version), group in sorted(strata.items())
        ],
        "queries": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True, help="one run's traces directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(load_traces(args.traces))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
