"""Compare matched real traces for observable tool-use and recovery behavior.

No GT is read. Query cue flags are review aids, not judgments of whether a
modality or final grounding was correct.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from tools.predict_aux_selection import load_manifest
from .audit_behavior import audit
from .report import load_traces


CAMERA_CUE = re.compile(
    r"\b(?:foreground|background|midground|frontmost|backmost|nearer|farther|farthest|furthest)\b"
    r"|\b(?:near|far|close|distant|middle)\s+(?:to|from|the\s+)?(?:camera|viewer)\b"
    r"|前景|中景|背景|相机距离|镜头距离|靠近镜头|远离镜头", re.I,
)
REFERENCE_RELATION_CUE = re.compile(
    r"\b(?:left|right)\s+of\b|\b(?:next to|beside|adjacent to|near the|behind the|in front of the)\b"
    r"|在.{0,8}(?:左边|右边|旁边|后面|前面)|靠近.{0,8}(?:物体|人|车)", re.I,
)


def _depth_review(query: str) -> dict:
    camera = bool(CAMERA_CUE.search(query))
    reference = bool(REFERENCE_RELATION_CUE.search(query))
    return {
        "explicit_camera_cue": camera,
        "reference_relation_cue": reference,
        "without_explicit_camera_cue": not camera,
        "possible_non_camera_relation": reference and not camera,
        "manual_review_required": True,
    }


def _recovery(actions: list[dict], index: int) -> dict:
    current = actions[index]
    following = actions[index + 1] if index + 1 < len(actions) else None
    actual = current["actual_action"]
    next_actual = following["actual_action"] if following else None
    next_model = following["model_action"] if following else None
    if next_actual:
        if next_actual["action"] != actual["action"]:
            kind = "changed_tool"
        elif following["modalities"] != current["modalities"]:
            kind = "changed_modality"
        else:
            kind = "same_tool_and_modality"
    elif (next_model or {}).get("action") == "finish":
        kind = "finished"
    else:
        kind = "no_next_executed_tool"
    return {
        "step": current["step"], "status": current["observation_status"],
        "from_action": actual, "next_model_action": next_model,
        "next_actual_action": next_actual, "response": kind,
    }


def _record(trace: dict, audited: dict) -> dict:
    actions = audited["actions"]
    adjacent = []
    for left, right in zip(actions, actions[1:]):
        if ((left["actual_action"] or {}).get("action"),
                (right["actual_action"] or {}).get("action")) == (
                "inspect_regions", "measure_depth"):
            adjacent.append({
                "inspect_step": left["step"], "depth_step": right["step"],
                "same_candidate_ids": left["actual_action"].get("candidate_ids") ==
                                      right["actual_action"].get("candidate_ids"),
            })
    depth = [{"step": action["step"], "action": action["actual_action"],
              "status": action["observation_status"], "query_cues": _depth_review(trace["query"]),
              "pair_status": (((trace["events"][index].get("observation") or {}).get("data") or {})
                              .get("pair") or {}).get("status")}
             for index, action in enumerate(actions)
             if (action["actual_action"] or {}).get("action") == "measure_depth"]
    recovery = [_recovery(actions, index) for index, action in enumerate(actions)
                if action["actual_action"] and action["observation_status"] in {"UNKNOWN", "EMPTY"}]
    sequence = [{"step": action["step"], "model_action": action["model_action"],
                 "actual_action": action["actual_action"], "status": action["observation_status"],
                 "evidence_note": action["evidence_note"],
                 "observation_text": action["observation_text"],
                 "observation_data": (trace["events"][index].get("observation") or {}).get("data"),
                 "returned_images": action["returned_images"]}
                for index, action in enumerate(actions)]
    return {
        "id": str(trace["id"]), "query": trace["query"],
        "mechanism": audited["mechanism"], "prompt_version": audited["prompt_version"],
        "final_status": audited["final_status"], "final_id": audited["final_id"],
        "action_sequence": sequence,
        "adjacent_inspect_then_depth": adjacent,
        "depth_calls_for_manual_review": depth,
        "unknown_empty_recovery": recovery,
        "ir_returned_images": audited["returned_images"].get("ir", 0),
        "ir_next_processor_consumed_images": audited["next_processor_consumed_images"].get("ir", 0),
    }


def _summary(records: list[dict]) -> dict:
    recoveries = Counter(item["response"] for record in records for item in record["unknown_empty_recovery"])
    depth = [item for record in records for item in record["depth_calls_for_manual_review"]]
    return {
        "queries": len(records),
        "adjacent_inspect_then_depth_calls": sum(len(record["adjacent_inspect_then_depth"]) for record in records),
        "adjacent_inspect_then_depth_queries": sum(bool(record["adjacent_inspect_then_depth"]) for record in records),
        "adjacent_inspect_then_depth_same_ids": sum(item["same_candidate_ids"] for record in records
                                                       for item in record["adjacent_inspect_then_depth"]),
        "depth_calls": len(depth),
        "depth_without_explicit_camera_cue_review": sum(item["query_cues"]["without_explicit_camera_cue"]
                                                        for item in depth),
        "depth_possible_non_camera_relation_review": sum(item["query_cues"]["possible_non_camera_relation"]
                                                         for item in depth),
        "unknown_empty_events": sum(recoveries.values()),
        "unknown_empty_next_decision": dict(recoveries),
        "ir_returned_queries": sum(record["ir_returned_images"] > 0 for record in records),
        "ir_next_processor_consumed_queries": sum(record["ir_next_processor_consumed_images"] > 0
                                                   for record in records),
        "ir_returned_images": sum(record["ir_returned_images"] for record in records),
        "ir_next_processor_consumed_images": sum(record["ir_next_processor_consumed_images"] for record in records),
    }


def compare(before: list[dict], after: list[dict], manifest: list[dict]) -> dict:
    expected = {str(row["id"]): row for row in manifest}
    if len(expected) != len(manifest):
        raise ValueError("fixed manifest has duplicate IDs")
    cohorts = []
    for name, traces in (("before", before), ("after", after)):
        ids = [str(trace["id"]) for trace in traces]
        if len(ids) != len(set(ids)) or set(ids) != set(expected):
            raise ValueError(f"{name} traces do not match the full fixed manifest ID set")
        for trace in traces:
            frozen = expected[str(trace["id"])]
            if trace["query"] != frozen["query"]:
                raise ValueError(f"{name} query differs from fixed manifest: {trace['id']}")
            if (frozen.get("images", {}).get("rgb") and
                    (trace.get("manifest_row") or {}).get("images", {}).get("rgb") != frozen["images"]["rgb"]):
                raise ValueError(f"{name} RGB image group differs from fixed manifest: {trace['id']}")
        audited = {record["id"]: record for record in audit(traces)["queries"]}
        cohorts.append({str(trace["id"]): _record(trace, audited[str(trace["id"])]) for trace in traces})
    first, second = cohorts
    before_summary, after_summary = _summary(list(first.values())), _summary(list(second.values()))
    samples = []
    for sample_id in sorted(expected):
        left, right = first[sample_id], second[sample_id]
        counts = lambda record: {
            "adjacent_inspect_then_depth": len(record["adjacent_inspect_then_depth"]),
            "depth_calls": len(record["depth_calls_for_manual_review"]),
            "depth_without_explicit_camera_cue_review": sum(
                item["query_cues"]["without_explicit_camera_cue"] for item in record["depth_calls_for_manual_review"]),
            "unknown_empty_events": len(record["unknown_empty_recovery"]),
            "ir_next_processor_consumed_images": record["ir_next_processor_consumed_images"],
        }
        old, new = counts(left), counts(right)
        samples.append({"id": sample_id, "query": expected[sample_id]["query"],
                        "before": left, "after": right,
                        "change": {key: new[key] - old[key] for key in old}})
    return {
        "schema_version": "visual-agent-recovery-comparison-v1",
        "interpretation": "Observed sequences/statuses and lexical review cues only; no GT or decision-correctness claim.",
        "before_summary": before_summary, "after_summary": after_summary,
        "summary_change": {key: after_summary[key] - before_summary[key] for key in before_summary
                           if isinstance(before_summary[key], int)},
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-traces", type=Path, required=True)
    parser.add_argument("--after-traces", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_manifest(args.manifest, require_images=True)
    result = compare(load_traces(args.before_traces), load_traces(args.after_traces), rows)
    result["manifest"] = str(args.manifest.resolve())
    result["before_traces"] = str(args.before_traces.resolve())
    result["after_traces"] = str(args.after_traces.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("before_summary", "after_summary", "summary_change")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
