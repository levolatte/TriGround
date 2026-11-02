"""CPU-only, raw-text audit of D's 32 completed G8 rollout groups.

The only counterfactual edit removes a complete Markdown fence around the
post-</think> answer. It never extracts a box from reasoning or alters a
JSON payload. No model or training state is loaded.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(CODE))

from tools.quant_grounding_reward import parse_thinking_bbox, score_completion


EVIDENCE = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = EVIDENCE / "d-rootcause-formal-diagnostic128.jsonl"
LOG = EVIDENCE / "d-timeout-signal/completed-updates-reward-metrics.jsonl"
ORIGINAL = EVIDENCE / "d-timeout-reward-metrics.jsonl"
REFERENCE = EVIDENCE / "warmup50-diagnostic32-g8-scored.jsonl"
TRAINING = EVIDENCE / "d-timeout-training-metrics.jsonl"
FENCE = re.compile(
    r"(?P<lead>\s*)```json[ \t]*\r?\n(?P<content>[\s\S]*?)\r?\n```(?P<trail>\s*)\Z"
)
BOX_LITERAL = re.compile(r"\[\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*\]")
RECONSIDER = re.compile(r"\b(?:wait|actually|re[- ]?evaluat\w*|re[- ]?check\w*|re[- ]?read\w*|double[- ]?check\w*|refin\w*)\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def payload_kind(content: str) -> str:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return "invalid_json"
    if isinstance(payload, dict):
        if set(payload) == {"bbox_2d"}:
            return "single_bbox_object"
        return "object_extra_or_missing_keys"
    if isinstance(payload, list):
        return f"array_{len(payload)}"
    return type(payload).__name__


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def candidate_single_box(text: str) -> tuple[str, str]:
    """Normalize one complete final object, without selecting or editing coordinates."""
    if text.count("</think>") != 1:
        return text, "no_unique_think_end"
    pre, answer = text.rsplit("</think>", 1)
    match = FENCE.fullmatch(answer)
    if match:
        answer = match.group("lead") + match.group("content") + match.group("trail")
    body = answer.strip()
    try:
        payload = json.loads(body, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError):
        return text, "invalid_final_json"
    unwrapped = False
    if isinstance(payload, list):
        if len(payload) != 1:
            return text, "multiple_or_zero_array_elements"
        payload = payload[0]
        unwrapped = True
    if not isinstance(payload, dict) or "bbox_2d" not in payload:
        return text, "not_single_bbox_object"
    keys = set(payload)
    if keys not in ({"bbox_2d"}, {"bbox_2d", "label"}):
        return text, "unapproved_extra_keys"
    if "label" in payload and not isinstance(payload["label"], str):
        return text, "non_string_label"
    canonical = json.dumps({"bbox_2d": payload["bbox_2d"]}, ensure_ascii=False)
    result = pre + "</think>" + "\n" + canonical
    if parse_thinking_bbox(result) is None:
        return text, "invalid_bbox"
    return result, "unwrap_one_and_drop_label" if unwrapped else (
        "drop_label" if "label" in payload else "remove_fence_only")


def main() -> None:
    batches = read_jsonl(LOG)
    original = read_jsonl(ORIGINAL)
    reference = {row["id"]: row for row in read_jsonl(REFERENCE)}
    formal_data = {row["id"]: row for row in read_jsonl(DATA)}
    items = [item for batch in batches for item in batch["items"]]
    assert len(batches) == 32 and len(items) == 256
    assert [batch["items"] for batch in batches] == [batch["items"] for batch in original[:32]]
    assert len(reference) == 32 and {item["id"] for item in items} == set(reference)
    assert all(len(batch["items"]) == 8 for batch in batches)
    assert all(len({item["occurrence_id"] for item in batch["items"]}) == 1 for batch in batches)
    assert all(item["reward"] == -0.2 and not item["valid"] for item in items)

    # Frozen snapshot of the formal 128-query diagnostic: the 32 rollout IDs
    # and all rollout metadata must match, then independently reproduce the
    # saved warmup scoring using the snapshot's exact GT and negative boxes.
    assert len(formal_data) == 128 and set(reference) <= set(formal_data)
    for item in items:
        row = formal_data[item["id"]]
        for key in ("id", "source", "group_id", "negative_kind", "occurrence_id", "sampling_pool"):
            assert item[key] == row[key], (item["id"], key)
    verified_samples = 0
    for item_id in reference:
        ref, formal = reference[item_id], formal_data[item_id]
        assert (ref["solution"], ref["negative_kind"], ref["source"], ref["group_id"]) == (
            formal["solution"], formal["negative_kind"], formal["source"], formal["group_id"])
        for sample in ref["samples"]:
            score = score_completion(
                sample["text"], formal["solution"], formal["negative_boxes"],
                answer_mode="short", finish_reason=sample["finish_reason"],
            )
            assert abs(score["reward"] - sample["score"]["reward"]) < 1e-12
            assert score["max_negative_iou"] == sample["score"]["max_negative_iou"]
            verified_samples += 1

    details = []
    by_group = defaultdict(list)
    for batch in batches:
        for index, item in enumerate(batch["items"]):
            raw = item["generated_text"]
            formal = formal_data[item["id"]]
            negatives = formal["negative_boxes"]
            baseline = score_completion(
                raw, formal["solution"], negatives, answer_mode="thinking",
                finish_reason=item["finish_reason"],
            )
            assert baseline["reward"] == item["reward"]
            assert baseline["valid"] == item["valid"]
            assert baseline["truncated"] == item["truncated"]
            assert baseline["format_valid"] == item["format_valid"]

            end_count = raw.count("</think>")
            fence_content = None
            changed = raw
            kind = None
            if end_count == 1:
                pre, answer = raw.rsplit("</think>", 1)
                match = FENCE.fullmatch(answer)
                if match:
                    fence_content = match.group("content")
                    kind = payload_kind(fence_content)
                    changed = pre + "</think>" + match.group("lead") + fence_content + match.group("trail")
            score = score_completion(
                changed, formal["solution"], negatives, answer_mode="thinking",
                finish_reason=item["finish_reason"],
            )
            candidate_text, candidate_action = candidate_single_box(raw)
            candidate_score = score_completion(
                candidate_text, formal["solution"], negatives, answer_mode="thinking",
                finish_reason=item["finish_reason"],
            )
            assert (parse_thinking_bbox(changed) is not None) == score["format_valid"]
            box_literals = len(BOX_LITERAL.findall(raw.split("</think>", 1)[0]))
            row = {
                "id": item["id"], "occurrence_id": item["occurrence_id"],
                "sample_index": index, "source": item["source"],
                "negative_kind": item["negative_kind"],
                "finish_reason": item["finish_reason"],
                "generated_tokens": item["generated_tokens"],
                "generated_text": raw, "think_end_count": end_count,
                "final_fence": fence_content is not None,
                "fenced_payload_kind": kind,
                "reasoning_box_literal_count": box_literals,
                "reasoning_reconsideration_marker_count": len(RECONSIDER.findall(raw.split("</think>", 1)[0])),
                "counterfactual_text": changed,
                "candidate_protocol_text": candidate_text,
                "candidate_protocol_action": candidate_action,
                "solution": formal["solution"], "negative_boxes": negatives,
                "negative_source": "frozen_formal_diagnostic128",
                "original": baseline, "fence_only": score,
                "candidate_protocol": candidate_score,
            }
            details.append(row)
            by_group[item["occurrence_id"]].append(row)

    assert len(by_group) == 32 and all(len(rows) == 8 for rows in by_group.values())
    assert all(item["think_end_count"] == 0 for item in details if item["finish_reason"] == "length")
    assert all(item["think_end_count"] == 1 and item["final_fence"] for item in details if item["finish_reason"] == "stop")
    assert all(item["finish_reason"] == "length" and item["generated_tokens"] == 384 for item in details if item["think_end_count"] == 0)

    group_details = []
    for group, rows in by_group.items():
        rewards = [r["fence_only"]["reward"] for r in rows]
        valid = [r["fence_only"] for r in rows if r["fence_only"]["valid"]]
        geometric_rewards = [r["hit_reward"] + r["iou_reward"] for r in valid]
        candidate_rewards = [r["candidate_protocol"]["reward"] for r in rows]
        candidate_valid = [r["candidate_protocol"] for r in rows if r["candidate_protocol"]["valid"]]
        candidate_geometric_rewards = [r["hit_reward"] + r["iou_reward"] for r in candidate_valid]
        group_details.append({
            "occurrence_id": group, "id": rows[0]["id"],
            "fence_only_valid": sum(r["fence_only"]["valid"] for r in rows),
            "fence_only_hit": sum(r["fence_only"]["hit"] for r in rows),
            "reward_population_variance": statistics.pvariance(rewards),
            "nonzero_advantage": len(set(rewards)) > 1,
            "valid_iou_differs": len({r["iou"] for r in valid}) > 1,
            "valid_geometric_reward_differs": len(set(geometric_rewards)) > 1,
            "valid_total_reward_differs": len({r["reward"] for r in valid}) > 1,
            "rewards": rewards,
            "candidate_valid": len(candidate_valid),
            "candidate_hit": sum(r["hit"] for r in candidate_valid),
            "candidate_reward_population_variance": statistics.pvariance(candidate_rewards),
            "candidate_nonzero_advantage": len(set(candidate_rewards)) > 1,
            "candidate_valid_iou_differs": len({r["iou"] for r in candidate_valid}) > 1,
            "candidate_valid_geometric_reward_differs": len(set(candidate_geometric_rewards)) > 1,
            "candidate_valid_total_reward_differs": len({r["reward"] for r in candidate_valid}) > 1,
            "candidate_rewards": candidate_rewards,
        })
    training = read_jsonl(TRAINING)
    summary = {
        "scope": "8 completed updates only; 32 G8 groups; no interrupted update",
        "source_rows": {"completed_batches": len(batches), "original_batches": len(original),
                        "answers": len(details), "frozen_formal_rows": len(formal_data),
                        "matched_formal_ids": len(reference), "verified_warmup_samples": verified_samples},
        "original": {
            "valid": sum(r["original"]["valid"] for r in details),
            "truncated": sum(r["original"]["truncated"] for r in details),
            "length_without_think_end": sum(r["finish_reason"] == "length" and r["think_end_count"] == 0 for r in details),
            "stopped_with_think_end_and_fence": sum(r["finish_reason"] == "stop" and r["think_end_count"] == 1 and r["final_fence"] for r in details),
            "all_equal_reward_groups": sum(len({r["original"]["reward"] for r in rows}) == 1 for rows in by_group.values()),
            "training_zero_loss_steps": sum(row["metrics"]["loss"] == 0 for row in training),
            "training_zero_grad_norm_steps": sum(row["metrics"]["grad_norm"] == 0 for row in training),
        },
        "reasoning_observations": {
            "unfinished_with_bbox_literal": sum(r["finish_reason"] == "length" and r["reasoning_box_literal_count"] > 0 for r in details),
            "unfinished_with_two_or_more_bbox_literals": sum(r["finish_reason"] == "length" and r["reasoning_box_literal_count"] >= 2 for r in details),
            "unfinished_with_reconsideration_marker": sum(r["finish_reason"] == "length" and r["reasoning_reconsideration_marker_count"] > 0 for r in details),
            "marker_definition": RECONSIDER.pattern,
            "bbox_literal_definition": BOX_LITERAL.pattern,
        },
        "fenced_payload_kind": dict(Counter(r["fenced_payload_kind"] for r in details if r["final_fence"])),
        "fence_only": {
            "changed_answers": sum(r["counterfactual_text"] != r["generated_text"] for r in details),
            "valid": sum(r["fence_only"]["valid"] for r in details),
            "hit": sum(r["fence_only"]["hit"] for r in details),
            "geometric_iou_mean_on_valid": statistics.mean(r["fence_only"]["iou"] for r in details if r["fence_only"]["valid"]),
            "geometric_iou_median_on_valid": statistics.median(r["fence_only"]["iou"] for r in details if r["fence_only"]["valid"]),
            "nonzero_advantage_groups": sum(g["nonzero_advantage"] for g in group_details),
            "multi_valid_groups": sum(g["fence_only_valid"] >= 2 for g in group_details),
            "multi_valid_iou_difference_groups": sum(g["fence_only_valid"] >= 2 and g["valid_iou_differs"] for g in group_details),
            "multi_valid_geometric_reward_difference_groups": sum(g["fence_only_valid"] >= 2 and g["valid_geometric_reward_differs"] for g in group_details),
            "multi_valid_total_reward_difference_groups": sum(g["fence_only_valid"] >= 2 and g["valid_total_reward_differs"] for g in group_details),
            "nonzero_advantage_from_valid_invalid_only_groups": sum(
                g["nonzero_advantage"] and 0 < g["fence_only_valid"] < 8 and not g["valid_total_reward_differs"]
                for g in group_details),
            "reward_variance_positive_groups": sum(g["reward_population_variance"] > 0 for g in group_details),
            "sum_group_population_variance": sum(g["reward_population_variance"] for g in group_details),
        },
        "candidate_protocol": {
            "definition": "one complete post-think JSON, optional one full json fence, unwrap only one-element array, drop only string label, preserve bbox coordinates; reject multiple boxes",
            "actions": dict(Counter(r["candidate_protocol_action"] for r in details)),
            "valid": sum(r["candidate_protocol"]["valid"] for r in details),
            "hit": sum(r["candidate_protocol"]["hit"] for r in details),
            "valid_iou_mean": statistics.mean(r["candidate_protocol"]["iou"] for r in details if r["candidate_protocol"]["valid"]),
            "nonzero_advantage_groups": sum(g["candidate_nonzero_advantage"] for g in group_details),
            "multi_valid_groups": sum(g["candidate_valid"] >= 2 for g in group_details),
            "multi_valid_iou_difference_groups": sum(g["candidate_valid"] >= 2 and g["candidate_valid_iou_differs"] for g in group_details),
            "multi_valid_geometric_reward_difference_groups": sum(g["candidate_valid"] >= 2 and g["candidate_valid_geometric_reward_differs"] for g in group_details),
            "multi_valid_total_reward_difference_groups": sum(g["candidate_valid"] >= 2 and g["candidate_valid_total_reward_differs"] for g in group_details),
            "nonzero_advantage_from_valid_invalid_only_groups": sum(
                g["candidate_nonzero_advantage"] and 0 < g["candidate_valid"] < 8
                and not g["candidate_valid_total_reward_differs"] for g in group_details),
        },
        "groups": group_details,
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "raw_answer_diagnostics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    (HERE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
