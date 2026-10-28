"""Summarize whether quant-grounding rollouts provide a useful learning signal."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_reward import parse_bbox  # noqa: E402

G8_SIZE = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            rows.append(row)
    return rows


def _dimension_value(value: Any, name: str) -> str:
    if name == "negative_kind":
        if value is None or value == [] or value == "":
            return "none"
        if isinstance(value, (list, tuple)):
            return "+".join(sorted(str(part) for part in value))
    return "unknown" if value is None or value == "" else str(value)


def _same_reward(items: list[dict[str, Any]]) -> bool:
    values = [item.get("reward") for item in items]
    return all(value == values[0] for value in values[1:])


def _group_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = []
    for record_index, row in enumerate(rows):
        items = row.get("items")
        if not isinstance(items, list):
            raise ValueError(f"reward metrics record {record_index + 1} has no items list")
        local_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"reward metrics record {record_index + 1} item {item_index} is not an object")
            for required_field in ("hit", "valid", "truncated", "reward"):
                if required_field not in item:
                    raise ValueError(
                        f"reward metrics record {record_index + 1} item {item_index} misses {required_field}"
                    )
            for boolean_field in ("hit", "valid", "truncated"):
                if not isinstance(item[boolean_field], bool):
                    raise ValueError(
                        f"reward metrics record {record_index + 1} item {item_index} "
                        f"has non-boolean {boolean_field}"
                    )
            reward = item["reward"]
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                raise ValueError(
                    f"reward metrics record {record_index + 1} item {item_index} has invalid reward"
                )
            occurrence_id = item.get("occurrence_id")
            group_field = "occurrence_id" if occurrence_id is not None else "id"
            group_value = item.get(group_field)
            if group_value is None:
                group_field, group_value = "item_index", str(item_index)
            local_groups.setdefault((group_field, str(group_value)), []).append(item)

        # A repeated occurrence in a later metrics record is a new rollout group.
        for (key_field, key), group_items in local_groups.items():
            first = group_items[0]
            ids = {str(item["id"]) for item in group_items if item.get("id") is not None}
            generated_texts = [item.get("generated_text") for item in group_items]
            inline_raw_samples = (
                [{"text": text, "finish_reason": item.get("finish_reason")}
                 for text, item in zip(generated_texts, group_items)]
                if all(isinstance(text, str) for text in generated_texts) else None
            )
            groups.append({
                "group_index": len(groups),
                "metrics_record": record_index,
                "key_field": key_field,
                "key": key,
                "occurrence_id": first.get("occurrence_id"),
                "id": first.get("id"),
                "ids": sorted(ids),
                "source": _dimension_value(first.get("source"), "source"),
                "negative_kind": _dimension_value(first.get("negative_kind"), "negative_kind"),
                "sampling_pool": _dimension_value(first.get("sampling_pool"), "sampling_pool"),
                "items": group_items,
                "raw_samples": inline_raw_samples,
                "answer_count": len(group_items),
                "complete": len(group_items) == G8_SIZE,
                "outcome": (
                    "allhit" if all(item.get("hit") is True for item in group_items)
                    else "allwrong" if all(item.get("hit") is not True for item in group_items)
                    else "mixed"
                ),
                "all_same_reward": _same_reward(group_items),
                "invalid_count": sum(item.get("valid") is False for item in group_items),
                "truncation_count": sum(item.get("truncated") is True for item in group_items),
            })
    return groups


def _raw_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Read eval-style id/samples JSONL and Swift completions.jsonl tables."""
    samples = []
    has_group_identity = False
    for row_index, row in enumerate(rows):
        row_id = row.get("id")
        row_occurrence = row.get("occurrence_id")
        sample_rows = row.get("samples")
        if isinstance(sample_rows, list):
            for sample_index, sample in enumerate(sample_rows):
                if isinstance(sample, str):
                    text, finish_reason = sample, None
                    sample_id, occurrence_id = row_id, row_occurrence
                elif isinstance(sample, dict):
                    text = sample.get("text", sample.get("completion"))
                    finish_reason = sample.get("finish_reason")
                    sample_id = sample.get("id", row_id)
                    occurrence_id = sample.get("occurrence_id", row_occurrence)
                else:
                    continue
                if isinstance(text, str):
                    samples.append({"id": sample_id, "occurrence_id": occurrence_id,
                                    "text": text, "finish_reason": finish_reason,
                                    "raw_record": row_index, "raw_index": sample_index})
                    has_group_identity |= occurrence_id is not None
            continue

        # Swift's log_completions format stores a columnar table with one array
        # per field (step, prompt, completion, rewards, advantages, ...).
        completions = row.get("completion", row.get("completions"))
        if isinstance(completions, str):
            completions = [completions]
        if isinstance(completions, list):
            ids = row_id if isinstance(row_id, list) else [row_id] * len(completions)
            occurrences = row_occurrence if isinstance(row_occurrence, list) else [row_occurrence] * len(completions)
            reasons = row.get("finish_reason", row.get("finish_reasons"))
            reasons = reasons if isinstance(reasons, list) else [reasons] * len(completions)
            for sample_index, text in enumerate(completions):
                if isinstance(text, str):
                    sample_id = ids[sample_index] if sample_index < len(ids) else None
                    occurrence_id = occurrences[sample_index] if sample_index < len(occurrences) else None
                    finish_reason = reasons[sample_index] if sample_index < len(reasons) else None
                    samples.append({"id": sample_id, "occurrence_id": occurrence_id,
                                    "text": text, "finish_reason": finish_reason,
                                    "raw_record": row_index, "raw_index": sample_index})
                    has_group_identity |= occurrence_id is not None
    return samples, has_group_identity


def _match_raw_samples(groups: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    by_occurrence_and_id: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample.get("occurrence_id") is not None and sample.get("id") is not None:
            by_occurrence_and_id[(str(sample["occurrence_id"]), str(sample["id"]))].append(sample)
        if sample.get("id") is not None:
            by_id[str(sample["id"])].append(sample)

    groups_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        for sample_id in group["ids"]:
            groups_by_id[sample_id].append(group)

    for group in groups:
        if group.get("raw_samples") is not None:
            continue
        matched = None
        if group["occurrence_id"] is not None and group["id"] is not None:
            direct = by_occurrence_and_id.get((str(group["occurrence_id"]), str(group["id"])), [])
            if len(direct) == group["answer_count"]:
                matched = direct
        if matched is None and len(group["ids"]) == 1:
            sample_id = group["ids"][0]
            candidates = by_id.get(sample_id, [])
            # A query-only raw record can be mapped only when it names exactly
            # one reward group, avoiding reuse across duplicate presentations.
            if len(groups_by_id[sample_id]) == 1 and len(candidates) == group["answer_count"]:
                matched = candidates
        group["raw_samples"] = matched

    # A Swift table may omit ids. Position mapping is safe only with exact
    # whole-file coverage; partial logs must not be assigned to arbitrary groups.
    if samples and all(sample.get("id") is None for sample in samples):
        item_slots = [(group, item_index) for group in groups for item_index in range(group["answer_count"])]
        if len(item_slots) == len(samples):
            cursor = 0
            for group in groups:
                if group["raw_samples"] is None:
                    group["raw_samples"] = samples[cursor:cursor + group["answer_count"]]
                cursor += group["answer_count"]


def _answer_mode(text: str, requested_mode: str, is_thinking_run: bool) -> str:
    if requested_mode != "auto":
        return requested_mode
    if is_thinking_run or "</think>" in text or "<think>" in text:
        return "thinking"
    return "short"


def _classify_d_answer(
    text: str | None,
    finish_reason: Any,
    item: dict[str, Any],
    mode: str,
    is_thinking_run: bool,
) -> str | None:
    if not is_thinking_run:
        return None
    if text is None:
        return "unknown_raw_text"
    if mode != "thinking":
        return "unfinished_thinking"
    has_think_end = "</think>" in text
    if not has_think_end:
        return "unfinished_thinking"
    if str(finish_reason).lower() == "length":
        return "token_limit_after_thinking"
    if parse_bbox(text, "thinking") is None:
        return "invalid_final_json"
    iou = item.get("iou")
    if isinstance(iou, (int, float)) and iou < 0.5:
        return "valid_bbox_wrong_target"
    if item.get("hit") is False and iou is None:
        return "unknown_target_result"
    return None


def _group_answer_details(
    group: dict[str, Any], requested_mode: str, is_thinking_run: bool
) -> dict[str, Any]:
    raw_samples = group.get("raw_samples")
    if raw_samples is None:
        group["d_failure_categories"] = ["unknown_raw_text"] * group["answer_count"] if is_thinking_run else []
        return {"status": "unavailable", "reason": "raw_text_unknown", "parsed_bbox_count": None,
                "unique_parsed_bbox_count": None, "parsed_bbox_variation": None}

    parsed_boxes = []
    failure_categories = []
    for index, item in enumerate(group["items"]):
        sample = raw_samples[index] if index < len(raw_samples) else None
        text = sample.get("text") if sample is not None else None
        finish_reason = (
            sample.get("finish_reason") or item.get("finish_reason")
        ) if sample else item.get("finish_reason")
        mode = _answer_mode(text or "", requested_mode, is_thinking_run)
        bbox = parse_bbox(text, mode) if isinstance(text, str) else None
        if bbox is not None:
            parsed_boxes.append(tuple(bbox))
        category = _classify_d_answer(text, finish_reason, item, mode, is_thinking_run)
        if category is not None:
            failure_categories.append(category)

    group["d_failure_categories"] = failure_categories
    unique_boxes = set(parsed_boxes)
    if not parsed_boxes:
        return {"status": "unavailable", "reason": "no_parsed_bbox", "parsed_bbox_count": 0,
                "unique_parsed_bbox_count": None, "parsed_bbox_variation": None}
    return {"status": "available", "reason": None, "parsed_bbox_count": len(parsed_boxes),
            "unique_parsed_bbox_count": len(unique_boxes),
            "parsed_bbox_variation": len(unique_boxes) > 1 if len(parsed_boxes) >= 2 else None}


def _summarize_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(group["outcome"] for group in groups)
    expected_slots = len(groups) * G8_SIZE
    observed_answers = sum(group["answer_count"] for group in groups)
    return {
        "groups": len(groups),
        "unique_queries": len({sample_id for group in groups for sample_id in group["ids"]}),
        "observed_answers": observed_answers,
        "expected_answer_slots": expected_slots,
        "unobserved_answer_slots": max(0, expected_slots - observed_answers),
        "complete_groups": sum(group["complete"] for group in groups),
        "incomplete_groups": sum(not group["complete"] for group in groups),
        "outcomes": {name: outcomes.get(name, 0) for name in ("allhit", "allwrong", "mixed")},
        "all_same_reward_groups": sum(group["all_same_reward"] for group in groups),
        "differing_reward_groups": sum(not group["all_same_reward"] for group in groups),
        "invalid_answers": sum(group["invalid_count"] for group in groups),
        "truncated_answers": sum(group["truncation_count"] for group in groups),
    }


def _by_dimension(groups: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        buckets[group[field]].append(group)
    return {key: _summarize_groups(buckets[key]) for key in sorted(buckets)}


def _kl_summary(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    if rows is None:
        return {"status": "unavailable", "steps": [], "count": 0}
    values = []
    for row in rows:
        metrics = row.get("metrics", row)
        if isinstance(metrics, dict) and isinstance(metrics.get("kl"), (int, float)):
            values.append({"step": row.get("step"), "kl": float(metrics["kl"])})
    if not values:
        return {"status": "unavailable", "steps": [], "count": 0}
    numbers = [row["kl"] for row in values]
    return {"status": "available", "steps": values, "count": len(values),
            "mean": statistics.fmean(numbers), "min": min(numbers), "max": max(numbers),
            "first": numbers[0], "last": numbers[-1]}


def _resource_summary(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    if rows is None:
        return {"status": "unavailable"}
    group_ids = [occurrence for row in rows for occurrence in row.get("rollout_occurrence_ids", [])]
    elapsed = [row.get("elapsed_seconds") for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
    answers = [row.get("rollout_answers") for row in rows if isinstance(row.get("rollout_answers"), (int, float))]
    return {"status": "available", "steps": len(rows), "rollout_groups_reported": len(group_ids),
            "unique_occurrence_ids": len(set(group_ids)), "rollout_answers_reported": sum(answers),
            "last_elapsed_seconds": elapsed[-1] if elapsed else None}


def analyze_signal(
    reward_rows: list[dict[str, Any]],
    gpu_hours: float,
    *,
    resource_rows: list[dict[str, Any]] | None = None,
    training_rows: list[dict[str, Any]] | None = None,
    raw_rows: list[dict[str, Any]] | None = None,
    answer_mode: str = "auto",
) -> dict[str, Any]:
    if not math.isfinite(gpu_hours) or gpu_hours <= 0:
        raise ValueError("gpu_hours must be a positive finite number")
    groups = _group_records(reward_rows)
    is_thinking_run = answer_mode == "thinking" or any(
        isinstance((row.get("metrics", row)), dict)
        and any("ThinkingReward" in key for key in row.get("metrics", row))
        for row in (training_rows or [])
    ) or any(
        isinstance(item.get("generated_text"), str)
        and ("</think>" in item["generated_text"] or "<think>" in item["generated_text"])
        for row in reward_rows
        for item in row.get("items", [])
        if isinstance(item, dict)
    )

    raw_sample_count = None
    if raw_rows is not None:
        raw_samples, _ = _raw_samples(raw_rows)
        _match_raw_samples(groups, raw_samples)
        raw_sample_count = max(
            len(raw_samples), sum(len(group["raw_samples"] or []) for group in groups)
        )
        is_thinking_run = is_thinking_run or any(
            "</think>" in sample["text"] or "<think>" in sample["text"] for sample in raw_samples
        )
    else:
        raw_sample_count = sum(len(group["raw_samples"] or []) for group in groups)

    answer_details = [_group_answer_details(group, answer_mode, is_thinking_run) for group in groups]
    base_summary = _summarize_groups(groups)
    base_summary["unique_queries"] = len({sample_id for group in groups for sample_id in group["ids"]})
    base_summary["differing_reward_groups_per_gpu_hour"] = base_summary["differing_reward_groups"] / gpu_hours
    base_summary["expected_answers_per_group"] = G8_SIZE

    available_diversity = [detail for detail in answer_details if detail["status"] == "available"]
    matched_raw_groups = sum(group.get("raw_samples") is not None for group in groups)
    all_parsed_counts = [detail["unique_parsed_bbox_count"] for detail in available_diversity]
    diversity = {
        "status": "available" if available_diversity else "unavailable",
        "reason": None if available_diversity else (
            "raw_completions_not_provided"
            if raw_rows is None and not raw_sample_count else "no_matched_parsed_bbox"
        ),
        "raw_completions_provided": raw_rows is not None,
        "raw_sample_count": raw_sample_count,
        "groups_with_raw_samples": matched_raw_groups,
        "groups_with_parsed_bbox": len(available_diversity) if matched_raw_groups else None,
        "groups_with_bbox_variation": (
            sum(detail["parsed_bbox_variation"] is True for detail in available_diversity)
            if any(detail["parsed_bbox_variation"] is not None for detail in available_diversity) else None
        ),
        "groups_with_variation_assessable": sum(
            detail["parsed_bbox_variation"] is not None for detail in available_diversity
        ),
        "mean_unique_parsed_bboxes_per_group": statistics.fmean(all_parsed_counts) if all_parsed_counts else None,
    }

    d_categories = Counter(category for group in groups for category in group.get("d_failure_categories", []))
    d_reason_names = ("unfinished_thinking", "token_limit_after_thinking", "invalid_final_json",
                      "valid_bbox_wrong_target", "unknown_raw_text", "unknown_target_result")
    d_classified_count = sum(
        len(group.get("raw_samples") or []) for group in groups
    ) if is_thinking_run else 0
    total_answer_count = sum(group["answer_count"] for group in groups)
    if not is_thinking_run:
        d_status = "not_applicable" if answer_mode != "thinking" else "unknown"
    elif d_classified_count == 0:
        d_status = "unknown"
    elif d_classified_count < total_answer_count:
        d_status = "partial"
    else:
        d_status = "available"

    return {
        "gpu_hours": gpu_hours,
        "answer_mode": "thinking" if is_thinking_run else (answer_mode if answer_mode != "auto" else "short_or_unknown"),
        "overall": base_summary,
        "by_source": _by_dimension(groups, "source"),
        "by_negative_kind": _by_dimension(groups, "negative_kind"),
        "by_sampling_pool": _by_dimension(groups, "sampling_pool"),
        "answer_diversity": diversity,
        "d_failure_categories": {"status": d_status,
                                 "counts": {name: d_categories.get(name, 0) for name in d_reason_names}},
        "kl": _kl_summary(training_rows),
        "resources": _resource_summary(resource_rows),
        "groups_detail": [
            {"group_index": group["group_index"], "metrics_record": group["metrics_record"],
             "key_field": group["key_field"], "key": group["key"], "id": group["id"],
             "occurrence_id": group["occurrence_id"], "source": group["source"],
             "negative_kind": group["negative_kind"], "sampling_pool": group["sampling_pool"],
             "answer_count": group["answer_count"], "complete": group["complete"],
             "outcome": group["outcome"], "all_same_reward": group["all_same_reward"],
             "invalid_answers": group["invalid_count"], "truncated_answers": group["truncation_count"],
             "answer_diversity": detail,
             "d_failure_categories": group.get("d_failure_categories", [])}
            for group, detail in zip(groups, answer_details)
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# Quant grounding learning-signal diagnosis",
        "",
        f"GPU hours: {report['gpu_hours']}",
        f"Answer mode: {report['answer_mode']}",
        "",
        "## Rollout groups",
        "",
        f"- Unique queries: {overall['unique_queries']}",
        f"- Groups: {overall['groups']} ({overall['complete_groups']} complete, {overall['incomplete_groups']} incomplete; denominator includes all groups)",
        f"- Outcomes: allhit {overall['outcomes']['allhit']}, allwrong {overall['outcomes']['allwrong']}, mixed {overall['outcomes']['mixed']}",
        f"- Same reward: {overall['all_same_reward_groups']}; differing reward: {overall['differing_reward_groups']} ({overall['differing_reward_groups_per_gpu_hour']:.4f} per GPU hour)",
        f"- Invalid answers: {overall['invalid_answers']}; truncated answers: {overall['truncated_answers']}",
        "",
        "## Answer diversity",
        "",
        f"- Status: {report['answer_diversity']['status']} ({report['answer_diversity']['reason'] or 'raw parsed boxes available'})",
        f"- Groups with raw samples: {report['answer_diversity']['groups_with_raw_samples']}; with parsed boxes: {report['answer_diversity']['groups_with_parsed_bbox']}; varying boxes: {report['answer_diversity']['groups_with_bbox_variation']}",
        "",
        "## D failure categories",
        "",
        f"- Status: {report['d_failure_categories']['status']}",
        "- " + ", ".join(f"{key}: {value}" for key, value in report["d_failure_categories"]["counts"].items()),
        "",
        "## Existing KL metrics",
        "",
    ]
    kl = report["kl"]
    lines.append(f"- Status: {kl['status']}; count: {kl['count']}")
    if kl["status"] == "available":
        lines.append(f"- Mean/min/max: {kl['mean']:.6g}/{kl['min']:.6g}/{kl['max']:.6g}; first/last: {kl['first']:.6g}/{kl['last']:.6g}")
    for dimension in ("by_source", "by_negative_kind", "by_sampling_pool"):
        lines.extend(["", f"## {dimension}", "", "| Grouping | Groups | Complete | Incomplete | All hit | All wrong | Mixed | Same reward | Different reward | Invalid | Truncated |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for name, summary in report[dimension].items():
            lines.append(
                f"| {name} | {summary['groups']} | {summary['complete_groups']} | {summary['incomplete_groups']} | "
                f"{summary['outcomes']['allhit']} | {summary['outcomes']['allwrong']} | {summary['outcomes']['mixed']} | "
                f"{summary['all_same_reward_groups']} | {summary['differing_reward_groups']} | "
                f"{summary['invalid_answers']} | {summary['truncated_answers']} |"
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-metrics", type=Path, required=True)
    parser.add_argument("--resource-steps", type=Path)
    parser.add_argument("--training-metrics", type=Path)
    parser.add_argument("--raw-completions", type=Path)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--answer-mode", choices=("auto", "short", "thinking"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_signal(
        read_jsonl(args.reward_metrics),
        args.gpu_hours,
        resource_rows=read_jsonl(args.resource_steps) if args.resource_steps else None,
        training_rows=read_jsonl(args.training_metrics) if args.training_metrics else None,
        raw_rows=read_jsonl(args.raw_completions) if args.raw_completions else None,
        answer_mode=args.answer_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "learning_signal_report.json"
    markdown_path = args.output_dir / "learning_signal_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
