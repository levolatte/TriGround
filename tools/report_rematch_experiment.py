"""Re-score rematch runs and report paired and candidate-pool diagnostics.

The evaluator that produced a predictions JSONL is deliberately not trusted for
ground-truth-dependent fields.  This small report tool reads the original
manifest, recomputes IoU in plain Python, and keeps candidate ordering
independent of the ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
ACC_THRESHOLD = 0.5
DEDUPE_THRESHOLD = 0.9
CANDIDATE_CAP = 8
IMAGE_FIELDS = ("visible", "infrared", "depth")


def box_iou(first: Iterable[float], second: Iterable[float]) -> float:
    """Return xyxy IoU without importing torch or another numeric package."""
    first_values = tuple(float(value) for value in first)
    second_values = tuple(float(value) for value in second)
    if len(first_values) != 4 or len(second_values) != 4:
        raise ValueError("boxes must contain four values")
    first_width = max(first_values[2] - first_values[0], 0.0)
    first_height = max(first_values[3] - first_values[1], 0.0)
    second_width = max(second_values[2] - second_values[0], 0.0)
    second_height = max(second_values[3] - second_values[1], 0.0)
    first_area = first_width * first_height
    second_area = second_width * second_height
    intersection_width = max(
        min(first_values[2], second_values[2])
        - max(first_values[0], second_values[0]),
        0.0,
    )
    intersection_height = max(
        min(first_values[3], second_values[3])
        - max(first_values[1], second_values[1]),
        0.0,
    )
    intersection = intersection_width * intersection_height
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _box(value: Any, *, field: str, allow_none: bool = True) -> list[float] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be a list of four numbers or null")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must contain finite numbers")
    return result


def _target_box(value: Any, sample_id: str) -> list[float]:
    target = _box(value, field=f"manifest[{sample_id!r}].bbox", allow_none=False)
    assert target is not None
    if not (target[0] < target[2] and target[1] < target[3]):
        raise ValueError(f"manifest[{sample_id!r}].bbox must be a valid xyxy box")
    return target


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("manifest must be a non-empty JSON object keyed by sample ID")
    manifest: dict[str, dict[str, Any]] = {}
    for raw_id, value in payload.items():
        sample_id = str(raw_id)
        if sample_id in manifest:
            raise ValueError(f"manifest contains duplicate ID {sample_id!r}")
        if not isinstance(value, dict):
            raise ValueError(f"manifest record {sample_id!r} must be an object")
        for field in IMAGE_FIELDS:
            if field not in value:
                raise ValueError(f"manifest record {sample_id!r} lacks {field!r}")
        record = dict(value)
        record["bbox"] = _target_box(value.get("bbox"), sample_id)
        manifest[sample_id] = record
    return manifest


def parse_run_spec(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError("--run must use NAME=predictions.jsonl")
    return name.strip(), Path(path.strip())


def _adjacent_metadata(path: Path) -> dict[str, Any] | None:
    """Read an existing sibling summary as report-only metadata."""
    candidates = [
        path.parent / "summary.json",
        path.parent / "metadata.json",
        path.parent / "run_metadata.json",
        path.with_suffix(".json"),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists() or candidate == path.resolve():
            continue
        seen.add(candidate)
        payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        return {"path": str(candidate), "data": payload}
    return None


def _row_candidates(row: dict[str, Any], sample_id: str) -> tuple[list[float] | None, list[list[float]]]:
    has_candidates = "candidates" in row and row["candidates"] is not None
    if has_candidates:
        values = row["candidates"]
        if not isinstance(values, list):
            raise ValueError(f"run row {sample_id!r}.candidates must be a list")
        candidates = [
            _box(candidate, field=f"run row {sample_id!r}.candidates[{index}]")
            for index, candidate in enumerate(values)
        ]
        normalized_candidates = [candidate for candidate in candidates if candidate is not None]
    else:
        prediction = _box(
            row.get("prediction"),
            field=f"run row {sample_id!r}.prediction",
        )
        normalized_candidates = [prediction] if prediction is not None else []
    explicit_prediction = _box(
        row.get("prediction"),
        field=f"run row {sample_id!r}.prediction",
    )
    primary = explicit_prediction
    if primary is None and normalized_candidates:
        primary = normalized_candidates[0]
    return primary, normalized_candidates


def load_run(
    name: str,
    path: Path,
    manifest_ids: set[str],
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"{path}:{line_number}: each row needs an id")
        sample_id = str(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
        primary, candidates = _row_candidates(row, sample_id)
        rows[sample_id] = {
            "id": sample_id,
            "primary": primary,
            "candidates": candidates,
            "raw": row,
        }
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    extra = sorted(set(rows) - manifest_ids)
    if extra:
        raise ValueError(f"{path} has IDs absent from the full manifest: {extra[:5]}")
    missing = sorted(manifest_ids - set(rows))
    if missing and not allow_partial:
        raise ValueError(
            f"{path} is missing {len(missing)} manifest IDs; use --allow-partial explicitly"
        )
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": rows,
        "missing_ids": missing,
        "partial": bool(missing),
        "metadata": _adjacent_metadata(path),
    }


def _metrics(ious: list[float], parsed: int) -> dict[str, float | int]:
    count = len(ious)
    if not count:
        return {
            "samples": 0,
            "parsed": 0,
            "parse_rate": None,
            "mean_iou": None,
            "acc_0.5": None,
            "acc_0.7": None,
        }
    return {
        "samples": count,
        "parsed": parsed,
        "parse_rate": parsed / count,
        "mean_iou": sum(ious) / count,
        "acc_0.5": sum(iou >= 0.5 for iou in ious) / count,
        "acc_0.7": sum(iou >= 0.7 for iou in ious) / count,
    }


def score_run(run: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered_ids = [sample_id for sample_id in manifest if sample_id in run["rows"]]
    ious = [
        box_iou(run["rows"][sample_id]["primary"], manifest[sample_id]["bbox"])
        if run["rows"][sample_id]["primary"] is not None
        else 0.0
        for sample_id in ordered_ids
    ]
    parsed = sum(run["rows"][sample_id]["primary"] is not None for sample_id in ordered_ids)
    return _metrics(ious, parsed)


def image_group(record: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(record[field]) for field in IMAGE_FIELDS)  # type: ignore[return-value]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_group_bootstrap(
    entries: list[tuple[str, float, float]],
    manifest: dict[str, dict[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for sample_id, left_iou, right_iou in entries:
        grouped.setdefault(image_group(manifest[sample_id]), []).append((left_iou, right_iou))
    groups = list(grouped.values())
    if not groups:
        return {"replicates": replicates, "seed": seed, "image_groups": 0}
    rng = random.Random(seed)
    deltas = {"mean_iou": [], "acc_0.5": [], "acc_0.7": []}
    for _ in range(replicates):
        sampled = [pair for _ in groups for pair in rng.choice(groups)]
        left = [pair[0] for pair in sampled]
        right = [pair[1] for pair in sampled]
        for metric, threshold in (("acc_0.5", 0.5), ("acc_0.7", 0.7)):
            deltas[metric].append(
                sum(value >= threshold for value in right) / len(right)
                - sum(value >= threshold for value in left) / len(left)
            )
        deltas["mean_iou"].append(sum(right) / len(right) - sum(left) / len(left))
    return {
        "replicates": replicates,
        "seed": seed,
        "image_groups": len(groups),
        **{
            f"{metric}_delta_95ci": [
                _percentile(values, 0.025),
                _percentile(values, 0.975),
            ]
            for metric, values in deltas.items()
        },
    }


def compare_runs(
    left: dict[str, Any],
    right: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    sample_ids = [sample_id for sample_id in manifest if sample_id in left["rows"] and sample_id in right["rows"]]
    entries: list[tuple[str, float, float]] = []
    wrong_to_right: list[str] = []
    right_to_wrong: list[str] = []
    for sample_id in sample_ids:
        left_box = left["rows"][sample_id]["primary"]
        right_box = right["rows"][sample_id]["primary"]
        left_iou = box_iou(left_box, manifest[sample_id]["bbox"]) if left_box is not None else 0.0
        right_iou = box_iou(right_box, manifest[sample_id]["bbox"]) if right_box is not None else 0.0
        entries.append((sample_id, left_iou, right_iou))
        if left_iou < ACC_THRESHOLD <= right_iou:
            wrong_to_right.append(sample_id)
        elif right_iou < ACC_THRESHOLD <= left_iou:
            right_to_wrong.append(sample_id)

    def transition_info(ids: list[str]) -> dict[str, Any]:
        groups = sorted({image_group(manifest[sample_id]) for sample_id in ids})
        return {
            "ids": ids,
            "image_count": len(groups),
            "image_groups": [list(group) for group in groups],
        }

    left_ious = [entry[1] for entry in entries]
    right_ious = [entry[2] for entry in entries]
    metrics_left = _metrics(
        left_ious,
        sum(left["rows"][sample_id]["primary"] is not None for sample_id in sample_ids),
    )
    metrics_right = _metrics(
        right_ious,
        sum(right["rows"][sample_id]["primary"] is not None for sample_id in sample_ids),
    )
    return {
        "left": left["name"],
        "right": right["name"],
        "samples": len(entries),
        "left_metrics": metrics_left,
        "right_metrics": metrics_right,
        "delta": {
            metric: metrics_right[metric] - metrics_left[metric]
            for metric in ("mean_iou", "acc_0.5", "acc_0.7")
        },
        "wrong_to_right": transition_info(wrong_to_right),
        "right_to_wrong": transition_info(right_to_wrong),
        "image_group_bootstrap": paired_group_bootstrap(
            entries, manifest, replicates=replicates, seed=seed
        ),
    }


def build_candidate_pool(
    runs: list[dict[str, Any]], sample_ids: Iterable[str], *, cap: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Merge source proposals in fixed run-order round-robin order.

    This function never receives the manifest and therefore cannot use GT to
    rank or choose proposals.  Deduplication compares proposals only.  The
    default retains all proposals for Recall@all; an explicit cap truncates
    the returned pool.
    """
    pool: dict[str, list[dict[str, Any]]] = {}
    for sample_id in sample_ids:
        streams = [run["rows"][sample_id]["candidates"] for run in runs]
        ordered: list[dict[str, Any]] = []
        max_length = max((len(stream) for stream in streams), default=0)
        for candidate_index in range(max_length):
            for run, stream in zip(runs, streams):
                if candidate_index >= len(stream):
                    continue
                candidate = stream[candidate_index]
                if any(box_iou(candidate, previous["box"]) >= DEDUPE_THRESHOLD for previous in ordered):
                    continue
                ordered.append(
                    {
                        "box": list(candidate),
                        "source": run["name"],
                        "source_rank": candidate_index + 1,
                    }
                )
        pool[sample_id] = ordered if cap is None else ordered[:cap]
    return pool


def _dedupe_stream(stream: list[list[float]]) -> list[list[float]]:
    result: list[list[float]] = []
    for candidate in stream:
        if not any(box_iou(candidate, previous) >= DEDUPE_THRESHOLD for previous in result):
            result.append(list(candidate))
    return result


def candidate_coverage(
    pool: dict[str, list[dict[str, Any]]],
    manifest: dict[str, dict[str, Any]],
    *,
    threshold: float = ACC_THRESHOLD,
) -> dict[str, Any]:
    sample_ids = list(manifest)
    coverage: dict[str, dict[str, Any]] = {}
    for label, limit in (("1", 1), ("4", 4), ("8", 8), ("all", None)):
        covered = []
        for sample_id in sample_ids:
            candidates = pool.get(sample_id, [])
            candidates = candidates if limit is None else candidates[:limit]
            if any(box_iou(candidate["box"], manifest[sample_id]["bbox"]) >= threshold for candidate in candidates):
                covered.append(sample_id)
        coverage[label] = {
            "covered_samples": len(covered),
            "total_samples": len(sample_ids),
            "recall": len(covered) / len(sample_ids) if sample_ids else None,
            "ids": covered,
        }
    return {
        "iou_threshold": threshold,
        "candidate_dedupe_iou": DEDUPE_THRESHOLD,
        "candidate_cap": CANDIDATE_CAP,
        "pool_truncated": False,
        "recall": coverage,
        "recall@1": coverage["1"]["recall"],
        "recall@4": coverage["4"]["recall"],
        "recall@8": coverage["8"]["recall"],
        "recall@all": coverage["all"]["recall"],
    }


def _run_summary(
    run: dict[str, Any],
    metrics: dict[str, Any],
    manifest_size: int,
) -> dict[str, Any]:
    return {
        "name": run["name"],
        "path": run["path"],
        "status": "partial" if run["partial"] else "complete",
        "partial": run["partial"],
        "samples": metrics["samples"],
        "manifest_samples": manifest_size,
        "missing_samples": len(run["missing_ids"]),
        "missing_ids": run["missing_ids"],
        "paired_eligible": not run["partial"],
        "gate_eligible": not run["partial"],
        "metrics": metrics,
        "metadata": run["metadata"],
    }


def _best_run(runs: list[dict[str, Any]], scored: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    complete = [run for run in runs if not run["partial"]]
    if not complete:
        return None
    return max(
        complete,
        key=lambda run: (
            float(scored[run["name"]]["acc_0.5"]),
            float(scored[run["name"]]["mean_iou"]),
            -runs.index(run),
        ),
    )


def build_report(
    manifest: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    scored = {run["name"]: score_run(run, manifest) for run in runs}
    run_summaries = [_run_summary(run, scored[run["name"]], len(manifest)) for run in runs]
    complete = [run for run in runs if not run["partial"]]
    pairs = [
        compare_runs(
            complete[left_index],
            complete[right_index],
            manifest,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        for left_index in range(len(complete))
        for right_index in range(left_index + 1, len(complete))
    ]
    best = _best_run(runs, scored)
    candidate_report: dict[str, Any] = {
        "status": "excluded_no_complete_runs",
        "eligible_runs": [],
        "recall": None,
    }
    if complete:
        sample_ids = list(manifest)
        pool = build_candidate_pool(complete, sample_ids)
        candidate_report = candidate_coverage(pool, manifest)
        candidate_report["status"] = "complete_runs_only"
        candidate_report["eligible_runs"] = [run["name"] for run in complete]
        candidate_report["pool"] = pool
        candidate_report["pool_top8"] = {
            sample_id: candidates[:CANDIDATE_CAP] for sample_id, candidates in pool.items()
        }
        if best is not None:
            best_streams = {
                sample_id: _dedupe_stream(best["rows"][sample_id]["candidates"])
                for sample_id in sample_ids
            }
            best_covered = {
                sample_id
                for sample_id, stream in best_streams.items()
                if any(box_iou(candidate, manifest[sample_id]["bbox"]) >= ACC_THRESHOLD for candidate in stream)
            }
            union_covered = set(candidate_report["recall"]["all"]["ids"])
            top8_covered = set(candidate_report["recall"]["8"]["ids"])
            new_ids = sorted(union_covered - best_covered)
            new_groups = sorted({image_group(manifest[sample_id]) for sample_id in new_ids})
            best_primary_covered = {
                sample_id for sample_id in sample_ids
                if best["rows"][sample_id]["primary"] is not None
                and box_iou(best["rows"][sample_id]["primary"], manifest[sample_id]["bbox"]) >= ACC_THRESHOLD
            }
            best_primary_hits = len(best_primary_covered)
            union_count = len(union_covered)
            new_primary_ids = sorted(union_covered - best_primary_covered)
            new_primary_groups = sorted({image_group(manifest[sample_id]) for sample_id in new_primary_ids})
            candidate_report["best_single_model"] = best["name"]
            candidate_report["best_single_model_metrics"] = scored[best["name"]]
            candidate_report["best_single_model_cover_samples"] = len(best_covered)
            candidate_report["union_cover_samples"] = union_count
            candidate_report["top8_cover_samples"] = len(top8_covered)
            candidate_report["new_cover_samples_vs_best"] = len(new_ids)
            candidate_report["new_cover_ids_vs_best"] = new_ids
            candidate_report["new_cover_image_groups_vs_best"] = [list(group) for group in new_groups]
            candidate_report["new_cover_image_count_vs_best"] = len(new_groups)
            candidate_report["new_cover_samples_vs_best_primary"] = len(new_primary_ids)
            candidate_report["new_cover_ids_vs_best_primary"] = new_primary_ids
            candidate_report["new_cover_image_groups_vs_best_primary"] = [list(group) for group in new_primary_groups]
            candidate_report["new_cover_image_count_vs_best_primary"] = len(new_primary_groups)
            candidate_report["B"] = best_primary_hits
            candidate_report["C"] = len(top8_covered)
            candidate_report["required_selection_rate"] = (
                (best_primary_hits + CANDIDATE_CAP) / len(top8_covered) if top8_covered else None
            )
    return {
        "manifest_samples": len(manifest),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "runs": run_summaries,
        "pairs": pairs,
        "candidate_union": candidate_report,
    }


CSV_FIELDS = (
    "kind",
    "name",
    "status",
    "partial",
    "paired_eligible",
    "gate_eligible",
    "samples",
    "manifest_samples",
    "missing_samples",
    "mean_iou",
    "acc_0.5",
    "acc_0.7",
    "parse_rate",
    "recall@1",
    "recall@4",
    "recall@8",
    "recall@all",
    "cover_samples",
    "new_cover_samples_vs_best",
    "new_cover_image_count_vs_best",
    "new_cover_samples_vs_best_primary",
    "new_cover_image_count_vs_best_primary",
    "B",
    "C",
    "required_selection_rate",
    "metadata_path",
)


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    candidate = report["candidate_union"]
    rows: list[dict[str, Any]] = []
    for run in report["runs"]:
        metrics = run["metrics"]
        rows.append(
            {
                "kind": "run",
                "name": run["name"],
                "status": run["status"],
                "partial": run["partial"],
                "paired_eligible": run["paired_eligible"],
                "gate_eligible": run["gate_eligible"],
                "samples": metrics["samples"],
                "manifest_samples": run["manifest_samples"],
                "missing_samples": run["missing_samples"],
                **{key: metrics[key] for key in ("mean_iou", "acc_0.5", "acc_0.7", "parse_rate")},
                "metadata_path": (run["metadata"] or {}).get("path"),
            }
        )
    recall = candidate.get("recall") or {}
    rows.append(
        {
            "kind": "candidate_union",
            "name": "+".join(candidate.get("eligible_runs", [])),
            "status": candidate["status"],
            "recall@1": (recall.get("1") or {}).get("recall"),
            "recall@4": (recall.get("4") or {}).get("recall"),
            "recall@8": (recall.get("8") or {}).get("recall"),
            "recall@all": (recall.get("all") or {}).get("recall"),
            "cover_samples": candidate.get("union_cover_samples"),
            "new_cover_samples_vs_best": candidate.get("new_cover_samples_vs_best"),
            "new_cover_image_count_vs_best": candidate.get("new_cover_image_count_vs_best"),
            "new_cover_samples_vs_best_primary": candidate.get("new_cover_samples_vs_best_primary"),
            "new_cover_image_count_vs_best_primary": candidate.get("new_cover_image_count_vs_best_primary"),
            "B": candidate.get("B"),
            "C": candidate.get("C"),
            "required_selection_rate": candidate.get("required_selection_rate"),
        }
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    text = [
        "# 复赛重算报告",
        "",
        f"完整 manifest 样本数：{report['manifest_samples']}。所有 IoU 与 ACC 均由原始 manifest bbox 重新计算。",
        f"图像组 bootstrap：{report['bootstrap']['replicates']} 次，seed={report['bootstrap']['seed']}。",
        "",
        "## 单模型",
        "",
        "| run | 状态 | 样本 | mean IoU | ACC@0.5 | ACC@0.7 | 解析率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run["metrics"]

        def fmt(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4f}"

        text.append(
            f"| {run['name']} | {run['status']} | {metrics['samples']} | "
            f"{fmt(metrics['mean_iou'])} | {fmt(metrics['acc_0.5'])} | "
            f"{fmt(metrics['acc_0.7'])} | {fmt(metrics['parse_rate'])} |"
        )
    text.extend(["", "## 成对比较", "", "仅完整 run 进入成对比较和门槛；partial run 只报告自身重算结果。", ""])
    for pair in report["pairs"]:
        bootstrap = pair["image_group_bootstrap"]
        text.extend(
            [
                f"### {pair['left']} vs {pair['right']}",
                "",
                f"wrong_to_right：{len(pair['wrong_to_right']['ids'])} IDs，涉及 {pair['wrong_to_right']['image_count']} 个图像组。",
                f"right_to_wrong：{len(pair['right_to_wrong']['ids'])} IDs，涉及 {pair['right_to_wrong']['image_count']} 个图像组。",
                f"ACC@0.5 delta={pair['delta']['acc_0.5']:.4f}，图像组 bootstrap 95% CI="
                f"[{bootstrap['acc_0.5_delta_95ci'][0]:.4f}, {bootstrap['acc_0.5_delta_95ci'][1]:.4f}]。",
                f"mean IoU delta={pair['delta']['mean_iou']:.4f}，95% CI="
                f"[{bootstrap['mean_iou_delta_95ci'][0]:.4f}, {bootstrap['mean_iou_delta_95ci'][1]:.4f}]。",
                "",
                f"wrong_to_right IDs：{', '.join(pair['wrong_to_right']['ids']) or '无'}",
                f"right_to_wrong IDs：{', '.join(pair['right_to_wrong']['ids']) or '无'}",
                "",
            ]
        )
    text.extend(["## 候选并集", "", "候选按固定 run 次序 round-robin 合流，候选间 IoU≥0.9 去重；pool 保留全部去重候选，pool_top8 是前 8 个。metadata 中 candidate_cap=8 是选择上限，Recall@all 使用全量候选；GT 只用于离线覆盖评分。", ""])
    if candidate["status"] == "excluded_no_complete_runs":
        text.append("没有完整 run，候选覆盖和门槛均未计算。")
    else:
        recall = candidate["recall"]
        text.extend(
            [
                "| 候选数 | 覆盖样本 | 覆盖率 |",
                "|---:|---:|---:|",
                *[
                    f"| {label} | {recall[label]['covered_samples']} | {recall[label]['recall']:.4f} |"
                    for label in ("1", "4", "8", "all")
                ],
                "",
                f"最好单模型：{candidate.get('best_single_model', 'NA')}；相对其完整候选流新增覆盖样本：{candidate.get('new_cover_samples_vs_best', 'NA')}，新增图像组：{candidate.get('new_cover_image_count_vs_best', 'NA')}。",
                f"相对其首选框新增覆盖样本：{candidate.get('new_cover_samples_vs_best_primary', 'NA')}，新增图像组：{candidate.get('new_cover_image_count_vs_best_primary', 'NA')}。",
                f"required_selection_rate=(B+8)/C，其中 B={candidate.get('B', 'NA')} 为最好单模型首选框命中数、C={candidate.get('C', 'NA')} 为前 8 个候选覆盖数：{candidate.get('required_selection_rate', 'NA')}。",
            ]
        )
    text.extend(["", "## Run metadata", "", "相邻的 summary/metadata JSON 只被读取并原样嵌入 summary.json；本报告不会修改源文件。"])
    for run in report["runs"]:
        if run["metadata"]:
            text.extend(["", f"`{run['name']}`：`{run['metadata']['path']}`"])
    (output_dir / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    specs = [parse_run_spec(spec) for spec in args.run]
    names = [name for name, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError("run names must be unique")
    runs = [
        load_run(name, path, set(manifest), allow_partial=args.allow_partial)
        for name, path in specs
    ]
    report = build_report(manifest, runs)
    write_outputs(report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "runs": names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
