"""Offline, full-denominator visual-agent scoring; never used by online inference."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SEED = 2026
SHARED_SEQUENCES = (
    "000002_025", "000002_041", "000002_043", "000002_074",
    "000013_016", "000014_025", "000014_052",
)
SHARED_RE = re.compile(r"(?<!\d)(?:" + "|".join(SHARED_SEQUENCES) + r")(?:_suppl)?_")
COST_SUM_FIELDS = ("model_calls", "input_tokens", "input_text_tokens", "visual_tokens", "output_tokens",
                   "tool_calls", "search_calls", "model_seconds", "tool_seconds",
                   "preprocess_seconds", "preparation_seconds", "elapsed_seconds", "depth_cache_hits", "depth_generated",
                   "dino_calls", "sam_loads", "dino_loads", "depth_cached_seconds",
                   "depth_generated_seconds", "loads_seconds")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def box(value, name: str, *, nullable: bool = True):
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name}: expected xyxy list")
    result = [float(x) for x in value]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in result):
        raise ValueError(f"{name}: non-finite or outside [0,1]")
    if not (result[0] < result[2] and result[1] < result[3]):
        raise ValueError(f"{name}: empty box")
    return result


def iou(a, b) -> float:
    if a is None:
        return 0.0
    x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def load_gt(path: Path) -> dict[str, dict]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict) or not result:
        raise ValueError("GT must be a nonempty object keyed by ID")
    for sample_id, row in result.items():
        box(row["bbox"], f"GT {sample_id}", nullable=False)
    return result


def load_run(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        sample_id = str(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path}: duplicate ID {sample_id}")
        result_box = row.get("bbox", row.get("prediction"))
        result_box = box(result_box, f"{path}:{sample_id}.bbox")
        if "final_status" in row:
            selected = row.get("selected_id")
            initial = box(row.get("initial_bbox"), f"{sample_id}.initial_bbox", nullable=False)
            if (row["final_status"] == "FINISHED") != (result_box is not None):
                raise ValueError(f"{sample_id}: final_status and bbox legality disagree")
            if result_box is not None:
                if selected is None:
                    raise ValueError(f"{sample_id}: nonnull bbox without selected_id")
                if str(selected) == "KEEP":
                    if result_box != initial:
                        raise ValueError(f"{sample_id}: KEEP changed initial bbox")
                else:
                    candidates = row.get("final_candidates", [])
                    matches = [c for c in candidates if str(c["id"]) == str(selected)]
                    if len(matches) != 1 or not finish_eligible(matches[0]):
                        raise ValueError(f"{sample_id}: selected ID is absent or has no eligible RGB box")
                    if result_box != box(matches[0]["bbox"], f"{sample_id}.selected_bbox", nullable=False):
                        raise ValueError(f"{sample_id}: final bbox differs from selected candidate")
            elif selected is not None:
                raise ValueError(f"{sample_id}: selected_id with null bbox")
        rows[sample_id] = row
    missing, extra = expected_ids - rows.keys(), rows.keys() - expected_ids
    if missing or extra:
        raise ValueError(f"{path}: incomplete denominator: missing={len(missing)}, extra={len(extra)}")
    return rows


def prediction_box(row: dict):
    return row.get("bbox", row.get("prediction"))


def group_key(record: dict) -> str:
    return str(record["visible"]).replace("\\", "/").split("/")[-1]


def finish_eligible(candidate: dict) -> bool:
    # Historical runs used a hard target-role restriction. New runs explicitly
    # expose which boxes can be returned in RGB; do not reinterpret old records.
    return candidate.get("finish_eligible", candidate.get("role") == "target")


def coverage(candidates: list[dict] | None, initial_bbox, gt_bbox) -> bool:
    if initial_bbox is not None and iou(initial_bbox, gt_bbox) >= .5:
        return True
    return any(finish_eligible(c) and iou(box(c["bbox"], "candidate", nullable=False), gt_bbox) >= .5
               for c in (candidates or []) if str(c.get("id")) != "KEEP")


def score_subset(rows: dict[str, dict], gt: dict[str, dict], ids: list[str]) -> dict:
    values, baseline_values = [], []
    initial_covered = final_covered = newly_covered = covered_wrong = valid = 0
    keep_count = keep_hits = 0
    statuses, observations = Counter(), Counter()
    cost = Counter()
    peak_memory = 0
    for sample_id in ids:
        row = rows[sample_id]
        target = gt[sample_id]["bbox"]
        pred = prediction_box(row)
        values.append(iou(pred, target))
        valid += pred is not None
        if str(row.get("selected_id")) == "KEEP":
            keep_count += 1
            keep_hits += values[-1] >= .5
        if "initial_bbox" in row:
            initial = row["initial_bbox"]
            baseline_values.append(iou(initial, target))
            has_initial = coverage(row.get("initial_candidates"), initial, target)
            initial_covered += has_initial
            has_final = coverage(row.get("final_candidates"), initial, target)
            final_covered += has_final
            newly_covered += has_final and not has_initial
            covered_wrong += has_final and values[-1] < .5
        statuses[str(row.get("final_status", "baseline"))] += 1
        metrics = row.get("metrics", {})
        for key in COST_SUM_FIELDS:
            cost[key] += metrics.get(key, 0) or 0
        peak_memory = max(peak_memory, metrics.get("peak_memory_bytes", 0) or 0)
        observations.update(metrics.get("tool_status_counts", {}))
        cost["invalid_actions"] += metrics.get("invalid_actions", 0) or 0
        cost["legal_finish"] += bool(metrics.get("legal_finish", pred is not None))
    n = len(ids)
    result = {
        "n": n, "hits_0.5": sum(x >= .5 for x in values),
        "acc_0.5": sum(x >= .5 for x in values) / n,
        "hits_0.7": sum(x >= .7 for x in values),
        "acc_0.7": sum(x >= .7 for x in values) / n,
        "mean_iou": sum(values) / n, "valid_final_bbox": valid,
        "invalid_final_bbox": n - valid, "keep_count": keep_count,
        "keep_hits_0.5": keep_hits,
        "final_status_counts": dict(statuses), "tool_status_counts": dict(observations),
        "cost_observed_queries": sum("elapsed_seconds" in rows[i].get("metrics", {}) for i in ids),
        "cost_total": dict(cost), "cost_mean_per_query": {k: v / n for k, v in cost.items()},
        "peak_memory_bytes": peak_memory,
    }
    if baseline_values:
        result.update({
            "initial_hits_0.5": sum(x >= .5 for x in baseline_values),
            "initial_coverage_hits_0.5": initial_covered,
            "initial_uncovered": n - initial_covered,
            "final_coverage_hits_0.5": final_covered,
            "final_uncovered": n - final_covered,
            "newly_covered_by_search": newly_covered,
            "covered_but_final_wrong": covered_wrong,
        })
    return result


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    p = q * (len(values) - 1)
    lo = math.floor(p)
    hi = math.ceil(p)
    return values[lo] + (values[hi] - values[lo]) * (p - lo)


def compare(left: dict[str, dict], right: dict[str, dict], gt: dict[str, dict],
            ids: list[str], replicates: int = 10_000) -> dict:
    by_group = defaultdict(list)
    corrected, damaged = [], []
    for sample_id in ids:
        target = gt[sample_id]["bbox"]
        a = iou(prediction_box(left[sample_id]), target)
        b = iou(prediction_box(right[sample_id]), target)
        by_group[group_key(gt[sample_id])].append((a, b))
        if a < .5 <= b:
            corrected.append(sample_id)
        if b < .5 <= a:
            damaged.append(sample_id)
    groups = list(by_group.values())
    rng = random.Random(SEED)
    draws = {"acc_0.5": [], "acc_0.7": [], "mean_iou": []}
    for _ in range(replicates):
        sampled = [pair for _ in groups for pair in rng.choice(groups)]
        denom = len(sampled)
        draws["acc_0.5"].append(sum((b >= .5) - (a >= .5) for a, b in sampled) / denom)
        draws["acc_0.7"].append(sum((b >= .7) - (a >= .7) for a, b in sampled) / denom)
        draws["mean_iou"].append(sum(b - a for a, b in sampled) / denom)
    n = len(ids)
    all_pairs = [p for g in groups for p in g]
    def cases(changed):
        items = []
        for sample_id in changed:
            target = gt[sample_id]["bbox"]
            a = iou(prediction_box(left[sample_id]), target)
            b = iou(prediction_box(right[sample_id]), target)
            items.append({"id": sample_id, "query": gt[sample_id].get("query"),
                          "before_iou": a, "after_iou": b,
                          "before_bbox": prediction_box(left[sample_id]),
                          "after_bbox": prediction_box(right[sample_id]),
                          "gt_bbox": target})
        return sorted(items, key=lambda x: -abs(x["after_iou"] - x["before_iou"]))[:5]
    delta = {"acc_0.5": sum((b >= .5) - (a >= .5) for a, b in all_pairs) / n,
             "acc_0.7": sum((b >= .7) - (a >= .7) for a, b in all_pairs) / n,
             "mean_iou": sum(b - a for a, b in all_pairs) / n}
    return {
        "n": n, "image_groups": len(groups), "corrected_ids": corrected,
        "damaged_ids": damaged, "corrected": len(corrected), "damaged": len(damaged),
        "net_hits_0.5": len(corrected) - len(damaged), "delta": delta,
        "representative_corrections": cases(corrected),
        "representative_damages": cases(damaged),
        "group_bootstrap": {"seed": SEED, "replicates": replicates,
                            "delta_95ci": {k: [percentile(v, .025), percentile(v, .975)]
                                           for k, v in draws.items()}},
    }


def evaluate(args) -> dict:
    gt = load_gt(args.gt)
    if args.manifest:
        manifest_rows = read_jsonl(args.manifest)
        manifest_ids = [str(row["id"]) for row in manifest_rows]
        if len(set(manifest_ids)) != len(manifest_ids) or not set(manifest_ids) <= set(gt):
            raise ValueError("scoring manifest has duplicate IDs or IDs absent from GT")
        gt = {sample_id: gt[sample_id] for sample_id in manifest_ids}
    ids = list(gt)
    runs = {}
    run_sources = {}
    specs = list(args.run)
    if args.baseline:
        specs.insert(0, f"C={args.baseline}")
    for spec in specs:
        name, sep, filename = spec.partition("=")
        if not sep or not name or name in runs:
            raise ValueError(f"invalid or duplicate --run: {spec}")
        runs[name] = load_run(Path(filename), set(gt))
        run_sources[name] = filename
    if not runs:
        raise ValueError("provide --run or --baseline")
    shared_ids = [i for i in ids if SHARED_RE.search(str(gt[i]["visible"]))]
    shared = set(shared_ids)
    splits = {"all": ids, "known_reused": shared_ids,
              "other_not_independent_test": [i for i in ids if i not in shared]}
    splits = {name: subset for name, subset in splits.items() if subset}
    scores = {name: {split: score_subset(rows, gt, subset) for split, subset in splits.items()}
              for name, rows in runs.items()}
    pairs = {}
    names = list(runs)
    pair_specs = args.pair or [f"{names[0]}:{name}" for name in names[1:]]
    for spec in pair_specs:
        left_name, sep, right_name = spec.partition(":")
        if not sep or left_name not in runs or right_name not in runs or left_name == right_name:
            raise ValueError(f"invalid --pair {spec}; use LEFT:RIGHT names supplied by --run")
        pairs[f"{right_name}-minus-{left_name}"] = {
            split: compare(runs[left_name], runs[right_name], gt, subset, args.bootstrap)
            for split, subset in splits.items()}
    result = {
        "gt": str(args.gt), "manifest": str(args.manifest) if args.manifest else None,
        "denominator": len(ids),
        "image_groups": len({group_key(r) for r in gt.values()}),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "runs": scores, "run_sources": run_sources, "pairs": pairs,
        "interpretation": ["All missing or illegal final boxes have IoU zero in the fixed denominator.",
                           "Known reused and other subsets are both development data; other is not certified independent.",
                           "Only matched weights, inputs and budgets isolate the corresponding training or tool change.",
                           "Cost differences remain possible across paired mechanisms."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="GT-free cohort JSONL; limits scoring to its complete frozen ID set")
    parser.add_argument("--run", action="append", default=[], help="NAME=predictions.jsonl; repeatable")
    parser.add_argument("--pair", action="append", default=[],
                        help="LEFT:RIGHT comparison; repeat for matched S/C/D and same controller/profile")
    parser.add_argument("--baseline", type=Path, help="Direct C predictions (old id/bbox or id/prediction format)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    result = evaluate(parser.parse_args())
    print(json.dumps({"denominator": result["denominator"], "split_counts": result["split_counts"],
                      "scores": {k: {s: v["hits_0.5"] for s, v in split.items()}
                                 for k, split in result["runs"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
