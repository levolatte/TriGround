from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = ("target_size_bin", "scale_bin", "target_class", "query_type")


def load_rows(path: Path, mode: str) -> dict[str, dict]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != mode:
            continue
        sample_id = str(row["id"])
        if sample_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
        rows[sample_id] = row
    if not rows:
        raise ValueError(f"{path} has no rows for mode {mode!r}")
    return rows


def metrics(rows: list[dict]) -> dict[str, float]:
    count = len(rows)
    return {
        "samples": count,
        "acc_0.5": sum(float(row["iou"]) >= 0.5 for row in rows) / count,
        "mean_iou": sum(float(row["iou"]) for row in rows) / count,
        "acc_0.7": sum(float(row["iou"]) >= 0.7 for row in rows) / count,
        "parse_rate": sum(bool(row["parsed"]) for row in rows) / count,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def scene_bootstrap(
    pairs: list[tuple[dict, dict]], replicates: int, seed: int
) -> dict[str, list[float]]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[str(pair[0].get("scene_group", pair[0]["id"]))].append(pair)
    groups = list(grouped.values())
    rng = random.Random(seed)
    deltas = defaultdict(list)
    for _ in range(replicates):
        sampled = [pair for _ in groups for pair in rng.choice(groups)]
        control_metrics = metrics([pair[0] for pair in sampled])
        treatment_metrics = metrics([pair[1] for pair in sampled])
        for name in ("acc_0.5", "mean_iou", "acc_0.7", "parse_rate"):
            deltas[name].append(treatment_metrics[name] - control_metrics[name])
    return {
        f"{name}_delta_95ci": [
            percentile(values, 0.025),
            percentile(values, 0.975),
        ]
        for name, values in deltas.items()
    }


def grouped_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    report = {}
    for field in GROUP_FIELDS:
        if not any(field in control for control, _ in pairs):
            continue
        values = sorted(
            {str(control[field]) for control, _ in pairs if field in control}
        )
        report[field] = {}
        for value in values:
            selected = [
                pair for pair in pairs if str(pair[0].get(field)) == value
            ]
            control_metrics = metrics([pair[0] for pair in selected])
            treatment_metrics = metrics([pair[1] for pair in selected])
            report[field][value] = {
                "control": control_metrics,
                "treatment": treatment_metrics,
                "acc_0.5_delta": (
                    treatment_metrics["acc_0.5"] - control_metrics["acc_0.5"]
                ),
                "mean_iou_delta": (
                    treatment_metrics["mean_iou"] - control_metrics["mean_iou"]
                ),
            }
    return report


def compare(
    control: dict[str, dict],
    treatment: dict[str, dict],
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    if set(control) != set(treatment):
        raise ValueError("control and treatment sample IDs differ")
    pairs = []
    per_sample = []
    transitions = {
        "incorrect_to_correct": 0,
        "correct_to_incorrect": 0,
        "correct_in_both": 0,
        "incorrect_in_both": 0,
    }
    for sample_id in sorted(control):
        control_row = control[sample_id]
        treatment_row = treatment[sample_id]
        if control_row["target"] != treatment_row["target"]:
            raise ValueError(f"target differs for {sample_id!r}")
        for field in ("scene_group", *GROUP_FIELDS):
            if control_row.get(field) != treatment_row.get(field):
                raise ValueError(f"{field} differs for {sample_id!r}")
        control_correct = float(control_row["iou"]) >= 0.5
        treatment_correct = float(treatment_row["iou"]) >= 0.5
        transition = {
            (False, True): "incorrect_to_correct",
            (True, False): "correct_to_incorrect",
            (True, True): "correct_in_both",
            (False, False): "incorrect_in_both",
        }[(control_correct, treatment_correct)]
        transitions[transition] += 1
        pairs.append((control_row, treatment_row))
        per_sample.append(
            {
                "id": sample_id,
                "scene_group": control_row.get("scene_group", sample_id),
                "control_iou": float(control_row["iou"]),
                "treatment_iou": float(treatment_row["iou"]),
                "iou_delta": float(treatment_row["iou"])
                - float(control_row["iou"]),
                "transition": transition,
            }
        )
    control_metrics = metrics([pair[0] for pair in pairs])
    treatment_metrics = metrics([pair[1] for pair in pairs])
    return {
        "control": control_metrics,
        "treatment": treatment_metrics,
        "delta": {
            name: treatment_metrics[name] - control_metrics[name]
            for name in ("acc_0.5", "mean_iou", "acc_0.7", "parse_rate")
        },
        "transitions": transitions,
        "scene_bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            **scene_bootstrap(pairs, bootstrap_replicates, seed),
        },
        "groups": grouped_metrics(pairs),
        "per_sample": per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired grounding runs")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="rgb_ir_depth")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    report = {
        "mode": args.mode,
        "control_rows": str(args.control),
        "treatment_rows": str(args.treatment),
        **compare(
            load_rows(args.control, args.mode),
            load_rows(args.treatment, args.mode),
            args.bootstrap_replicates,
            args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("mode", "control", "treatment", "delta", "transitions", "scene_bootstrap")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
