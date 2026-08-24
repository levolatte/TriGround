from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def select_indices(records: list[dict], limit: int, seed: int) -> list[int]:
    """Match train.py's deterministic class/scale-stratified subset selection."""
    if limit >= len(records):
        return list(range(len(records)))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.get("class_name", ""), record.get("scale_bin", ""))].append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[i] for i in order]
    exact = {key: len(indices) * limit / len(records) for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key])),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    selected: list[int] = []
    for key in sorted(groups):
        selected.extend(groups[key][: quotas[key]])
    return selected


def box_iou(prediction: list[float], target: list[float]) -> float:
    px1, py1, px2, py2 = prediction
    tx1, ty1, tx2, ty2 = target
    intersection = max(0.0, min(px2, tx2) - max(px1, tx1)) * max(
        0.0, min(py2, ty2) - max(py1, ty1)
    )
    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    union = pred_area + target_area - intersection
    return intersection / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [{"id": sample_id, **record} for sample_id, record in manifest.items()]
    prediction_by_id = {}
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result = json.loads(line)
                prediction_by_id[result["id"]] = result

    selected = [items[index] for index in select_indices(items, args.limit, args.seed)]
    missing = [item["id"] for item in selected if item["id"] not in prediction_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} predictions; first missing id: {missing[0]}")

    ious = []
    coordinate_errors = []
    parsed = 0
    cap_hits = 0
    for item in selected:
        result = prediction_by_id[item["id"]]
        prediction = result.get("prediction")
        if prediction is None:
            ious.append(0.0)
            continue
        parsed += 1
        target = item["bbox"]
        ious.append(box_iou(prediction, target))
        coordinate_errors.extend(abs(float(a) - float(b)) for a, b in zip(prediction, target))
        cap_hits += int(bool(result.get("generation_cap_hit", False)))

    count = len(selected)
    report = {
        "eval_samples": count,
        "seed": args.seed,
        "mean_iou": sum(ious) / count,
        "acc_0.5": sum(iou >= 0.5 for iou in ious) / count,
        "acc_0.7": sum(iou >= 0.7 for iou in ious) / count,
        "l1_coordinate_error": sum(coordinate_errors) / len(coordinate_errors),
        "parse_rate": parsed / count,
        "generation_cap_hit_rate": cap_hits / count,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
