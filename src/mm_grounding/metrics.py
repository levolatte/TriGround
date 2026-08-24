from __future__ import annotations

import torch

from .boxes import box_iou_aligned


@torch.no_grad()
def grounding_metrics(predicted_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> dict[str, float]:
    iou = box_iou_aligned(predicted_xyxy, target_xyxy)
    return {
        "mean_iou": float(iou.mean().item()),
        "acc_0.5": float((iou >= 0.5).float().mean().item()),
        "acc_0.7": float((iou >= 0.7).float().mean().item()),
        "l1_coordinate_error": float((predicted_xyxy - target_xyxy).abs().mean().item()),
    }


def merge_metric_sums(rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    count = sum(size for size, _ in rows)
    if count == 0:
        return {name: 0.0 for name in ("mean_iou", "acc_0.5", "acc_0.7", "l1_coordinate_error")}
    names = rows[0][1]
    return {name: sum(size * values[name] for size, values in rows) / count for name in names}

