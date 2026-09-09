from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def load_state(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint, checkpoint["model"]


def parameter_group(name: str) -> str:
    if "fusion.ir_adapters." in name:
        return "ir_adapters"
    if "fusion.depth_adapters." in name:
        return "depth_adapters"
    if "fusion.ir_query_encoder." in name:
        return "ir_query_encoder"
    if "fusion.depth_query_encoder." in name:
        return "depth_query_encoder"
    if "fusion.stage_fusions." in name and ".ir." in name:
        return "ir_stage_fusion"
    if "fusion.stage_fusions." in name and ".depth." in name:
        return "depth_stage_fusion"
    return "other"


def state_summary(
    reference: dict[str, torch.Tensor], state: dict[str, torch.Tensor]
) -> dict:
    totals = defaultdict(lambda: {"tensors": 0, "parameters": 0, "norm_sq": 0.0,
                                 "delta_sq": 0.0, "reference_sq": 0.0})
    scalars = {}
    for name, value in state.items():
        if not torch.is_tensor(value):
            continue
        group = parameter_group(name)
        work = value.float()
        row = totals[group]
        row["tensors"] += 1
        row["parameters"] += value.numel()
        row["norm_sq"] += float(work.square().sum())
        if name in reference and reference[name].shape == value.shape:
            baseline = reference[name].float()
            row["delta_sq"] += float((work - baseline).square().sum())
            row["reference_sq"] += float(baseline.square().sum())
        if name.endswith("residual_scale") or name.endswith("gate.2.bias"):
            scalars[name] = {
                "mean": float(work.mean()),
                "min": float(work.min()),
                "max": float(work.max()),
            }
    groups = {}
    for group, row in sorted(totals.items()):
        delta_norm = math.sqrt(row["delta_sq"])
        reference_norm = math.sqrt(row["reference_sq"])
        groups[group] = {
            "tensors": row["tensors"],
            "parameters": row["parameters"],
            "norm": math.sqrt(row["norm_sq"]),
            "delta_norm_from_stage1": delta_norm,
            "relative_delta_from_stage1": delta_norm / max(reference_norm, 1e-12),
        }
    return {"groups": groups, "scalars": scalars}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1a", type=Path, required=True)
    parser.add_argument("--stage1b", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    args = parser.parse_args()
    _, ir = load_state(args.stage1a)
    _, depth = load_state(args.stage1b)
    reference = dict(ir)
    reference.update(depth)
    reports = []
    for path in args.checkpoint:
        checkpoint, state = load_state(path)
        reports.append({
            "checkpoint": str(path),
            "epoch": checkpoint.get("epoch"),
            "score": checkpoint.get("score"),
            **state_summary(reference, state),
        })
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
