from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before = torch.load(args.before, map_location="cpu", weights_only=False)["model"]
    after = torch.load(args.after, map_location="cpu", weights_only=False)["model"]
    groups = {
        "restore": ".restore.",
        "modality_attention": ".modality_attention.",
        "reliability": "reliability",
        "query_encoder": "query_encoder",
        "residual_scale": "residual_scale",
    }
    report = {}
    for group, pattern in groups.items():
        deltas = [
            (after[name] - value).float()
            for name, value in before.items()
            if pattern in name and name in after
        ]
        report[group] = {
            "parameter_tensors": len(deltas),
            "delta_l2": sum(float(delta.square().sum()) for delta in deltas) ** 0.5,
            "delta_max_abs": max(
                (float(delta.abs().max()) for delta in deltas), default=0.0
            ),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
