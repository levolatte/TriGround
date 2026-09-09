from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_grounding.data import GroundingDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Open and validate grounding manifests")
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--stage", choices=("ir", "depth", "joint"), default="joint")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-clip", type=float, default=20.0)
    args = parser.parse_args()
    report = {}
    for manifest in args.manifests:
        dataset = GroundingDataset(
            manifest,
            stage=args.stage,
            depth_scale=args.depth_scale,
            depth_clip=args.depth_clip,
        )
        for index in range(len(dataset)):
            dataset[index]
        report[str(manifest)] = {"samples": len(dataset), "validated": len(dataset)}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
