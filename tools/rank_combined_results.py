from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "mean_iou",
    "acc_0.5",
    "acc_0.7",
    "l1_coordinate_error",
    "parse_rate",
)
SELECTION_ORDER = ("acc_0.5", "mean_iou", "acc_0.7", "parse_rate")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", nargs=4, action="append", metavar=("NAME", "CHECKPOINT", "OLD", "NEW"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path)
    args = parser.parse_args()
    if args.selected_output is not None:
        parser.error(
            "combined284 is diagnostic-only; --selected-output is disabled"
        )
    rows = []
    for name, checkpoint, old_path, new_path in args.candidate:
        old = load(old_path)
        new = load(new_path)
        old_n = int(old["diagnostic_metadata"]["samples"])
        new_n = int(new["diagnostic_metadata"]["samples"])
        total = old_n + new_n
        combined = {
            metric: (
                old["rgb_ir_depth"][metric] * old_n
                + new["rgb_ir_depth"][metric] * new_n
            ) / total
            for metric in METRICS
        }
        rgb = {
            metric: (
                old["rgb_baseline"][metric] * old_n
                + new["rgb_baseline"][metric] * new_n
            ) / total
            for metric in METRICS
        }
        rows.append({
            "name": name,
            "checkpoint": checkpoint,
            "old_result": old_path,
            "new_result": new_path,
            "samples": total,
            "rgb_ir_depth": combined,
            "rgb_baseline": rgb,
            "mean_iou_gain_over_rgb": combined["mean_iou"] - rgb["mean_iou"],
        })
    rows.sort(
        key=lambda row: tuple(
            row["rgb_ir_depth"][metric] for metric in SELECTION_ORDER
        ),
        reverse=True,
    )
    report = {
        "purpose": "diagnostic_only",
        "ranking_order": [f"combined284.rgb_ir_depth.{name}" for name in SELECTION_ORDER],
        "top_diagnostic": rows[0],
        "ranking": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
