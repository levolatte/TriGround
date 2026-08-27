from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = ("mean_iou", "acc_0.5", "acc_0.7", "parse_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two multimodal evaluation reports")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fine-tune-samples", type=int, required=True)
    parser.add_argument("--selection-samples", type=int, required=True)
    args = parser.parse_args()

    before_report = json.loads(args.before.read_text(encoding="utf-8"))
    after_report = json.loads(args.after.read_text(encoding="utf-8"))
    before = {key: before_report["rgb_ir_depth"][key] for key in METRIC_KEYS}
    after = {key: after_report["rgb_ir_depth"][key] for key in METRIC_KEYS}
    summary = {
        "model": args.model,
        "test_manifest": before_report["diagnostic_metadata"]["manifest"],
        "fine_tune_samples": args.fine_tune_samples,
        "selection_samples": args.selection_samples,
        "before": before,
        "after": after,
        "delta": {key: after[key] - before[key] for key in METRIC_KEYS},
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

