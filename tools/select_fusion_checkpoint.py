from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SELECTION_ORDER = ("acc_0.5", "mean_iou", "acc_0.7", "parse_rate")


def best_epoch_key(path: Path) -> tuple[float, ...]:
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event", "").startswith("early_probe"):
            continue
        if record.get("eval_scope") in {"subset", "full"}:
            keys.append(tuple(float(record[name]) for name in SELECTION_ORDER))
    if not keys:
        raise ValueError(f"no epoch evaluation score in {path}")
    return max(keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_key = best_epoch_key(args.reference_metrics)
    candidate_key = best_epoch_key(args.candidate_metrics)
    if candidate_key > reference_key:
        selected = args.candidate_checkpoint
        selected_key = candidate_key
        source = "extended"
    else:
        selected = args.reference_checkpoint
        selected_key = reference_key
        source = "pre_extension"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, args.output)
    report = {
        "selection_order": list(SELECTION_ORDER),
        "reference_metrics": dict(zip(SELECTION_ORDER, reference_key)),
        "candidate_metrics": dict(zip(SELECTION_ORDER, candidate_key)),
        "selected_metrics": dict(zip(SELECTION_ORDER, selected_key)),
        "selected_source": source,
        "selected_checkpoint": str(selected),
        "output": str(args.output),
    }
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
