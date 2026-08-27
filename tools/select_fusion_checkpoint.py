from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def best_epoch_score(path: Path) -> float:
    scores = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event", "").startswith("early_probe"):
            continue
        if record.get("eval_scope") in {"subset", "full"}:
            scores.append(float(record["mean_iou"]))
    if not scores:
        raise ValueError(f"no epoch evaluation score in {path}")
    return max(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_score = best_epoch_score(args.reference_metrics)
    candidate_score = best_epoch_score(args.candidate_metrics)
    if candidate_score > reference_score:
        selected = args.candidate_checkpoint
        selected_score = candidate_score
        source = "extended"
    else:
        selected = args.reference_checkpoint
        selected_score = reference_score
        source = "pre_extension"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected, args.output)
    report = {
        "reference_score": reference_score,
        "candidate_score": candidate_score,
        "selected_score": selected_score,
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
