from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RUN_DIR", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = [
        record
        for record in records
        if record.get("eval_scope") == "subset" and record.get("phase") == "A"
    ]
    epoch3 = [record for record in subset if record.get("epoch") == 3]
    if not epoch3:
        return 1
    expected_best_epoch = max(subset, key=lambda record: record["mean_iou"])["epoch"]
    try:
        last = torch.load(run_dir / "last_phase_a.pt", map_location="cpu", weights_only=False)
        best = torch.load(run_dir / "best_phase_a.pt", map_location="cpu", weights_only=False)
    except (FileNotFoundError, EOFError, RuntimeError):
        return 1
    if last.get("epoch") != 3 or best.get("epoch") != expected_best_epoch:
        return 1
    print(json.dumps({
        "epoch3_mean_iou": epoch3[-1]["mean_iou"],
        "best_epoch": best.get("epoch"),
        "best_score": best.get("score"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
