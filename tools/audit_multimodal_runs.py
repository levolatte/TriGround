from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    for run in sorted(Path("runs").glob("multimodal*")):
        if not run.is_dir():
            continue
        records = []
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "mean_iou" in record:
                    records.append(record)
        best = max(records, key=lambda row: row["mean_iou"], default=None)
        checkpoints = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
            }
            for path in sorted(run.glob("*.pt"))
        ]
        print(json.dumps({
            "run": str(run),
            "records": len(records),
            "best_metric": best,
            "last_metric": records[-1] if records else None,
            "checkpoints": checkpoints,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
