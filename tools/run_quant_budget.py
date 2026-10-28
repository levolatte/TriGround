"""Execute one experiment command and charge its wall time to the 72 GPU-hour budget.

One runner at a time. Model loading, evaluation and failures are charged. Cloud
idle/download time outside this runner must be added with --record-hours.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


STAGE_HOURS = {"prepare": 6.0, "initial": 32.0, "optimization": 2.0,
               "exposure_eval": 2.0, "replicate": 16.0, "extend": 8.0, "final": 6.0}
LABEL_HOURS = {"initial": 8.0, "replicate": 8.0, "extend": 4.0}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_ledger(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remaining_hours(rows: list[dict], stage: str, label: str) -> float:
    if any(row["status"] == "running" for row in rows):
        raise ValueError("Ledger contains a running command. Finish it or reconcile its actual cloud bill before continuing.")
    used = sum(row["gpu_hours"] for row in rows)
    stage_used = sum(row["gpu_hours"] for row in rows if row["stage"] == stage)
    limits = [72.0 - used, STAGE_HOURS[stage] - stage_used]
    if stage in LABEL_HOURS:
        label_used = sum(row["gpu_hours"] for row in rows if row["stage"] == stage and row["label"] == label)
        limits.append(LABEL_HOURS[stage] - label_used)
    return max(0.0, min(limits))


def stop_child(process: subprocess.Popen) -> None:
    """Stop only this runner's child process tree at the budget boundary."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_command(command: list[str], path: Path, *, stage: str, label: str,
                gpu_model: str, gpu_count: int = 1, price_per_gpu_hour: float | None = None,
                max_hours: float | None = None) -> int:
    rows = read_ledger(path)
    remaining = remaining_hours(rows, stage, label)
    allowed = min(remaining, max_hours) if max_hours is not None else remaining
    if allowed <= 0:
        raise ValueError(f"No GPU hours remain for {stage}/{label}")
    if gpu_count < 1 or (price_per_gpu_hour is not None and
                         (price_per_gpu_hour < 0 or not math.isfinite(price_per_gpu_hour))):
        raise ValueError("GPU count must be positive; price must be finite and nonnegative")
    row = {"stage": stage, "label": label, "gpu_model": gpu_model, "gpu_count": gpu_count,
           "price_per_gpu_hour": price_per_gpu_hour, "command": command, "started_at": utc_now(),
           "status": "running", "gpu_hours": 0.0}
    rows.append(row)
    write_ledger(path, rows)
    started = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(command, start_new_session=os.name != "nt")
        try:
            code = process.wait(timeout=allowed * 3600 / gpu_count)
            row["status"] = "completed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            stop_child(process)
            row["status"] = "budget_exhausted"
            code = 124
        except KeyboardInterrupt:
            stop_child(process)
            row["status"] = "interrupted"
            code = 130
        row["exit_code"] = code
        return code
    finally:
        elapsed = time.monotonic() - started
        row.update(ended_at=utc_now(), wall_seconds=elapsed, gpu_hours=elapsed * gpu_count / 3600,
                   cost=elapsed * gpu_count * price_per_gpu_hour / 3600 if price_per_gpu_hour is not None else None)
        if process is None:
            row["status"] = "launch_failed"
        write_ledger(path, rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_HOURS, required=True)
    parser.add_argument("--label", required=True, help="Reuse the same arm label for training and its evaluations")
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--price-per-gpu-hour", type=float)
    parser.add_argument("--max-hours", type=float, help="GPU hours for this command, further bounded by stage/arm/total")
    parser.add_argument("--record-hours", type=float, help="Add already incurred GPU-hours outside the runner, including idle time")
    parser.add_argument("--note", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.record_hours is not None:
        if args.command or not math.isfinite(args.record_hours) or args.record_hours <= 0:
            parser.error("--record-hours needs positive hours and no command")
        rows = read_ledger(args.ledger)
        if any(row["status"] == "running" for row in rows):
            parser.error("Finish/reconcile the running record first")
        rows.append({"stage": args.stage, "label": args.label, "gpu_model": args.gpu_model,
                     "gpu_count": args.gpu_count, "price_per_gpu_hour": args.price_per_gpu_hour,
                     "status": "external_charge", "gpu_hours": args.record_hours,
                     "cost": args.record_hours * args.price_per_gpu_hour if args.price_per_gpu_hour is not None else None,
                     "recorded_at": utc_now(), "note": args.note})
        write_ledger(args.ledger, rows)
        print(json.dumps({"remaining_gpu_hours": remaining_hours(rows, args.stage, args.label)}))
        return
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply -- COMMAND, or --record-hours")
    if args.max_hours is not None and (not math.isfinite(args.max_hours) or args.max_hours <= 0):
        parser.error("--max-hours must be positive and finite")
    raise SystemExit(run_command(command, args.ledger, stage=args.stage, label=args.label,
                                 gpu_model=args.gpu_model, gpu_count=args.gpu_count,
                                 price_per_gpu_hour=args.price_per_gpu_hour, max_hours=args.max_hours))


if __name__ == "__main__":
    main()
