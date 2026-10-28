"""Complete warmup50 and C29 development evaluation with the existing evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path("/root/autodl-tmp/quant_rl_20261026")
DATA_ROOT = Path("/root/autodl-tmp/rematch_20260922/data")
MODEL = "/root/autodl-tmp/rematch_models/Qwen3.6-27B"
DATASETS = {"city378": ("city400_dev.jsonl", 378), "external88": ("external94_dev.jsonl", 88)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_wait(ledger):
    rows = read(ledger)
    if any(row["status"] == "running" for row in rows):
        raise RuntimeError("An experiment is already running")
    last = max(datetime.fromisoformat(row.get("ended_at", row.get("recorded_at"))) for row in rows)
    hours = max(0.0, (datetime.now(timezone.utc) - last).total_seconds() / 3600)
    if hours:
        rows.append({"stage": "occupancy", "label": "retained_instance_wait",
                     "gpu_model": "RTX4090-48GB", "gpu_count": 1, "status": "external_charge",
                     "gpu_hours": hours, "recorded_at": now(), "price_per_gpu_hour": None,
                     "cost": None, "note": "Retained-instance elapsed time; no experiment-hour cutoff"})
        write(ledger, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root, runs = args.root, args.root / "runs"
    python = root / "train_env/bin/python"
    evaluator = root / "code/tools/eval_quant_grounding.py"
    dataset_dir = args.data_root / "quant_grounding_20261026_formal"
    jobs = [
        ("warmup50", "final", runs / "shared_format_warmup/training/checkpoint-50",
         runs / "warmup50_development_baseline", 50),
        ("c_matched", "exposure_eval", runs / "optimized_C_seed2026/train/matched-exposure-checkpoint-29",
         runs / "optimized_C_seed2026/matched_exposure", 29),
    ]
    ledger = runs / "budget.json"
    if any(row["status"] == "running" for row in read(ledger)):
        raise RuntimeError("An experiment is already running")
    gpu_jobs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if gpu_jobs:
        raise RuntimeError(f"GPU already in use: {gpu_jobs}")
    for filename, count in DATASETS.values():
        rows = [json.loads(line) for line in (dataset_dir / filename).read_text(encoding="utf-8").splitlines() if line]
        if len(rows) != count or len({row["id"] for row in rows}) != count:
            raise RuntimeError(f"Development coverage differs for {filename}")
    for name, stage, adapter, output, step in jobs:
        if read(adapter / "trainer_state.json")["global_step"] != step:
            raise RuntimeError(f"Wrong checkpoint: {adapter}")
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter / "adapter_model.safetensors")
        if output.exists():
            raise FileExistsError(output)
    if args.check_only:
        print(json.dumps({"ready": True, "jobs": [name for name, *_ in jobs],
                          "new_answers": 932, "hour_cutoff": None}, ensure_ascii=True))
        return

    state_path = runs / "fair_evaluation_state.json"
    state = {"status": "running", "started_at": now(), "completed": [],
             "hour_cutoff": None, "instance_power_action": "none"}
    write(state_path, state)
    try:
        for name, stage, adapter, output, step in jobs:
            output.mkdir(parents=True)
            summaries = {}
            for split, (filename, count) in DATASETS.items():
                record_wait(ledger)
                target = output / split
                command = [str(python), str(evaluator), "--dataset", str(dataset_dir / filename),
                           "--data-root", str(args.data_root), "--model", args.model, "--adapter", str(adapter),
                           "--engine", "hf", "--precision", "nf4", "--answer-mode", "short",
                           "--reward-mode", "contrast", "--format-policy", "strict",
                           "--samples-per-query", "1", "--hf-max-batch-size", "2", "--seed", "2026",
                           "--max-tokens", "64", "--max-length", "8192",
                           "--min-pixels", "200704", "--max-pixels", "1204224", "--output-dir", str(target)]
                state.update(model=name, split=split, stage_started_at=now(), output=str(target))
                write(state_path, state)
                rows = read(ledger)
                entry = {"stage": stage, "label": name + "_development", "gpu_model": "RTX4090-48GB",
                         "gpu_count": 1, "status": "running", "gpu_hours": 0.0, "started_at": now(),
                         "command": command, "price_per_gpu_hour": None, "cost": None}
                rows.append(entry)
                write(ledger, rows)
                started = time.monotonic()
                try:
                    with (output / (split + ".log")).open("x", encoding="utf-8") as log:
                        process = subprocess.Popen(command, cwd=root / "code", stdout=log, stderr=subprocess.STDOUT)
                        state["child_pid"] = process.pid
                        write(state_path, state)
                        returncode = process.wait()
                    entry.update(exit_code=returncode, status="completed" if returncode == 0 else "failed")
                    if returncode:
                        raise RuntimeError(f"{name}/{split} exited {returncode}; see its log")
                    result = read(target / "summary.json")
                    predictions = [json.loads(line) for line in (target / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
                    expected = {json.loads(line)["id"] for line in (dataset_dir / filename).read_text(encoding="utf-8").splitlines() if line}
                    if result["overall"]["queries"] != count or len(predictions) != count or {row["id"] for row in predictions} != expected:
                        raise RuntimeError(f"Incomplete evaluation: {name}/{split}")
                    if not (target / "predictions_raw.jsonl").is_file():
                        raise FileNotFoundError(target / "predictions_raw.jsonl")
                    summaries[split] = str(target / "summary.json")
                    state["completed"].append({"model": name, "split": split, "summary": summaries[split]})
                except BaseException:
                    entry["status"] = "failed"
                    raise
                finally:
                    elapsed = time.monotonic() - started
                    entry.update(ended_at=now(), wall_seconds=elapsed, gpu_hours=elapsed / 3600)
                    write(ledger, rows)
                    write(state_path, state)
            marker = "baseline_completed.json" if name == "warmup50" else "matched_completed.json"
            write(output / marker, {"status": "completed", "checkpoint": str(adapter),
                                   "global_step": step, "datasets": summaries, "completed_at": now()})
        state.update(status="completed", ended_at=now())
    except BaseException as error:
        state.update(status="failed", error=str(error), ended_at=now())
        raise
    finally:
        state.pop("child_pid", None)
        write(state_path, state)
        record_wait(ledger)


if __name__ == "__main__":
    main()
