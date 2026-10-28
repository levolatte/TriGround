"""Run initial A/B/C/D NF4 arms. Legacy city400/external94 filenames have 378/88 queries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ARMS = ("A", "B", "C", "D")
LEDGER = "/root/autodl-tmp/quant_rl_20261026/runs/budget.json"


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_stage(args, code: Path, state_path: Path, state: dict, arm: str, stage: str,
              command: list[str], hours: float, results: dict) -> None:
    label = f"{arm}_seed2026"
    ledger_rows = json.loads(args.ledger.read_text(encoding="utf-8"))
    assert all(row["status"] != "running" for row in ledger_rows)
    latest = max(datetime.fromisoformat(row.get("ended_at", row.get("recorded_at")))
                 for row in ledger_rows)
    gap_hours = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
    subprocess.run([args.python, str(code / "tools/run_quant_budget.py"),
                    "--ledger", str(args.ledger), "--stage", "initial", "--label", label,
                    "--gpu-model", args.gpu_model, "--record-hours", str(gap_hours),
                    "--note", f"Occupied interval before {stage}"], check=True)
    budget_command = [args.python, str(code / "tools/run_quant_budget.py"),
                      "--ledger", str(args.ledger), "--stage", "initial", "--label", label,
                      "--gpu-model", args.gpu_model, "--max-hours", str(hours)]
    if args.price_per_gpu_hour is not None:
        budget_command += ["--price-per-gpu-hour", str(args.price_per_gpu_hour)]
    budget_command += ["--", *command]
    state.update(status="running", current_arm=arm, stage=stage, label=label,
                 pid=None, exit_code=None, command=budget_command, result_paths=results)
    save_state(state_path, state)
    process = subprocess.Popen(budget_command, cwd=code)
    state["pid"] = process.pid
    save_state(state_path, state)
    code_status = process.wait()
    state["pid"] = None
    state["exit_code"] = code_status
    save_state(state_path, state)
    if code_status:
        state["status"] = "failed"
        save_state(state_path, state)
        raise SystemExit(code_status)


def select_checkpoint(train_dir: Path) -> tuple[Path, int]:
    checkpoints = []
    for path in train_dir.glob("checkpoint-*"):
        if not path.is_dir() or not path.name.removeprefix("checkpoint-").isdigit():
            continue
        state_path = path / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checkpoints.append((state["global_step"], path))
    step = max(item[0] for item in checkpoints)
    paths = [path for current_step, path in checkpoints if current_step == step]
    if len(paths) != 1:
        raise RuntimeError(f"checkpoint global_step={step} 不唯一：{paths}")
    checkpoint = paths[0]
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_config.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    weights = list(checkpoint.glob("*.safetensors"))
    empty = [name for name in required if (checkpoint / name).is_file() and (checkpoint / name).stat().st_size == 0]
    if missing or not weights or empty or any(weight.stat().st_size == 0 for weight in weights):
        raise RuntimeError(f"最终 checkpoint 缺少可恢复状态或 LoRA 权重：{checkpoint}; missing={missing}; empty={empty}")
    return checkpoint, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--code", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path,
                        help=f"全局72 GPU小时账本，必须是 {LEDGER}")
    parser.add_argument("--start-arm", choices=ARMS, default="A")
    parser.add_argument("--adapter", type=Path, help="共享格式 warmup LoRA；训练工具将其设为 ref_adapter")
    parser.add_argument("--baseline-city", type=Path,
                        help="可选评估基线 JSONL，格式 id + 归一化 prediction 或 bbox")
    parser.add_argument("--baseline-external", type=Path,
                        help="可选评估基线 JSONL，格式 id + 归一化 prediction 或 bbox")
    parser.add_argument("--gpu-model", default="RTX4090-48GB")
    parser.add_argument("--hf-max-batch-size", type=int, choices=(1, 2), default=1,
                        help="单次评价的跨题并发；2须先完成真实双题显存检查")
    parser.add_argument("--train-hf-max-batch-size", type=int, choices=(1, 2), default=1,
                        help="RL生成并发；2须先完成生成及反传组合显存检查")
    parser.add_argument("--price-per-gpu-hour", type=float)
    parser.add_argument("--arm-hours", type=float, default=8.0)
    parser.add_argument("--train-hours", type=float, default=7.0)
    parser.add_argument("--d-train-hours", type=float, default=None,
                        help="D臂训练时长；默认沿用 --train-hours")
    parser.add_argument("--save-margin-hours", type=float, default=0.25)
    args = parser.parse_args()

    code, data_root = args.code.resolve(), args.data_root.resolve()
    dataset_dir, run_root = args.dataset_dir.resolve(), args.run_root.resolve()
    args.ledger = args.ledger.resolve()
    if args.ledger.as_posix() != LEDGER:
        parser.error(f"--ledger 必须使用统一账本 {LEDGER}")
    if not 0 < args.train_hours < args.arm_hours <= 8:
        parser.error("需满足 0 < train-hours < arm-hours <= 8")
    if args.d_train_hours is not None and not 0 < args.d_train_hours < args.arm_hours:
        parser.error("d-train-hours 必须大于0且小于 arm-hours")
    if not 0 <= args.save_margin_hours < min(args.train_hours, args.d_train_hours or args.train_hours):
        parser.error("save-margin-hours 必须非负且小于 train-hours")
    city_dataset = dataset_dir / "city400_dev.jsonl"
    external_dataset = dataset_dir / "external94_dev.jsonl"
    city_count = sum(1 for line in city_dataset.open(encoding="utf-8") if line.strip())
    external_count = sum(1 for line in external_dataset.open(encoding="utf-8") if line.strip())
    total_eval_queries = city_count + external_count
    start = ARMS.index(args.start_arm)
    selected = ARMS[start:]
    labels = {arm: f"{arm}_seed2026" for arm in selected}
    existing = [str(run_root / labels[arm]) for arm in selected if (run_root / labels[arm]).exists()]
    if existing:
        parser.error("为保护已有输出，不覆盖这些臂目录：" + ", ".join(existing))

    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "queue_state.json"
    completed = json.loads(state_path.read_text(encoding="utf-8")).get("completed_arms", []) if state_path.exists() else []
    state = {"status": "running", "current_arm": None, "stage": "starting",
             "pid": None, "completed_arms": completed, "result_paths": {},
             "budget_ledger": str(args.ledger)}
    save_state(state_path, state)

    train_dataset = dataset_dir / "schedule.jsonl"
    for arm in selected:
        label = labels[arm]
        arm_dir = run_root / label
        arm_train_hours = (args.d_train_hours if arm == "D" and args.d_train_hours is not None
                           else args.train_hours)
        remaining_eval = args.arm_hours - arm_train_hours
        city_eval_hours = remaining_eval * city_count / total_eval_queries
        external_eval_hours = remaining_eval * external_count / total_eval_queries
        train_dir, city_dir, external_dir = arm_dir / "train", arm_dir / "city378", arm_dir / "external88"
        arm_dir.mkdir()
        state.update(arm_train_hours=arm_train_hours, city_eval_hours=city_eval_hours,
                     external_eval_hours=external_eval_hours)
        save_state(state_path, state)
        config_path = code / "configs/quant_grounding" / f"arm_{arm.lower()}.json"
        if args.adapter or args.train_hf_max_batch_size != 1:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if args.adapter:
                config["adapters"] = str(args.adapter.resolve())
            config["hf_max_batch_size"] = args.train_hf_max_batch_size
            config_path = arm_dir / "arm_config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        results = {"train": str(train_dir),
                   "city378_raw": str(city_dir / "predictions_raw.jsonl"),
                   "city378_summary": str(city_dir / "summary.json"),
                   "external88_raw": str(external_dir / "predictions_raw.jsonl"),
                   "external88_summary": str(external_dir / "summary.json")}
        train_command = [args.python, str(code / "tools/train_quant_grounding.py"),
                         "--config", str(config_path), "--model", args.model,
                         "--dataset", str(train_dataset), "--data-root", str(data_root),
                         "--output-dir", str(train_dir), "--max-gpu-hours",
                         str(arm_train_hours - args.save_margin_hours)]
        run_stage(args, code, state_path, state, arm, "training", train_command, arm_train_hours, results)
        state.update(stage="checkpoint_audit", pid=None)
        save_state(state_path, state)
        try:
            checkpoint, step = select_checkpoint(train_dir)
        except Exception as error:
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            save_state(state_path, state)
            raise
        results["selected_checkpoint"] = str(checkpoint)
        state["selected_checkpoint_global_step"] = step

        answer = "thinking" if arm == "D" else "short"
        reward = "acc05" if arm == "B" else "contrast"
        evals = (("eval_city378", city_dataset, city_dir, args.baseline_city, city_eval_hours),
                 ("eval_external88", external_dataset, external_dir, args.baseline_external, external_eval_hours))
        for stage, dataset, output_dir, baseline, hours in evals:
            command = [args.python, str(code / "tools/eval_quant_grounding.py"),
                       "--dataset", str(dataset), "--data-root", str(data_root),
                       "--model", args.model, "--adapter", str(checkpoint),
                       "--engine", "hf", "--precision", "nf4", "--answer-mode", answer,
                       "--reward-mode", reward, "--samples-per-query", "1",
                       "--hf-max-batch-size", str(args.hf_max_batch_size), "--output-dir", str(output_dir)]
            if baseline:
                command += ["--baseline", str(baseline.resolve())]
            run_stage(args, code, state_path, state, arm, stage, command, hours, results)
        completed.append(arm)
        state.update(status="running", stage="arm_complete", pid=None, completed_arms=completed,
                     result_paths=results)
        save_state(state_path, state)

    state.update(status="completed", stage="all_selected_arms_complete", current_arm=None, pid=None,
                 completed_arms=completed)
    save_state(state_path, state)


if __name__ == "__main__":
    main()
