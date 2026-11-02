"""Run optimized B/C/D from shared warmup50, then evaluate matched B/C exposure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from run_initial_arms import select_checkpoint


DEFAULT_ROOT = Path("/root/autodl-tmp/quant_rl_20261026")
DATA_ROOT = Path("/root/autodl-tmp/rematch_20260922/data")
MODEL = "/root/autodl-tmp/rematch_models/Qwen3.6-27B"
EXPECTED_COUNTS = {"city378": 378, "external88": 88}
DATASETS = {"city378": "city400_dev.jsonl", "external88": "external94_dev.jsonl"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_full_summary(path: Path, expected: int) -> dict:
    summary = read_json(path)
    if summary["overall"]["queries"] != expected:
        raise RuntimeError(f"评价未覆盖全部 {expected} 题：{path}")
    return summary


def require_checkpoint(path: Path, step: int) -> None:
    state = read_json(path / "trainer_state.json")
    required = ("optimizer.pt", "scheduler.pt", "adapter_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    weights = list(path.glob("*.safetensors"))
    if (state["global_step"] != step or missing or not weights
            or any((path / name).stat().st_size == 0 for name in required)
            or any(weight.stat().st_size == 0 for weight in weights)):
        raise RuntimeError(f"匹配曝光 checkpoint 不完整或步数错误：{path}; expected_step={step}; missing={missing}")


def rl_exposures(train_dir: Path, final_step: int) -> int:
    rows = [json.loads(line) for line in (train_dir / "resource_steps.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [row for row in rows if row["step"] <= final_step]
    if ([row["step"] for row in rows] != list(range(1, final_step + 1))
            or any(row["rollout_groups"] != 4 or row["rollout_answers"] != 32 for row in rows)):
        raise RuntimeError(f"RL 题目曝光与每步 4Q/G8 不一致：{train_dir}")
    return sum(row["rollout_groups"] for row in rows)


def charge_gap(args, stage: str, label: str, note: str) -> None:
    rows = json.loads(args.ledger.read_text(encoding="utf-8"))
    if any(row["status"] == "running" for row in rows):
        raise RuntimeError("统一账本仍有运行中的命令")
    last = max(datetime.fromisoformat(row.get("ended_at", row.get("recorded_at"))) for row in rows)
    hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    if hours > 0:
        command = [args.python, str(args.budget_tool), "--ledger", str(args.ledger),
                   "--stage", stage, "--label", label, "--gpu-model", args.gpu_model,
                   "--record-hours", str(hours), "--note", note]
        if args.price_per_gpu_hour is not None:
            command += ["--price-per-gpu-hour", str(args.price_per_gpu_hour)]
        subprocess.run(command, cwd=args.code, check=True)


def run_stage(args, state_path: Path, state: dict, arm: str, stage: str, budget_stage: str,
              hours: float, command: list[str]) -> None:
    label = f"optimized_{arm}_seed2026"
    charge_gap(args, budget_stage, label, f"Occupied interval before {arm} {stage}")
    budget_command = [args.python, str(args.budget_tool), "--ledger", str(args.ledger),
                      "--stage", budget_stage, "--label", label,
                      "--gpu-model", args.gpu_model, "--max-hours", str(hours)]
    if args.price_per_gpu_hour is not None:
        budget_command += ["--price-per-gpu-hour", str(args.price_per_gpu_hour)]
    budget_command += ["--", *command]
    state.update(status="running", current_arm=arm, stage=stage, budget_stage=budget_stage,
                 label=label, command=budget_command, pid=None, exit_code=None)
    save_state(state_path, state)
    process = subprocess.Popen(budget_command, cwd=args.code)
    state["pid"] = process.pid
    save_state(state_path, state)
    exit_code = process.wait()
    state.update(pid=None, exit_code=exit_code)
    save_state(state_path, state)
    if exit_code:
        raise RuntimeError(f"{arm} {stage} 退出码 {exit_code}")


def eval_command(args, checkpoint: Path, arm: str, dataset: Path, output_dir: Path) -> list[str]:
    return [args.python, str(args.code / "tools/eval_quant_grounding.py"),
            "--dataset", str(dataset), "--data-root", str(args.data_root),
            "--model", args.model, "--adapter", str(checkpoint),
            "--engine", "hf", "--precision", "nf4",
            "--answer-mode", "thinking" if arm == "D" else "short",
            "--reward-mode", "acc05" if arm == "B" else "contrast",
            "--samples-per-query", "1", "--hf-max-batch-size", "2",
            "--output-dir", str(output_dir)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--gpu-model", default="RTX4090-48GB")
    parser.add_argument("--price-per-gpu-hour", type=float)
    parser.add_argument("--check-only", action="store_true", help="核对决议、A记录、数据和预算状态，不启动队列")
    cli = parser.parse_args()

    cli.root = cli.root.resolve()
    cli.data_root = cli.data_root.resolve()
    cli.ledger = cli.root / "runs/budget.json"
    cli.budget_tool = cli.root / "code/tools/run_quant_budget.py"
    selection_path = cli.root / "runs/optimization_selection.json"
    completed_a = read_json(cli.root / "runs/a_completed.json")
    selection = read_json(selection_path)
    if selection["status"] not in {"selected", "fallback"}:
        raise RuntimeError(f"优化决议不明确：{selection_path}")
    settings = selection["selected_config"]
    if (int(settings["train_micro_batch_size"]) not in {1, 2}
            or int(settings["gradient_checkpointing_every_n_layers"]) not in {1, 2}
            or int(settings.get("hf_max_batch_size", 2)) not in {1, 2}):
        raise RuntimeError("优化决议中的微批/检查点配置无效")
    if selection["status"] == "fallback" and settings.get("runtime_optimization") is not False:
        raise RuntimeError("fallback 必须显式关闭 runtime_optimization")
    if selection["status"] == "fallback" and (
            int(settings["train_micro_batch_size"]) != 1
            or int(settings["gradient_checkpointing_every_n_layers"]) != 1
            or Path(selection["python"]).absolute() != (cli.root / "train_env/bin/python").absolute()):
        raise RuntimeError("fallback 必须使用原 train_env、micro batch 1 和逐层梯度检查点")
    if not Path(selection["python"]).is_absolute() or not Path(selection["code"]).is_absolute():
        raise RuntimeError("优化决议必须明确绝对 Python 与代码路径")
    # Preserve the venv launcher path: resolving its symlink changes sys.prefix.
    cli.python = str(Path(selection["python"]).absolute())
    cli.code = Path(selection["code"]).resolve()
    if not Path(cli.python).is_file() or not (cli.code / "tools/train_quant_grounding.py").is_file():
        raise RuntimeError("优化决议指定的 Python 环境或代码目录不存在")
    if "matched-exposure-checkpoint" not in (cli.code / "tools/train_quant_grounding.py").read_text(encoding="utf-8"):
        raise RuntimeError("所选训练入口无法保护 B/C 匹配曝光 checkpoint")

    n_a = int(completed_a["completed_query_exposures"])
    if (completed_a["arm"] != "A" or n_a <= 0 or n_a % 4
            or int(completed_a["global_step"]) * 4 != n_a
            or int(completed_a["completed_answers"]) != n_a * 8):
        raise RuntimeError("A 完成记录的曝光量与更新步数不一致")
    for name, count in EXPECTED_COUNTS.items():
        if completed_a[name]["overall"]["queries"] != count:
            raise RuntimeError(f"A 的 {name} 完成记录非全量")

    adapter = cli.root / "runs/shared_format_warmup/training/checkpoint-50"
    require_checkpoint(adapter, 50)
    dataset_dir = cli.data_root / "quant_grounding_20261026_formal"
    train_dataset = dataset_dir / "schedule.jsonl"
    if not train_dataset.is_file():
        raise FileNotFoundError(train_dataset)
    for name, count in EXPECTED_COUNTS.items():
        dataset = dataset_dir / DATASETS[name]
        actual = sum(bool(line.strip()) for line in dataset.open(encoding="utf-8"))
        if actual != count:
            raise RuntimeError(f"{dataset}: {actual} 条，预期 {count} 条")

    run_root = cli.root / "runs"
    state_path = run_root / "optimized_queue_state.json"
    if state_path.exists() or any((run_root / f"optimized_{arm}_seed2026").exists() for arm in "BCD"):
        raise RuntimeError("优化队列已有状态或输出目录；拒绝重复启动")
    if any(row["status"] == "running" for row in json.loads(cli.ledger.read_text(encoding="utf-8"))):
        raise RuntimeError("统一账本仍有运行中的命令")

    matched = {"B": n_a, "C": (n_a // 8) * 8}
    if matched["C"] <= 0:
        raise RuntimeError("A 曝光量不足 C 臂一个更新")
    if cli.check_only:
        print(json.dumps({"status": "ready", "selection": str(selection_path),
                          "python": cli.python, "code": str(cli.code), "warmup50": str(adapter),
                          "a_query_exposures": n_a, "matched_query_exposures": matched,
                          "ledger": str(cli.ledger), "train_outer_hours": {"B": 7, "C": 7, "D": 3},
                          "train_inner_hours": {"B": 6.75, "C": 6.75, "D": 2.75},
                          "eval_hours": {"B": 1, "C": 1, "D": 5},
                          "matched_eval_stage_hours": 2}, ensure_ascii=False, indent=2))
        return
    state = {"status": "running", "stage": "starting", "completed_arms": [],
             "a_final_checkpoint": completed_a["checkpoint"], "a_query_exposures": n_a,
             "matched_query_exposures": matched, "selection": selection,
             "budget_ledger": str(cli.ledger), "results": {}}
    state_path.touch(exist_ok=False)
    save_state(state_path, state)
    try:
        for arm in "BCD":
            arm_dir = run_root / f"optimized_{arm}_seed2026"
            train_dir = arm_dir / "train"
            arm_dir.mkdir()
            config = read_json(cli.code / "configs/quant_grounding" / f"arm_{arm.lower()}.json")
            config.update(adapters=str(adapter), seed=2026, output_dir=str(train_dir),
                          train_micro_batch_size=int(settings["train_micro_batch_size"]),
                          gradient_checkpointing_every_n_layers=int(
                              settings["gradient_checkpointing_every_n_layers"]),
                          hf_max_batch_size=int(settings.get("hf_max_batch_size", 2)),
                          save_total_limit=max(4, int(config.get("save_total_limit", 4))))
            if "runtime_optimization" in settings:
                config["runtime_optimization"] = settings["runtime_optimization"]
            if "deterministic_training" in settings:
                config["deterministic_training"] = settings["deterministic_training"]
            if arm in matched:
                config["matched_query_exposures"] = matched[arm]
            config_path = arm_dir / "arm_config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            train_hours = 3.0 if arm == "D" else 7.0
            train_command = [cli.python, str(cli.code / "tools/train_quant_grounding.py"),
                             "--config", str(config_path), "--model", cli.model,
                             "--dataset", str(train_dataset), "--data-root", str(cli.data_root),
                             "--output-dir", str(train_dir), "--max-gpu-hours", str(train_hours - 0.25)]
            run_stage(cli, state_path, state, arm, "training", "initial", train_hours, train_command)
            checkpoint, final_step = select_checkpoint(train_dir)
            actual_exposures = final_step * 8 if arm == "C" else rl_exposures(train_dir, final_step)
            results = {"final_checkpoint": str(checkpoint), "final_step": final_step,
                       "final_query_exposures": actual_exposures,
                       "final_evaluation": {}}
            if arm in matched:
                target_step = matched[arm] // (8 if arm == "C" else 4)
                protected = train_dir / f"matched-exposure-checkpoint-{target_step}"
                if actual_exposures < matched[arm]:
                    results["matched_checkpoint_missing_reason"] = (
                        f"training reached {actual_exposures}/{matched[arm]} query exposures")
                elif protected.exists():
                    require_checkpoint(protected, target_step)
                    results["matched_checkpoint"] = str(protected)
                else:
                    raise RuntimeError(f"{arm} 达到匹配曝光但受保护 checkpoint 缺失：{protected}")
            state["results"][arm] = results
            save_state(state_path, state)

            eval_hours = 5.0 if arm == "D" else 1.0
            for name, count in EXPECTED_COUNTS.items():
                output = arm_dir / name
                hours = eval_hours * count / 466
                run_stage(cli, state_path, state, arm, f"final_eval_{name}", "initial", hours,
                          eval_command(cli, checkpoint, arm, dataset_dir / DATASETS[name], output))
                require_full_summary(output / "summary.json", count)
                results["final_evaluation"][name] = str(output / "summary.json")
                save_state(state_path, state)
            state["completed_arms"].append(arm)
            state.update(stage=f"{arm}_complete", current_arm=None)
            save_state(state_path, state)

        state["stage"] = "matched_exposure_evaluation"
        save_state(state_path, state)
        unmatched = []
        for arm in "BC":
            arm_dir = run_root / f"optimized_{arm}_seed2026"
            results = state["results"][arm]
            if "matched_checkpoint" not in results:
                results["matched_evaluation"] = {
                    "status": "unavailable", "reason": results["matched_checkpoint_missing_reason"]}
                unmatched.append(arm)
                save_state(state_path, state)
                continue
            results["matched_evaluation"] = {}
            if results["final_step"] == matched[arm] // (8 if arm == "C" else 4):
                for name in EXPECTED_COUNTS:
                    results["matched_evaluation"][name] = {
                        "summary": results["final_evaluation"][name], "reused_final": True}
                save_state(state_path, state)
                continue
            for name, count in EXPECTED_COUNTS.items():
                output = arm_dir / "matched_exposure" / name
                hours = count / 466
                run_stage(cli, state_path, state, arm, f"matched_eval_{name}", "exposure_eval", hours,
                          eval_command(cli, Path(results["matched_checkpoint"]), arm,
                                       dataset_dir / DATASETS[name], output))
                require_full_summary(output / "summary.json", count)
                results["matched_evaluation"][name] = {
                    "summary": str(output / "summary.json"), "reused_final": False}
                save_state(state_path, state)
        charge_gap(cli, "exposure_eval", "optimized_BC_seed2026", "Occupied queue tail")
        state.update(status="completed_with_unmatched" if unmatched else "completed",
                     unmatched_arms=unmatched,
                     stage="all_BCD_final_evaluations_complete_with_unmatched" if unmatched
                     else "all_BCD_and_matched_evaluations_complete",
                     current_arm=None, pid=None, command=None)
        save_state(state_path, state)
    except Exception as error:
        state.update(status="failed", pid=None, error=f"{type(error).__name__}: {error}")
        save_state(state_path, state)
        raise


if __name__ == "__main__":
    main()
