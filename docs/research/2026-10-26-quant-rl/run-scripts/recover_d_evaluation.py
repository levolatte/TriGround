"""Recover the timed-out optimized D arm at its last complete checkpoint, then finish matched evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from run_initial_arms import select_checkpoint
from run_optimized_arms import (
    DATA_ROOT, DATASETS, DEFAULT_ROOT, EXPECTED_COUNTS, MODEL,
    charge_gap, eval_command, read_json, require_checkpoint,
    require_full_summary, rl_exposures, run_stage, save_state,
)


TAIL_HOURS = 18 / 3600


def budget_room(rows: list[dict], stage: str, label: str) -> float:
    used = sum(row["gpu_hours"] for row in rows)
    stage_used = sum(row["gpu_hours"] for row in rows if row["stage"] == stage)
    limits = [72 - used, {"initial": 32, "exposure_eval": 2}[stage] - stage_used]
    if stage == "initial":
        limits.append(8 - sum(row["gpu_hours"] for row in rows
                              if row["stage"] == stage and row["label"] == label))
    return min(limits)


def ledger_rows(cli) -> list[dict]:
    rows = read_json(cli.ledger)
    if any(row["status"] == "running" for row in rows):
        raise RuntimeError("统一账本仍有运行中的命令")
    return rows


def idle_hours(rows: list[dict]) -> float:
    latest = max(datetime.fromisoformat(row.get("ended_at", row.get("recorded_at"))) for row in rows)
    return max(0.0, (datetime.now(timezone.utc) - latest).total_seconds() / 3600)


def preflight(cli, state: dict, run_root: Path) -> dict:
    if (state.get("status") != "failed" or state.get("stage") != "training"
            or state.get("current_arm") != "D" or state.get("exit_code") != 124
            or state.get("completed_arms") != ["B", "C"]):
        raise RuntimeError("队列不是 D 训练超时且 B/C 已完成的待恢复状态")
    if "D" in state.get("results", {}):
        raise RuntimeError("D 已有结果；拒绝重复恢复")

    rows = ledger_rows(cli)
    d_label = "optimized_D_seed2026"
    timed_out = [row for row in rows if row.get("stage") == "initial"
                 and row.get("label") == d_label and row.get("status") == "budget_exhausted"
                 and row.get("exit_code") == 124]
    if len(timed_out) != 1 or any(row.get("label") == d_label and row.get("stage") != "initial"
                                  for row in rows):
        raise RuntimeError("D 的统一账本记录与单次训练超时不一致")

    for arm in "BC":
        result = state["results"][arm]
        require_checkpoint(Path(result["final_checkpoint"]), int(result["final_step"]))
        for name, count in EXPECTED_COUNTS.items():
            require_full_summary(Path(result["final_evaluation"][name]), count)
    matched = state["matched_query_exposures"]
    b_result = state["results"]["B"]
    if int(b_result["final_step"]) != int(matched["B"]) // 4:
        raise RuntimeError("B 最终评价不能作为匹配曝光评价复用")
    require_checkpoint(Path(b_result["matched_checkpoint"]), int(b_result["final_step"]))
    c_checkpoint = Path(state["results"]["C"]["matched_checkpoint"])
    if int(matched["C"]) != 232:
        raise RuntimeError("C 匹配曝光量不是 232 次")
    require_checkpoint(c_checkpoint, 29)
    if state["results"]["C"].get("matched_evaluation"):
        raise RuntimeError("C 匹配评价已有结果；拒绝重复恢复")
    if any((run_root / "optimized_C_seed2026" / "matched_exposure" / name).exists()
           for name in EXPECTED_COUNTS):
        raise RuntimeError("C 匹配评价输出已存在；拒绝覆盖")

    arm_dir = run_root / d_label
    if any((arm_dir / name).exists() for name in EXPECTED_COUNTS):
        raise RuntimeError("D 最终评价输出已存在；拒绝覆盖")
    train_dir = arm_dir / "train"
    lines = [line for line in (train_dir / "resource_steps.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if len(lines) != 8 or rl_exposures(train_dir, 8) != 32:
        raise RuntimeError("D 的 8 步资源日志不完整或不符合每步 4Q/G8")
    checkpoint, final_step = select_checkpoint(train_dir)
    require_checkpoint(checkpoint, final_step)
    if final_step != 6 or rl_exposures(train_dir, final_step) != 24:
        raise RuntimeError(f"D 的最后完整权重不是第 6 步：{checkpoint}")

    gap = idle_hours(rows)
    projected = [*rows, {"stage": "initial", "label": d_label, "gpu_hours": gap}]
    d_hours = min(5.0, budget_room(projected, "initial", d_label))
    exposure_room = budget_room(projected, "exposure_eval", "optimized_C_seed2026")
    total_after_d = 72 - sum(row["gpu_hours"] for row in projected) - d_hours
    if d_hours <= TAIL_HOURS or exposure_room <= 1 + TAIL_HOURS or total_after_d <= 1 + TAIL_HOURS:
        raise RuntimeError("扣除待记占机间隔后，D 评价或 C 全量匹配评价预算不足")
    return {"checkpoint": checkpoint, "train_dir": train_dir, "c_checkpoint": c_checkpoint,
            "idle_hours": gap, "d_eval_hours": d_hours, "exposure_eval_room": exposure_room,
            "total_after_d_hours": total_after_d}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--gpu-model", default="RTX4090-48GB")
    parser.add_argument("--price-per-gpu-hour", type=float)
    parser.add_argument("--check-only", action="store_true")
    cli = parser.parse_args()

    cli.root = cli.root.absolute()
    cli.data_root = cli.data_root.absolute()
    cli.ledger = cli.root / "runs/budget.json"
    cli.budget_tool = cli.root / "code/tools/run_quant_budget.py"
    selection = read_json(cli.root / "runs/optimization_selection.json")
    if selection["status"] not in {"selected", "fallback"}:
        raise RuntimeError("优化决议不明确")
    # Keep the venv launcher path: resolving the symlink changes sys.prefix.
    cli.python = str(Path(selection["python"]).absolute())
    cli.code = Path(selection["code"]).resolve()
    if (Path(cli.python) != (cli.root / "train_env/bin/python")
            or not Path(cli.python).is_file()
            or not (cli.code / "tools/eval_quant_grounding.py").is_file()
            or not cli.budget_tool.is_file()):
        raise RuntimeError("优化决议中的解释器或代码路径不符合当前云端实验")
    dataset_dir = cli.data_root / "quant_grounding_20261026_formal"
    for name, count in EXPECTED_COUNTS.items():
        dataset = dataset_dir / DATASETS[name]
        if sum(bool(line.strip()) for line in dataset.open(encoding="utf-8")) != count:
            raise RuntimeError(f"数据集不是完整 {count} 题：{dataset}")

    run_root = cli.root / "runs"
    state_path = run_root / "optimized_queue_state.json"
    state = read_json(state_path)
    info = preflight(cli, state, run_root)
    if cli.check_only:
        print(json.dumps({"status": "ready", "checkpoint": str(info["checkpoint"]),
                          "executed_training_step": 8, "executed_query_exposures": 32,
                          "final_step": 6, "final_query_exposures": 24,
                          "unrecorded_idle_hours": info["idle_hours"],
                          "d_eval_hours": info["d_eval_hours"],
                          "exposure_eval_room": info["exposure_eval_room"],
                          "total_after_d_hours": info["total_after_d_hours"],
                          "python": cli.python, "code": str(cli.code)}, ensure_ascii=False, indent=2))
        return

    original_failure = {key: state.get(key) for key in
                        ("status", "stage", "current_arm", "budget_stage", "label", "command",
                         "pid", "exit_code", "error", "updated_at")}
    state["recovery"] = {"source_failure": original_failure,
                         "started_at": datetime.now(timezone.utc).isoformat()}
    state.pop("error", None)
    result = {"training": {"status": "budget_exhausted", "recovery_status": "recovered_checkpoint"},
              "executed_training_step": 8, "executed_query_exposures": 32,
              "final_checkpoint": str(info["checkpoint"]), "final_step": 6,
              "final_query_exposures": 24, "final_evaluation": {}}
    state["results"]["D"] = result
    state.update(status="running", stage="D_checkpoint_recovered", current_arm="D",
                 pid=None, command=None, exit_code=None)
    save_state(state_path, state)
    try:
        d_label = "optimized_D_seed2026"
        charge_gap(cli, "initial", d_label, "Occupied recovery preparation after timed-out D training")
        d_hours = min(5.0, budget_room(ledger_rows(cli), "initial", d_label))
        if d_hours <= TAIL_HOURS:
            raise RuntimeError("D 评价预算已耗尽")
        arm_dir = run_root / d_label
        for index, (name, count) in enumerate(EXPECTED_COUNTS.items()):
            if index:
                charge_gap(cli, "initial", d_label, "Interval between D final evaluations")
            remaining = budget_room(ledger_rows(cli), "initial", d_label)
            hours = min(d_hours * count / 466, remaining) - (TAIL_HOURS if index else 0)
            if hours <= 0:
                raise RuntimeError(f"D {name} 评价预算已耗尽")
            output = arm_dir / name
            run_stage(cli, state_path, state, "D", f"final_eval_{name}", "initial", hours,
                      eval_command(cli, info["checkpoint"], "D", dataset_dir / DATASETS[name], output))
            require_full_summary(output / "summary.json", count)
            result["final_evaluation"][name] = str(output / "summary.json")
            save_state(state_path, state)
        state["completed_arms"].append("D")
        state.update(stage="matched_exposure_evaluation", current_arm=None)
        save_state(state_path, state)

        b_result = state["results"]["B"]
        b_result["matched_evaluation"] = {
            name: {"summary": b_result["final_evaluation"][name], "reused_final": True}
            for name in EXPECTED_COUNTS}
        save_state(state_path, state)

        c_result = state["results"]["C"]
        c_result["matched_evaluation"] = {}
        c_label = "optimized_C_seed2026"
        charge_gap(cli, "exposure_eval", c_label, "Interval before C matched-exposure evaluation")
        c_hours = min(1.0, budget_room(ledger_rows(cli), "exposure_eval", c_label))
        if c_hours < 1.0:
            raise RuntimeError("C 全量匹配评价的 1 GPU小时预算不足")
        for index, (name, count) in enumerate(EXPECTED_COUNTS.items()):
            if index:
                charge_gap(cli, "exposure_eval", c_label, "Interval between C matched evaluations")
            remaining = budget_room(ledger_rows(cli), "exposure_eval", c_label)
            hours = min(c_hours * count / 466, remaining) - (TAIL_HOURS if index else 0)
            if hours <= 0:
                raise RuntimeError(f"C {name} 匹配评价预算已耗尽")
            output = run_root / c_label / "matched_exposure" / name
            run_stage(cli, state_path, state, "C", f"matched_eval_{name}", "exposure_eval", hours,
                      eval_command(cli, info["c_checkpoint"], "C", dataset_dir / DATASETS[name], output))
            require_full_summary(output / "summary.json", count)
            c_result["matched_evaluation"][name] = {
                "summary": str(output / "summary.json"), "reused_final": False}
            save_state(state_path, state)
        charge_gap(cli, "exposure_eval", "optimized_BC_seed2026", "Occupied queue tail")
        state.update(status="completed", stage="all_BCD_and_matched_evaluations_complete",
                     current_arm=None, pid=None, command=None, exit_code=0, unmatched_arms=[])
        state.pop("error", None)
        state["recovery"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state)
    except Exception as error:
        state.update(status="failed", pid=None, error=f"{type(error).__name__}: {error}")
        save_state(state_path, state)
        raise


if __name__ == "__main__":
    main()
