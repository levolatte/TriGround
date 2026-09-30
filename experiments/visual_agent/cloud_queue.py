"""Finite queue for the dedicated clone while the local workstation is off.

Uses the existing experiment CLIs. Stops on errors; it does not train, run the
full 412, choose final methods, restart processes, or touch the mainline host.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


class BudgetReached(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root, exp = args.root.resolve(), args.experiment.resolve()
    code, runs = exp / "code_v3", exp / "schema_v3"
    queue = exp / "offline_queue"
    queue.mkdir(exist_ok=True)
    state_path = queue / "state.json"
    if state_path.exists():
        raise FileExistsError("This finite queue has already started; inspect its state before continuing manually.")
    model = "/root/rematch_models/Qwen3-VL-8B-Instruct"
    adapter = root / "results/next_stage_20260925/c_phase2/checkpoint-500"
    dino, sam = root / "models/grounding-dino-tiny", root / "models/sam2.1-hiera-tiny"
    manifests = exp / "manifests"
    cache412 = root / "results/aux_selection_refine_20260927/city412"
    gt412 = root / "results/aux_selection_refine_20260927/manifests/scoring/city412_gt.json"
    state = {"pid": os.getpid(), "started_unix": time.time(), "status": "running", "completed_stages": []}

    def save(stage, **extra):
        state.update(current_stage=stage, updated_unix=time.time(), **extra)
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps({"stage": stage, **extra}), flush=True)

    def module(name, *argv):
        return [sys.executable, "-u", "-m", name, *map(str, argv)]

    def used(line):
        path = exp / f"{line.lower()}_budget.jsonl"
        return sum(json.loads(row)["elapsed_seconds"] for row in path.read_text().splitlines() if row) if path.exists() else 0

    def execute(name, command, *, direct_budget=None):
        if direct_budget and used(direct_budget) >= {"Z": 12, "T": 8}[direct_budget] * 3600:
            raise BudgetReached(f"{direct_budget} budget exhausted before {name}")
        save(name)
        with (queue / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": name, "argv": command}) + "\n")
        started = time.perf_counter()
        with (queue / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        if direct_budget:
            with (exp / f"{direct_budget.lower()}_budget.jsonl").open("a") as handle:
                handle.write(json.dumps({"phase": "offline_queue", "task": name,
                    "elapsed_seconds": elapsed, "budget_hours": {"Z": 12, "T": 8}[direct_budget],
                    "exit_code": result.returncode}) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed with exit {result.returncode}; see {queue / (name + '.log')}")
        if "experiments.visual_agent.batch" in command:
            batch_result = json.loads((queue / f"{name}.log").read_text())
            if batch_result["status"] == "budget_interrupted":
                raise BudgetReached(f"{name}: {batch_result}")
        state["completed_stages"].append({"name": name, "elapsed_seconds": elapsed})
        save(name, last_exit_code=0)

    def batch(phase, cache, *, line="Z", evidence=None, extras=()):
        command = module("experiments.visual_agent.batch", "--phase", phase,
            "--run-root", runs, "--code-root", code, "--manifests", manifests,
            "--candidate-cache", cache, "--model", model, "--c-adapter", adapter,
            "--dino-model", dino, "--sam-model", sam, "--python", sys.executable,
            "--budget-hours", 12 if line == "Z" else 8,
            "--budget-ledger", exp / f"{line.lower()}_budget.jsonl")
        if evidence:
            command += ["--evidence-cache", str(evidence)]
        return command + list(map(str, extras))

    def evaluate(name, cohort, paths):
        command = module("experiments.visual_agent.evaluate", "--gt",
            exp / "scoring/train32_gt.json" if cohort == "train32" else gt412,
            "--manifest", manifests / f"{cohort}.jsonl", "--baseline",
            exp / "baseline32/predictions.jsonl" if cohort == "train32" else manifests / "dev16groups_c_baseline.jsonl",
            "--output-dir", exp / "scores" / name)
        for label, path in paths:
            command += ["--run", f"{label}={path}"]
        execute("score_" + name, command)
        return json.loads((exp / "scores" / name / "summary.json").read_text())

    try:
        save("wait_t250_initial_predictions")
        deadline = time.monotonic() + 3600
        while not (exp / "t_prepare_complete.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("T250 preparation marker did not appear within one hour; no GPU job started")
            time.sleep(10)
        cpu_command = module("experiments.visual_agent.bootstrap", "--stage", "candidates",
            "--manifest", manifests / "t250.jsonl", "--baseline", exp / "baseline250/predictions.jsonl",
            "--query-info", exp / "query250/query_info.jsonl", "--data-root", root / "data/city/train",
            "--model", dino, "--output", exp / "t250/candidates.jsonl", "--threads", 4)
        cpu_log = (queue / "t250_cpu_candidates.log").open("w")
        cpu = subprocess.Popen(cpu_command, cwd=code, stdout=cpu_log, stderr=subprocess.STDOUT)
        with (queue / "commands.jsonl").open("a") as handle:
            handle.write(json.dumps({"stage": "t250_cpu_candidates", "argv": cpu_command, "pid": cpu.pid}) + "\n")
        save("capacity_diagnostic", cpu_candidate_pid=cpu.pid)
        first = json.loads((manifests / "train32.jsonl").read_text().splitlines()[0])["id"]
        execute("capacity_diagnostic", module("experiments.visual_agent.diagnostic",
            "--manifest", manifests / "train32.jsonl", "--sample-id", first,
            "--candidate-cache", exp / "train32/candidates.jsonl", "--model", model,
            "--dino-model", dino, "--sam-model", sam, "--profile", "capacity24",
            "--output-dir", runs / "capacity_diagnostic"), direct_budget="Z")
        diagnostic = json.loads((runs / "capacity_diagnostic/diagnostic.json").read_text())
        capacity_ok = diagnostic["coverage"]["all_six_executed"] and diagnostic["final_generation"]["status"] == "GENERATED"
        save("capacity_diagnostic", capacity_verified=capacity_ok)
        for controller in ("native", "c-lora"):
            for mechanism in ("A", "B", "S"):
                command = module("experiments.visual_agent.run", "--mechanism", mechanism,
                    "--controller", controller, "--profile", "fast", "--model", model,
                    "--manifest", manifests / "train32.jsonl", "--candidate-cache", exp / "train32/candidates.jsonl",
                    "--dino-model", dino, "--sam-model", sam, "--limit", 4,
                    "--max-run-seconds", 600, "--output-dir", runs / "smoke" / f"{controller}_{mechanism}")
                if controller == "c-lora":
                    command += ["--adapter", str(adapter)]
                execute(f"smoke_{controller}_{mechanism}", command, direct_budget="Z")
        execute("debug_v3", batch("debug", exp / "train32/candidates.jsonl"))
        debug_paths = [(f"{c}_{m}_fast", runs / "runs/train32/debug" / f"{c}_{m}_fast/predictions.jsonl")
                       for c in ("native", "c-lora") for m in ("C", "D")]
        evaluate("debug_v3", "train32", debug_paths)
        for name, path in debug_paths:
            execute("audit_debug_" + name, module("experiments.visual_agent.report", "--traces", path,
                "--output", exp / "scores/debug_v3" / name / "behavior_audit.json"))
        execute("dev_fast", batch("dev-fast", cache412 / "candidates.jsonl", evidence=cache412 / "evidence.jsonl"))
        dev_paths = [(f"{c}_{m}_fast", runs / "runs/dev16groups/dev-fast" / f"{c}_{m}_fast/predictions.jsonl")
                     for c in ("native", "c-lora") for m in ("A", "B", "S", "C", "D")]
        scores = evaluate("dev_fast_v3", "dev16groups", dev_paths)
        choices = [(c, m, scores["runs"][f"{c}_{m}_fast"]["all"])
                   for c in ("native", "c-lora") for m in ("C", "D")]
        priority = min(choices, key=lambda x: (-x[2]["acc_0.5"], -x[2]["mean_iou"],
                       x[2]["cost_mean_per_query"]["elapsed_seconds"], x[0], x[1]))[0]
        # Keep the plan's 4.5 GPU-hour full-412 allocation intact. This finite
        # queue does not select the final 2 methods or start full evaluation.
        estimated_capacity = sum(scores["runs"][f"{priority}_{m}_fast"]["all"]["cost_total"]["elapsed_seconds"]
                                 for m in ("B", "S", "C", "D")) * 3 + 120
        if capacity_ok and 12 * 3600 - used("Z") >= 4.5 * 3600 + estimated_capacity:
            execute("dev_capacity", batch("dev-capacity", cache412 / "candidates.jsonl",
                evidence=cache412 / "evidence.jsonl", extras=("--controllers", priority, "--profile", "capacity24")))
            capacity_paths = [(f"{priority}_{m}_capacity24", runs / "runs/dev16groups/dev-capacity" /
                              f"{priority}_{m}_capacity24/predictions.jsonl") for m in ("B", "S", "C", "D")]
            evaluate("dev_combined_v3", "dev16groups", dev_paths + capacity_paths)
        else:
            save("capacity_extension_deferred", capacity_verified=capacity_ok,
                 reason="capacity diagnostic incomplete or reserved full-evaluation budget")
        save("wait_t250_cpu_candidates")
        if cpu.wait() != 0:
            raise RuntimeError("T250 CPU candidate generation failed; see t250_cpu_candidates.log")
        cpu_log.close()
        execute("t250_generate", batch("t-generate", exp / "t250/candidates.jsonl", line="T", extras=("--mechanisms", "D")))
        t_run = runs / "runs/t250/t-generate/native_D_sft"
        execute("t250_audit", module("experiments.visual_agent.report", "--traces", t_run / "predictions.jsonl",
            "--output", exp / "scores/t250/behavior_audit.json"))
        execute("t250_export", module("experiments.visual_agent.export_trajectories", "--traces", t_run / "traces",
            "--train-manifest", manifests / "t_train200.jsonl", "--holdout-manifest", manifests / "t_holdout50.jsonl",
            "--debug-manifest", manifests / "train32.jsonl", "--gt", exp / "scoring/t250_gt.json",
            "--output-dir", exp / "t_sft_export"))
        execute("t250_preflight", module("experiments.visual_agent.train", "--train", exp / "t_sft_export/train.jsonl",
            "--model", model, "--output-dir", exp / "t_preflight", "--preflight-only"))
        save("await_local_review", status="complete", z_used_hours=used("Z") / 3600,
             t_used_hours=used("T") / 3600,
             next="Review development scores and trajectory labels; choose full-412 methods and run independent SFT.")
    except BudgetReached as error:
        save(state.get("current_stage", "startup"), status="budget_exhausted", reason=str(error))
    except Exception as error:
        save(state.get("current_stage", "startup"), status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
