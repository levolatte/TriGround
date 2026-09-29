"""Serial A/B/V execution with a cumulative 12-hour process-time budget.

The default is a command preview. Execution requires the human-approved data
release produced by prepare_triground_abv_data; no historical queue is resumed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ARMS = ("A", "B", "V")
BUDGET_SECONDS = 12 * 3600


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_release(release):
    if release.get("status") != "ready":
        raise ValueError("Human-approved data release is not ready; prepare and review the data first")
    if release["steps"] not in (400, 600):
        raise ValueError("The frozen schedule must have 400 or 600 optimizer steps")
    if release["seed"] not in (2026, 2027):
        raise ValueError("Only the planned seeds 2026 and 2027 are supported")
    if set(release["diagnostics"]) != {"normal", "ir_missing", "depth_missing", "both_missing"}:
        raise ValueError("Diagnostic release must contain normal and the three planned missing-modality conditions")
    for arm in ARMS:
        rows = read_json(release["manifests"][arm])
        if len(rows) != release["steps"] * 8:
            raise ValueError(f"{arm}: manifest length differs from the frozen training horizon")
    if read_json(release["manifests"]["B"]) != read_json(release["manifests"]["V"]):
        raise ValueError("B and V must consume exactly the same manifest")


def train_stage(config, release, out, arm, name, *, stop=None, resume=None, init_only=False):
    output = out / arm / name
    env = {
        "QWEN_FINETUNE_DIR": config["qwen_finetune_dir"],
        "PYTHON_EXECUTABLE": config["python"],
        "DATA_ROOT": config["data_root"], "MODEL_PATH": config["model"],
        "ANNOTATION_PATH": release["manifests"][arm], "OUTPUT_DIR": str(output),
        "DATASET_VARIANT": "trimodal", "MAX_STEPS": str(release["steps"]),
        "SEED": str(release["seed"]), "LEARNING_RATE": "5e-6",
        "VISUAL_LORA_LR": "2e-5", "LORA_SCOPE": "language_merger" if arm == "V" else "language",
        "MIN_PIXELS": "200704", "MAX_PIXELS": "602112", "PRESERVE_MANIFEST_ORDER": "1",
        "SAVE_STEPS": "16" if stop else "200", "SAVE_TOTAL_LIMIT": "5",
        "INIT_ADAPTER": "" if resume else config["initial_adapter"],
        "RESUME_FROM_CHECKPOINT": str(resume) if resume else "",
        "STOP_AFTER_STEP": str(stop) if stop else "", "INIT_ONLY": "1" if init_only else "0",
    }
    return {"name": f"{arm}_{name}_{'init' if init_only else stop or 'final'}",
            "command": ["bash", str(Path(config["repo"]) / "scripts/run_qwen3vl_native_lora.sh")],
            "env": env, "cwd": config["repo"], "gpu": True}


def evaluate_stage(config, name, adapter, manifest, gt, output, limit=0):
    command = [config["python"], "-m", "tools.evaluate_pretrained_grounder",
               "--model", config["model"], "--adapter", str(adapter),
               "--manifest", str(manifest), "--target-manifest", str(gt),
               "--data-root", config["data_root"], "--output-dir", str(output),
               "--prompt-style", "native", "--min-pixels", "200704", "--max-pixels", "602112",
               "--max-new-tokens", "128"]
    if limit:
        command += ["--limit", str(limit)]
    return {"name": name, "command": command, "env": {}, "cwd": config["repo"], "gpu": True}


def build_stages(config, release, out, phase, arms=ARMS):
    steps = release["steps"]
    has_diagnostics = release.get("counts", {}).get("approved_diagnostic", 96) > 0
    stages = []
    if phase == "initialize":
        for arm in arms:
            stages.append(train_stage(config, release, out, arm, "initial", init_only=True))
        for arm in ("C0", *arms):
            adapter = config["initial_adapter"] if arm == "C0" else out / arm / "initial"
            stages.append(evaluate_stage(config, f"{arm}_env8", adapter, config["city_manifest"],
                                         config["city_gt"], out / "environment" / arm, limit=8))
    elif phase == "preflight":
        for arm in arms:
            stages.append(train_stage(config, release, out, arm, "continuous", stop=32))
            stages.append(train_stage(config, release, out, arm, "main", stop=16))
            stages.append(train_stage(config, release, out, arm, "main", stop=32,
                                      resume=out / arm / "main/checkpoint-16"))
            stages.append({"name": f"{arm}_compare_resume", "env": {}, "cwd": config["repo"], "gpu": False,
                           "command": [config["python"], "-m", "tools.compare_lora_resume",
                                       "--continuous16", str(out / arm / "continuous/checkpoint-16"),
                                       "--continuous32", str(out / arm / "continuous/checkpoint-32"),
                                       "--resumed16", str(out / arm / "main/checkpoint-16"),
                                       "--resumed32", str(out / arm / "main/checkpoint-32"),
                                       "--expected-groups", "2" if arm == "V" else "1",
                                       "--expected-steps", str(steps),
                                       "--output", str(out / arm / "resume_comparison.json")]})
    elif phase == "train":
        for arm in arms:
            stages.append(train_stage(config, release, out, arm, "main",
                                      resume=out / arm / "main/checkpoint-32"))
    elif phase in ("evaluate", "report"):
        for arm in (("C0", *arms) if phase == "evaluate" else ()):
            adapter = config["initial_adapter"] if arm == "C0" else out / arm / "main" / f"checkpoint-{steps}"
            stages.append(evaluate_stage(config, f"{arm}_city412", adapter, config["city_manifest"],
                                         config["city_gt"], out / "evaluation" / arm / "city412"))
            for condition, manifest in (release["diagnostics"].items() if has_diagnostics else ()):
                stages.append(evaluate_stage(config, f"{arm}_diagnostic_{condition}", adapter,
                                             manifest, release["gt_manifest"],
                                             out / "evaluation" / arm / condition))
        cohorts = [("city412", config["city_gt"])]
        if has_diagnostics:
            cohorts.append(("diagnostics", release["gt_manifest"]))
        for cohort, manifest in cohorts:
            command = [config["python"], "-m", "tools.report_triground_abv", "--manifest", manifest,
                       "--require-run-config", "--output-dir", str(out / "reports" / cohort)]
            atlas_command = [config["python"], "-m", "tools.render_triground_abv_atlas",
                             "--report", str(out / "reports" / cohort / "summary.json"),
                             "--gt", manifest, "--output-dir", str(out / "reports" / cohort / "atlas"),
                             "--native-manifest", config["city_manifest"] if cohort == "city412" else release["diagnostics"]["normal"],
                             "--image-root", config["data_root"]]
            for arm in ("C0", *arms):
                condition = "city412" if cohort == "city412" else "normal"
                run_spec = f"{arm}={out / 'evaluation' / arm / condition / 'predictions.jsonl'}"
                command += ["--run", run_spec]
                atlas_command += ["--run", run_spec]
                if cohort == "diagnostics":
                    for condition in release["diagnostics"]:
                        if condition != "normal":
                            modal_spec = f"{arm}:{condition}={out / 'evaluation' / arm / condition / 'predictions.jsonl'}"
                            command += ["--modal-run", modal_spec]
                            atlas_command += ["--run", modal_spec]
            if cohort == "diagnostics":
                command += ["--scene-map", release["scene_map"], "--class-map", release["class_map"]]
            stages.append({"name": f"report_{cohort}", "command": command, "env": {},
                           "cwd": config["repo"], "gpu": False})
            stages.append({"name": f"atlas_{cohort}", "command": atlas_command, "env": {},
                           "cwd": config["repo"], "gpu": False})
    else:
        raise ValueError(phase)
    return stages


def compare_environment(out, arms=ARMS):
    def rows(arm):
        return [json.loads(line) for line in (out / "environment" / arm / "predictions.jsonl").read_text().splitlines() if line]
    baseline = rows("C0")
    if len(baseline) != 8 or len({r["id"] for r in baseline}) != 8:
        raise ValueError("C0 environment check must have exactly eight unique predictions")
    fields = ("id", "prompt", "prediction", "raw_text", "image_grid_thw", "input_tokens")
    differences = []
    for arm in arms:
        actual = rows(arm)
        if len(actual) != 8:
            differences.append({"arm": arm, "reason": "incomplete"})
            continue
        for expected, observed in zip(baseline, actual):
            changed = [key for key in fields if expected[key] != observed[key]]
            if changed:
                differences.append({"arm": arm, "id": expected["id"], "fields": changed})
    result = {"passed": not differences, "samples": 8, "differences": differences,
              "historical_C_292_reproduced": False}
    write_json(out / "environment_check.json", result)
    if differences:
        raise ValueError("Zero-step outputs differ; see environment_check.json before training")


def load_budget(out):
    path = out / "gpu_budget.json"
    state = read_json(path) if path.exists() else {"limit_seconds": BUDGET_SECONDS, "spent_seconds": 0.0, "stages": []}
    if state["limit_seconds"] != BUDGET_SECONDS or state.get("running"):
        raise ValueError("Budget has an unfinished process or a changed limit; reconcile its actual status first")
    return state


def execute_stage(stage, out, state, budget_dir=None):
    budget_dir = budget_dir or out
    remaining = state["limit_seconds"] - state["spent_seconds"]
    if remaining <= 0 and stage.get("gpu", True):
        raise RuntimeError("The cumulative 12-hour GPU budget is exhausted")
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log = log_dir / (stage["name"] + ".log")
    state["running"] = {"name": stage["name"], "started_epoch": started, "run_dir": str(out)}
    write_json(budget_dir / "gpu_budget.json", state)
    status = "failed"
    with log.open("a", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen(stage["command"], cwd=stage["cwd"],
                                       env={**os.environ, **stage["env"]}, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state["running"]["pid"] = process.pid
            write_json(budget_dir / "gpu_budget.json", state)
            returncode = process.wait(timeout=remaining if stage["gpu"] else None)
            status = "complete" if returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status, returncode = "budget_exhausted", -1
        finally:
            elapsed = time.time() - started
            if stage["gpu"]:
                state["spent_seconds"] += elapsed
            state["stages"].append({"name": stage["name"], "status": status,
                                    "elapsed_seconds": elapsed, "log": str(log),
                                    "gpu": stage["gpu"], "run_dir": str(out)})
            state.pop("running", None)
            write_json(budget_dir / "gpu_budget.json", state)
    if returncode:
        raise RuntimeError(f"{stage['name']} {status}; see {log}")


def forecast_remaining(state, out, release, arms=ARMS):
    """Conservative estimate from this run's 16-step resume and eight-query probes.

    Model-loading time remains in the per-step estimate. A result is an estimate,
    never authority to extend the budget or change a frozen scheduler mid-run.
    """
    records = {s["name"]: s for s in state["stages"]
               if s.get("run_dir") == str(out) and s["status"] == "complete"}
    seconds_per_step = {arm: records[f"{arm}_main_32"]["elapsed_seconds"] / 16 for arm in arms}
    latency_estimates = []
    load_estimates = []
    for arm in ("C0", *arms):
        rows = [json.loads(line) for line in (out / "environment" / arm / "predictions.jsonl").read_text().splitlines() if line]
        generation = sum(row["latency_seconds"] for row in rows)
        latency_estimates.append(generation / len(rows))
        load_estimates.append(max(0.0, records[f"{arm}_env8"]["elapsed_seconds"] - generation))
    # Charge model initialization once per evaluator process, not once per eight queries.
    seconds_per_query = max(latency_estimates) * 1.15
    diagnostic_n = len(read_json(release["gt_manifest"]))
    evaluation_queries = (412 + diagnostic_n * len(release["diagnostics"])) * (len(arms) + 1)
    evaluation_jobs = (1 + len(release["diagnostics"])) * (len(arms) + 1)
    evaluation_seconds = seconds_per_query * evaluation_queries + max(load_estimates) * evaluation_jobs
    available = state["limit_seconds"] - state["spent_seconds"]
    estimates = {}
    for steps in (600, 400):
        # A different horizon starts again and needs its own extra continuous
        # 32-step control; the current horizon already has 32 useful steps.
        remaining_updates = steps - 32 if release["steps"] == steps else steps + 32
        training = sum(seconds_per_step.values()) * remaining_updates
        estimates[str(steps)] = training + evaluation_seconds
    recommendation = next((steps for steps in (600, 400) if estimates[str(steps)] <= available * 0.9), None)
    return {"available_seconds": available, "seconds_per_optimizer_step": seconds_per_step,
            "evaluation_queries": evaluation_queries, "estimated_evaluation_seconds": evaluation_seconds,
            "estimated_core_seconds": estimates, "recommended_steps": recommendation,
            "current_schedule_fits": estimates[str(release["steps"])] <= available * 0.9,
            "safety_margin_fraction": 0.1,
            "note": "Includes model-load overhead; excludes optional second-seed repeat. A 400-step restart needs a new frozen release/output and the SAME budget_dir."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("initialize", "preflight", "train", "evaluate", "report"), required=True)
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the commands only")
    parser.add_argument("--resume", action="store_true", help="Skip completed stages in this new experiment only")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS),
                        help="For seed 2027, select only the shortlisted arm; share config budget_dir")
    args = parser.parse_args()
    config, release = read_json(args.config), read_json(args.release)
    validate_release(release)
    out = args.output_dir.resolve()
    if release["seed"] == 2026 and tuple(args.arms) != ARMS:
        raise ValueError("The first seed must run all A/B/V arms")
    stages = build_stages(config, release, out, args.phase, tuple(args.arms))
    if not args.execute:
        print(json.dumps({"phase": args.phase, "steps": release["steps"], "stages": stages}, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("GPU execution runs on the Linux cloud host; local command preview is supported")
    if args.phase in ("preflight", "train") and not read_json(out / "environment_check.json")["passed"]:
        raise ValueError("The zero-step environment check has not passed")
    if args.phase == "train":
        for arm in args.arms:
            check = read_json(out / arm / "resume_comparison.json")
            if check.get("status") != "pass":
                raise ValueError(f"{arm}: continuous/resumed comparison has not passed")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "execution_config.json"
    record = {"config": config, "release": release, "arms": args.arms}
    if frozen.exists() and read_json(frozen) != record:
        raise ValueError("Execution config differs from this experiment's frozen configuration")
    write_json(frozen, record)
    budget_dir = Path(config["budget_dir"]).resolve()
    state = load_budget(budget_dir)
    if args.phase == "train":
        estimate = forecast_remaining(state, out, release, tuple(args.arms))
        write_json(out / "budget_forecast.json", estimate)
        if not estimate["current_schedule_fits"]:
            raise RuntimeError("Frozen horizon does not fit with full evaluation; see budget_forecast.json. Do not change an active scheduler.")
    completed = {item["name"] for item in state["stages"] if item["status"] == "complete" and item.get("run_dir") == str(out)}
    for stage in stages:
        if stage["gpu"] and stage["name"] in completed:
            if args.resume:
                continue
            raise FileExistsError(f"{stage['name']} already completed; use --resume to skip it")
        execute_stage(stage, out, state, budget_dir)
    if args.phase == "initialize":
        compare_environment(out, tuple(args.arms))


if __name__ == "__main__":
    main()
