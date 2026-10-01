"""Serial T/M preflight, training, and evaluation under a separate 12-hour budget.

Commands are printed by default. Execute only a frozen, reviewed release on Linux.
The optional IR two-pass gate is reserved for a later measured decision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from tools.run_triground_abv import BUDGET_SECONDS, execute_stage, load_budget, read_json, write_json


ARMS = ("T", "M")
DIAGNOSTICS = ("normal", "ir_missing", "depth_missing", "both_missing")
INITIAL_STEP_SECONDS = 18.0  # A/B/V measured about 12 s at accumulation 8; allow for 10.
INITIAL_QUERY_SECONDS = 2.0  # A/B/V full-evaluation estimate was about 1.3 s/query.
INITIAL_LOAD_SECONDS = 100.0


def count_records(path: str) -> int:
    path = Path(path)
    if path.suffix == ".jsonl":
        return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines())
    return len(read_json(path))


def validate_release(release: dict) -> None:
    if release.get("status") != "ready":
        raise ValueError("T/M data release must be reviewed and ready")
    if release["steps"] not in (400, 600) or release["seed"] not in (2028, 2029):
        raise ValueError("T/M release requires a frozen 400/600-step horizon and seed 2028/2029")
    if set(release["manifests"]) != set(ARMS):
        raise ValueError("T/M release must contain both training manifests")
    for arm in ARMS:
        if count_records(release["manifests"][arm]) != release["steps"] * 10:
            raise ValueError(f"{arm} manifest must contain exactly steps × 10 samples")
    if set(release["diagnostics"]) != set(DIAGNOSTICS):
        raise ValueError("T/M release must contain normal and three missing-modality diagnostics")
    if set(release["probes"]) != {"ir_read", "depth_read"}:
        raise ValueError("T/M release must contain IR and depth reading probes")
    for path in release["diagnostics"].values():
        if count_records(path) == 0:
            raise ValueError(f"empty diagnostic manifest: {path}")


def validate_config(config: dict) -> None:
    required = ("repo", "python", "qwen_finetune_dir", "data_root", "model",
                "initial_adapter", "city_manifest", "city_gt", "budget_dir",
                "baseline_adapters")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"T/M run config lacks {missing}")
    if set(config["baseline_adapters"]) != {"C", "A", "B", "V"}:
        raise ValueError("baseline_adapters must identify historical C/A/B/V")
    if config["initial_adapter"] != config["baseline_adapters"]["C"]:
        raise ValueError("T/M initial adapter must be the historical C adapter")
    if "triground_abv_20260927/gpu_budget" in str(config["budget_dir"]):
        raise ValueError("T/M requires a new 12-hour budget_dir, separate from A/B/V")


def train_stage(config: dict, release: dict, out: Path, arm: str, name: str,
                *, stop: int | None = None, resume: Path | None = None,
                save_steps: int = 100) -> dict:
    env = {
        "QWEN_FINETUNE_DIR": config["qwen_finetune_dir"],
        "PYTHON_EXECUTABLE": config["python"],
        "DATA_ROOT": config["data_root"], "MODEL_PATH": config["model"],
        "ANNOTATION_PATH": release["manifests"][arm],
        "OUTPUT_DIR": str(out / arm / name), "DATASET_VARIANT": "trimodal",
        "MAX_STEPS": str(release["steps"]), "SEED": str(release["seed"]),
        "LEARNING_RATE": "5e-6", "VISUAL_LORA_LR": "2e-5",
        "LORA_SCOPE": "language", "MIN_PIXELS": "200704", "MAX_PIXELS": "602112",
        "PRESERVE_MANIFEST_ORDER": "1", "SAVE_STEPS": str(save_steps),
        "SAVE_TOTAL_LIMIT": "8", "GRADIENT_ACCUMULATION_STEPS": "10",
        "LOSS_REDUCTION": "sample_mean", "CHECKPOINT_STEPS": "2,4" if arm == "M" else "",
        "INIT_ADAPTER": "" if resume else config["initial_adapter"],
        "RESUME_FROM_CHECKPOINT": str(resume) if resume else "",
        "STOP_AFTER_STEP": str(stop) if stop else "",
        "AUDIT_SAMPLE_COUNT": "4" if stop else "1",
    }
    return {"name": f"{arm}_{name}_{stop or 'final'}", "gpu": True,
            "command": ["bash", str(Path(config["repo"]) / "scripts/run_qwen3vl_native_lora.sh")],
            "env": env, "cwd": config["repo"],
            "expected_artifact": str(out / arm / name / f"checkpoint-{stop or release['steps']}"),
            "updates": (stop if not resume else stop - int(resume.name.split("-")[-1]))
            if stop else release["steps"] - (int(resume.name.split("-")[-1]) if resume else 0)}


def grounding_stage(config: dict, name: str, adapter: str | Path, manifest: str,
                    gt: str, output: Path, *, resume: bool) -> dict:
    command = [config["python"], "-m", "tools.evaluate_pretrained_grounder",
               "--model", config["model"], "--adapter", str(adapter),
               "--manifest", manifest, "--target-manifest", gt,
               "--data-root", config["data_root"], "--output-dir", str(output),
               "--prompt-style", "native", "--inference-mode", "direct",
               "--min-pixels", "200704", "--max-pixels", "602112",
               "--max-new-tokens", "128"]
    if resume:
        command.append("--resume")
    return {"name": name, "gpu": True, "command": command, "env": {},
            "cwd": config["repo"], "queries": count_records(manifest),
            "expected_artifact": str(output)}


def reading_stage(config: dict, out: Path, arm: str, adapter: Path,
                  probe: str, manifest: str) -> dict:
    queries = count_records(manifest)
    command = [config["python"], "-m", "tools.evaluate_aux_reading",
               "--model", config["model"], "--adapter", str(adapter),
               "--manifest", manifest, "--data-root", config["data_root"],
               "--output-dir", str(out / "evaluation" / arm / probe),
               "--min-pixels", "200704", "--max-pixels", "602112",
               "--max-new-tokens", "64"]
    return {"name": f"{arm}_{probe}", "gpu": bool(queries), "command": command,
            "env": {}, "cwd": config["repo"], "queries": queries,
            "expected_artifact": str(out / "evaluation" / arm / probe)}


def build_stages(config: dict, release: dict, out: Path, phase: str,
                 *, resume_evaluation: bool = False) -> list[dict]:
    steps = release["steps"]
    m_main = out / "M" / "main"
    if phase == "preflight":
        return [
            train_stage(config, release, out, "M", "continuous", stop=4),
            train_stage(config, release, out, "M", "main", stop=2),
            train_stage(config, release, out, "M", "main", stop=4,
                        resume=m_main / "checkpoint-2"),
            {"name": "M_compare_resume", "gpu": False, "env": {}, "cwd": config["repo"],
             "expected_artifact": str(out / "M/resume_comparison.json"),
             "command": [config["python"], "-m", "tools.compare_lora_resume",
                         "--continuous-split", str(out / "M/continuous/checkpoint-2"),
                         "--continuous-final", str(out / "M/continuous/checkpoint-4"),
                         "--resumed-split", str(m_main / "checkpoint-2"),
                         "--resumed-final", str(m_main / "checkpoint-4"),
                         "--split-step", "2", "--final-step", "4",
                         "--microbatches-per-step", "10", "--expected-groups", "1",
                         "--expected-steps", str(steps),
                         "--output", str(out / "M/resume_comparison.json")]},
        ]
    if phase == "train":
        return [
            train_stage(config, release, out, "M", "main",
                        resume=m_main / "checkpoint-4"),
            train_stage(config, release, out, "T", "main"),
        ]
    if phase == "evaluate":
        stages = []
        for arm in ARMS:
            adapter = out / arm / "main" / f"checkpoint-{steps}"
            stages.append(grounding_stage(config, f"{arm}_city412", adapter,
                                          config["city_manifest"], config["city_gt"],
                                          out / "evaluation" / arm / "city412",
                                          resume=resume_evaluation))
            for condition in DIAGNOSTICS:
                stages.append(grounding_stage(
                    config, f"{arm}_diagnostic_{condition}", adapter,
                    release["diagnostics"][condition], release["gt_manifest"],
                    out / "evaluation" / arm / condition, resume=resume_evaluation,
                ))
            for probe in ("ir_read", "depth_read"):
                stages.append(reading_stage(config, out, arm, adapter, probe,
                                            release["probes"][probe]))
        for arm in ("C", "A"):
            for condition in DIAGNOSTICS:
                stages.append(grounding_stage(
                    config, f"{arm}_diagnostic_{condition}",
                    config["baseline_adapters"][arm], release["diagnostics"][condition],
                    release["gt_manifest"], out / "evaluation" / arm / condition,
                    resume=resume_evaluation,
                ))
        return stages
    raise ValueError(phase)


def complete_checkpoints(output: Path) -> list[Path]:
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors")
    checkpoints = []
    for checkpoint in output.glob("checkpoint-*"):
        if not checkpoint.is_dir() or not all((checkpoint / name).is_file() for name in required):
            continue
        if not list(checkpoint.glob("rng_state*.pth")):
            continue
        step = int(checkpoint.name.split("-")[-1])
        if read_json(checkpoint / "trainer_state.json")["global_step"] == step:
            checkpoints.append(checkpoint)
    return sorted(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def assert_stage_artifact(stage: dict) -> None:
    path = Path(stage["expected_artifact"])
    if "updates" in stage:
        if path not in complete_checkpoints(path.parent):
            raise RuntimeError(f"completed training stage lacks a full checkpoint: {path}")
    elif stage["name"] == "M_compare_resume":
        if read_json(path)["status"] != "pass":
            raise RuntimeError("completed resume comparison is not passing")
    elif not (path / "summary.json").is_file() or count_records(path / "predictions.jsonl") != stage["queries"]:
        raise RuntimeError(f"completed evaluation is incomplete: {path}")


def budget_forecast(config: dict, release: dict, out: Path, state: dict,
                    stages: list[dict]) -> dict:
    completed = {row["name"] for row in state["stages"]
                 if row["status"] == "complete" and row.get("run_dir") == str(out)}
    measured = next((row for row in state["stages"]
                     if row["name"] == "M_continuous_4" and row["status"] == "complete"
                     and row.get("run_dir") == str(out)), None)
    step_seconds = float(config.get("initial_step_seconds", INITIAL_STEP_SECONDS))
    load_seconds = float(config.get("initial_model_load_seconds", INITIAL_LOAD_SECONDS))
    source = "historical A/B/V with accumulation-10 allowance"
    if measured:
        trainer_state = out / "M/continuous/trainer_state.json"
        history = read_json(trainer_state).get("log_history", []) if trainer_state.is_file() else []
        runtimes = [row["train_runtime"] for row in history if "train_runtime" in row]
        if runtimes:
            train_seconds = float(runtimes[-1])
        else:
            log_text = Path(measured["log"]).read_text(encoding="utf-8")
            printed = re.findall(r"['\"]train_runtime['\"]\s*:\s*['\"]?([0-9.]+)", log_text)
            train_seconds = float(printed[-1]) if printed else measured["elapsed_seconds"]
        step_seconds = 1.15 * train_seconds / 4
        load_seconds = max(load_seconds, 1.15 * max(0.0, measured["elapsed_seconds"] - train_seconds))
        source = "M continuous 4-step measured runtime with 15% allowance"
    query_seconds = float(config.get("initial_query_seconds", INITIAL_QUERY_SECONDS))
    pending = [stage for stage in stages if stage["gpu"] and stage["name"] not in completed]
    training = [stage for stage in pending if "updates" in stage]
    evaluation = [stage for stage in pending if "queries" in stage]
    training_updates = 0
    for stage in training:
        updates = stage["updates"]
        if stage["name"].endswith("_main_final"):
            checkpoints = complete_checkpoints(Path(stage["expected_artifact"]).parent)
            if checkpoints:
                updates = release["steps"] - int(checkpoints[-1].name.split("-")[-1])
        training_updates += updates
    estimated = (step_seconds * training_updates
                 + query_seconds * sum(stage["queries"] for stage in evaluation)
                 + load_seconds * len(pending))
    available = state["limit_seconds"] - state["spent_seconds"]
    return {"horizon": release["steps"], "available_seconds": available,
            "estimated_required_seconds": estimated, "fits_with_10_percent_margin": estimated <= available * 0.9,
            "step_seconds": step_seconds, "query_seconds": query_seconds,
            "process_load_seconds": load_seconds, "rate_source": source,
            "pending_training_updates": training_updates,
            "pending_evaluation_queries": sum(stage["queries"] for stage in evaluation),
            "pending_gpu_jobs": len(pending), "safety_margin_fraction": 0.1,
            "note": "Estimate only. If the frozen horizon does not fit, stop and decide whether a new 400-step release/run is warranted; never change MAX_STEPS in place."}


def run_phase(config: dict, release: dict, out: Path, phase: str, *, resume: bool,
              retry_reason: str, state: dict, budget_dir: Path, all_stages: list[dict]) -> None:
    if phase == "train" and read_json(out / "M/resume_comparison.json")["status"] != "pass":
        raise ValueError("M 4 versus 2+2 checkpoint comparison must pass before formal training")
    if phase == "evaluate":
        for arm in ARMS:
            checkpoint = out / arm / "main" / f"checkpoint-{release['steps']}"
            if not (checkpoint / "trainer_state.json").is_file():
                raise FileNotFoundError(f"formal {arm} checkpoint is incomplete: {checkpoint}")
    forecast = budget_forecast(config, release, out, state, all_stages)
    write_json(out / "budget_forecast.json", forecast)
    if not forecast["fits_with_10_percent_margin"]:
        raise RuntimeError("Frozen horizon plus required evaluations exceeds 90% of remaining GPU budget; see budget_forecast.json")
    completed = {row["name"] for row in state["stages"]
                 if row["status"] == "complete" and row.get("run_dir") == str(out)}
    for stage in build_stages(config, release, out, phase, resume_evaluation=True):
        if stage["name"] in completed:
            if resume:
                assert_stage_artifact(stage)
                continue
            raise FileExistsError(f"{stage['name']} completed already; use --resume")
        failures = sum(row["name"] == stage["name"] and row["status"] != "complete"
                       and row.get("run_dir") == str(out) for row in state["stages"])
        if failures and not retry_reason:
            raise RuntimeError(f"{stage['name']} failed previously; diagnose and pass --retry-reason")
        if failures >= 3:
            raise RuntimeError(f"{stage['name']} exhausted the original attempt plus two diagnosed retries")
        if phase == "preflight" and "updates" in stage:
            target = Path(stage["expected_artifact"])
            if target in complete_checkpoints(target.parent):
                raise RuntimeError(f"checkpoint exists without a completed stage record; inspect before retry: {target}")
        if phase == "train" and resume:
            main_dir = out / stage["name"][0] / "main"
            checkpoints = complete_checkpoints(main_dir)
            if checkpoints:
                latest = checkpoints[-1]
                latest_step = int(latest.name.split("-")[-1])
                if latest_step >= release["steps"]:
                    raise RuntimeError(f"final checkpoint exists but training stage is not recorded complete: {latest}")
                stage["env"]["RESUME_FROM_CHECKPOINT"] = str(latest)
                stage["env"]["INIT_ADAPTER"] = ""
        execute_stage(stage, out, state, budget_dir)
        assert_stage_artifact(stage)
        if stage["name"] == "M_continuous_4":
            forecast = budget_forecast(config, release, out, state, all_stages)
            write_json(out / "budget_forecast.json", forecast)
            if not forecast["fits_with_10_percent_margin"]:
                raise RuntimeError("M 4-step measured forecast does not fit the frozen horizon; see budget_forecast.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True,
                        choices=("preflight", "train", "evaluate", "run-all", "ir-two-pass"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-reason", default="")
    args = parser.parse_args()
    config, release = read_json(args.config), read_json(args.release)
    validate_config(config)
    validate_release(release)
    out = args.output_dir.resolve()
    if args.phase == "ir-two-pass":
        raise NotImplementedError("IR two-pass gate requires measured A/M diagnostic comparison and is not in the automatic T/M queue")
    phases = ("preflight", "train", "evaluate") if args.phase == "run-all" else (args.phase,)
    all_stages = [stage for phase in ("preflight", "train", "evaluate")
                  for stage in build_stages(config, release, out, phase, resume_evaluation=True)]
    if not args.execute:
        preview = [stage for phase in phases
                   for stage in build_stages(config, release, out, phase,
                                             resume_evaluation=True)]
        forecast = budget_forecast(
            config, release, out, load_budget(Path(config["budget_dir"])), all_stages
        )
        print(json.dumps({"phases": phases, "steps": release["steps"],
                          "budget_forecast": forecast, "stages": preview},
                         ensure_ascii=False, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("GPU execution requires the Linux cloud host; local dry-run is supported")
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "task_manifest.json"
    record = read_json(frozen) if frozen.exists() else {
        "config": config, "release": release, "stages": {}
    }
    if record["config"] != config or record["release"] != release:
        raise ValueError("T/M task manifest differs from the frozen run")
    for stage in all_stages:
        previous = record["stages"].get(stage["name"])
        if previous is not None and previous != stage:
            raise ValueError(f"T/M task stage changed after freezing: {stage['name']}")
        record["stages"][stage["name"]] = stage
    write_json(frozen, record)
    budget_dir = Path(config["budget_dir"]).resolve()
    state = load_budget(budget_dir)
    if state["limit_seconds"] != BUDGET_SECONDS:
        raise ValueError("T/M GPU budget must be an independent 12-hour ledger")
    for phase in phases:
        run_phase(config, release, out, phase, resume=args.resume,
                  retry_reason=args.retry_reason, state=state,
                  budget_dir=budget_dir, all_stages=all_stages)


if __name__ == "__main__":
    main()
