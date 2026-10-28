"""Continue A/B checkpoint-59 to step 550, evaluating both full dev sets every 50 steps.

Run with the original train_env/bin/python. --check-only checks inputs without starting jobs.
Each training segment and evaluation gets a fresh process; elapsed time never stops a job.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path("/root/autodl-tmp/quant_rl_20261026")
DATA_ROOT = Path("/root/autodl-tmp/rematch_20260922/data")
MODEL = "/root/autodl-tmp/rematch_models/Qwen3.6-27B"
TARGETS = tuple(range(100, 551, 50))
SPLITS = {"city378": ("city400_dev.jsonl", 378), "external88": ("external94_dev.jsonl", 88)}
# Exact scores recomputed from evidence/punctuation/{a,b,c}-final-{city378,external88}/predictions.jsonl.
# Ordering: City ACC, external ACC, City mean IoU, external mean IoU; invalid answers have zero IoU.
INITIAL_SCORES = {
    "A": [295 / 378, 85 / 88, 0.6697149564558392, 0.8678023893374338],
    "B": [300 / 378, 85 / 88, 0.6731227471914035, 0.867479967231464],
}
SELECTION = "city378 ACC@0.5, external88 ACC@0.5, city378 mean IoU, external88 mean IoU"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_step(path: Path) -> int:
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in required if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Incomplete resume checkpoint: {path}; missing={missing}")
    return int(read(path / "trainer_state.json")["global_step"])


def segment_config(source: dict, arm: str, output: Path, checkpoint: Path, target: int,
                   args) -> dict:
    if source["arm"] != arm or source["precision"] != "nf4":
        raise ValueError(f"Expected original {arm}/NF4 config")
    config = dict(source)
    config.pop("matched_query_exposures", None)
    config.update(model=args.model, dataset=str(args.train_dataset), data_root=str(args.data_root),
                  adapters=str(args.warmup), output_dir=str(output), resume_from_checkpoint=str(checkpoint),
                  max_steps=target, max_gpu_hours=None, save_hours=[], save_steps=50, save_total_limit=2)
    return config


def eval_command(args, arm: str, checkpoint: Path, dataset: Path, output: Path) -> list[str]:
    return [args.python, str(args.code / "tools/eval_quant_grounding.py"),
            "--dataset", str(dataset), "--data-root", str(args.data_root),
            "--model", args.model, "--adapter", str(checkpoint),
            "--engine", "hf", "--precision", "nf4", "--answer-mode", "short",
            "--reward-mode", "acc05" if arm == "B" else "contrast",
            "--format-policy", "punctuation", "--samples-per-query", "1",
            "--hf-max-batch-size", "2", "--seed", "2026", "--max-tokens", "64",
            "--max-length", "8192", "--min-pixels", "200704", "--max-pixels", "1204224",
            "--output-dir", str(output)]


def full_summary(output: Path, expected_ids: set[str]) -> dict:
    summary = read(output / "summary.json")
    if (summary["overall"]["queries"] != len(expected_ids)
            or summary["samples_per_query"] != 1 or summary["format_policy"] != "punctuation"
            or summary["answer_mode"] != "short"):
        raise RuntimeError(f"Incomplete evaluation or wrong scoring policy: {output}")
    for filename in ("predictions.jsonl", "predictions_raw.jsonl"):
        predictions = rows(output / filename)
        if (len(predictions) != len(expected_ids) or {row["id"] for row in predictions} != expected_ids
                or any(len(row["samples"]) != 1 for row in predictions)):
            raise RuntimeError(f"Incomplete single-answer coverage: {output / filename}")
        if filename == "predictions.jsonl":
            samples = [row["samples"][0] for row in predictions]
            summary["overall"]["mean_iou"] = sum(
                sample["iou"] if sample["score"]["valid"] else 0.0 for sample in samples) / len(samples)
    return summary


def write_curve(output: Path, state: dict) -> None:
    lines = ["# A/B续训曲线", "", "mIoU按完整开发集计算，无效回答记零。", "",
             "| 状态 | 步数 | City ACC@0.5 | 外部 ACC@0.5 | City mIoU | 外部 mIoU |",
             "|---|---:|---:|---:|---:|---:|"]
    entries = [(arm, 59, score) for arm, score in INITIAL_SCORES.items()]
    entries += [("C参考", 550, [303 / 378, 85 / 88, 0.682655, 0.876148])]
    entries += [(arm, target, state["results"][arm][str(target)]["score"])
                for target in TARGETS for arm in "AB"
                if state["results"][arm].get(str(target), {}).get("status") == "completed"]
    for arm, step, score in entries:
        lines.append(f"| {arm} | {step} | {score[0]:.2%} | {score[1]:.2%} | {score[2]:.6f} | {score[3]:.6f} |")
    (output / "curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_job(args, state: dict, state_path: Path, label: str, command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    job = {"label": label, "command": command, "log": str(log), "started_at": now(), "status": "running"}
    state["jobs"].append(job)
    state.update(status="running", current_job=label)
    write(state_path, state)
    started = time.monotonic()
    try:
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=args.code, stdout=stream, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            write(state_path, state)
            code = process.wait()
        job.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code:
            raise RuntimeError(f"{label} exited {code}; see {log}")
    finally:
        job.update(ended_at=now(), wall_seconds=time.monotonic() - started)
        job.pop("pid", None)
        write(state_path, state)


def train_segment(config_path: Path, code: Path) -> None:
    """Use the existing training implementation, with explicit full-resume flags."""
    sys.path.insert(0, str(code))
    from tools import train_quant_grounding as training
    from tools.quant_grounding_config import load_config, prepare_dataset
    from transformers import TrainerCallback

    config = load_config(config_path)
    expected_step = checkpoint_step(Path(config["resume_from_checkpoint"]))
    target = int(config["max_steps"])
    if not expected_step < target:
        raise ValueError(f"Resume step {expected_step} must precede target {target}")
    training.normalize_runtime_optimization_config(config)
    training.configure_deterministic_training(config)
    output = Path(config["output_dir"])
    source = Path(config["dataset"])
    if read(source.parent / "summary.json")["mode"] != "formal":
        raise ValueError("Training requires the original formal dataset")
    os.environ["MAX_PIXELS"] = str(config.get("max_pixels", 1204224))
    os.environ["MIN_PIXELS"] = str(config.get("min_pixels", 200704))
    os.environ["QUANT_REWARD_METRICS_JSONL"] = str(output / "reward_metrics.jsonl")
    prepared = output / "prepared_train.jsonl"
    prepare_dataset(source, prepared, Path(config["data_root"]), config["arm"])
    original_arguments = training.swift_arguments

    def full_resume_arguments(*positional, **keywords):
        # Swift 4.5.3 already defaults to these values. State them explicitly so old
        # args.json cannot replace the new target or switch to weights-only resume.
        return [*original_arguments(*positional, **keywords), "--load_args", "false",
                "--resume_only_model", "false", "--ignore_data_skip", "false"]

    training.swift_arguments = full_resume_arguments

    def hook(trainer):
        class ResumeCheck(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if (state.global_step != expected_step or state.max_steps != target
                        or args.max_steps != target or args.resume_only_model or args.ignore_data_skip):
                    raise RuntimeError("Swift did not apply the expected full-state resume/target")
                write(output / f"resume_to_{target}.json", {
                    "resumed_step": state.global_step, "target_step": state.max_steps,
                    "resume_from_checkpoint": config["resume_from_checkpoint"],
                    "reference_adapter": config["adapters"], "load_args": False,
                    "resume_only_model": args.resume_only_model, "ignore_data_skip": args.ignore_data_skip,
                    "checked_at": now()})

        trainer.add_callback(ResumeCheck())

    training.run_swift(config, prepared, trainer_hook=hook)
    if checkpoint_step(output / f"checkpoint-{target}") != target:
        raise RuntimeError(f"Training did not save the requested step {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a-config", type=Path)
    parser.add_argument("--b-config", type=Path)
    parser.add_argument("--a-checkpoint", type=Path)
    parser.add_argument("--b-checkpoint", type=Path)
    parser.add_argument("--warmup", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-dataset", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--arms", choices=("AB", "B"), default="AB")
    parser.add_argument("--train-segment", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.absolute()
    args.code = (args.code or args.root / "code").absolute()
    if args.train_segment:
        train_segment(args.train_segment, args.code)
        return
    # Do not resolve the virtualenv Python symlink: that changes sys.prefix.
    args.python = str(Path(args.python or args.root / "train_env/bin/python").absolute())
    args.output = (args.output or args.root / "runs/ab550_20261028").absolute()
    args.data_root = args.data_root.absolute()
    dataset_dir = args.data_root / "quant_grounding_20261026_formal"
    args.train_dataset = (args.train_dataset or dataset_dir / "schedule.jsonl").absolute()
    args.warmup = (args.warmup or args.root / "runs/shared_format_warmup/training/checkpoint-50").absolute()
    sources = {"A": args.root / "runs/initial_seed2026/A_seed2026",
               "B": args.root / "runs/optimized_B_seed2026"}
    initial = {arm: (getattr(args, f"{arm.lower()}_checkpoint") or path / "train/checkpoint-59").absolute()
               for arm, path in sources.items()}
    configs = {arm: read(getattr(args, f"{arm.lower()}_config") or path / "arm_config.json")
               for arm, path in sources.items()}
    for arm, path in initial.items():
        if checkpoint_step(path) != 59:
            raise ValueError(f"{arm} must start from its original checkpoint-59: {path}")
    if checkpoint_step(args.warmup) != 50:
        raise ValueError("The frozen reference must remain shared warmup50")
    for path in (Path(args.python), args.code / "tools/train_quant_grounding.py",
                 args.code / "tools/eval_quant_grounding.py", args.train_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {}
    for split, (filename, count) in SPLITS.items():
        dataset = rows(dataset_dir / filename)
        expected[split] = {row["id"] for row in dataset}
        if len(dataset) != count or len(expected[split]) != count:
            raise ValueError(f"{split} must contain {count} distinct queries")
    for arm in "AB":
        segment_config(configs[arm], arm, args.output / arm / "train", initial[arm], 100, args)
    if args.check_only:
        print(json.dumps({"ready": True, "output": str(args.output), "targets": TARGETS,
                          "arms": args.arms,
                          "python": args.python, "initial": {arm: str(path) for arm, path in initial.items()},
                          "reference": str(args.warmup), "hour_cutoff": None,
                          "save_total_limit": 2, "evaluation_queries_per_arm_per_step": 466}, indent=2))
        return

    state_path = args.output / "state.json"
    state = read(state_path) if state_path.exists() else {
        "status": "starting", "started_at": now(), "hour_cutoff": None,
        "initial_checkpoints": {arm: str(path) for arm, path in initial.items()},
        "reference_adapter": str(args.warmup), "jobs": [], "results": {"A": {}, "B": {}}}
    state["active_arms"] = list(args.arms)
    for arm in "AB":
        best_path = args.output / arm / "best.json"
        if not best_path.exists():
            write(best_path, {"global_step": 59, "adapter": str(initial[arm]), "score": INITIAL_SCORES[arm],
                              "selection": SELECTION,
                              "source": "Saved punctuation evaluation: evidence/punctuation/"
                                        f"{arm.lower()}-final-{{city378,external88}}/predictions.jsonl"})
    write_curve(args.output, state)
    try:
        for target in TARGETS:
            for arm in args.arms:
                arm_dir = args.output / arm
                train = arm_dir / "train"
                result = state["results"][arm].setdefault(str(target), {})
                if result.get("status") == "completed":
                    for split in SPLITS:
                        full_summary(Path(result[split]), expected[split])
                    continue
                checkpoint = train / f"checkpoint-{target}"
                if not checkpoint.exists():
                    candidates = [initial[arm], *train.glob("checkpoint-*")]
                    previous = max(candidates, key=checkpoint_step)
                    if checkpoint_step(previous) >= target:
                        raise RuntimeError(f"Missing unevaluated checkpoint-{target}; latest={previous}")
                    config = segment_config(configs[arm], arm, train, previous, target, args)
                    config_path = arm_dir / f"configs/to_{target}.json"
                    write(config_path, config)
                    run_job(args, state, state_path, f"{arm}_train_to_{target}",
                            [args.python, str(Path(__file__).absolute()), "--code", str(args.code),
                             "--train-segment", str(config_path)], arm_dir / f"logs/train_to_{target}.log")
                if checkpoint_step(checkpoint) != target:
                    raise RuntimeError(f"Wrong checkpoint step: {checkpoint}")
                result.update(checkpoint=str(checkpoint), global_step=target)
                summaries = {}
                for split, (filename, _) in SPLITS.items():
                    base = arm_dir / f"eval/step-{target}/{split}"
                    attempts = sorted(base.glob("attempt-*"), key=lambda path: int(path.name.split("-")[-1]))
                    complete = [path for path in attempts if (path / "summary.json").is_file()]
                    if complete:
                        output = complete[-1]
                    else:
                        # Keep partial raw answers from an interrupted evaluation; start a fresh process/output.
                        output = base / f"attempt-{len(attempts) + 1}"
                        run_job(args, state, state_path, f"{arm}_{target}_{split}",
                                eval_command(args, arm, checkpoint, dataset_dir / filename, output),
                                output / "evaluation.log")
                    summaries[split] = full_summary(output, expected[split])
                    result[split] = str(output)
                    write(state_path, state)
                score = ([summaries[split]["overall"]["acc05_first"] for split in SPLITS]
                         + [summaries[split]["overall"]["mean_iou"] for split in SPLITS])
                best_path = arm_dir / "best.json"
                best = read(best_path) if best_path.exists() else None
                if best is None or score > best["score"]:
                    best_adapter = arm_dir / "best_adapter"
                    best_adapter.mkdir(exist_ok=True)
                    for filename in ("adapter_config.json", "adapter_model.safetensors"):
                        shutil.copy2(checkpoint / filename, best_adapter / filename)
                    write(best_path, {"global_step": target, "adapter": str(best_adapter), "score": score,
                                      "selection": SELECTION,
                                      "summaries": {split: str(Path(result[split]) / "summary.json") for split in SPLITS}})
                result.update(status="completed", score=score, completed_at=now())
                write(state_path, state)
                write_curve(args.output, state)
        state.update(status="completed", current_job=None, ended_at=now())
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", ended_at=now())
        raise
    finally:
        write(state_path, state)


if __name__ == "__main__":
    main()
