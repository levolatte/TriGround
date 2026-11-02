"""Wait for B550/diagnostics, then alternate ACC correction and matched SFT.

This coordinator imports only the standard library. All model work runs in the
configured native Python, after both existing jobs have released the GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


TARGETS = (800, 1600, 2400, 3200, 4000, 4400)
MODES = ("correction", "sft")
SPLITS = {"city378": 378, "external88": 88}
SELECTION = "City378 hits, external88 hits; exact ties keep the earlier checkpoint"
CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors", "loop_state.json",
                    "optimizer.pt", "rng.pt")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def gate_status(b_state: dict | None, diagnostic_state: dict | None) -> tuple[bool, str]:
    if b_state is None:
        return False, "waiting_for_b_state"
    if b_state.get("status") in {"failed", "cancelled"}:
        raise RuntimeError(f"B550 queue ended as {b_state['status']}: {b_state.get('error', '')}")
    if b_state.get("status") == "paused_by_user" and diagnostic_state is not None and diagnostic_state.get("status") == "paused_by_user":
        return True, "ready"
    b550 = b_state.get("results", {}).get("B", {}).get("550", {})
    if b_state.get("status") != "completed" or b550.get("status") != "completed":
        return False, "waiting_for_b550"
    if diagnostic_state is None or diagnostic_state.get("status") not in {
            "completed", "failed", "cancelled", "cancelled_by_user"}:
        return False, "waiting_for_diagnostic"
    if "original_queue_resumed" in diagnostic_state and diagnostic_state["original_queue_resumed"] is not True:
        return False, "waiting_for_original_queue_resume"
    return True, "ready"


def checkpoint_state(path: Path) -> dict | None:
    """Only a complete native optimizer/RNG checkpoint can resume training."""
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in CHECKPOINT_FILES):
        return None
    state = read(path / "loop_state.json")
    if int(state["presentations"]) != int(path.name.removeprefix("checkpoint-")):
        raise RuntimeError(f"Checkpoint presentation counter disagrees with directory: {path}")
    return state


def latest_checkpoint(train: Path, target: int) -> Path | None:
    complete = [(int(path.name.removeprefix("checkpoint-")), path)
                for path in train.glob("checkpoint-*") if checkpoint_state(path) is not None]
    if complete and max(number for number, _ in complete) > target:
        raise RuntimeError(f"A later checkpoint exists before unevaluated target {target}: {train}")
    return max(complete, default=(0, None), key=lambda item: item[0])[1]


def train_command(config: dict, mode: str, segment: Path, train: Path,
                  resume: Path | None) -> list[str]:
    command = [config["python"], str(Path(config["code_root"]) / "tools/train_quant_grounding.py"),
               "--mode", mode, "--config", str(segment), "--output-dir", str(train)]
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def eval_command(config: dict, checkpoint: Path, dataset: Path, output: Path) -> list[str]:
    return [config["python"], str(Path(config["code_root"]) / "tools/eval_quant_grounding.py"),
            "--dataset", str(dataset), "--data-root", config["media_root"],
            "--model", config["model"], "--adapter", str(checkpoint),
            "--engine", "hf", "--precision", "nf4", "--answer-mode", "short",
            "--reward-mode", "acc05", "--format-policy", "punctuation",
            "--samples-per-query", "1", "--hf-max-batch-size", "2", "--seed", "2026",
            "--max-tokens", "64", "--max-length", "8192", "--min-pixels", "200704",
            "--max-pixels", "1204224", "--output-dir", str(output)]


def full_summary(output: Path, expected_ids: set[str], *, require_raw: bool = True) -> dict:
    summary = read(output / "summary.json")
    if (summary["overall"]["queries"] != len(expected_ids) or summary["samples_per_query"] != 1
            or summary["format_policy"] != "punctuation" or summary["answer_mode"] != "short"):
        raise RuntimeError(f"Incomplete evaluation or wrong protocol: {output}")
    filenames = ["predictions.jsonl"] + (["predictions_raw.jsonl"] if require_raw else [])
    for filename in filenames:
        predictions = rows(output / filename)
        if (len(predictions) != len(expected_ids) or {row["id"] for row in predictions} != expected_ids
                or any(len(row["samples"]) != 1 for row in predictions)):
            raise RuntimeError(f"Incomplete single-answer ID coverage: {output / filename}")
    scored = rows(output / "predictions.jsonl")
    hits = sum(row["samples"][0]["hit"] for row in scored)
    if not math.isclose(summary["overall"]["acc05_first"], hits / len(expected_ids), abs_tol=1e-12):
        raise RuntimeError(f"Summary and prediction hits disagree: {output}")
    return {"hits": hits, "queries": len(expected_ids), "acc05": hits / len(expected_ids),
            "seconds": summary["overall"].get("timed_seconds"), "output": str(output),
            "summary": str(output / "summary.json"), "predictions": str(output / "predictions.jsonl")}


def score(result: dict) -> tuple[int, int]:
    return tuple(result[split]["hits"] for split in SPLITS)


def select_best(candidates: list[dict]) -> dict:
    # Iteration is chronological. max keeps the first item for an exact tie.
    return max(candidates, key=score)


def paired_comparison(current: Path, baseline: Path) -> dict:
    current_rows, baseline_rows = rows(current), rows(baseline)
    after = {row["id"]: row["samples"][0]["hit"] for row in current_rows}
    before = {row["id"]: row["samples"][0]["hit"] for row in baseline_rows}
    if len(after) != len(current_rows) or len(before) != len(baseline_rows) or after.keys() != before.keys():
        raise RuntimeError("Paired comparison requires complete, unique, identical ID sets")
    pairs = [{"id": row["id"], "baseline_hit": before[row["id"]], "current_hit": after[row["id"]],
              "change": "rescue" if after[row["id"]] and not before[row["id"]] else
              "regression" if before[row["id"]] and not after[row["id"]] else "unchanged"}
             for row in current_rows]
    return {"queries": len(pairs), "baseline_hits": sum(before.values()), "current_hits": sum(after.values()),
            "baseline_acc05": sum(before.values()) / len(pairs), "current_acc05": sum(after.values()) / len(pairs),
            "rescue": sum(row["change"] == "rescue" for row in pairs),
            "regression": sum(row["change"] == "regression" for row in pairs),
            "net_hits": sum(after.values()) - sum(before.values()), "per_query": pairs}


def run_job(config: dict, state: dict, state_path: Path, label: str, command: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    job = {"label": label, "command": command, "log": str(log), "started_at": now(), "status": "running"}
    state["jobs"].append(job)
    state.update(status="running", current_job=label)
    write(state_path, state)
    started = time.monotonic()
    process = None
    try:
        with log.open("a", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=config["code_root"], stdout=stream, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            write(state_path, state)
            code = process.wait()
        job.update(exit_code=code, status="completed" if code == 0 else "failed")
        if code:
            raise RuntimeError(f"{label} exited {code}; see {log}")
    except BaseException:
        job["status"] = "failed"
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        raise
    finally:
        job.update(ended_at=now(), wall_seconds=time.monotonic() - started)
        job.pop("pid", None)
        write(state_path, state)
    return job["wall_seconds"]


def rebuild_b_best(config: dict, b_state: dict, expected: dict) -> dict:
    initial = config.get("comparisons", {}).get("B59")
    if initial is None:
        raise ValueError("comparisons.B59 is required to rebuild Bbest with hits-only selection")
    sources = [(59, initial)]
    sources += [(int(step), result) for step, result in b_state["results"]["B"].items()
                if result.get("status") == "completed"]
    candidates = [{"legacy_step": step,
                   **{split: full_summary(Path(source[split]), expected[split], require_raw=False)
                      for split in SPLITS}} for step, source in sorted(sources, key=lambda item: item[0])]
    best = select_best(candidates)
    return {**{split: best[split]["output"] for split in SPLITS}, "legacy_step": best["legacy_step"],
            "selection": SELECTION, "selection_candidates": candidates}


def comparison_sources(config: dict, b_state: dict, expected: dict) -> dict:
    sources = {name: source for name, source in config.get("comparisons", {}).items()
               if name not in {"B59", "Bbest"}}
    completed = [(int(step), result) for step, result in b_state["results"]["B"].items()
                 if result.get("status") == "completed"]
    if completed:
        step, result = max(completed, key=lambda item: item[0])
    else:
        step, result = 59, config["comparisons"]["B59"]
    if b_state["results"]["B"].get("550", {}).get("status") != "completed":
        sources.pop("B550", None)
    name = f"B{step}"
    if name not in sources:
        sources[name] = {split: result[split] for split in SPLITS}
    sources["Bbest"] = rebuild_b_best(config, b_state, expected)
    return sources


def archive_baselines(config: dict, output: Path, state: dict, expected: dict, b_state: dict) -> None:
    if "C550" not in config["comparisons"]:
        raise ValueError("comparisons.C550 must provide the complete initialization baseline")
    for name, source in comparison_sources(config, b_state, expected).items():
        if not all((Path(source[split]) / "summary.json").is_file() and
                   (Path(source[split]) / "predictions.jsonl").is_file() for split in SPLITS):
            raise RuntimeError(f"Missing full {name} comparison archive: {source}")
        entry = {"name": name, "presentations": 0, "updates": 0, "training_seconds": 0.0}
        if name == "Bbest":
            entry.update({key: source[key] for key in ("legacy_step", "selection", "selection_candidates")})
        if name == "C550":
            entry["checkpoint"] = read(Path(config["base_train_config"]["correction"]))["init_adapter"]
        for split in SPLITS:
            full_summary(Path(source[split]), expected[split], require_raw=False)
            destination = output / "baselines" / name / split
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("summary.json", "predictions.jsonl", "predictions_raw.jsonl"):
                path = Path(source[split]) / filename
                if path.is_file():
                    shutil.copy2(path, destination / filename)
            entry[split] = full_summary(destination, expected[split], require_raw=False)
        state["baselines"][name] = entry


def update_reports(output: Path, state: dict) -> None:
    entries = list(state["baselines"].values())
    for target in TARGETS:
        entries.extend(state["results"][mode][str(target)] for mode in MODES
                       if state["results"][mode].get(str(target), {}).get("status") == "completed")
    lines = ["# ACC 纠错与 SFT 曲线", "", SELECTION, "",
             "| 状态 | 问题呈现数 | 更新数 | City 命中 / 378 | City ACC@0.5 | 外部命中 / 88 | 外部 ACC@0.5 | 训练秒 | 评价秒 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in entries:
        city, external = result["city378"], result["external88"]
        evaluation_seconds = sum(result[split].get("seconds") or 0 for split in SPLITS)
        lines.append(f"| {result['name']} | {result['presentations']} | {result.get('updates', '—')} | "
                     f"{city['hits']} | {city['acc05']:.2%} | {external['hits']} | {external['acc05']:.2%} | "
                     f"{result.get('training_seconds', 0):.1f} | {evaluation_seconds:.1f} |")
    (output / "curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    all_best = {}
    for mode in MODES:
        candidates = ([state["baselines"]["C550"]] if "C550" in state["baselines"] else [])
        candidates += [state["results"][mode][str(target)] for target in TARGETS
                       if state["results"][mode].get(str(target), {}).get("status") == "completed"]
        if candidates:
            best = dict(select_best(candidates), selection=SELECTION)
            write(output / mode / "best.json", best)
            all_best[mode] = best
    write(output / "best.json", all_best)


def wait_for_existing_jobs(config: dict, output: Path, state: dict) -> dict:
    while True:
        b_path, diagnostic_path = Path(config["b_state"]), Path(config["diagnostic_state"])
        b_state = read(b_path) if b_path.is_file() else None
        diagnostic = read(diagnostic_path) if diagnostic_path.is_file() else None
        ready, status = gate_status(b_state, diagnostic)
        state.update(status=status, current_job=None, gate_checked_at=now(),
                     existing_b_status=b_state.get("status") if b_state else None,
                     diagnostic_status=diagnostic.get("status") if diagnostic else None)
        write(output / "state.json", state)
        if ready:
            return b_state
        time.sleep(min(float(config.get("poll_seconds", 60)), 60))


def run_queue(config: dict, output: Path, expected: dict) -> dict:
    state_path = output / "state.json"
    state = read(state_path) if state_path.is_file() else {
        "status": "starting", "started_at": now(), "jobs": [], "results": {mode: {} for mode in MODES},
        "baselines": {}, "baseline_unavailable": {}, "selection": SELECTION}
    write(output / "queue_config.json", config)
    try:
        b_state = wait_for_existing_jobs(config, output, state)
        archive_baselines(config, output, state, expected, b_state)
        update_reports(output, state)
        if state.get("smoke", {}).get("status") != "completed":
            seconds = run_job(config, state, state_path, "native_gpu_smoke", config["smoke_command"], output / "logs/smoke.log")
            report = read(Path(config["smoke_report"]))
            if report.get("status") != "passed":
                raise RuntimeError(f"Native GPU smoke did not pass: {config['smoke_report']}")
            state["smoke"] = {"status": "completed", "report": config["smoke_report"], "seconds": seconds}
            write(state_path, state)
        elif read(Path(config["smoke_report"])).get("status") != "passed":
            raise RuntimeError("Saved smoke report is no longer passed")
        datasets = {"city378": Path(config["city_dataset"]), "external88": Path(config["external_dataset"])}
        for target in TARGETS:
            for mode in MODES:
                arm_dir = output / mode
                train = arm_dir / "train"
                result = state["results"][mode].setdefault(str(target), {})
                if result.get("status") == "completed":
                    for split in SPLITS:
                        full_summary(Path(result[split]["output"]), expected[split])
                    continue
                checkpoint = train / f"checkpoint-{target}"
                if checkpoint_state(checkpoint) is None:
                    resume = latest_checkpoint(train, target)
                    segment = read(Path(config["base_train_config"][mode]))
                    segment.update(stop_presentations=target, checkpoint_every=400)
                    if mode == "sft":
                        sequence = output / "correction/train/order.jsonl"
                        order = rows(sequence)
                        if len(order) < target or [item["presentation"] for item in order[:target]] != list(range(1, target + 1)):
                            raise RuntimeError(f"Correction sequence does not cover SFT target {target}: {sequence}")
                        segment["sequence_path"] = str(sequence)
                    segment_path = arm_dir / f"configs/to_{target}.json"
                    write(segment_path, segment)
                    seconds = run_job(config, state, state_path, f"{mode}_train_to_{target}",
                                      train_command(config, mode, segment_path, train, resume),
                                      arm_dir / f"logs/train_to_{target}.log")
                    result["segment_seconds"] = result.get("segment_seconds", 0.0) + seconds
                    completed = read(train / "completed.json")
                    if int(completed["presentations"]) != target or Path(completed["checkpoint"]) != checkpoint:
                        raise RuntimeError(f"Training did not complete target {target}: {train}")
                saved = checkpoint_state(checkpoint)
                if saved is None:
                    raise RuntimeError(f"Training did not save a complete checkpoint: {checkpoint}")
                result.update(name=mode, presentations=target, updates=int(saved["updates"]), checkpoint=str(checkpoint))
                result["training_seconds"] = sum(job.get("wall_seconds", 0) for job in state["jobs"]
                                                 if job["label"].startswith(f"{mode}_train_to_")
                                                 and int(job["label"].rsplit("_", 1)[1]) <= target)
                for split in SPLITS:
                    prior = result.get(split, {})
                    if prior.get("status") == "completed":
                        full_summary(Path(prior["output"]), expected[split])
                        continue
                    base = arm_dir / f"eval/presentations-{target}/{split}"
                    attempts = list(base.glob("attempt-*"))
                    attempt = max((int(path.name.removeprefix("attempt-")) for path in attempts), default=0) + 1
                    evaluation = base / f"attempt-{attempt}"
                    run_job(config, state, state_path, f"{mode}_{target}_{split}",
                            eval_command(config, checkpoint, datasets[split], evaluation), evaluation / "evaluation.log")
                    result[split] = dict(full_summary(evaluation, expected[split]), status="completed")
                    write(state_path, state)
                result.update(status="completed", completed_at=now())
                pairs = {name: {split: paired_comparison(Path(result[split]["predictions"]),
                            Path(baseline[split]["predictions"])) for split in SPLITS}
                         for name, baseline in state["baselines"].items()}
                comparison_path = arm_dir / f"comparisons/presentations-{target}.json"
                write(comparison_path, pairs)
                result["comparisons"] = str(comparison_path)
                write(state_path, state)
                update_reports(output, state)
        comparisons = {}
        lines = ["# 完整 ID 配对比较", "", "挽救：基线未命中、当前命中；退步：基线命中、当前未命中。", "",
                 "| 方案 | 检查点 | 基线 | 开发集 | 当前命中 | 当前 ACC@0.5 | 基线命中 | 基线 ACC@0.5 | 挽救 | 退步 |",
                 "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for mode in MODES:
            comparisons[mode] = {}
            checkpoints = {"best": read(output / mode / "best.json"),
                           f"final{TARGETS[-1]}": state["results"][mode][str(TARGETS[-1])]}
            for selected, current in checkpoints.items():
                comparison = {}
                for name, baseline in state["baselines"].items():
                    comparison[name] = {split: paired_comparison(Path(current[split]["predictions"]),
                                                               Path(baseline[split]["predictions"])) for split in SPLITS}
                    for split, pair in comparison[name].items():
                        lines.append(f"| {mode} | {selected} ({current['presentations']}) | {name} | {split} | "
                                     f"{pair['current_hits']} | {pair['current_acc05']:.2%} | {pair['baseline_hits']} | "
                                     f"{pair['baseline_acc05']:.2%} | {pair['rescue']} | {pair['regression']} |")
                comparisons[mode][selected] = comparison
        comparisons["matched_correction_vs_sft"] = {
            str(target): {split: paired_comparison(Path(state["results"]["correction"][str(target)][split]["predictions"]),
                                                  Path(state["results"]["sft"][str(target)][split]["predictions"]))
                          for split in SPLITS} for target in TARGETS}
        selected_correction = read(output / "correction/best.json")
        selected_sft = read(output / "sft/best.json")
        comparisons["best_correction_vs_best_sft"] = {
            split: paired_comparison(Path(selected_correction[split]["predictions"]),
                                     Path(selected_sft[split]["predictions"])) for split in SPLITS}
        write(output / "paired_comparisons.json", comparisons)
        (output / "paired_comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        state.update(status="completed", current_job=None, ended_at=now(), comparisons=str(output / "paired_comparisons.json"))
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", ended_at=now())
        raise
    finally:
        write(state_path, state)
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true", help="CPU-only config, dataset, and dependency-state check")
    args = parser.parse_args(argv)
    config = read(args.config)
    for key in ("python", "code_root", "old_root", "b_state", "diagnostic_state", "base_train_config", "city_dataset",
                "external_dataset", "model", "media_root", "smoke_command", "smoke_report"):
        if key not in config:
            raise ValueError(f"Missing queue configuration: {key}")
    if not isinstance(config["smoke_command"], list) or not config["smoke_command"]:
        raise ValueError("smoke_command must be a nonempty argument list")
    if not 0 < float(config.get("poll_seconds", 60)) <= 60:
        raise ValueError("poll_seconds must be in (0, 60]")
    expected = {}
    for split, key in (("city378", "city_dataset"), ("external88", "external_dataset")):
        dataset = rows(Path(config[key]))
        expected[split] = {row["id"] for row in dataset}
        if len(dataset) != SPLITS[split] or len(expected[split]) != SPLITS[split]:
            raise ValueError(f"{split} must contain exactly {SPLITS[split]} distinct IDs")
    if args.check_only:
        b_path, diagnostic_path = Path(config["b_state"]), Path(config["diagnostic_state"])
        ready, status = gate_status(read(b_path) if b_path.exists() else None,
                                    read(diagnostic_path) if diagnostic_path.exists() else None)
        print(json.dumps({"ready": ready, "status": status, "targets": TARGETS, "modes": MODES,
                          "queries_per_checkpoint": sum(SPLITS.values())}, indent=2))
        return
    run_queue(config, args.output_dir.absolute(), expected)


if __name__ == "__main__":
    main()
