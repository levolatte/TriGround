"""Run the approved D repair sequence once, without experiment-hour cutoffs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from evaluate_warmup50_baseline import now, read, record_wait, write


ROOT = Path("/root/autodl-tmp/quant_rl_20261026")
DATA = Path("/root/autodl-tmp/rematch_20260922/data")
MODEL = "/root/autodl-tmp/rematch_models/Qwen3.6-27B"
DATASETS = {"city378": ("city400_dev.jsonl", 378), "external88": ("external94_dev.jsonl", 88)}
SCORE_FIELDS = ("valid", "format_valid", "truncated", "iou", "max_negative_iou",
                "hit", "contrast", "reward")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def coverage(rows, dataset, samples):
    expected = {row["id"] for row in dataset}
    if len(expected) != len(dataset) or len(rows) != len(dataset) or {row["id"] for row in rows} != expected:
        raise RuntimeError("Prediction IDs must exactly cover the fixed dataset")
    targets = {row["id"]: row["solution"] for row in dataset}
    for row in rows:
        if len(row["samples"]) != samples:
            raise RuntimeError(f"{row['id']}: expected {samples} answers")
        if "solution" in row and row["solution"] != targets[row["id"]]:
            raise RuntimeError(f"{row['id']}: prediction GT differs from the fixed dataset")


def metrics(rows):
    samples = [sample for row in rows for sample in row["samples"]]
    count = len(samples)
    valid = sum(bool(sample["score"]["valid"]) for sample in samples)
    hits = sum(bool(sample["hit"]) for sample in samples)
    return {"queries": len(rows), "answers": count, "valid": valid, "valid_rate": valid / count,
            "invalid": count - valid, "truncated": sum(sample["score"]["truncated"] for sample in samples),
            "hits": hits, "acc05": hits / count,
            "miou": sum(float(sample["iou"]) if sample["score"]["valid"] else 0.0
                        for sample in samples) / count,
            "miou_invalid_value": 0.0}


def export_first_rollout(dataset, rollout_path, output):
    """Reuse the actual on-policy first batch; never generate a second 256 baseline."""
    from tools.eval_quant_grounding import score_rows, summarize

    exported = read_rows(rollout_path)
    if len(dataset) != 4 or len(exported) != 32:
        raise RuntimeError("The 256 baseline requires the complete first 4xG8 training rollout")
    by_id = {row["id"]: row for row in dataset}
    groups = {row["id"]: {} for row in dataset}
    for item in exported:
        sample_id, index = item["id"], item["sample_index"]
        if sample_id not in by_id or item["dataset_record"] != by_id[sample_id]:
            raise RuntimeError("First-rollout dataset record differs from the fixed train4 record")
        if index not in range(8) or index in groups[sample_id]:
            raise RuntimeError(f"{sample_id}: invalid or duplicate first-rollout sample index")
        if item["scored"]["id"] != sample_id or item["raw_text"] != item["scored"]["generated_text"]:
            raise RuntimeError("First-rollout text or ID differs from the actual reward log")
        if item["finish_reason"] != item["scored"]["finish_reason"]:
            raise RuntimeError("First-rollout finish reason differs from the actual reward log")
        groups[sample_id][index] = item
    if any(set(group) != set(range(8)) for group in groups.values()):
        raise RuntimeError("The first training batch must contain eight answers for each train4 ID")
    raw = [{"id": row["id"], "samples": [
        {"text": item["raw_text"], "finish_reason": item["finish_reason"],
         "tokens": len(item["token_ids"]), "seconds": None,
         "token_ids": item["token_ids"], "thinking_trace": item["thinking_trace"]}
        for item in (groups[row["id"]][index] for index in range(8))]} for row in dataset]
    scored = score_rows(dataset, raw, "thinking", "contrast", "single_box")
    for row in scored:
        for index, sample in enumerate(row["samples"]):
            logged = groups[row["id"]][index]["scored"]
            for key in SCORE_FIELDS:
                actual, expected = sample["score"][key], logged[key]
                equal = (math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                         if isinstance(actual, float) and expected is not None else actual == expected)
                if not equal:
                    raise RuntimeError(f"{row['id']}/{index}: CPU score {key} differs from training reward")
    output.mkdir()
    write_rows(output / "predictions_raw.jsonl", raw)
    write_rows(output / "predictions.jsonl", scored)
    summary = summarize(scored)
    summary.update(answer_mode="thinking", reward_mode="contrast", format_policy="single_box",
                   samples_per_query=8, bounded_thinking={"thought_limit": 256, "answer_limit": 64},
                   resource={"source": str(rollout_path), "rescore_only": True,
                             "reward_fields_checked": list(SCORE_FIELDS), "reused_first_training_batch": True})
    write(output / "summary.json", summary)
    return scored


def select_setting(rows256, rows1024):
    from tools.run_quant_d_repair_gate import gate_summary

    settings = {}
    for budget, rows in ((256, rows256), (1024, rows1024)):
        value = metrics(rows)
        value["qualified"] = value["valid_rate"] >= 0.95 and value["truncated"] == 0
        value["geometry_gate"] = gate_summary(rows, min_valid_rate=0.95,
                                               max_truncated_rate=0.0, min_geometry_groups=2)
        settings[str(budget)] = value
    candidates = [budget for budget in (256, 1024) if settings[str(budget)]["qualified"]]
    winner = max(candidates, key=lambda budget: (settings[str(budget)]["hits"],
                 settings[str(budget)]["miou"], -budget)) if candidates else None
    signal = winner is not None and settings[str(winner)]["geometry_gate"]["geometry_reward_groups"] >= 2
    return {"status": "selected" if signal else "no_signal" if winner else "no_qualified_setting",
            "selection_order": ["valid_rate>=0.95 and truncated=0", "hits", "miou_invalid_zero", "prefer_256"],
            "settings": settings, "thought_limit": winner, "answer_limit": 64,
            "formal_training_allowed": signal, "reward_difference_used_for_ranking": False}


def check_probe(report, output):
    if report["status"] != "passed" or not report["train_gate"]["passed"]:
        raise RuntimeError(f"{output.name}: actual training gate did not pass")
    update, saved = report["parameter_update"], report["saved_adapter_tensor_check"]
    if min(update["changed_tensors"], update["changed_elements"], update["max_abs_delta"]) <= 0:
        raise RuntimeError("Probe did not change LoRA parameters")
    if not saved["saved_values_equal"] or saved["saved_tensor_count"] <= 0:
        raise RuntimeError("Probe saved tensors differ from post-update parameters")
    checkpoint = output / "checkpoint-1"
    if read(checkpoint / "trainer_state.json")["global_step"] != 1:
        raise RuntimeError("Probe checkpoint must contain exactly one completed optimizer step")
    if not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "adapter_model.safetensors")
    return checkpoint


def paired_report(dataset, before, after):
    coverage(before, dataset, 1)
    coverage(after, dataset, 1)
    prior = {row["id"]: row for row in before}
    current = {row["id"]: row for row in after}
    items = []
    for row in dataset:
        left, right = prior[row["id"]]["samples"][0], current[row["id"]]["samples"][0]
        left_iou = float(left["iou"]) if left["score"]["valid"] else 0.0
        right_iou = float(right["iou"]) if right["score"]["valid"] else 0.0
        change = "rescue" if right["hit"] and not left["hit"] else (
            "regression" if left["hit"] and not right["hit"] else "unchanged")
        items.append({"id": row["id"], "solution": row["solution"], "change": change,
                      "before_hit": left["hit"], "after_hit": right["hit"],
                      "before_valid": left["score"]["valid"], "after_valid": right["score"]["valid"],
                      "before_truncated": left["score"]["truncated"], "after_truncated": right["score"]["truncated"],
                      "before_iou": left_iou, "after_iou": right_iou, "iou_change": right_iou - left_iou,
                      "before_bbox": left["bbox"], "after_bbox": right["bbox"]})
    rescues = [item["id"] for item in items if item["change"] == "rescue"]
    regressions = [item["id"] for item in items if item["change"] == "regression"]
    return {"queries": len(dataset), "ids_and_gt_aligned": True, "before": metrics(before),
            "after": metrics(after), "rescues": len(rescues), "regressions": len(regressions),
            "rescue_ids": rescues, "regression_ids": regressions, "items": items}


def eval_command(code, data, model, dataset, adapter, output, samples, thought_limit):
    return [sys.executable, str(code / "tools/eval_quant_grounding.py"),
            "--dataset", str(dataset), "--data-root", str(data), "--model", model,
            "--adapter", str(adapter), "--engine", "hf", "--precision", "nf4",
            "--answer-mode", "thinking", "--bounded-thinking", "--thought-limit", str(thought_limit),
            "--answer-limit", "64", "--format-policy", "single_box", "--reward-mode", "contrast",
            "--samples-per-query", str(samples), "--hf-max-batch-size", "2", "--seed", "2026",
            "--max-length", "8192", "--min-pixels", "200704", "--max-pixels", "1204224",
            "--output-dir", str(output)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    root, data = args.root, args.data_root
    code = root / "code"
    run = args.run_dir or root / "runs/d_repair_20261028"
    warmup = root / "runs/shared_format_warmup/training/checkpoint-50"
    ledger = root / "runs/budget.json"
    state_path = run / "state.json"
    sys.path.insert(0, str(code))
    from tools.eval_quant_grounding import score_rows, summarize

    # The caller waits for fair evaluation; this script only checks readiness once.
    if read(root / "runs/fair_evaluation_state.json")["status"] != "completed":
        raise RuntimeError("Complete all four fair evaluations before launching D repair")
    if any(row["status"] == "running" for row in read(ledger)):
        raise RuntimeError("An experiment is already running")
    gpu_jobs = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    if gpu_jobs:
        raise RuntimeError(f"GPU already in use: {gpu_jobs}")
    if state_path.exists():
        raise FileExistsError("D repair state already exists; this runner never resumes automatically")
    train4, train12 = read_rows(run / "train4.jsonl"), read_rows(run / "train12.jsonl")
    if len(train4) != 4 or len({row["id"] for row in train4}) != 4:
        raise RuntimeError("train4 must contain exactly four unique fixed queries")
    if len(train12) != 12 or len({row["id"] for row in train12}) != 12:
        raise RuntimeError("train12 must contain exactly twelve unique queries")
    if {row["id"] for row in train4} & {row["id"] for row in train12}:
        raise RuntimeError("Reload12 must exclude the four one-step probe queries")
    config = read(code / "configs/quant_grounding/arm_d_repair.json")
    if Path(config["adapters"]).resolve() != warmup.resolve() or config.get("resume_from_checkpoint"):
        raise ValueError("All D training must restart from the shared warmup50 adapter")
    if config["max_gpu_hours"] is not None or config["thought_limit"] != 256 or config["answer_limit"] != 64:
        raise ValueError("D repair template must use 256/64 and no experiment-hour cutoff")
    state = {"status": "running", "started_at": now(), "completed_stages": [],
             "hour_cutoff": None, "instance_power_action": "none"}
    write(state_path, state)

    @contextmanager
    def stage(name, command=None):
        record_wait(ledger)
        state.update(stage=name, stage_started_at=now())
        if command is not None:
            state["command"] = command
        else:
            state.pop("command", None)
        write(state_path, state)
        rows = read(ledger)
        entry = {"stage": "d_repair", "label": name, "gpu_model": "RTX4090-48GB", "gpu_count": 1,
                 "status": "running", "gpu_hours": 0.0, "started_at": now(),
                 "price_per_gpu_hour": None, "cost": None}
        if command is not None:
            entry["command"] = command
        rows.append(entry)
        write(ledger, rows)
        started = time.monotonic()
        try:
            yield entry
            entry.setdefault("exit_code", 0)
            entry["status"] = "completed" if entry["exit_code"] == 0 else "failed"
            if entry["status"] == "completed":
                state["completed_stages"].append(name)
        except BaseException:
            entry["status"] = "failed"
            raise
        finally:
            elapsed = time.monotonic() - started
            entry.update(ended_at=now(), wall_seconds=elapsed, gpu_hours=elapsed / 3600)
            write(ledger, rows)
            state["stage_status"] = entry["status"]
            state.pop("child_pid", None)
            write(state_path, state)

    def launch(name, command, allow_failed_probe=False):
        with stage(name, command) as entry:
            with (run / (name + ".log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=code, stdout=log, stderr=subprocess.STDOUT)
                state["child_pid"] = process.pid
                write(state_path, state)
                entry["exit_code"] = process.wait()
            if entry["exit_code"] and not allow_failed_probe:
                raise RuntimeError(f"{name} exited {entry['exit_code']}; inspect {name}.log")
            return entry["exit_code"]

    gate_tool = code / "tools/run_quant_d_repair_gate.py"
    try:
        with stage("prepare_configs"):
            for budget in (256, 1024):
                setting = dict(config, model=args.model, dataset=str(run / "train4.jsonl"), data_root=str(data),
                               output_dir=str(run / f"probe{budget}"), thought_limit=budget,
                               max_completion_length=budget + 64 + 2, max_gpu_hours=None)
                write(run / f"config{budget}.json", setting)
        exit256 = launch("probe256", [sys.executable, str(gate_tool), "train-one", "--config",
                         str(run / "config256.json"), "--dataset", str(run / "train4.jsonl"),
                         "--data-root", str(data), "--output-dir", str(run / "probe256")], allow_failed_probe=True)
        with stage("baseline256_from_first_rollout"):
            report256 = read(run / "probe256/gate_result.json")
            if exit256:
                no_signal = (report256["status"] == "failed" and
                             report256.get("train_gate", {}).get("passed") is False and
                             report256.get("first_rollout_groups") == 4 and "parameter_update" not in report256)
                if not no_signal:
                    raise RuntimeError(f"Probe256 stopped on a computation error: {report256.get('error')}")
                state["probe256_no_learning_signal"] = True
                write(run / "probe256_no_learning_signal.json", {
                    "reason": "no-learning-signal", "train_gate": report256["train_gate"],
                    "next_action": "one fixed 1024 counterfactual", "parameter_update": False})
            else:
                check_probe(report256, run / "probe256")
            rows256 = export_first_rollout(train4, run / "probe256/first_rollout.jsonl", run / "baseline256")
        launch("baseline1024", eval_command(code, data, args.model, run / "train4.jsonl", warmup,
               run / "baseline1024", 8, 1024))
        with stage("select_thinking_budget"):
            rows1024 = read_rows(run / "baseline1024/predictions.jsonl")
            coverage(rows256, train4, 8)
            coverage(rows1024, train4, 8)
            selection = select_setting(rows256, rows1024)
            write(run / "selection.json", selection)
            if not selection["formal_training_allowed"]:
                raise RuntimeError(f"{selection['status']}: selected outputs lack the required localization signal")
            budget = selection["thought_limit"]
            state["selected_thought_limit"] = budget
        if budget == 1024:
            launch("probe1024", [sys.executable, str(gate_tool), "train-one", "--config",
                   str(run / "config1024.json"), "--dataset", str(run / "train4.jsonl"),
                   "--data-root", str(data), "--output-dir", str(run / "probe1024")])
        with stage("selected_probe_check"):
            selected_probe = run / f"probe{budget}"
            checkpoint = check_probe(read(selected_probe / "gate_result.json"), selected_probe)
        launch("reload12", eval_command(code, data, args.model, run / "train12.jsonl", checkpoint,
               run / "reload12", 1, budget))
        with stage("reload12_check"):
            reloaded = read_rows(run / "reload12/predictions.jsonl")
            coverage(reloaded, train12, 1)
            reload_gate = metrics(reloaded)
            reload_gate.update(passed=reload_gate["valid_rate"] >= 0.95 and reload_gate["truncated"] == 0,
                               checkpoint=str(checkpoint), mechanism_only=True)
            write(run / "reload12_gate.json", reload_gate)
            if not reload_gate["passed"]:
                raise RuntimeError("New-process checkpoint reload failed the 12-query output protocol gate")
        schedule = data / "quant_grounding_20261026_formal/schedule.jsonl"
        with stage("prepare_formal"):
            formal_config = dict(read(run / f"config{budget}.json"), adapters=str(warmup), dataset=str(schedule),
                                 output_dir=str(run / "formal"), max_steps=59, save_steps=59, max_gpu_hours=None)
            write(run / "formal_config.json", formal_config)
        launch("formal", [sys.executable, str(gate_tool), "train-formal", "--config",
               str(run / "formal_config.json"), "--schedule", str(schedule),
               "--data-root", str(data), "--output-dir", str(run / "formal")])
        with stage("formal_check"):
            formal = read(run / "formal/gate_result.json")
            checkpoint = run / "formal/checkpoint-59"
            if formal["status"] != "passed" or not formal["first12_gate"]["passed"]:
                raise RuntimeError("Formal D did not pass its first-12 localization gate")
            if read(checkpoint / "trainer_state.json")["global_step"] != 59:
                raise RuntimeError("Formal D checkpoint must contain exactly 59 completed optimizer steps")
            if not (checkpoint / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(checkpoint / "adapter_model.safetensors")
        paired = {"thought_limit": budget, "answer_limit": 64, "samples_per_query": 1,
                  "d0_adapter": str(warmup), "d_adapter": str(checkpoint), "datasets": {}}
        for name, adapter in (("d0", warmup), ("d", checkpoint)):
            for split, (filename, count) in DATASETS.items():
                dataset_path = data / "quant_grounding_20261026_formal" / filename
                dataset = read_rows(dataset_path)
                if len(dataset) != count or len({row["id"] for row in dataset}) != count:
                    raise RuntimeError(f"Wrong development dataset coverage: {split}")
                output = run / "development" / name / split
                launch(f"development_{name}_{split}", eval_command(
                    code, data, args.model, dataset_path, adapter, output, 1, budget))
                with stage(f"development_{name}_{split}_rescore"):
                    predictions = read_rows(output / "predictions.jsonl")
                    coverage(predictions, dataset, 1)
                    raw = read_rows(output / "predictions_raw.jsonl")
                    strict = score_rows(dataset, raw, "thinking", "contrast", "strict")
                    write_rows(output / "predictions_strict.jsonl", strict)
                    write(output / "summary_strict.json", summarize(strict))
        with stage("paired_development"):
            for split, (filename, _) in DATASETS.items():
                dataset = read_rows(data / "quant_grounding_20261026_formal" / filename)
                paired["datasets"][split] = {}
                for policy, filename in (("single_box", "predictions.jsonl"), ("strict", "predictions_strict.jsonl")):
                    before = read_rows(run / "development/d0" / split / filename)
                    after = read_rows(run / "development/d" / split / filename)
                    paired["datasets"][split][policy] = paired_report(dataset, before, after)
            write(run / "paired_development.json", paired)
        state.update(status="completed", ended_at=now())
    except BaseException as error:
        state.update(status="failed", error=str(error), ended_at=now())
        raise
    finally:
        state.pop("child_pid", None)
        write(state_path, state)
        record_wait(ledger)
    return state


if __name__ == "__main__":
    main()
