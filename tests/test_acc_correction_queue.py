"""CPU-only checks for queue gating, matched order, resume, and ACC selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_acc_correction as queue


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


def complete_checkpoint(train, number, updates=1):
    path = train / f"checkpoint-{number}"
    path.mkdir(parents=True, exist_ok=True)
    for filename in queue.CHECKPOINT_FILES:
        (path / filename).write_text("saved", encoding="utf-8")
    queue.write(path / "loop_state.json", {"presentations": number, "updates": updates})
    return path


def evaluation(output, hits=(True, False), ids=("q1", "q2")):
    queue.write(output / "summary.json", {
        "overall": {"queries": len(ids), "acc05_first": sum(hits) / len(ids), "timed_seconds": 3.0},
        "samples_per_query": 1, "format_policy": "punctuation", "answer_mode": "short",
    })
    predictions = [{"id": key, "samples": [{"hit": hit, "iou": 1.0 if hit else 0.0}]} for key, hit in zip(ids, hits)]
    write_jsonl(output / "predictions.jsonl", predictions)
    write_jsonl(output / "predictions_raw.jsonl", predictions)


def ready_b():
    return {"status": "completed", "results": {"B": {"550": {"status": "completed"}}}}


@pytest.mark.parametrize("diagnostic_status", ["completed", "failed", "cancelled", "cancelled_by_user"])
def test_gate_requires_b550_and_a_terminal_diagnostic(diagnostic_status):
    assert queue.gate_status(ready_b(), {"status": diagnostic_status}) == (True, "ready")
    assert queue.gate_status(ready_b(), {"status": "running"}) == (False, "waiting_for_diagnostic")
    assert queue.gate_status(ready_b(), None) == (False, "waiting_for_diagnostic")
    assert queue.gate_status({"status": "running"}, {"status": diagnostic_status}) == (False, "waiting_for_b550")
    assert queue.gate_status(None, {"status": diagnostic_status}) == (False, "waiting_for_b_state")


@pytest.mark.parametrize("diagnostic_status", ["completed", "failed", "cancelled_by_user"])
def test_gate_waits_for_diagnostic_finally_to_resume_original_queue(diagnostic_status):
    assert queue.gate_status(ready_b(), {"status": diagnostic_status, "original_queue_resumed": False}) == (
        False, "waiting_for_original_queue_resume")
    assert queue.gate_status(ready_b(), {"status": diagnostic_status, "original_queue_resumed": True}) == (True, "ready")


def test_explicitly_paused_b_and_diagnostic_release_gate_without_b550_or_old_parent_resume():
    paused = {"status": "paused_by_user", "results": {"B": {}}}
    diagnostic = {"status": "paused_by_user", "original_queue_resumed": False}
    assert queue.gate_status(paused, diagnostic) == (True, "ready")
    assert queue.gate_status(paused, {"status": "running"}) == (False, "waiting_for_b550")
    with pytest.raises(RuntimeError, match="B550 queue ended as failed"):
        queue.gate_status(paused | {"status": "failed"}, diagnostic)


def test_failed_b_is_an_explicit_failure_even_when_diagnostic_failed():
    with pytest.raises(RuntimeError, match="B550 queue ended as failed"):
        queue.gate_status({"status": "failed", "error": "out of memory"}, {"status": "failed"})
    with pytest.raises(RuntimeError, match="cancelled"):
        queue.gate_status({"status": "cancelled"}, {"status": "completed"})
    failed_after_550 = ready_b() | {"status": "failed"}
    with pytest.raises(RuntimeError, match="B550 queue ended as failed"):
        queue.gate_status(failed_after_550, {"status": "failed", "original_queue_resumed": True})


def test_resume_uses_last_complete_native_checkpoint_and_rejects_missing_ladder_point(tmp_path):
    first = complete_checkpoint(tmp_path, 400)
    partial = complete_checkpoint(tmp_path, 800)
    (partial / "rng.pt").unlink()
    assert queue.latest_checkpoint(tmp_path, 800) == first
    assert queue.checkpoint_state(partial) is None
    complete_checkpoint(tmp_path, 1200)
    with pytest.raises(RuntimeError, match="later checkpoint"):
        queue.latest_checkpoint(tmp_path, 800)


def test_checkpoint_counter_must_match_its_native_name(tmp_path):
    checkpoint = complete_checkpoint(tmp_path, 400)
    queue.write(checkpoint / "loop_state.json", {"presentations": 399, "updates": 20})
    with pytest.raises(RuntimeError, match="counter disagrees"):
        queue.checkpoint_state(checkpoint)


def test_initial_checkpoint_zero_is_a_valid_native_resume_point(tmp_path):
    zero = complete_checkpoint(tmp_path, 0, updates=0)
    assert queue.checkpoint_state(zero) == {"presentations": 0, "updates": 0}
    assert queue.latest_checkpoint(tmp_path, 800) == zero


def test_commands_use_configured_native_python_and_exact_existing_eval_protocol(tmp_path):
    config = {"python": "/native/venv/bin/python", "code_root": "/independent/code", "old_root": "/old/code",
              "media_root": "/media", "model": "/models/Qwen3.6-27B"}
    command = queue.train_command(config, "sft", tmp_path / "segment.json", tmp_path / "train", tmp_path / "checkpoint-400")
    assert command == ["/native/venv/bin/python", str(Path(config["code_root"]) / "tools/train_quant_grounding.py"),
                       "--mode", "sft", "--config", str(tmp_path / "segment.json"), "--output-dir",
                       str(tmp_path / "train"), "--resume", str(tmp_path / "checkpoint-400")]
    evaluation_command = queue.eval_command(config, tmp_path / "checkpoint-800", tmp_path / "city.jsonl", tmp_path / "eval")
    flags = dict(zip(evaluation_command[2::2], evaluation_command[3::2]))
    assert flags == {"--dataset": str(tmp_path / "city.jsonl"), "--data-root": "/media", "--model": "/models/Qwen3.6-27B",
                     "--adapter": str(tmp_path / "checkpoint-800"), "--engine": "hf", "--precision": "nf4",
                     "--answer-mode": "short", "--reward-mode": "acc05", "--format-policy": "punctuation",
                     "--samples-per-query": "1", "--hf-max-batch-size": "2", "--seed": "2026",
                     "--max-tokens": "64", "--max-length": "8192", "--min-pixels": "200704",
                     "--max-pixels": "1204224", "--output-dir": str(tmp_path / "eval")}


def test_selection_uses_only_hits_and_keeps_earliest_exact_tie():
    earlier = {"name": "early", "city378": {"hits": 303, "mean_iou": 0.1}, "external88": {"hits": 85}}
    later = {"name": "late", "city378": {"hits": 303, "mean_iou": 0.99}, "external88": {"hits": 85}}
    external_better = {"name": "ext", "city378": {"hits": 303}, "external88": {"hits": 86}}
    city_better = {"name": "city", "city378": {"hits": 304}, "external88": {"hits": 80}}
    assert queue.select_best([earlier, later]) is earlier
    assert queue.select_best([earlier, later, external_better]) is external_better
    assert queue.select_best([earlier, external_better, city_better]) is city_better


def test_eval_skip_requires_full_ids_policy_and_both_prediction_files(tmp_path):
    evaluation(tmp_path)
    assert queue.full_summary(tmp_path, {"q1", "q2"})["hits"] == 1
    (tmp_path / "predictions_raw.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        queue.full_summary(tmp_path, {"q1", "q2"})
    assert queue.full_summary(tmp_path, {"q1", "q2"}, require_raw=False)["hits"] == 1
    write_jsonl(tmp_path / "predictions.jsonl", [{"id": "q1", "samples": [{"hit": True}]}] * 2)
    with pytest.raises(RuntimeError, match="ID coverage"):
        queue.full_summary(tmp_path, {"q1", "q2"}, require_raw=False)


def test_full_id_pairing_counts_rescues_and_regressions_without_position_matching(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    evaluation(before, (True, False, True), ("a", "b", "c"))
    evaluation(after, (True, True, False), ("c", "b", "a"))
    comparison = queue.paired_comparison(after / "predictions.jsonl", before / "predictions.jsonl")
    assert comparison["rescue"] == comparison["regression"] == 1
    assert comparison["baseline_hits"] == comparison["current_hits"] == 2
    assert len(comparison["per_query"]) == 3
    write_jsonl(after / "predictions.jsonl", [{"id": "a", "samples": [{"hit": True}]}])
    with pytest.raises(RuntimeError, match="identical ID"):
        queue.paired_comparison(after / "predictions.jsonl", before / "predictions.jsonl")


def test_wait_is_cpu_only_and_does_not_touch_existing_processes(tmp_path, monkeypatch):
    config = {"b_state": str(tmp_path / "b.json"), "diagnostic_state": str(tmp_path / "diagnostic.json"), "poll_seconds": 60}
    queue.write(Path(config["b_state"]), {"status": "running"})
    queue.write(Path(config["diagnostic_state"]), {"status": "running"})
    sleeps = []

    def finish_dependency(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            queue.write(Path(config["b_state"]), ready_b())
        else:
            queue.write(Path(config["diagnostic_state"]), {"status": "failed"})

    monkeypatch.setattr(queue.time, "sleep", finish_dependency)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("GPU job started while waiting"))
    result = queue.wait_for_existing_jobs(config, tmp_path, {"jobs": []})
    assert result == ready_b()
    assert sleeps == [60, 60]


def queue_fixture(tmp_path):
    baseline = tmp_path / "baseline"
    for split in queue.SPLITS:
        evaluation(baseline / split)
    for mode in queue.MODES:
        queue.write(tmp_path / f"{mode}.json", {"init_adapter": "/saved/C550", "total_presentations": 4400})
    config = {"python": "/native/bin/python", "code_root": str(tmp_path), "old_root": "/old",
              "b_state": str(tmp_path / "b.json"), "diagnostic_state": str(tmp_path / "diagnostic.json"),
              "base_train_config": {mode: str(tmp_path / f"{mode}.json") for mode in queue.MODES},
              "city_dataset": "/city.jsonl", "external_dataset": "/external.jsonl", "model": "/model", "media_root": "/media",
              "smoke_command": ["native-smoke"], "smoke_report": str(tmp_path / "smoke/report.json"),
              "comparisons": {name: {split: str(baseline / split) for split in queue.SPLITS} for name in ("C550", "B550", "B59")}}
    b_state = ready_b()
    b_state["results"]["B"]["550"].update({split: str(baseline / split) for split in queue.SPLITS})
    queue.write(Path(config["b_state"]), b_state)
    queue.write(Path(config["diagnostic_state"]), {"status": "completed"})
    return config, {split: {"q1", "q2"} for split in queue.SPLITS}


def test_bbest_rebuilds_from_b59_and_all_completed_b_results_using_hits_and_earliest_step(tmp_path):
    expected = {split: {"q1", "q2"} for split in queue.SPLITS}

    def source(step, city, external):
        paths = {split: str(tmp_path / str(step) / split) for split in queue.SPLITS}
        evaluation(Path(paths["city378"]), city)
        evaluation(Path(paths["external88"]), external)
        return paths

    initial = source(59, (True, False), (True, False))
    first_improvement = source(100, (True, False), (True, True))
    later_tie = source(200, (True, False), (True, True))
    summary = queue.read(Path(later_tie["city378"]) / "summary.json")
    summary["overall"]["mean_iou"] = 1.0
    queue.write(Path(later_tie["city378"]) / "summary.json", summary)
    final = source(550, (True, False), (True, False))
    config = {"comparisons": {"B59": initial, "Bbest": {"city378": "obsolete", "external88": "obsolete"}}}
    b_state = {"results": {"B": {"550": final | {"status": "completed"},
                                         "200": later_tie | {"status": "completed"},
                                         "300": {"status": "running"},
                                         "100": first_improvement | {"status": "completed"}}}}
    best = queue.rebuild_b_best(config, b_state, expected)
    assert best["legacy_step"] == 100
    assert best["city378"] == first_improvement["city378"]
    assert [item["legacy_step"] for item in best["selection_candidates"]] == [59, 100, 200, 550]
    assert queue.comparison_sources(config, b_state, expected)["Bbest"]["legacy_step"] == 100
    b_state["results"]["B"]["100"] = initial | {"status": "completed"}
    b_state["results"]["B"]["200"] = initial | {"status": "completed"}
    assert queue.rebuild_b_best(config, b_state, expected)["legacy_step"] == 59
    b_state["results"]["B"]["300"] = source(300, (True, True), (False, False)) | {"status": "completed"}
    assert queue.rebuild_b_best(config, b_state, expected)["legacy_step"] == 300


def test_paused_queue_compares_latest_completed_b_under_its_actual_step(tmp_path):
    config, expected = queue_fixture(tmp_path)
    initial = config["comparisons"]["B59"]
    state = {"status": "paused_by_user", "results": {"B": {
        "100": initial | {"status": "completed"}, "150": {"status": "running"}}}}
    sources = queue.comparison_sources(config, state, expected)
    assert "B550" not in sources
    assert sources["B100"] == initial
    assert sources["Bbest"]["legacy_step"] == 59
    state["results"]["B"] = {}
    sources = queue.comparison_sources(config, state, expected)
    assert "B550" not in sources
    assert sources["B59"] == initial


def test_queue_alternates_matched_presentations_and_completed_restart_skips_every_job(tmp_path, monkeypatch):
    config, expected = queue_fixture(tmp_path)
    monkeypatch.setattr(queue, "TARGETS", (800, 1600))
    jobs = []

    def fake_job(config, state, state_path, label, command, log):
        jobs.append(label)
        if label == "native_gpu_smoke":
            queue.write(Path(config["smoke_report"]), {"status": "passed"})
        elif "_train_to_" in label:
            flags = dict(zip(command[2::2], command[3::2]))
            mode = flags["--mode"]
            segment = queue.read(Path(flags["--config"]))
            target = segment["stop_presentations"]
            train = Path(flags["--output-dir"])
            assert segment["checkpoint_every"] == 400
            if target == 1600:
                assert Path(flags["--resume"]) == train / "checkpoint-800"
            else:
                assert "--resume" not in flags
            if mode == "sft":
                assert Path(segment["sequence_path"]) == tmp_path / "output/correction/train/order.jsonl"
                assert len(queue.rows(Path(segment["sequence_path"]))) == target
            else:
                write_jsonl(train / "order.jsonl", [{"presentation": index, "id": f"question{index}"} for index in range(1, target + 1)])
            checkpoint = complete_checkpoint(train, target, updates=target // 2)
            queue.write(train / "completed.json", {"presentations": target, "updates": target // 2, "checkpoint": str(checkpoint)})
        else:
            evaluation(Path(command[command.index("--output-dir") + 1]), (True, True))
        return 2.0

    monkeypatch.setattr(queue, "run_job", fake_job)
    output = tmp_path / "output"
    result = queue.run_queue(config, output, expected)
    assert result["status"] == "completed"
    assert jobs == ["native_gpu_smoke", "correction_train_to_800", "correction_800_city378", "correction_800_external88",
                    "sft_train_to_800", "sft_800_city378", "sft_800_external88", "correction_train_to_1600",
                    "correction_1600_city378", "correction_1600_external88", "sft_train_to_1600", "sft_1600_city378", "sft_1600_external88"]
    assert queue.read(output / "best.json")["correction"]["presentations"] == 800
    assert queue.read(output / "best.json")["sft"]["presentations"] == 800
    assert "mIoU" not in (output / "curve.md").read_text(encoding="utf-8")
    jobs.clear()
    assert queue.run_queue(config, output, expected)["status"] == "completed"
    assert jobs == []


@pytest.mark.parametrize("dependency_status", ["completed", "paused_by_user"])
def test_smoke_failure_prevents_training_and_is_saved(tmp_path, monkeypatch, dependency_status):
    config, expected = queue_fixture(tmp_path)
    if dependency_status == "paused_by_user":
        b_state = queue.read(Path(config["b_state"])) | {"status": "paused_by_user"}
        queue.write(Path(config["b_state"]), b_state)
        queue.write(Path(config["diagnostic_state"]), {"status": "paused_by_user", "original_queue_resumed": False})
    jobs = []

    def smoke_only(config, state, state_path, label, command, log):
        jobs.append(label)
        queue.write(Path(config["smoke_report"]), {"status": "failed"})
        return 1.0

    monkeypatch.setattr(queue, "run_job", smoke_only)
    with pytest.raises(RuntimeError, match="smoke did not pass"):
        queue.run_queue(config, tmp_path / "output", expected)
    assert jobs == ["native_gpu_smoke"]
    assert queue.read(tmp_path / "output/state.json")["status"] == "failed"


def test_resume_keeps_complete_city_eval_and_restarts_partial_external_in_fresh_output(tmp_path, monkeypatch):
    config, expected = queue_fixture(tmp_path)
    monkeypatch.setattr(queue, "TARGETS", (800,))
    output = tmp_path / "output"
    train = output / "correction/train"
    complete_checkpoint(train, 800, 200)
    write_jsonl(train / "order.jsonl", [{"presentation": index, "id": f"q{index}"} for index in range(1, 801)])
    city = output / "correction/eval/presentations-800/city378/attempt-1"
    evaluation(city)
    partial = output / "correction/eval/presentations-800/external88/attempt-1"
    write_jsonl(partial / "predictions_raw.jsonl", [{"id": "q1", "samples": []}])
    queue.write(Path(config["smoke_report"]), {"status": "passed"})
    queue.write(output / "state.json", {
        "status": "failed", "jobs": [], "results": {"correction": {"800": {
            "city378": dict(queue.full_summary(city, expected["city378"]), status="completed")}}, "sft": {}},
        "baselines": {}, "baseline_unavailable": {}, "smoke": {"status": "completed"},
    })
    jobs = []

    def remaining_jobs(config, state, state_path, label, command, log):
        jobs.append(label)
        if label == "sft_train_to_800":
            sft_train = output / "sft/train"
            checkpoint = complete_checkpoint(sft_train, 800, 800)
            queue.write(sft_train / "completed.json", {"presentations": 800, "updates": 800, "checkpoint": str(checkpoint)})
        else:
            destination = Path(command[command.index("--output-dir") + 1])
            if label == "correction_800_external88":
                assert destination.name == "attempt-2"
            evaluation(destination)
        return 1.0

    monkeypatch.setattr(queue, "run_job", remaining_jobs)
    assert queue.run_queue(config, output, expected)["status"] == "completed"
    assert jobs == ["correction_800_external88", "sft_train_to_800", "sft_800_city378", "sft_800_external88"]
    assert queue.rows(partial / "predictions_raw.jsonl") == [{"id": "q1", "samples": []}]
