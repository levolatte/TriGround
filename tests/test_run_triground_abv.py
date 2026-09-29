import json
from pathlib import Path

import pytest

from tools import run_triground_abv as runner


def config():
    return {"repo": "/work/code", "python": "/env/python", "qwen_finetune_dir": "/work/Qwen",
            "data_root": "/work/data", "model": "/models/Qwen", "initial_adapter": "/results/C",
            "city_manifest": "/manifests/city.json", "city_gt": "/manifests/gt.json"}


def release():
    return {"status": "ready", "steps": 600, "seed": 2026,
            "manifests": {a: f"/manifests/{a}.json" for a in runner.ARMS},
            "diagnostics": {name: f"/manifests/{name}.json" for name in ("normal", "ir_missing", "depth_missing", "both_missing")},
            "gt_manifest": "/manifests/dgt.json", "scene_map": "/manifests/scenes.json", "class_map": "/manifests/classes.json"}


def test_unapproved_data_cannot_start():
    with pytest.raises(ValueError, match="Human-approved"):
        runner.validate_release({"status": "draft"})


def test_release_requires_exact_bv_rows_and_horizon(tmp_path):
    value = release()
    value["steps"] = 400
    for arm in runner.ARMS:
        target = tmp_path / f"{arm}.json"
        runner.write_json(target, [{"id": str(i)} for i in range(3200)])
        value["manifests"][arm] = str(target)
    runner.validate_release(value)
    rows = runner.read_json(value["manifests"]["V"])
    rows[-1]["id"] = "wrong"
    runner.write_json(value["manifests"]["V"], rows)
    with pytest.raises(ValueError, match="exactly the same"):
        runner.validate_release(value)


def test_preflight_keeps_full_horizon_and_original_resume_state():
    stages = runner.build_stages(config(), release(), Path("/out"), "preflight")
    training = [s for s in stages if s["gpu"]]
    assert len(training) == 9
    for index, stage in enumerate(training):
        env = stage["env"]
        assert env["MAX_STEPS"] == "600"
        assert env["PYTHON_EXECUTABLE"] == config()["python"]
        assert env["STOP_AFTER_STEP"] == ("16" if index % 3 == 1 else "32")
        assert bool(env["INIT_ADAPTER"]) != bool(env["RESUME_FROM_CHECKPOINT"])
    assert len([s for s in stages if not s["gpu"]]) == 3
    assert training[-1]["env"]["LORA_SCOPE"] == "language_merger"
    assert training[-1]["env"]["VISUAL_LORA_LR"] == "2e-5"


def test_final_training_continues_verified_32_without_resetting_optimizer():
    stages = runner.build_stages(config(), release(), Path("/out"), "train")
    for stage in stages:
        assert stage["env"]["INIT_ADAPTER"] == ""
        assert stage["env"]["RESUME_FROM_CHECKPOINT"].endswith("checkpoint-32")
        assert stage["env"]["STOP_AFTER_STEP"] == ""


def test_partial_reports_are_available_without_any_gpu_stage():
    stages = runner.build_stages(config(), release(), Path("/out"), "report")
    assert len(stages) == 4
    assert all(not stage["gpu"] for stage in stages)
    assert "--modal-run" in stages[2]["command"]
    assert stages[3]["name"] == "atlas_diagnostics"
    assert any("V:depth_missing=" in value for value in stages[3]["command"])


def test_no_approved_diagnostics_never_launches_an_empty_gpu_manifest():
    value = release()
    value["counts"] = {"approved_diagnostic": 0}
    stages = runner.build_stages(config(), value, Path("/out"), "evaluate")
    assert [stage["name"] for stage in stages if stage["gpu"]] == [
        f"{arm}_city412" for arm in ("C0", *runner.ARMS)
    ]
    assert len(stages) == 6


def test_environment_check_requires_identical_prompt_grid_and_raw_text(tmp_path):
    rows = [{"id": str(i), "prompt": "same prompt", "prediction": [0.1, 0.2, 0.3, 0.4],
             "raw_text": "same", "image_grid_thw": [[1, 2, 3]], "input_tokens": 8} for i in range(8)]
    for arm in ("C0", *runner.ARMS):
        p = tmp_path / "environment" / arm / "predictions.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    runner.compare_environment(tmp_path)
    assert runner.read_json(tmp_path / "environment_check.json")["passed"]
    rows[0]["raw_text"] = "different"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Zero-step"):
        runner.compare_environment(tmp_path)


def test_budget_cannot_silently_reset_or_resume_unknown_process(tmp_path):
    state = runner.load_budget(tmp_path)
    state["spent_seconds"] = runner.BUDGET_SECONDS
    with pytest.raises(RuntimeError, match="exhausted"):
        runner.execute_stage({}, tmp_path, state)
    state["running"] = {"pid": 42}
    runner.write_json(tmp_path / "gpu_budget.json", state)
    with pytest.raises(ValueError, match="unfinished"):
        runner.load_budget(tmp_path)


def test_failed_gpu_process_is_charged_to_shared_budget(tmp_path, monkeypatch):
    class Process:
        pid = 123

        def wait(self, timeout):
            return 7

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: Process())
    clock = iter([100.0, 103.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(clock))
    state = {"limit_seconds": runner.BUDGET_SECONDS, "spent_seconds": 5.0, "stages": []}
    stage = {"name": "failed_gpu", "gpu": True, "command": ["unused"], "cwd": str(tmp_path), "env": {}}
    with pytest.raises(RuntimeError, match="failed_gpu failed"):
        runner.execute_stage(stage, tmp_path / "seed2027", state, tmp_path / "shared")
    saved = runner.read_json(tmp_path / "shared/gpu_budget.json")
    assert saved["spent_seconds"] == 8.0
    assert "running" not in saved
    assert saved["stages"][0]["status"] == "failed"
    assert saved["stages"][0]["run_dir"].endswith("seed2027")


def test_budget_forecast_reserves_full_evaluation_and_loads_once_per_job(tmp_path):
    value = release()
    value["diagnostics"] = {name: "unused" for name in ("normal", "ir_missing", "depth_missing", "both_missing")}
    value["gt_manifest"] = str(tmp_path / "gt.json")
    runner.write_json(value["gt_manifest"], {str(i): {} for i in range(96)})
    stages = []
    for arm in ("C0", *runner.ARMS):
        path = tmp_path / "environment" / arm / "predictions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps({"latency_seconds": 1.0}) for _ in range(8)))
        stages.append({"name": f"{arm}_env8", "elapsed_seconds": 18,
                       "run_dir": str(tmp_path), "status": "complete"})
    for arm in runner.ARMS:
        stages.append({"name": f"{arm}_main_32", "elapsed_seconds": 160,
                       "run_dir": str(tmp_path), "status": "complete"})
    state = {"limit_seconds": runner.BUDGET_SECONDS, "spent_seconds": 1000, "stages": stages}
    forecast = runner.forecast_remaining(state, tmp_path, value)
    assert forecast["evaluation_queries"] == 3184
    assert forecast["estimated_evaluation_seconds"] == pytest.approx(3184 * 1.15 + 20 * 10)
    assert forecast["estimated_core_seconds"]["400"] == pytest.approx(
        30 * (400 + 32) + forecast["estimated_evaluation_seconds"]
    )
    assert forecast["recommended_steps"] == 600
    state["spent_seconds"] = 40000
    assert runner.forecast_remaining(state, tmp_path, value)["recommended_steps"] is None
