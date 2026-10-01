"""CPU contract checks for the isolated T/M experiment queue."""

import json
from pathlib import Path
import sys

import pytest

from tools import run_triground_tm as tm


def config():
    return {
        "repo": "/work/code", "python": "/env/python",
        "qwen_finetune_dir": "/work/Qwen", "data_root": "/work/data",
        "model": "/models/Qwen", "initial_adapter": "/adapters/C",
        "baseline_adapters": {arm: f"/adapters/{arm}" for arm in ("C", "A", "B", "V")},
        "city_manifest": "/inputs/city412.json", "city_gt": "/inputs/city_gt.json",
        "budget_dir": "/runs/tm/new_budget",
    }


def release():
    return {
        "status": "ready", "steps": 600, "seed": 2028,
        "manifests": {arm: f"/release/{arm}.json" for arm in tm.ARMS},
        "diagnostics": {condition: f"/release/{condition}.json" for condition in tm.DIAGNOSTICS},
        "gt_manifest": "/release/gt.json",
        "probes": {probe: f"/release/{probe}.json" for probe in ("ir_read", "depth_read")},
    }


def test_release_and_config_require_isolated_full_run(monkeypatch):
    value = release()
    monkeypatch.setattr(tm, "count_records", lambda path: 6000 if path.endswith(("/T.json", "/M.json")) else 10)
    tm.validate_release(value)
    tm.validate_config(config())
    value["steps"] = 400
    with pytest.raises(ValueError, match="steps × 10"):
        tm.validate_release(value)
    old = config()
    old["budget_dir"] = "/root/results/triground_abv_20260927/gpu_budget"
    with pytest.raises(ValueError, match="new 12-hour"):
        tm.validate_config(old)


def test_preflight_is_four_vs_two_plus_two_on_formal_horizon():
    stages = tm.build_stages(config(), release(), Path("/out"), "preflight")
    assert [stage["name"] for stage in stages] == [
        "M_continuous_4", "M_main_2", "M_main_4", "M_compare_resume"
    ]
    assert [stage["updates"] for stage in stages[:3]] == [4, 2, 2]
    for stage in stages[:3]:
        env = stage["env"]
        assert env["MAX_STEPS"] == "600"
        assert env["GRADIENT_ACCUMULATION_STEPS"] == "10"
        assert env["LOSS_REDUCTION"] == "sample_mean"
        assert env["SAVE_STEPS"] == "100"
        assert env["CHECKPOINT_STEPS"] == "2,4"
        assert env["LORA_SCOPE"] == "language"
    assert stages[2]["env"]["RESUME_FROM_CHECKPOINT"].replace("\\", "/") == "/out/M/main/checkpoint-2"
    assert stages[2]["env"]["INIT_ADAPTER"] == ""
    assert stages[3]["command"][-2].replace("\\", "/") == "--output"
    assert stages[3]["command"][-1].replace("\\", "/") == "/out/M/resume_comparison.json"
    assert stages[3]["command"][stages[3]["command"].index("--microbatches-per-step") + 1] == "10"


def test_formal_training_continues_m_and_starts_t_from_c():
    stages = tm.build_stages(config(), release(), Path("/out"), "train")
    assert [stage["name"] for stage in stages] == ["M_main_final", "T_main_final"]
    assert stages[0]["env"]["RESUME_FROM_CHECKPOINT"].replace("\\", "/") == "/out/M/main/checkpoint-4"
    assert stages[0]["env"]["INIT_ADAPTER"] == ""
    assert stages[1]["env"]["INIT_ADAPTER"] == "/adapters/C"
    assert stages[1]["env"]["RESUME_FROM_CHECKPOINT"] == ""
    assert [stage["updates"] for stage in stages] == [596, 600]
    assert all(stage["env"]["SAVE_STEPS"] == "100" for stage in stages)
    assert stages[0]["env"]["CHECKPOINT_STEPS"] == "2,4"
    assert stages[1]["env"]["CHECKPOINT_STEPS"] == ""


def test_evaluation_uses_raw_gt_direct_mode_and_reading_probes(monkeypatch):
    monkeypatch.setattr(tm, "count_records", lambda path: 412 if "city" in path else 12)
    stages = tm.build_stages(config(), release(), Path("/out"), "evaluate",
                             resume_evaluation=True)
    assert len(stages) == 22  # T/M: City + 4 diagnostic + 2 probes; C/A: 4 diagnostic.
    city = next(stage for stage in stages if stage["name"] == "M_city412")
    assert ["--target-manifest", "/inputs/city_gt.json"] == city["command"][
        city["command"].index("--target-manifest"):city["command"].index("--target-manifest") + 2
    ]
    assert "--resume" in city["command"]
    assert "direct" in city["command"]
    assert next(stage for stage in stages if stage["name"] == "A_diagnostic_ir_missing")["command"][
        -1
    ] == "--resume"
    assert next(stage for stage in stages if stage["name"] == "T_ir_read")["command"][2] == "tools.evaluate_aux_reading"
    assert not any(stage["name"].startswith(("C_city", "A_city")) for stage in stages)


def test_empty_ir_read_probe_is_cpu_only(monkeypatch):
    monkeypatch.setattr(tm, "count_records", lambda path: 0 if path.endswith("ir_read.json") else 12)
    stages = tm.build_stages(config(), release(), Path("/out"), "evaluate")
    assert all(not stage["gpu"] and stage["queries"] == 0
               for stage in stages if stage["name"].endswith("ir_read"))


def test_complete_checkpoint_discovery_ignores_partial_state(tmp_path):
    output = tmp_path / "M/main"
    for step in (4, 100):
        checkpoint = output / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"state")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    partial = output / "checkpoint-200"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step": 200}', encoding="utf-8")
    assert [path.name for path in tm.complete_checkpoints(output)] == ["checkpoint-4", "checkpoint-100"]


def test_formal_retry_uses_latest_complete_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "count_records", lambda path: 12)
    (tmp_path / "M").mkdir()
    tm.write_json(tmp_path / "M/resume_comparison.json", {"status": "pass"})
    for arm, step in (("M", 100), ("T", 200)):
        checkpoint = tmp_path / arm / "main" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors", "rng_state.pth"):
            (checkpoint / name).write_bytes(b"state")
        tm.write_json(checkpoint / "trainer_state.json", {"global_step": step})
    monkeypatch.setattr(tm, "budget_forecast", lambda *args: {"fits_with_10_percent_margin": True})
    monkeypatch.setattr(tm, "assert_stage_artifact", lambda stage: None)
    commands = []
    monkeypatch.setattr(tm, "execute_stage", lambda stage, *args: commands.append(stage))
    state = {"limit_seconds": tm.BUDGET_SECONDS, "spent_seconds": 0, "stages": []}
    tm.run_phase(config(), release(), tmp_path, "train", resume=True, retry_reason="fixed",
                 state=state, budget_dir=tmp_path / "budget", all_stages=[])
    assert [stage["env"]["RESUME_FROM_CHECKPOINT"].replace("\\", "/").split("/")[-1]
            for stage in commands] == ["checkpoint-100", "checkpoint-200"]
    assert all(stage["env"]["INIT_ADAPTER"] == "" for stage in commands)


def test_run_all_dry_run_lists_order_and_initial_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tm, "count_records", lambda path: 6000 if path.endswith(("/T.json", "/M.json")) else 12)
    setup = config()
    setup["budget_dir"] = str(tmp_path / "new_budget")
    tm.write_json(tmp_path / "config.json", setup)
    tm.write_json(tmp_path / "release.json", release())
    monkeypatch.setattr(sys, "argv", ["run_triground_tm", "--config", str(tmp_path / "config.json"),
                                         "--release", str(tmp_path / "release.json"),
                                         "--output-dir", str(tmp_path / "run"), "--phase", "run-all"])
    tm.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["phases"] == ["preflight", "train", "evaluate"]
    assert preview["stages"][0]["name"] == "M_continuous_4"
    assert preview["stages"][-1]["name"] == "A_diagnostic_both_missing"
    assert preview["budget_forecast"]["fits_with_10_percent_margin"]
    assert not (tmp_path / "new_budget").exists()


def test_forecast_uses_measured_four_steps_and_preserves_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "count_records", lambda path: 412 if "city" in path else 12)
    monkeypatch_path = tmp_path / "m.log"
    monkeypatch_path.write_text("{'train_runtime': 48.0, 'train_steps_per_second': 0.083}\n", encoding="utf-8")
    state = {"limit_seconds": tm.BUDGET_SECONDS, "spent_seconds": 200,
             "stages": [{"name": "M_continuous_4", "status": "complete",
                         "run_dir": str(tmp_path), "elapsed_seconds": 70,
                         "log": str(monkeypatch_path)}]}
    stages = [stage for phase in ("preflight", "train", "evaluate")
              for stage in tm.build_stages(config(), release(), tmp_path, phase)]
    forecast = tm.budget_forecast(config(), release(), tmp_path, state, stages)
    assert forecast["rate_source"].startswith("M continuous")
    assert forecast["step_seconds"] == pytest.approx(13.8)
    assert forecast["pending_training_updates"] == 1200
    assert forecast["fits_with_10_percent_margin"]
    checkpoint = tmp_path / "M/main/checkpoint-100"
    checkpoint.mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "adapter_model.safetensors", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"state")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 100}', encoding="utf-8")
    assert tm.budget_forecast(config(), release(), tmp_path, state, stages)["pending_training_updates"] == 1104
    state["spent_seconds"] = 40000
    assert not tm.budget_forecast(config(), release(), tmp_path, state, stages)["fits_with_10_percent_margin"]


def test_forecast_prefers_real_trainer_runtime_over_stage_loading_time(tmp_path):
    log = tmp_path / "continuous.log"
    log.write_text("{'train_runtime': '65.1'}\n", encoding="utf-8")
    trainer_state = tmp_path / "M/continuous/trainer_state.json"
    trainer_state.parent.mkdir(parents=True)
    tm.write_json(trainer_state, {"log_history": [
        {"loss": 0.8, "step": 4}, {"train_runtime": 65.1002, "step": 4}
    ]})
    state = {"limit_seconds": tm.BUDGET_SECONDS, "spent_seconds": 102.44,
             "stages": [{"name": "M_continuous_4", "status": "complete",
                         "run_dir": str(tmp_path), "elapsed_seconds": 102.44,
                         "log": str(log)}]}
    stages = [{"name": "M_main_final", "gpu": True, "updates": 596,
               "expected_artifact": str(tmp_path / "M/main/checkpoint-600")}]
    forecast = tm.budget_forecast(config(), release(), tmp_path, state, stages)
    assert forecast["step_seconds"] == pytest.approx(65.1002 * 1.15 / 4)
    assert forecast["process_load_seconds"] == 100
    trainer_state.unlink()
    fallback = tm.budget_forecast(config(), release(), tmp_path, state, stages)
    assert fallback["step_seconds"] == pytest.approx(65.1 * 1.15 / 4)
