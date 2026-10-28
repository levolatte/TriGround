import json
import sys

import pytest

from tools.run_quant_budget import STAGE_HOURS, remaining_hours, run_command


def test_optimization_reallocation_keeps_total_and_separate_caps():
    assert sum(STAGE_HOURS.values()) == 72
    rows = [{"stage": "optimization", "label": "probe", "gpu_hours": 1.5, "status": "failed"},
            {"stage": "exposure_eval", "label": "B", "gpu_hours": 0.5, "status": "completed"}]
    assert remaining_hours(rows, "optimization", "new") == 0.5
    assert remaining_hours(rows, "exposure_eval", "C") == 1.5
    assert remaining_hours(rows, "extend", "A") == 4


def test_stage_and_arm_cost_include_failed_evaluation():
    rows = [
        {"stage": "initial", "label": "A", "gpu_hours": 6.5, "status": "completed"},
        {"stage": "initial", "label": "A", "gpu_hours": 0.5, "status": "failed"},
        {"stage": "initial", "label": "B", "gpu_hours": 8, "status": "completed"},
        {"stage": "prepare", "label": "probe", "gpu_hours": 6, "status": "completed"},
    ]
    assert remaining_hours(rows, "initial", "A") == 1
    assert remaining_hours(rows, "initial", "B") == 0
    assert remaining_hours(rows, "prepare", "other") == 0
    rows.append({"stage": "final", "label": "eval", "gpu_hours": 51, "status": "external_charge"})
    assert remaining_hours(rows, "initial", "A") == 0


def test_unfinished_charge_cannot_be_ignored():
    with pytest.raises(ValueError, match="running"):
        remaining_hours([{"status": "running"}], "initial", "A")


def test_failed_process_is_charged(tmp_path):
    path = tmp_path / "ledger.json"
    code = run_command([sys.executable, "-c", "raise SystemExit(3)"], path,
                       stage="prepare", label="cpu_fixture", gpu_model="test-no-gpu", max_hours=0.1,
                       gpu_count=2, price_per_gpu_hour=2.5)
    row = json.loads(path.read_text())[0]
    assert code == 3 and row["status"] == "failed"
    assert row["gpu_hours"] > 0
    assert row["gpu_hours"] == pytest.approx(row["wall_seconds"] * 2 / 3600)
    assert row["cost"] == pytest.approx(row["gpu_hours"] * 2.5)


def test_budget_timeout_stops_own_command(tmp_path):
    path = tmp_path / "ledger.json"
    code = run_command([sys.executable, "-c", "import time; time.sleep(30)"], path,
                       stage="prepare", label="cpu_timeout", gpu_model="test-no-gpu", max_hours=0.00003)
    row = json.loads(path.read_text())[0]
    assert code == 124 and row["status"] == "budget_exhausted"
    assert row["wall_seconds"] < 10
