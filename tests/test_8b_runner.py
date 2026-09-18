import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Exercises the Linux cloud runner")


def runner_fixture(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / "scripts/run_8b_experiment.sh", scripts / "run_8b_experiment.sh")
    run = tmp_path / "run"
    (run / "results").mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"max_pixels": 802816}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {
        "python": 'if [[ "$1" == -c ]]; then echo "$3"; else echo "$*" >> "$ACTIONS"; fi',
        "nvidia-smi": "echo 0",
        "sleep": "exit 0",
    }.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(bin_dir / "python"),
               PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               ACTIONS=str(tmp_path / "actions.txt"))
    return scripts / "run_8b_experiment.sh", run, env


def test_oom_retries_all_pressure_before_training_and_preserves_evidence(tmp_path):
    script, run, env = runner_fixture(tmp_path)
    pressure = (
        f'if [[ ! -f "{run}/first_seen" ]]; then '
        f'touch "{run}/first_seen"; echo failed > "{run}/results/pressure_ir.json"; '
        'exit 42; fi'
    )
    result = subprocess.run(["bash", str(script), str(run), pressure], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    status = (run / "stage_status.tsv").read_text()
    assert "failed:42" in status
    assert status.index("pressure_602112") < status.index("native_rgb") < status.index("stage1_ir")
    assert (run / "results/pressure_802816/pressure_ir.json").read_text().strip() == "failed"
    assert "--set-max-pixels 602112" in Path(env["ACTIONS"]).read_text()
    assert (run / ".t4_fixed4_eval.ok").exists()


def test_runner_refuses_new_training_after_44_hours(tmp_path):
    script, run, env = runner_fixture(tmp_path)
    (run / "experiment_started_unix.txt").write_text(str(int(time.time()) - 44 * 3600 - 1))
    (run / ".pressure.ok").touch()
    (run / ".native_rgb.ok").touch()
    result = subprocess.run(["bash", str(script), str(run), "true"], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 5
    assert "44-hour training cutoff" in result.stderr
    assert not Path(env["ACTIONS"]).exists()
