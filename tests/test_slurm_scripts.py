"""Exercise Bash ordering and failure propagation with model commands replaced."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASH = ("C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash"))
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash unavailable")


def bash_path(path):
    value = Path(path).as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


FAKE_PYTHON = '''import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["CALL_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "-":
    code = sys.stdin.read()
    if "torch.cuda" not in code:
        sys.argv = args
        exec(code)
elif args[0] == "-c":
    sys.argv = ["-c", *args[2:]]
    exec(args[1])
elif args[0] == "tools/prepare_slurm_run.py":
    run = pathlib.Path(args[args.index("--run-dir") + 1])
    (run / "configs").mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"eval_manifest": "held out.json"}))
elif args[0] == "train.py":
    config = pathlib.Path(args[args.index("--config") + 1])
    stage = config.stem.removeprefix("qwen3_vl_8b_")
    if os.environ.get("FAIL_STAGE") == stage:
        sys.exit(23)
    output = config.parent.parent / stage
    output.mkdir()
    (output / "best_phase_a.pt").write_text("fake checkpoint")
elif args[0] == "evaluate.py":
    output = pathlib.Path(args[args.index("--output") + 1])
    rate = 0 if os.environ.get("FAIL_PARSE") else 1
    output.write_text(json.dumps({mode: {"parse_rate": rate} for mode in
        ("rgb_baseline", "rgb_ir", "rgb_depth", "rgb_ir_depth")}))
'''


def run_script(tmp_path, kind, **extra_env):
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    fake = bin_dir / "fake_python.py"
    fake.write_text(FAKE_PYTHON, encoding="utf-8")
    wrapper = bin_dir / "python"
    wrapper.write_text(
        f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{fake.as_posix()}" "$@"\n',
        encoding="utf-8", newline="\n",
    )
    gpu = bin_dir / "nvidia-smi"
    gpu.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
    wrapper.chmod(0o755)
    gpu.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, REPO_DIR=bash_path(ROOT), PYTHON=bash_path(wrapper),
               RUN_ROOT=bash_path(tmp_path / "run with spaces"), CALL_LOG=str(log),
               SLURM_JOB_ID="123", **extra_env)
    # Set PATH in Bash so Git Bash and Unix both see the fake GPU command.
    command = 'export PATH="$1:$PATH"; bash "$2"'
    result = subprocess.run(
        [BASH, "-c", command, "test", bash_path(bin_dir),
         bash_path(ROOT / f"scripts/qwen3_vl_8b_{kind}.slurm")],
        env=env, capture_output=True, text=True, timeout=60,
    )
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return result, calls


def test_smoke_runs_three_two_step_checks_and_generation(tmp_path):
    result, calls = run_script(tmp_path, "smoke")
    assert result.returncode == 0, result.stdout + result.stderr
    checks = [call for call in calls if call[0] == "tools/preflight.py"]
    assert len(checks) == 3
    assert all(call[call.index("--optimizer-steps") + 1] == "2" for call in checks)
    assert not any(call[0] == "train.py" for call in calls)
    assert len([call for call in calls if call[0] == "evaluate.py"]) == 1


def test_formal_orders_five_stages_then_full_evaluation(tmp_path):
    result, calls = run_script(tmp_path, "formal")
    assert result.returncode == 0, result.stdout + result.stderr
    work = [call for call in calls if call[0] in ("tools/preflight.py", "train.py", "evaluate.py")]
    assert [call[0] for call in work] == ["tools/preflight.py", "train.py"] * 5 + ["evaluate.py"] * 2
    trains = [call for call in work if call[0] == "train.py"]
    assert [Path(call[call.index("--config") + 1]).stem for call in trains] == [
        "qwen3_vl_8b_" + stage for stage in
        ("stage1a_ir", "stage1b_depth", "stage2_joint", "stage2_weak", "stage2_clean")
    ]
    assert all("--subset-size" not in call for call in work[-2:])
    assert "--checkpoint" not in work[-2] and "--rgb-only" in work[-2]
    assert "--checkpoint" in work[-1]


def test_failed_training_stops_pipeline_through_tee(tmp_path):
    result, calls = run_script(tmp_path, "formal", FAIL_STAGE="stage2_joint")
    assert result.returncode == 23
    assert len([call for call in calls if call[0] == "train.py"]) == 3
    assert not any(call[0] == "evaluate.py" for call in calls)


def test_smoke_rejects_unparseable_generation(tmp_path):
    result, _ = run_script(tmp_path, "smoke", FAIL_PARSE="1")
    assert result.returncode != 0
    assert "No generated bbox parsed successfully" in result.stderr
