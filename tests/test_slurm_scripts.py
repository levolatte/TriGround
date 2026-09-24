"""Exercise launcher ordering without GPU work or filesystem artifacts."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
pytestmark = pytest.mark.skipif(not BASH or not Path(BASH).is_file(), reason="Bash unavailable")


def bash_path(path):
    value = Path(path).as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


def run_script(**overrides):
    # Replace side effects, but retain the real script's control flow and pipefail.
    harness = r'''
source() { python_bin=fake_python; }
mkdir() { :; }
tee() { cat; }
fake_python() {
    if [[ "$1" == - ]]; then cat >/dev/null; return 0; fi
    if [[ "$1" == -c ]]; then
        printf '/__rdt_launcher_test__/manifest.json\n'
        return 0
    fi
    printf 'CALL:%s\n' "$1"
    if [[ "$1" == "${FAIL_COMMAND:-}" ]]; then return 23; fi
    return 0
}
builtin source "$1"
'''
    env = dict(os.environ, REPO_DIR=bash_path(ROOT),
               RUN_ROOT="/__rdt_launcher_test__/run", PREFLIGHT_ONLY="0")
    env.update(overrides)
    return subprocess.run(
        [BASH, "-c", harness, "test", bash_path(ROOT / "scripts/qwen3_vl_8b_rdt_qwen_city.slurm")],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )


def calls(result):
    return [line[5:] for line in result.stdout.splitlines() if line.startswith("CALL:")]


def test_only_current_training_launcher_remains():
    assert sorted(p.name for p in (ROOT / "scripts").glob("*.slurm")) == [
        "qwen3_vl_8b_rdt_qwen_city.slurm"
    ]
    script = ROOT / "scripts/qwen3_vl_8b_rdt_qwen_city.slurm"
    result = subprocess.run([BASH, "-n", bash_path(script)], capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_preflight_only_never_starts_training():
    result = run_script(PREFLIGHT_ONLY="1")
    assert result.returncode == 0, result.stderr
    assert calls(result) == ["tools/audit_manifest_overlap.py", "tools/preflight.py"]


def test_formal_checks_then_trains_then_evaluates():
    result = run_script()
    assert result.returncode == 0, result.stderr
    assert calls(result) == [
        "tools/audit_manifest_overlap.py", "tools/preflight.py", "train.py", "evaluate.py"
    ]


@pytest.mark.parametrize("command", ["tools/audit_manifest_overlap.py", "tools/preflight.py", "train.py"])
def test_failure_stops_pipeline_even_through_tee(command):
    result = run_script(FAIL_COMMAND=command)
    assert result.returncode == 23, result.stderr
    assert calls(result)[-1] == command
    assert "evaluate.py" not in calls(result)


def test_invalid_preflight_mode_is_rejected():
    result = run_script(PREFLIGHT_ONLY="yes")
    assert result.returncode == 2
    assert not calls(result)
