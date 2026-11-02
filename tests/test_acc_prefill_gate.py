"""实测门禁必须只启用合格候选，所有失败分支恢复原父队列。"""

import json
import signal
import subprocess
from pathlib import Path

import pytest

from scripts import run_acc_prefill_gate as gate


def setup_gate(tmp_path, monkeypatch, report, code=0, timeout=False):
    # 监督器部署于Linux；本地Windows只模拟进程控制，不发送真实信号。
    for name, number in (("SIGSTOP", 19), ("SIGCONT", 18)):
        monkeypatch.setattr(signal, name, number, raising=False)
    output = tmp_path / "gate"
    train_config = tmp_path / "correction.json"
    gate.write(train_config, {"seed": 2026, "learning_rate": 2e-6})
    config = {"parent_pid": 10, "child_pid": 11, "target": 800,
              "code_root": str(tmp_path / "code"), "run_root": str(tmp_path / "run"),
              "python": "python", "train_config": str(train_config)}
    parent = {"pid": 10, "start_ticks": "100"}
    child = {"pid": 11, "start_ticks": "101"}
    calls = []
    monkeypatch.setattr(gate, "check_initial_processes", lambda _: (parent, child))
    monkeypatch.setattr(gate, "same_process", lambda pid, _: pid == 10)
    monkeypatch.setattr(gate, "check_boundary", lambda _: tmp_path / "run/correction/train/checkpoint-800")
    monkeypatch.setattr(gate.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(gate.os, "killpg", lambda pid, sig: calls.append((pid, sig)), raising=False)

    class Process:
        pid = 12
        returncode = None

        def wait(self, timeout=None):
            if timeout == 1800 and timeout_enabled:
                raise subprocess.TimeoutExpired("benchmark", timeout)
            self.returncode = code
            gate.write(output / "benchmark/report.json", report)
            return code

        def poll(self):
            return self.returncode

    timeout_enabled = timeout
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *args, **kwargs: Process())
    return config, output, train_config, calls


@pytest.mark.parametrize("report,code,accepted", [
    ({"status": "passed", "accepted": True}, 0, True),
    ({"status": "passed", "accepted": False, "reason": "no speedup"}, 0, False),
    ({"status": "failed", "accepted": True}, 1, False),
    ({"status": "passed", "accepted": True}, 1, False),
])
def test_gate_decides_and_always_resumes_original_queue(tmp_path, monkeypatch, report, code, accepted):
    config, output, train_config, calls = setup_gate(tmp_path, monkeypatch, report, code)
    state = gate.run_gate(config, output)
    assert state["accepted"] is accepted
    assert state["original_queue_resumed"] is True
    assert calls == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]
    saved = gate.read(train_config)
    assert saved == {"seed": 2026, "learning_rate": 2e-6} | ({"generation_prefill_reuse": True} if accepted else {})
    assert gate.read(output / "decision.json")["accepted"] is accepted


def test_timeout_stops_probe_before_resuming_original_queue(tmp_path, monkeypatch):
    config, output, train_config, calls = setup_gate(tmp_path, monkeypatch, {}, timeout=True)
    state = gate.run_gate(config, output)
    assert state["status"] == "failed" and state["accepted"] is False
    assert state["original_queue_resumed"] is True
    assert calls == [(10, signal.SIGSTOP), (12, signal.SIGTERM), (10, signal.SIGCONT)]
    assert "generation_prefill_reuse" not in gate.read(train_config)


def test_incomplete_checkpoint_restores_parent_without_running_probe(tmp_path, monkeypatch):
    config, output, _, calls = setup_gate(tmp_path, monkeypatch, {})
    def incomplete(_):
        raise RuntimeError("保存点不完整")
    monkeypatch.setattr(gate, "check_boundary", incomplete)
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *a, **k: pytest.fail("must not run probe"))
    state = gate.run_gate(config, output)
    assert state["status"] == "failed" and state["original_queue_resumed"] is True
    assert calls == [(10, signal.SIGSTOP), (10, signal.SIGCONT)]


def test_changed_process_identity_is_not_resumed(tmp_path, monkeypatch):
    config, output, _, calls = setup_gate(tmp_path, monkeypatch, {"status": "passed", "accepted": False})
    monkeypatch.setattr(gate, "same_process", lambda pid, _: False)
    state = gate.run_gate(config, output)
    assert state["original_queue_resumed"] is False
    assert calls == [(10, signal.SIGSTOP)]
