"""在现有训练段完整结束后实测提示复用，并无条件恢复原云端队列。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "rng.pt", "loop_state.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def process_info(pid):
    root = Path(f"/proc/{pid}")
    try:
        stat = (root / "stat").read_text()
        command = (root / "cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "start_ticks": fields[19], "command": command[:-1]}


def same_process(pid, identity):
    current = process_info(pid)
    return current is not None and current["start_ticks"] == identity["start_ticks"] and current["state"] != "Z"


def check_initial_processes(config):
    parent, child = process_info(config["parent_pid"]), process_info(config["child_pid"])
    if parent is None or child is None or parent["state"] == "Z" or child["state"] == "Z":
        raise RuntimeError("指定的现役父队列或训练子进程已退出")
    if str(Path(config["code_root"]) / "scripts/run_acc_correction.py") not in parent["command"]:
        raise RuntimeError("父进程不是指定的新实验队列")
    if str(Path(config["code_root"]) / "tools/train_quant_grounding.py") not in child["command"]:
        raise RuntimeError("子进程不是指定的新实验训练")
    if child["ppid"] != parent["pid"]:
        raise RuntimeError("训练子进程不属于指定父队列")
    run = Path(config["run_root"])
    if str(run) not in parent["command"] or str(run / "correction/train") not in child["command"]:
        raise RuntimeError("父子进程不属于当前纠错输出目录")
    state = read(run / "state.json")
    expected = f"correction_train_to_{config['target']}"
    if state["current_job"] != expected or state["jobs"][-1].get("pid") != child["pid"]:
        raise RuntimeError("当前作业已改变；未暂停任何进程")
    return parent, child


def check_boundary(config):
    train = Path(config["run_root"]) / "correction/train"
    checkpoint = train / f"checkpoint-{config['target']}"
    for name in CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"保存点不完整: {path}")
    saved, completed = read(checkpoint / "loop_state.json"), read(train / "completed.json")
    if saved["presentations"] != config["target"] or saved["mode"] != "correction":
        raise RuntimeError("保存点题量或训练模式不符")
    if completed["presentations"] != config["target"] or Path(completed["checkpoint"]) != checkpoint:
        raise RuntimeError("原训练段未正常完整结束")
    gpu_pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if gpu_pids:
        raise RuntimeError(f"GPU仍有计算进程，跳过提速检测: {gpu_pids}")
    return checkpoint


def stop_benchmark(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def apply_decision(report, exit_code, train_config):
    accepted = exit_code == 0 and report.get("status") == "passed" and report.get("accepted") is True
    if accepted:
        config = read(train_config)
        config["generation_prefill_reuse"] = True
        write(train_config, config)
    return accepted


def run_gate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {"status": "starting", "started_at": now(), "parent_pid": config["parent_pid"],
             "child_pid": config["child_pid"], "target": config["target"],
             "accepted": False, "original_queue_resumed": False}
    write(output / "config.json", config)
    write(output / "state.json", state)
    parent, child = None, None
    paused, benchmark = False, None
    try:
        parent, child = check_initial_processes(config)
        # 只暂停调度父进程；现役训练子进程自然完成整段并释放GPU。
        os.kill(parent["pid"], signal.SIGSTOP)
        paused = True
        state.update(status="waiting_for_boundary", parent_identity=parent, child_identity=child)
        write(output / "state.json", state)
        while same_process(child["pid"], child):
            if not same_process(parent["pid"], parent):
                raise RuntimeError("等待期间原父队列退出")
            time.sleep(config.get("poll_seconds", 15))
        checkpoint = check_boundary(config)
        state.update(status="benchmarking", checkpoint=str(checkpoint), boundary_completed_at=now())
        write(output / "original_train_config.json", read(config["train_config"]))
        command = [config["python"], str(Path(config["code_root"]) / "tools/benchmark_quant_prefill.py"),
                   "--config", config["train_config"], "--adapter", str(checkpoint),
                   "--output-dir", str(output / "benchmark")]
        state["benchmark_command"] = command
        write(output / "state.json", state)
        with (output / "benchmark.log").open("a", encoding="utf-8") as stream:
            benchmark = subprocess.Popen(command, cwd=config["code_root"], stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state["benchmark_pid"] = benchmark.pid
            write(output / "state.json", state)
            try:
                code = benchmark.wait(timeout=config.get("benchmark_timeout_seconds", 1800))
            except subprocess.TimeoutExpired:
                stop_benchmark(benchmark)
                raise RuntimeError("提示复用检测超时；保留原训练路径")
        report_path = output / "benchmark/report.json"
        report = read(report_path) if report_path.is_file() else {"status": "failed", "reason": "检测未产生报告"}
        state.update(accepted=apply_decision(report, code, config["train_config"]),
                     benchmark_exit_code=code, benchmark_report=str(report_path),
                     benchmark_reason=report.get("reason"), status="completed")
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        # 检测失败不得把原队列留在T状态，也不得让残留检测与训练抢GPU。
        try:
            stop_benchmark(benchmark)
        finally:
            if paused and same_process(parent["pid"], parent):
                os.kill(parent["pid"], signal.SIGCONT)
                state["original_queue_resumed"] = True
            state.update(ended_at=now())
            write(output / "state.json", state)
            write(output / "decision.json", {key: state.get(key) for key in (
                "status", "accepted", "benchmark_reason", "error", "checkpoint", "benchmark_report",
                "original_queue_resumed", "ended_at")})
            outcome = "启用双答提示复用，后续纠错训练段使用新路径。" if state["accepted"] else "保持原生成路径，继续原训练与评价。"
            (output / "decision.md").write_text(
                "# 双答提示复用实测决策\n\n" + outcome + "\n\n"
                + f"原父队列已恢复：{state['original_queue_resumed']}。\n\n"
                + f"详情：decision.json、benchmark/report.json、benchmark.log。\n", encoding="utf-8")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"检测监督收到信号 {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(json.dumps(run_gate(read(args.config), args.output_dir), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
