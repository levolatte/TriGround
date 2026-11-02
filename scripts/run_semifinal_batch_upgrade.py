"""暂停现有半决赛进程，短测批量并从已完成 Query 继续。

**迁移期一次性工具，已不在关键路径（2026-10-30 审计）**。它的用途是当年在 HF 与 vLLM
之间热切换引擎：暂停父进程、向子进程发 SIGTERM、等它退出、跑完探针再按选中的批大小
继续。为了做到中途换引擎，这里造了 PID 信号编排（SIGSTOP/SIGTERM/进程组 SIGKILL）、
超时返回 124、以及失败后回退 batch=2 的完整状态机——对竞赛代码属于过度工程。

当前半决赛推理走直连入口 `run_correction1600_k20_nf4_semifinal.py`，不需要本文件。
保留仅因为它记录了 vLLM 被采纳的完整过程（probe → deploy → adopt 三个阶段）。
**不要把它当作当前入口使用。**
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--code-dir', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--old-parent-pid', type=int, required=True)
    parser.add_argument('--old-child-pid', type=int, required=True)
    parser.add_argument('--probe', choices=['batch', 'vllm', 'vllm_deploy', 'vllm_adopt'], default='batch')
    args = parser.parse_args()
    root, code = args.run_dir, args.code_dir
    full = root / 'full'
    config = json.loads((full / 'run_config.json').read_text())
    state = json.loads((root / 'state.json').read_text())
    original_state = root / f'state_before_{args.probe}_upgrade.json'
    if original_state.exists():
        raise FileExistsError(original_state)
    original_state.write_text(json.dumps(state, indent=2) + '\n')

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        (root / 'state.json').write_text(json.dumps(state, indent=2) + '\n')

    def run(command, log, timeout=None):
        with log.open('w') as stream:
            child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save(child_pid=child.pid, command=command)
            try:
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                return 124

    try:
        parent_name = 'run_correction1600_semifinal.py' if args.probe == 'batch' else 'run_semifinal_batch_upgrade.py'
        for pid, expected in ((args.old_parent_pid, parent_name),
                              (args.old_child_pid, 'predict_quant_submission.py')):
            command = Path(f'/proc/{pid}/cmdline').read_bytes().decode()
            if expected not in command:
                raise RuntimeError(f'PID {pid} does not match {expected}')
        os.kill(args.old_parent_pid, signal.SIGSTOP)
        os.kill(args.old_child_pid, signal.SIGTERM)
        for _ in range(30):
            proc = Path(f'/proc/{args.old_child_pid}/stat')
            if not proc.exists() or proc.read_text().split(') ', 1)[1].startswith('Z'):
                break
            time.sleep(1)
        else:
            raise RuntimeError('old inference did not exit after SIGTERM')
        os.kill(args.old_parent_pid, signal.SIGTERM)
        os.kill(args.old_parent_pid, signal.SIGCONT)
        raw = (full / 'predictions_raw.jsonl').read_bytes()
        if not raw.endswith(b'\n'):
            raise RuntimeError('interrupted raw output has an incomplete final line')
        completed = len(raw.splitlines())
        save(status=f'benchmarking_{args.probe}', supervisor_pid=os.getpid(), preserved_queries=completed)
        benchmark = root / {'batch': 'batch_benchmark', 'vllm': 'vllm_nf4_probe',
                            'vllm_deploy': 'vllm_validation', 'vllm_adopt': 'vllm_user_adoption'}[args.probe]
        if args.probe == 'batch':
            probe_command = [args.python, str(code / 'tools/benchmark_quant_submission.py'),
                             '--run-dir', str(full), '--output-dir', str(benchmark)]
        elif args.probe == 'vllm':
            probe_command = ['/root/vllm_env/bin/python', str(code / 'tools/probe_vllm_nf4_submission.py'),
                             '--run-dir', str(root), '--output-dir', str(benchmark)]
        else:
            probe_command = ['/root/vllm_env/bin/python', '-u', str(code / 'tools/validate_vllm_and_resume_submission.py'),
                             '--run-dir', str(root), '--code-dir', str(code)]
        if args.probe == 'vllm_adopt':
            probe_command.append('--adopt-existing-validation')
        timeout = {'batch': 600, 'vllm': 900, 'vllm_deploy': None, 'vllm_adopt': None}[args.probe]
        rc = run(probe_command, root / f'{args.probe}_benchmark.log', timeout=timeout)
        if args.probe in ('vllm_deploy', 'vllm_adopt'):
            decision_path = benchmark / 'decision.json'
            deployment = json.loads(decision_path.read_text()) if decision_path.exists() else {}
            if deployment.get('accepted'):
                if rc or not (full / 'submission.zip').is_file():
                    raise RuntimeError(f'accepted vLLM inference failed with exit {rc}; see vllm_deploy_benchmark.log')
                return
        if args.probe == 'vllm_adopt':
            raise RuntimeError(f'user-adopted vLLM exited {rc}; see vllm_adopt_benchmark.log')
        report_file = benchmark / 'report.json'
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        selected = report.get('selected_batch_size', 2) if rc == 0 and report.get('status') == 'completed' else 2
        decision = {'benchmark_exit_code': rc, 'selected_batch_size': selected,
                    'preserved_queries': completed, 'report': str(report_file),
                    'reason': 'exact-output matched and faster' if selected > 2 else 'no accepted acceleration; continue baseline'}
        (root / f'{args.probe}_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
        command = [args.python, str(code / 'tools/predict_quant_submission.py'), '--resume']
        for key in ('queries', 'data_root', 'model', 'adapter', 'min_pixels', 'max_pixels',
                    'max_length', 'max_tokens', 'seed', 'format_policy'):
            command.extend(['--' + key.replace('_', '-'), str(config[key])])
        command.extend(['--output-dir', str(full), '--hf-max-batch-size', str(selected)])
        save(status='running', selected_batch_size=selected, batch_decision=decision)
        rc = run(command, root / f'inference_resumed_{args.probe}.log')
        if rc:
            raise RuntimeError(f'resumed inference exited {rc}; see inference_resumed.log')
        predictions = [json.loads(line) for line in (full / 'predictions_raw.jsonl').read_text().splitlines()]
        if len(predictions) != state['expected_queries'] or not (full / 'submission.zip').is_file():
            raise RuntimeError('full prediction output or submission is incomplete')
        save(status='completed', completed_queries=len(predictions),
             submission=str(full / 'submission.zip'), finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        save(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
