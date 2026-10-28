"""Finish the existing A arm after its original queue parent was stopped."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/quant_rl_20261026')
CODE = ROOT / 'code'
DATA = Path('/root/autodl-tmp/rematch_20260922/data')
MODEL = '/root/autodl-tmp/rematch_models/Qwen3.6-27B'
DATASET = DATA / 'quant_grounding_20261026_formal'
RUN = ROOT / 'runs/initial_seed2026/A_seed2026'
LEDGER = ROOT / 'runs/budget.json'
STATE = ROOT / 'runs/optimization_state.json'
PARENT = 180935
sys.path[:0] = [str(CODE), str(ROOT)]
from run_initial_arms import select_checkpoint


def update(**fields):
    state = json.loads(STATE.read_text())
    state.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
    STATE.write_text(json.dumps(state, indent=2) + '\n')


def ledger():
    return json.loads(LEDGER.read_text())


def charge_gap():
    rows = ledger()
    assert not any(row['status'] == 'running' for row in rows)
    last = max(datetime.fromisoformat(row.get('ended_at', row.get('recorded_at'))) for row in rows)
    hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
    if hours > 0:
        subprocess.run([sys.executable, 'tools/run_quant_budget.py', '--ledger', str(LEDGER),
                        '--stage', 'initial', '--label', 'A_seed2026', '--gpu-model', 'RTX4090-48GB',
                        '--record-hours', str(hours), '--note', 'Occupied handoff between A stages'],
                       cwd=CODE, check=True)


def main():
    update(continuation_pid=os.getpid(), phase='awaiting_A_training')
    while True:
        active = [row for row in ledger() if row['status'] == 'running']
        if not active:
            break
        assert len(active) == 1 and active[0]['label'] == 'A_seed2026'
        time.sleep(15)
    training = [row for row in ledger() if row['label'] == 'A_seed2026' and
                any('train_quant_grounding.py' in part for part in row.get('command', []))][-1]
    if training['status'] != 'completed':
        raise RuntimeError(f'A training ended as {training["status"]}')
    checkpoint, step = select_checkpoint(RUN / 'train')
    for name, filename, expected, hours in (
        ('city378', 'city400_dev.jsonl', 378, 378 / 466),
        ('external88', 'external94_dev.jsonl', 88, 88 / 466),
    ):
        out = RUN / name
        summary_path = out / 'summary.json'
        if summary_path.exists():
            assert json.loads(summary_path.read_text())['overall']['queries'] == expected
            continue
        charge_gap()
        update(phase=f'A_evaluation_{name}', checkpoint=str(checkpoint), final_A_step=step)
        command = [sys.executable, 'tools/run_quant_budget.py', '--ledger', str(LEDGER),
                   '--stage', 'initial', '--label', 'A_seed2026', '--gpu-model', 'RTX4090-48GB',
                   '--max-hours', str(hours), '--', sys.executable, 'tools/eval_quant_grounding.py',
                   '--dataset', str(DATASET / filename), '--data-root', str(DATA), '--model', MODEL,
                   '--adapter', str(checkpoint), '--engine', 'hf', '--precision', 'nf4',
                   '--answer-mode', 'short', '--reward-mode', 'contrast', '--samples-per-query', '1',
                   '--hf-max-batch-size', '2', '--output-dir', str(out)]
        subprocess.run(command, cwd=CODE, check=True)
        assert json.loads(summary_path.read_text())['overall']['queries'] == expected
    resources = [json.loads(line) for line in (RUN / 'train/resource_steps.jsonl').read_text().splitlines()]
    resources = [row for row in resources if row['step'] <= step]
    assert len(resources) == step and all(row['rollout_groups'] == 4 and row['rollout_answers'] == 32
                                         for row in resources)
    manifest = {'arm': 'A', 'checkpoint': str(checkpoint), 'global_step': step,
                'completed_query_exposures': sum(row['rollout_groups'] for row in resources),
                'completed_answers': sum(row['rollout_answers'] for row in resources),
                'city378': json.loads((RUN / 'city378/summary.json').read_text()),
                'external88': json.loads((RUN / 'external88/summary.json').read_text()),
                'completed_at': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'runs/a_completed.json').write_text(json.dumps(manifest, indent=2) + '\n')
    process_dir = Path(f'/proc/{PARENT}')
    if process_dir.exists():
        command = (process_dir / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert str(ROOT / 'run_initial_arms.py') in command
        assert not any(row['status'] == 'running' for row in ledger())
        os.kill(PARENT, signal.SIGKILL)  # Only the stopped, now idle queue coordinator.
    queue_path = ROOT / 'runs/initial_seed2026/queue_state.json'
    queue = json.loads(queue_path.read_text())
    queue.update(status='completed', stage='A_complete_before_optimization', pid=None,
                 completed_arms=['A'], selected_checkpoint_global_step=step)
    queue_path.write_text(json.dumps(queue, indent=2) + '\n')
    update(phase='A_complete_ready_for_optimization', final_A_step=step,
           matched_query_exposures=manifest['completed_query_exposures'], continuation_pid=None)
    print(json.dumps({'A_complete': True, 'step': step,
                      'matched_query_exposures': manifest['completed_query_exposures']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        update(phase='A_continuation_failed', error=f'{type(error).__name__}: {error}')
        raise
