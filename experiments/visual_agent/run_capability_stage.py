"""Run one finite dedicated-GPU stage and charge the combined experiment budget.

This is a subprocess wrapper, not an experiment scheduler. The caller chooses
the concrete command only after the preceding data/model artifacts exist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--old-budget-root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a concrete command is required after --')
    ledger = args.root / 'ledger.jsonl'
    history = [*read_rows(args.old_budget_root / 'z_budget.jsonl'),
               *read_rows(args.old_budget_root / 't_budget.jsonl'), *read_rows(ledger)]
    used = sum(row['elapsed_seconds'] for row in history)
    if used + args.seconds > 20 * 3600:
        raise RuntimeError(f'Combined budget has {(72000-used)/3600:.4f} h; stage requests {args.seconds/3600:.4f} h')
    stage = args.root / 'stages' / args.name
    stage.mkdir(parents=True, exist_ok=False)
    state_path = stage / 'state.json'
    state = {'name': args.name, 'pid': os.getpid(), 'status': 'running',
             'started_unix': time.time(), 'command': command,
             'seconds_limit': args.seconds, 'used_seconds_before': used}
    def save():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    save()
    started = time.monotonic()
    try:
        with (stage / 'output.log').open('w') as log:
            result = subprocess.run(command, cwd=args.cwd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.seconds,
                                    env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        state.update(status='complete' if result.returncode == 0 else 'failed',
                     exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        state.update(status='timeout', exit_code=124)
    finally:
        elapsed = time.monotonic() - started
        state.update(elapsed_seconds=elapsed, ended_unix=time.time())
        with ledger.open('a') as handle:
            handle.write(json.dumps({'task': args.name, 'elapsed_seconds': elapsed,
                                     'status': state['status'], 'stage': str(stage)}) + '\n')
        save()
    print(json.dumps(state, ensure_ascii=False))
    raise SystemExit(state['exit_code'])


if __name__ == '__main__':
    main()
