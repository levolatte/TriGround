"""Run one bounded optimization measurement in the single experiment ledger."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/quant_rl_20261026')
CODE = ROOT / 'code'
LEDGER = ROOT / 'runs/budget.json'
STATE = ROOT / 'runs/optimization_state.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--max-hours', type=float, required=True)
    parser.add_argument('--wait-a', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    while args.wait_a and not (ROOT / 'runs/a_completed.json').is_file():
        state = json.loads(STATE.read_text())
        if state['phase'] == 'A_continuation_failed':
            raise RuntimeError(state)
        time.sleep(15)
    completed = json.loads((ROOT / 'runs/a_completed.json').read_text())
    exposures = int(completed['completed_query_exposures'])
    if (completed['arm'] != 'A' or exposures <= 0 or exposures % 4
            or int(completed['global_step']) * 4 != exposures
            or int(completed['completed_answers']) != exposures * 8
            or completed['city378']['overall']['queries'] != 378
            or completed['external88']['overall']['queries'] != 88):
        raise RuntimeError('A completion record is not a full 378/88 evaluated run')
    rows = json.loads(LEDGER.read_text())
    assert not any(row['status'] == 'running' for row in rows)
    latest = max(datetime.fromisoformat(row.get('ended_at', row.get('recorded_at'))) for row in rows)
    gap = (datetime.now(timezone.utc) - latest).total_seconds() / 3600
    prefix = [sys.executable, str(CODE / 'tools/run_quant_budget.py'), '--ledger', str(LEDGER),
              '--stage', 'optimization', '--label', args.label, '--gpu-model', 'RTX4090-48GB']
    if gap > 0:
        subprocess.run([*prefix, '--record-hours', str(gap), '--note', 'Occupied optimization preparation interval'],
                       cwd=CODE, check=True)
    state = json.loads(STATE.read_text())
    state.update(phase='optimization_measurement', active_measurement=args.label,
                 updated_at=datetime.now(timezone.utc).isoformat())
    STATE.write_text(json.dumps(state, indent=2) + '\n')
    process = subprocess.Popen([*prefix, '--max-hours', str(args.max_hours), '--', *command], cwd=CODE)
    def sample_memory():
        line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        used, total = [int(value.strip()) for value in line.split(',')]
        return {'time': datetime.now(timezone.utc).isoformat(), 'used_mib': used, 'total_mib': total}

    memory = [sample_memory()]
    while process.poll() is None:
        time.sleep(1)
        memory.append(sample_memory())
    output = ROOT / 'runs/optimization_measurements'
    output.mkdir(exist_ok=True)
    report = {'label': args.label, 'exit_code': process.returncode, 'command': command,
              'peak_device_used_mib': max(item['used_mib'] for item in memory),
              'minimum_device_headroom_mib': min(item['total_mib'] - item['used_mib'] for item in memory),
              'device_total_mib': memory[0]['total_mib'], 'memory_sample_interval_seconds': 1,
              'samples': memory}
    (output / (args.label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    state.update(phase='optimization_measurement_complete' if process.returncode == 0 else 'optimization_measurement_failed',
                 measurement_exit_code=process.returncode, measurement_report=str(output / (args.label + '.json')),
                 updated_at=datetime.now(timezone.utc).isoformat())
    STATE.write_text(json.dumps(state, indent=2) + '\n')
    raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
