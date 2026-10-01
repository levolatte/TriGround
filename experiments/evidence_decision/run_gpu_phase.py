"""Concrete GPU commands for the evidence-decision experiment; print unless executed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path('/root/autodl-tmp/rematch_20260922')
RESULT = ROOT / 'results/visual_agent/evidence_decision_20260930'
MODEL = Path('/root/rematch_models/Qwen3-VL-8B-Instruct')
C_ADAPTER = ROOT / 'results/next_stage_20260925/c_phase2/checkpoint-500'
PHASES = ('reference_holdout', 'capacity', 'untrained_holdout', 'untrained_city412',
          'train', 'trained_holdout', 'trained_city412')


def command_for(phase: str, context_tokens: int, result: Path = RESULT,
                training_data: str = 'train_full.jsonl'):
    code = result / 'code'
    module = [sys.executable, '-m', 'experiments.evidence_decision.']
    data = result / 'data'
    if phase == 'reference_holdout':
        command = [*module[:-1], module[-1] + 'prepare_object_assets', '--stage', 'baseline',
                   '--manifest', str(data / 'holdout120_manifest.jsonl'), '--model', str(MODEL),
                   '--adapter', str(C_ADAPTER), '--output', str(result / 'reference_holdout/predictions.jsonl'),
                   '--max-run-seconds', '800']
        seconds = 900
    elif phase in ('capacity', 'train'):
        command = [*module[:-1], module[-1] + 'train', '--train', str(data / training_data),
                   '--model', str(MODEL), '--init-adapter', str(C_ADAPTER), '--max-length', str(context_tokens),
                   '--validated-lengths', str(result / f'cpu_preflight_full_{context_tokens}/preflight_lengths.json'),
                   '--output-dir', str(result / (f'capacity_{context_tokens}' if phase == 'capacity' else 'model')),
                   '--loss-mode', 'token-mean']
        if phase == 'capacity':
            command += ['--capacity-only']
            seconds = 720
        else:
            command += ['--epochs', '1', '--max-run-seconds', '8500', '--save-steps', '50']
            seconds = 9000
    else:
        trained = phase.startswith('trained_')
        cohort = phase.split('_', 1)[1]
        manifest = (data / 'holdout120_manifest.jsonl' if cohort == 'holdout'
                    else ROOT / 'results/visual_agent/capability_rebuild_20260929/data/city412.jsonl')
        candidates = (data / 'holdout120_candidates_frozenC.jsonl' if cohort == 'holdout'
                      else ROOT / 'results/aux_selection_refine_20260927/city412/candidates.jsonl')
        seconds = 900 if cohort == 'holdout' else 3960
        command = [*module[:-1], module[-1] + 'run_objects', '--manifest', str(manifest),
                   '--candidate-cache', str(candidates), '--model', str(MODEL),
                   '--adapter', str(result / 'model/adapter' if trained else C_ADAPTER),
                   '--context-tokens', str(context_tokens), '--output-dir', str(result / phase),
                   '--dino-model', str(ROOT / 'models/grounding-dino-tiny'),
                   '--sam-model', str(ROOT / 'models/sam2.1-hiera-tiny'),
                   '--max-run-seconds', str(seconds - 100)]
    wrapped = [sys.executable, '-m', 'experiments.evidence_decision.run_capability_stage',
               '--root', str(result), '--old-budget-root', str(ROOT / 'results/visual_agent/implementation_20260928'),
               '--prior-ledger', str(ROOT / 'results/visual_agent/capability_rebuild_20260929/ledger.jsonl'),
               '--name', f'{phase}_{context_tokens}', '--seconds', str(seconds), '--cwd', str(code), '--', *command]
    return code, wrapped, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--context-tokens', type=int, choices=(6144, 8192), default=8192)
    parser.add_argument('--result-root', type=Path, default=RESULT)
    parser.add_argument('--training-data', default='train_full.jsonl',
                        help='Frozen merged training JSONL under result-root/data')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cwd, command, seconds = command_for(args.phase, args.context_tokens, args.result_root,
                                       args.training_data)
    print(json.dumps({'phase': args.phase, 'command': command, 'cwd': str(cwd),
                      'seconds_limit': seconds, 'execute': args.execute}, ensure_ascii=False), flush=True)
    if args.execute:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('The dedicated host has no GPU; this phase has not started.')
        subprocess.run(command, cwd=cwd, check=True)


if __name__ == '__main__':
    main()
