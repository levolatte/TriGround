"""Measure real generation/update times and complete save/reload for one variant."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/root/autodl-tmp/quant_rl_20261026')
sys.path.insert(0, str(ROOT / 'code'))

import torch
from transformers import TrainerCallback
from tools.quant_grounding_config import load_config, prepare_dataset
from tools.train_quant_grounding import run_swift


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    config = load_config(args.config)
    config.update(output_dir=str(args.output), save_steps=args.steps, save_total_limit=1,
                  save_hours=[], max_gpu_hours=1.5, benchmark_mode=False)
    config.pop('matched_query_exposures', None)
    assert Path(config['adapters']).name == 'checkpoint-50'
    args.output.mkdir(parents=True)
    dataset = args.output / 'prepared_probe.jsonl'
    prepare_dataset(Path(config['dataset']), dataset, Path(config['data_root']), config['arm'], probe=True)
    os.environ['MAX_PIXELS'] = str(config['max_pixels'])
    os.environ['MIN_PIXELS'] = str(config.get('min_pixels', 200704))
    os.environ['QUANT_REWARD_METRICS_JSONL'] = str(args.output / 'reward_metrics.jsonl')
    measured = []

    def hook(trainer):
        class Clock(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                self.started = time.monotonic()

            def on_step_end(self, args, state, control, **kwargs):
                measured.append({'step': state.global_step, 'complete_update_seconds': time.monotonic() - self.started})
        trainer.add_callback(Clock())

    start = time.monotonic()
    run_swift(config, dataset, max_steps=args.steps, save_steps=args.steps, trainer_hook=hook)
    checkpoint = args.output / f'checkpoint-{args.steps}'
    for name in ['optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'adapter_config.json']:
        assert (checkpoint / name).is_file()
    assert json.loads((checkpoint / 'trainer_state.json').read_text())['global_step'] == args.steps
    forward_updates = list(measured)
    if args.reload:
        gc.collect()
        torch.cuda.empty_cache()
        run_swift(config, dataset, max_steps=args.steps + 1, save_steps=1,
                  resume=str(checkpoint), trainer_hook=hook)
        reloaded = args.output / f'checkpoint-{args.steps + 1}'
        assert json.loads((reloaded / 'trainer_state.json').read_text())['global_step'] == args.steps + 1
    resources = [json.loads(line) for line in (args.output / 'resource_steps.jsonl').read_text().splitlines()]
    metrics = [json.loads(line) for line in (args.output / 'training_metrics.jsonl').read_text().splitlines()]
    grad_norms = {row['step']: float(row['metrics']['grad_norm']) for row in metrics
                  if 'grad_norm' in row['metrics']}
    update_steps = [row['step'] for row in measured]
    if set(grad_norms) != set(update_steps) or len(grad_norms) != len(update_steps):
        raise RuntimeError(f'每步 grad_norm 记录不完整：updates={update_steps}, norms={grad_norms}')
    if config['arm'] != 'C':
        assert all(row['rollout_groups'] == 4 and row['rollout_answers'] == 32 for row in resources)
    report = {'arm': config['arm'], 'config': config, 'updates': measured,
              'first_segment_median_complete_update_seconds': statistics.median(
                  row['complete_update_seconds'] for row in forward_updates),
              'saved_and_reloaded': args.reload,
              'total_seconds_including_loading_and_saving': time.monotonic() - start,
              'update_grad_norms': grad_norms,
              'all_updates_finite_nonzero': all(math.isfinite(grad_norms[step]) and grad_norms[step] > 0
                                                for step in update_steps),
              'peak_allocated_bytes': max(row['allocated_peak_bytes'] for row in resources),
              'peak_reserved_bytes': max(row['reserved_peak_bytes'] for row in resources)}
    if config['arm'] != 'D':
        assert report['all_updates_finite_nonzero'], report
    (args.output / 'live_result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
