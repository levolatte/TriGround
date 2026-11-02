"""固定半决赛题目比较 HF 批量吞吐；只生成诊断文件，不改正式预测。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import create_engine, generate_rows, read_jsonl
from tools.predict_quant_submission import build_parser
from tools.quant_grounding_postprocess import normalize_bbox_completion
from tools.quant_grounding_reward import parse_bbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=16)
    parser.add_argument('--batch-sizes', default='2,4,8')
    cli = parser.parse_args()
    import torch

    config = json.loads((cli.run_dir / 'run_config.json').read_text())
    sizes = [int(x) for x in cli.batch_sizes.split(',')]
    if sizes[0] != 2 or any(x <= 0 for x in sizes) or cli.sample_count < max(sizes):
        raise ValueError('start with baseline batch 2; sample count must cover every batch')
    all_rows = read_jsonl(cli.run_dir / 'unlabeled.jsonl')
    indices = [round(i * (len(all_rows) - 1) / (cli.sample_count - 1))
               for i in range(cli.sample_count)]
    rows = [all_rows[i] for i in indices]
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    args = build_parser().parse_args([
        '--queries', config['queries'], '--data-root', config['data_root'],
        '--output-dir', str(cli.output_dir), '--model', config['model'],
        '--adapter', config['adapter'], '--hf-max-batch-size', '2',
    ])
    for key in ('min_pixels', 'max_pixels', 'max_length', 'max_tokens', 'seed', 'format_policy'):
        setattr(args, key, config[key])
    args.engine, args.precision, args.answer_mode = 'hf', 'nf4', 'short'
    args.samples_per_query, args.resume = 1, False
    report = {'status': 'running', 'sample_ids': [x['id'] for x in rows],
              'sizes': sizes, 'trials': [], 'selected_batch_size': 2}
    report_path = cli.output_dir / 'report.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2) + '\n')

    save()
    os.environ['MIN_PIXELS'] = str(args.min_pixels)
    os.environ['MAX_PIXELS'] = str(args.max_pixels)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine = create_engine(args)
    baseline_text = baseline_boxes = None

    def trial(size, label, selected_rows):
        nonlocal baseline_text, baseline_boxes
        run_args = deepcopy(args)
        run_args.hf_max_batch_size = size
        run_args.output_dir = cli.output_dir / label
        run_args.output_dir.mkdir()
        engine.max_batch_size = size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        predictions, resource = generate_rows(run_args, selected_rows, engine=engine)
        elapsed = time.monotonic() - started
        texts = [row['samples'][0]['text'] for row in predictions]
        boxes = [parse_bbox(normalize_bbox_completion(x, 'short', args.format_policy), 'short')
                 for x in texts]
        result = {'label': label, 'batch_size': size, 'queries': len(predictions),
                  'seconds': elapsed, 'seconds_per_query': elapsed / len(predictions),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'valid_boxes': sum(box is not None for box in boxes), 'resource': resource}
        if label == 'batch2':
            baseline_text, baseline_boxes = texts, boxes
        if baseline_text is not None and len(texts) == len(baseline_text):
            result.update(text_equal=texts == baseline_text, boxes_equal=boxes == baseline_boxes,
                          changed_ids=[row['id'] for row, x, y in zip(selected_rows, texts, baseline_text) if x != y])
        report['trials'].append(result)
        save()
        print(json.dumps(result), flush=True)
        return result

    trial(2, 'warmup', rows[:2])
    base = trial(2, 'batch2', rows)
    candidates = []
    for size in sizes[1:]:
        try:
            result = trial(size, f'batch{size}', rows)
        except torch.OutOfMemoryError as error:
            report['trials'].append({'batch_size': size, 'status': 'oom', 'error': str(error)})
            save()
            torch.cuda.empty_cache()
            break
        if (result['text_equal'] and result['boxes_equal'] and result['valid_boxes'] == len(rows)
                and result['seconds'] < base['seconds'] * .95):
            candidates.append(result)
    if candidates:
        best = min(candidates, key=lambda x: x['seconds'])
        confirm = trial(best['batch_size'], 'confirm_candidate', rows)
        confirm_base = trial(2, 'confirm_baseline', rows)
        if (confirm['text_equal'] and confirm['boxes_equal'] and confirm_base['text_equal']
                and confirm['seconds'] < confirm_base['seconds'] * .95):
            report['selected_batch_size'] = best['batch_size']
            report['confirmed_speedup'] = confirm_base['seconds'] / confirm['seconds']
    report['status'] = 'completed'
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
