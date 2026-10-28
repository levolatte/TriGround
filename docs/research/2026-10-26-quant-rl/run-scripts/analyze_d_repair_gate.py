"""CPU diagnosis of the completed first D repair probe; no reward changes."""
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))
from tools.eval_quant_grounding import read_jsonl, score_rows
from tools.quant_grounding_reward import score_completion

report_dir = CODE / 'docs/research/2026-10-26-quant-rl/evidence/d-repair-20261027'
dataset = read_jsonl(report_dir / 'train4.jsonl')
raw = read_jsonl(report_dir / 'gate4/predictions_raw.jsonl')
scored = score_rows(dataset, raw, 'thinking', 'contrast', 'single_box')
saved = read_jsonl(report_dir / 'gate4/predictions.jsonl')
if scored != saved:
    raise RuntimeError('CPU score replay differs from recorded GPU-probe scoring')
groups = []
for row, target in zip(scored, dataset):
    samples = row['samples']
    valid = [sample for sample in samples if sample['score']['valid']]
    invalid = []
    for index, sample in enumerate(samples):
        if sample['score']['valid']:
            continue
        # Counterfactual only: the saved thinking-policy result remains invalid.
        final_only = sample['text'].rsplit('</think>', 1)[-1].strip()
        final_score = score_completion(final_only, target['solution'], target.get('negative_boxes', []),
                                       answer_mode='short', finish_reason=sample['finish_reason'],
                                       require_termination=True)
        invalid.append({'sample_index': index, 'closing_delimiters': sample['text'].count('</think>'),
                        'final_only': final_only, 'final_only_score': final_score,
                        'counterfactual_only': True})
    groups.append({'id': row['id'], 'valid': len(valid),
                   'valid_iou_min': min(sample['iou'] for sample in valid),
                   'valid_iou_max': max(sample['iou'] for sample in valid),
                   'valid_rewards': sorted({sample['score']['reward'] for sample in valid}),
                   'valid_contrasts': sorted({sample['score']['contrast'] for sample in valid}),
                   'all_valid_boxes_hit': all(sample['hit'] for sample in valid),
                   'invalid': invalid})
result = {'queries': len(scored), 'answers': sum(len(row['samples']) for row in scored),
          'valid': sum(group['valid'] for group in groups),
          'all_valid_boxes_hit': all(group['all_valid_boxes_hit'] for group in groups),
          'forced_boundaries': sum(sample['thinking_trace']['thought_boundary']=='forced'
                                   for row in raw for sample in row['samples']),
          'groups': groups, 'actual_optimizer_updates': 0,
          'conclusion': 'Format/completion recovery observed; geometry-reward gate failed because valid outputs saturate the original reward.'}
(report_dir / 'cpu_diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
