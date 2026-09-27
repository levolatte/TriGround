"""Offline full-412 prediction geometry; no model execution or label changes."""
from pathlib import Path
import json
import statistics
import sys

sys.path.insert(0, 'F:/AIC/code')
from tools.report_rematch_experiment import load_manifest, load_run, box_iou

root = Path('F:/AIC/results/gu_diagnosis_20260927')
gt = load_manifest(root / 'source_snapshot/city_gt.json')
paths = {
    'M2': Path('F:/AIC/results/next_stage_20260925/baseline_m2/predictions.jsonl'),
    'C': Path('F:/AIC/results/next_stage_20260925/c_step1500/predictions.jsonl'),
}
paths.update({n: root / f'cloud_snapshot/{folder}_city412/predictions.jsonl'
              for n, folder in [('B', 'b'), ('Gstar', 'gstar'), ('Ustar', 'ustar')]})
runs = {n: load_run(n, p, set(gt), allow_partial=False)['rows'] for n, p in paths.items()}
ious = {n: {i: box_iou(r['primary'], gt[i]['bbox']) for i, r in rs.items()}
        for n, rs in runs.items()}
pairs = {}
for a, b in [('M2', 'B'), ('M2', 'Gstar'), ('M2', 'Ustar'), ('C', 'B'),
             ('C', 'Gstar'), ('C', 'Ustar'), ('B', 'Gstar'), ('B', 'Ustar'), ('Gstar', 'Ustar')]:
    mutual = {i: box_iou(runs[a][i]['primary'], runs[b][i]['primary']) for i in gt}
    flips = [i for i in gt if (ious[a][i] >= .5) != (ious[b][i] >= .5)]
    pairs[a + '->' + b] = {
        'identical_bbox': sum(runs[a][i]['primary'] == runs[b][i]['primary'] for i in gt),
        'mutual_iou_ge07': sum(x >= .7 for x in mutual.values()),
        'mutual_iou_ge09': sum(x >= .9 for x in mutual.values()),
        'mutual_iou_lt05_ids': [i for i, x in mutual.items() if x < .5],
        'flip_count': len(flips),
        'flips_both_gt_iou_035_065': [i for i in flips if .35 <= ious[a][i] <= .65 and .35 <= ious[b][i] <= .65],
        'corrected': [i for i in gt if ious[a][i] < .5 <= ious[b][i]],
        'regressed': [i for i in gt if ious[b][i] < .5 <= ious[a][i]],
        'positive_iou_changes': sum(ious[b][i] > ious[a][i] + 1e-9 for i in gt),
        'negative_iou_changes': sum(ious[b][i] < ious[a][i] - 1e-9 for i in gt),
        'mean_iou_delta': statistics.mean(ious[b][i] - ious[a][i] for i in gt),
    }
common = {}
for ref in ['M2', 'C']:
    misses = {n: set(pairs[ref + '->' + n]['regressed']) for n in ['B', 'Gstar', 'Ustar']}
    common[ref] = {'all_three_regression_ids': sorted(set.intersection(*misses.values())),
                   'any_regression_ids': sorted(set.union(*misses.values()))}
tiny = [i for i, r in gt.items() if min(r['bbox'][2]-r['bbox'][0], r['bbox'][3]-r['bbox'][1]) < .03]
other = [i for i in gt if i not in tiny]
size = {n: {label: {'n': len(ids), 'hits': sum(ious[n][i] >= .5 for i in ids),
                    'miou': statistics.mean(ious[n][i] for i in ids)}
            for label, ids in [('minside_lt003', tiny), ('other', other)]} for n in runs}
out = {'note': 'Full412 descriptive geometry. Mutual-box IoU does not establish visual instance identity. The 0.35-0.65 band is post hoc.',
       'pairs': pairs, 'common_regressions': common, 'target_size': size}
Path(__file__).with_name('full412_geometry.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
for key, val in pairs.items():
    print(key, {k: len(v) if isinstance(v, list) else v for k, v in val.items()})
print('common', common)
print('size', size)
