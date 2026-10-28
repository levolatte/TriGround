"""固定方法/阈值的旧8B缓存实验，不在本集选择参数后声称独立性能。"""
import importlib.metadata
import itertools
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools import ensemble_grounding as eg

OUT = ROOT.parent / 'tmp/structural_20261023/ensemble_cpu'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'docs/audit-20260929/evidence/evaluation'
config = {'members': [
    {'name': name, 'path': str(BASE/name/'city412/predictions.jsonl'),
     'family': name, 'weight': 1} for name in ['A', 'C0', 'B', 'V']]}
eg.write_json(OUT/'members.json', config)
members, tables = eg.load_members(OUT/'members.json')
gt_path = ROOT / 'docs/audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json'
gt = eg.load_gt(gt_path)
assert all(row['target'] == gt[sid] for table in tables.values() for sid, row in table.items())
threshold = .55
outputs = {method: eg.aggregate_table(members, tables, method, threshold) for method in eg.METHODS}
for method, rows in outputs.items():
    path = OUT/method/'predictions.jsonl'
    path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows.values()), encoding='utf-8')
evaluation = eg.evaluate(members, tables, outputs, gt, baseline='A')
individual = evaluation['individual_first_valid_candidate']
pairs = {}
for a, b in itertools.combinations(individual, 2):
    ha, hb = individual[a]['hit_by_id'], individual[b]['hit_by_id']
    pairs[f'{a}+{b}'] = {
        'both': sum(ha[sid] and hb[sid] for sid in gt),
        'first_only': sum(ha[sid] and not hb[sid] for sid in gt),
        'second_only': sum(hb[sid] and not ha[sid] for sid in gt),
        'union': sum(ha[sid] or hb[sid] for sid in gt)}
groups = defaultdict(list)
for sid in gt:
    groups[sid.split('_')[1]].append(sid)
keys = sorted(groups)
rng = random.Random(20261023)
bootstrap_samples = [[sid for key in rng.choices(keys, k=len(keys)) for sid in groups[key]] for _ in range(2000)]
for method, score in evaluation['methods'].items():
    h, baseline = score['hit_by_id'], individual['A']['hit_by_id']
    deltas = sorted(sum(int(h[sid])-int(baseline[sid]) for sid in sample)/len(sample)
                    for sample in bootstrap_samples)
    score['scene_bootstrap_delta_acc_95pct'] = [deltas[50], deltas[1950]]
    score['newly_correct_beyond_original_oracle'] = sum(
        h[sid] and not any(individual[n]['hit_by_id'][sid] for n in individual) for sid in gt)
for score in list(individual.values()) + list(evaluation['methods'].values()):
    del score['hit_by_id']
report_path = ROOT/'docs/research/2026-10-23-structural/ensemble_cpu.json'
portable_members = [dict(row, path=os.path.relpath(row['path'], report_path.parent).replace('\\','/'))
                    for row in config['members']]
report = {
    'purpose': '旧Qwen3-VL-8B四训练臂固定参数CPU聚合实验；非当前27B、非官方成绩、未调参',
    'members': portable_members,
    'path_convention': 'members.path 相对本JSON；gt/outputs相对仓库根目录code，源文件原位于工作区F:/AIC/code',
    'gt': gt_path.relative_to(ROOT).as_posix(), 'threshold': threshold,
    'priority': 'A,C0,B,V；A为历史已知最好单臂，平局固定优先，不由本轮评分动态决定',
    'family_definition': '此实验将不同训练臂视为四个family；未来同一权重的不同尺度/采样必须共享family',
    'groups': {'scene_proxy_count': len(keys), 'definition': 'city ID 的第二段地点字段',
               'image_group_count': len({sid.rsplit('_',1)[0] for sid in gt}),
               'bootstrap_rounds': 2000, 'seed': 20261023},
    'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'ensemble-boxes', 'numba', 'pandas']},
    'evaluation': evaluation, 'pairs': pairs,
    'reproduce': 'python tools/ensemble_grounding.py --config docs/research/2026-10-23-structural/ensemble_cpu.json --compare-all --iou-threshold 0.55 --gt docs/audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json --baseline A --out-dir ../tmp/structural_20261023/ensemble_cpu_rerun',
    'reproduce_with_scene_bootstrap': 'python docs/research/2026-10-23-structural/run_ensemble_cpu.py',
    'outputs': os.path.relpath(OUT, ROOT).replace('\\','/'),
    'interpretation': 'City412已反复选模；场景重采样仅描述本集配对差，不恢复独立泛化。原框oracle不限制融合新框。'}
eg.write_json(report_path, report)
print(json.dumps({'individual': {n:s['hits'] for n,s in individual.items()},
                  'oracle': evaluation['original_candidate_oracle_hits'],
                  'methods': {n:{k:s[k] for k in ['hits','acc_0.5','rescued','damaged','net_hits', 'newly_correct_beyond_original_oracle']} for n,s in evaluation['methods'].items()}}, ensure_ascii=False,indent=2))
