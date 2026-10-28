"""Count ACC transitions and training sampling opportunities; no new model inference."""
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
OLD = HERE.parent / '2026-10-26-quant-rl/evidence'

def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def band(sample):
    if not sample['score']['valid']:
        return 'invalid'
    u = sample['iou']
    if u == 0:
        return 'zero_overlap'
    if u < .25:
        return 'overlap_below_0.25'
    if u < .4:
        return '0.25_to_0.4'
    if u < .5:
        return '0.4_to_0.5'
    return 'hit'

report = {}
for split in ('city378', 'external88'):
    paths = {name: OLD / f'punctuation/{stem}-{split}/predictions.jsonl' for name, stem in
             [('warmup50', 'warmup50'), ('B59', 'b-final'), ('C550', 'c-final')]}
    paths.update({arm+'100': OLD/f'ab550-20261028/{arm}100/{split}/predictions.jsonl' for arm in 'AB'})
    data = {name: {row['id']: row for row in read(path)} for name, path in paths.items()}
    keys = data['B100'].keys()
    assert all(rows.keys() == keys for rows in data.values())
    out = {'counts': {}, 'transitions': {}}
    for name, rows in data.items():
        out['counts'][name] = dict(Counter(band(row['samples'][0]) for row in rows.values()))
    for ref in ('warmup50', 'B59', 'A100', 'C550'):
        groups = defaultdict(list)
        for ident, row in data['B100'].items():
            a, b = data[ref][ident]['samples'][0], row['samples'][0]
            label = ('hit' if a['hit'] else 'miss')+'->'+('hit' if b['hit'] else 'miss')
            if label == 'hit->hit':
                continue
            groups[label].append({'id': ident, 'before_band': band(a), 'after_band': band(b)})
        out['transitions'][ref] = {
            k: {'count': len(v), 'before_bands': dict(Counter(x['before_band'] for x in v)),
                'after_bands': dict(Counter(x['after_band'] for x in v)), 'items': v}
            for k, v in groups.items()
        }
    report[split] = out

groups = []
for row in read(HERE/'evidence/B-reward_metrics.jsonl'):
    items = row['items']
    idx = int(items[0]['occurrence_id'].split(':')[-1])
    if 236 <= idx < 400:
        assert len(items) == 8 and len({x['occurrence_id'] for x in items}) == 1
        groups.append(items)
assert len(groups) == 164
by_source = defaultdict(list)
for items in groups:
    by_source[items[0]['source']].append(items)

def sampling_summary(rows):
    hs = [sum(x['hit'] for x in items) for items in rows]
    exp_mixed2 = sum(h*(8-h)/comb(8, 2) for h in hs)
    exp_atleast1_2 = sum(1-comb(8-h,2)/comb(8,2) for h in hs)
    return {'groups':len(rows), 'answers':8*len(rows), 'hit_answers':sum(hs),
            'hit_count_histogram':dict(sorted(Counter(hs).items())),
            'mixed_G8':sum(0<h<8 for h in hs), 'any_hit_G8':sum(h>0 for h in hs),
            'expected_mixed_G2_subsets':exp_mixed2,
            'expected_any_hit_G2_subsets':exp_atleast1_2,
            'mixed_groups_per_100_answers_G8':100*sum(0<h<8 for h in hs)/(8*len(rows)),
            'mixed_groups_per_100_answers_G2':100*exp_mixed2/(2*len(rows))}

report['training_60_100'] = {'all':sampling_summary(groups),
    'by_source':{s:sampling_summary(rows) for s, rows in by_source.items()},
    'limits':'G2 is exact subset enumeration of existing G8 answers, not a new training result; training sample hit rates are not development ACC.'}
(HERE/'evidence/acc-priority.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for split in ('city378','external88'):
    print(split, json.dumps(report[split]['counts'],ensure_ascii=False))
    for ref, transitions in report[split]['transitions'].items():
        print(ref, json.dumps({k:{a:b for a,b in v.items() if a!='items'} for k,v in transitions.items()},ensure_ascii=False))
print(json.dumps(report['training_60_100'],ensure_ascii=False,indent=2))
