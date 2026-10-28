"""Recompute paired development metrics and original schedule coverage; no GPU work."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OLD = HERE.parent / '2026-10-26-quant-rl/evidence'

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

report = {}
for split in ('city378', 'external88'):
    paths = {name: OLD / f'punctuation/{stem}-{split}/predictions.jsonl' for name, stem in
             [('warmup50', 'warmup50'), ('A59', 'a-final'), ('B59', 'b-final'),
              ('C29', 'c-matched'), ('C550', 'c-final')]}
    paths.update({arm+'100': OLD / f'ab550-20261028/{arm}100/{split}/predictions.jsonl' for arm in 'AB'})
    data = {name: {row['id']: row for row in read(path)} for name,path in paths.items()}
    keys = data['B100'].keys()
    assert all(rows.keys() == keys for rows in data.values())
    def iou(row):
        s = row['samples'][0]
        return s['iou'] if s['score']['valid'] else 0.
    summary = {name: {'n':len(rows), 'hits':sum(r['samples'][0]['hit'] for r in rows.values()),
                       'miou':sum(map(iou,rows.values()))/len(rows),
                       'iou75':sum(iou(r)>=.75 for r in rows.values()),
                       'iou90':sum(iou(r)>=.90 for r in rows.values())} for name,rows in data.items()}
    pairs = {}
    for reference in ('warmup50','B59','C550'):
        groups = {}
        for ident,row in data['B100'].items():
            prior=data[reference][ident]
            label=('hit' if prior['samples'][0]['hit'] else 'miss')+'->'+('hit' if row['samples'][0]['hit'] else 'miss')
            groups.setdefault(label,[]).append((ident,iou(row)-iou(prior)))
        pairs[reference] = {k:{'n':len(v),'sum_iou_delta':sum(d for _,d in v),
                               'mean_iou_delta':sum(d for _,d in v)/len(v),'ids':[i for i,_ in v]} for k,v in groups.items()}
    report[split]={'summary':summary,'B100_pairs':pairs,
                   'group_ids':len({r['group_id'] for r in data['B100'].values()}),
                   'inference_limit':'one seed, correlated queries/frames; no independent-scene significance claimed'}
schedule=read(HERE/'evidence/formal-schedule.jsonl')
report['coverage']={}
for label,n in [('C29',232),('B59',236),('B100',400),('B550_planned',2200),('C550',4400)]:
    rows=schedule[:n]
    report['coverage'][label]={'presentations':len(rows),'unique_queries':len({r['id'] for r in rows}),
        'image_groups':len({r['group_id'] for r in rows}),
        'source_counts':dict(Counter(r['source'] for r in rows)),
        'pool_counts':dict(Counter(r['sampling_pool'] for r in rows))}
(HERE/'evidence/paired-and-coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v['summary'] for k,v in report.items() if k!='coverage'},indent=2))
print(json.dumps(report['coverage'],indent=2))
for split in ('city378','external88'):
    for ref,pairs in report[split]['B100_pairs'].items():
        print(split,ref,{k:{a:b for a,b in v.items() if a!='ids'} for k,v in pairs.items()})
