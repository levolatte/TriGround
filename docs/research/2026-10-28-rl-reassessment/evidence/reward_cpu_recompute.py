import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('F:/AIC/code')
sys.path.insert(0, str(ROOT))
from tools.quant_grounding_reward import score_completion
from tools.verify_acc05_dev import iou_xyxy


def summary(path, lower=0, upper=10**12):
    groups = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        items=json.loads(line)['items']
        oid=items[0]['occurrence_id']
        occurrence=int(oid.split(':')[1])
        if lower<=occurrence<=upper:
            groups.append(items)
    result = dict(path=str(path), occurrence_min=lower, occurrence_max=upper, groups=len(groups), answers=sum(map(len, groups)))
    types = Counter()
    sources = defaultdict(Counter)
    sums = Counter()
    examples = []
    cf_examples=[]
    for group in groups:
        rs = [i['reward'] for i in group]
        n = len(group)
        hit = sum(i['hit'] for i in group)
        typ = 'all_hit' if hit == n else 'mixed' if hit else 'no_hit'
        active = max(rs) - min(rs) > 1e-10
        types[typ] += 1
        types[typ + '_active'] += active
        sources[group[0]['source']]['groups'] += 1
        sources[group[0]['source']]['active'] += active
        sources[group[0]['source']]['answers'] += n
        sources[group[0]['source']]['hits'] += hit
        av = [r - statistics.mean(rs) for r in rs]
        sums['abs_adv'] += sum(map(abs, av))
        sums['sq_adv'] += sum(a*a for a in av)
        sums['policy_signal_token_mass'] += sum(abs(a)*item['generated_tokens'] for a,item in zip(av,group))
        sums['all_token_mass'] += sum(item['generated_tokens'] for item in group)
        sums['zero_group_token_mass'] += sum(item['generated_tokens'] for item in group) * (not active)
        sums['invalid'] += sum(not i['valid'] for i in group)
        sums['hits'] += hit
        if typ == 'all_hit':
            vals = [i['iou'] for i in group]
            sums['all_hit_iou_span_sum'] += max(vals)-min(vals)
        crs = [1 + .2*i['iou'] if i['valid'] and i['hit'] else .2*i['iou'] if i['valid'] else -.2 for i in group]
        cav = [r-statistics.mean(crs) for r in crs]
        brs = [float(i['hit']) + .2*min(i['iou'],.5) if i['valid'] else -.2 for i in group]
        ars = [b + .1*i['contrast'] if i['valid'] else -.2 for b,i in zip(brs,group)]
        aav = [r-statistics.mean(ars) for r in ars]
        bav = [r-statistics.mean(brs) for r in brs]
        sums['cf_a_active_groups'] += max(ars)-min(ars)>1e-10
        sums['cf_b_active_groups'] += max(brs)-min(brs)>1e-10
        sums['cf_a_abs_adv'] += sum(map(abs,aav))
        sums['cf_b_abs_adv'] += sum(map(abs,bav))
        sums['cf_a_sq_adv'] += sum(x*x for x in aav)
        sums['cf_b_sq_adv'] += sum(x*x for x in bav)
        sums['negative_groups'] += any(i.get('negative_kind') for i in group)
        sums['contrast_varies_groups'] += max(i['contrast'] for i in group)-min(i['contrast'] for i in group)>1e-10
        sums['continuous_abs_adv'] += sum(map(abs,cav))
        sums['continuous_sq_adv'] += sum(a*a for a in cav)
        sums['continuous_active_groups'] += max(crs)-min(crs)>1e-10
        if typ == 'all_hit':
            sums['continuous_allhit_abs_adv'] += sum(map(abs,cav))
            sums['continuous_allhit_sq_adv'] += sum(a*a for a in cav)
        for (xi,x),(yi,y) in itertools.combinations(enumerate(group),2):
            if x['valid'] and y['valid'] and abs(x['iou']-y['iou']) > 1e-9:
                sums['pairs_different_iou'] += 1
                if (x['iou']-y['iou'])*(x['reward']-y['reward']) < -1e-12:
                    sums['iou_reward_rank_inversions'] += 1
                    if len(examples)<8:
                        examples.append(dict(id=x['id'],occurrence_id=x['occurrence_id'], x=x,y=y))
                if (x['iou']-y['iou'])*(ars[xi]-ars[yi]) < -1e-12:
                    sums['cf_a_iou_reward_rank_inversions'] += 1
                    sums['cf_a_allhit_iou_reward_rank_inversions'] += x['hit'] and y['hit']
                    if len(cf_examples)<8:
                        cf_examples.append(dict(id=x['id'],occurrence_id=x['occurrence_id'],x=x,y=y,reward_a_x=ars[xi],reward_a_y=ars[yi]))
        if not active:
            # Equal reward is exactly zero in real arithmetic. Float32 subtraction can
            # contain representation-level residuals, hence retain range-based count.
            sums['zero_groups'] += 1
    result.update(types=dict(types),sources=dict(sources),sums=dict(sums),inversion_examples=examples,cf_a_inversion_examples=cf_examples)
    return result


def inspect_index(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines()]
    c = Counter()
    overlaps=[]
    all_stats = defaultdict(Counter)
    for row in rows:
        source = row['source']
        c[source] += 1
        all_stats[source]['rows'] += 1
        all_stats[source]['with_negative'] += bool(row['negative_boxes'])
        if row['negative_boxes']:
            gtneg=max(iou_xyxy(row['solution'],n) for n in row['negative_boxes'])
            all_stats[source]['gt_negative_iou_ge_05'] += gtneg>=.5
            all_stats[source]['gt_negative_iou_ge_08'] += gtneg>=.8
            overlaps.append((gtneg,row))
    examples=[]
    for gtneg,row in sorted(overlaps,key=lambda x:x[0],reverse=True)[:4]:
        best=None
        gt=row['solution']
        gt_text=json.dumps({'bbox_2d':[v*1000 for v in gt]})
        gt_reward=score_completion(gt_text,gt,row['negative_boxes'])
        width,height=gt[2]-gt[0],gt[3]-gt[1]
        for dx,dy,scale in itertools.product([-.2,-.1,-.05,0,.05,.1,.2],[-.2,-.1,-.05,0,.05,.1,.2],[.8,.9,1,1.1,1.2]):
            cx,cy=(gt[0]+gt[2])/2+dx*width,(gt[1]+gt[3])/2+dy*height
            pred=[max(0,cx-width*scale/2),max(0,cy-height*scale/2),min(1,cx+width*scale/2),min(1,cy+height*scale/2)]
            text=json.dumps({'bbox_2d':[v*1000 for v in pred]})
            sc=score_completion(text,gt,row['negative_boxes'])
            if sc['hit'] and sc['reward']>gt_reward['reward']+1e-8 and (best is None or sc['reward']>best['score']['reward']):
                best=dict(pred=pred,score=sc)
        examples.append(dict(id=row['id'],source=row['source'],negative_kind=row['negative_kind'],gt=gt,negative=row['negative_boxes'],gt_negative_iou=gtneg,gt_reward=gt_reward,better_reward_than_gt=best))
    return dict(path=str(path),rows=len(rows),sources=dict(all_stats),high_overlap_examples=examples)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--log',action='append',default=[])
    p.add_argument('--index')
    p.add_argument('--min',type=int,default=0)
    p.add_argument('--max',type=int,default=10**12)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    result=dict(logs=[summary(x,a.min,a.max) for x in a.log])
    if a.index: result['index']=inspect_index(a.index)
    Path(a.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:([{kk:vv for kk,vv in x.items() if 'examples' not in kk} for x in v] if k=='logs' else {kk:vv for kk,vv in v.items() if 'examples' not in kk}) for k,v in result.items()},ensure_ascii=False,indent=2))
