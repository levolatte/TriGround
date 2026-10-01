"""A/R/S/S+G evaluation: full denominator, literal KEEP, at most two calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from tools.train_grounding_structure import read, write, load_model
from tools.grounding_structure import StructureBundle
from tools.structured_grounder import QwenStructureRuntime, generate, finalized_revision
from tools.cache_grounding_proposals import load_cache
from tools.prepare_grounding_revision import box_iou


def comparison(rows, arm):
    delta = np.array([int(r[arm]['hit'])-int(r['A']['hit']) for r in rows])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['scene_id']].append(index)
    values = [delta[index].sum() for index in groups.values()]
    counts = [len(index) for index in groups.values()]
    rng = np.random.default_rng(2030)
    indices = rng.integers(len(values), size=(5000,len(values)))
    bootstrap = np.asarray(values)[indices].sum(1)/np.asarray(counts)[indices].sum(1)
    return {'denominator':len(rows), 'correct':sum(r[arm]['hit'] for r in rows),
            'acc_05':float(np.mean([r[arm]['hit'] for r in rows])),
            'miou':float(np.mean([r[arm]['iou'] for r in rows])),
            'rescue':int((delta>0).sum()), 'harm':int((delta<0).sum()),
            'net':int(delta.sum()), 'scene_bootstrap_delta_95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'parse_failures':sum(r[arm]['prediction'] is None for r in rows)}


def summarize(rows, arms):
    report = {arm:comparison(rows,arm) for arm in arms}
    for arm in arms:
        city=[r for r in rows if r['task_category']=='old_city' and r['condition']=='normal']
        diagnostic=[r for r in rows if r['task_category']!='old_city' and r['condition']=='normal']
        if city: report[arm]['city412']=comparison(city,arm)
        if diagnostic: report[arm]['diagnostic119']=comparison(diagnostic,arm)
        report[arm]['groups']={}
        for field in ('task_category','condition','city_subset'):
            groups=defaultdict(list)
            for row in rows:
                if field=='task_category' and row['condition']!='normal': continue
                if row.get(field): groups[row[field]].append(row)
            report[arm]['groups'][field]={key:comparison(group,arm) for key,group in groups.items()}
        by_pair=defaultdict(dict)
        for row in rows: by_pair[row['evaluation_mother_id']][row['condition']]=row
        modal={}
        for modality in ('ir','depth'):
            pairs=[(v['normal'],v['without_'+modality]) for v in by_pair.values()
                   if 'normal' in v and 'without_'+modality in v]
            if pairs:
                modal[modality]={'pairs':len(pairs),'normal_correct':sum(a[arm]['hit'] for a,b in pairs),
                    'blank_correct':sum(b[arm]['hit'] for a,b in pairs),
                    'benefit':sum(a[arm]['hit'] and not b[arm]['hit'] for a,b in pairs),
                    'harm':sum(b[arm]['hit'] and not a[arm]['hit'] for a,b in pairs),
                    'net':sum(int(a[arm]['hit'])-int(b[arm]['hit']) for a,b in pairs)}
                by_group=defaultdict(list)
                for a,b in pairs:by_group[a['task_category']].append((a,b))
                modal[modality]['groups']={key:{'pairs':len(group),
                    'normal_correct':sum(a[arm]['hit'] for a,b in group),'blank_correct':sum(b[arm]['hit'] for a,b in group),
                    'net':sum(int(a[arm]['hit'])-int(b[arm]['hit']) for a,b in group)} for key,group in by_group.items()}
        report[arm]['modality_effect']=modal
    return report


def evaluate(args):
    if not torch.cuda.is_available(): raise RuntimeError('GPU evaluation is pending')
    rows=read(args.inputs)
    cache=load_cache(args.proposals,args.model,args.initial_adapter,rows)
    if set(r['id'] for r in rows) != set(cache['rows']): raise ValueError('A evaluation cache is incomplete')
    bundle=StructureBundle.load(args.checkpoint).to('cuda').float().eval()
    model,processor=load_model(args.model,args.initial_adapter,trainable=False)
    model.load_adapter(args.checkpoint,adapter_name='revision',is_trainable=False,local_files_only=True)
    model.set_adapter('revision')
    model.eval().requires_grad_(False)
    runtime=QwenStructureRuntime(model,bundle)
    arm='S' if bundle.modal is not None else 'R'
    arms=['A',arm]+(['S+G'] if bundle.geometry is not None else [])
    config={'inputs':str(args.inputs),'checkpoint':str(args.checkpoint),'cache_metadata':cache['metadata']}
    if (args.output/'run_config.json').exists() and read(args.output/'run_config.json') != config:
        raise ValueError('cannot resume evaluation with changed weights or input')
    write(args.output/'run_config.json',config)
    output=args.output/'predictions.json'
    results=read(output) if output.exists() else []
    if [r['id'] for r in results] != [r['id'] for r in rows[:len(results)]]:
        raise ValueError('evaluation resume is not an input prefix')
    for row in rows[len(results):]:
        first=cache['rows'][row['id']]
        second=generate(model,processor,runtime,row,context=first['context'],proposal=first['prediction'],revision=True)
        record={k:row[k] for k in ('id','scene_id','task_category','condition','evaluation_mother_id','city_subset') if k in row}
        record['target']=row['rgb_gt_bbox']
        record['A']={'prediction':first['prediction'],'raw_text':first['raw_text']}
        for name in arms[1:]:
            raw,box,action=finalized_revision(first,second,continuous=name=='S+G')
            record[name]={'prediction':box,'raw_text':raw,'action':action,'revision_text':second['raw_text'],
                          'calls':2,'latency_seconds':first['latency_seconds']+second['latency_seconds']}
        for name in arms:
            record[name]['iou']=box_iou(record[name]['prediction'],row['rgb_gt_bbox'])
            record[name]['hit']=record[name]['iou']>=.5
        results.append(record)
        if len(results)%10==0 or len(results)==len(rows):
            write(output,results)
            print(f'{arm} evaluation {len(results)}/{len(rows)}',flush=True)
    write(args.output/'summary.json',summarize(results,arms))
    runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','initial-adapter','checkpoint','inputs','proposals','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    evaluate(parser.parse_args())
