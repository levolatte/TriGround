"""Unique, order-balanced oracle depth-reading probes; not full grounding data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from tools.prepare_triground_tm_data import _depth_pair, _write_json


def swapped(row):
    row=deepcopy(row)
    row['id'] += ':swapped'
    row['image'].reverse()
    row['roi_source_ids'].reverse()
    row['expected_answer']={'A_nearer':'B_nearer','B_nearer':'A_nearer'}[row['expected_answer']]
    row['conversations'][1]['value']=row['expected_answer']
    row['pair_orientation']='reverse'
    return row


def pixel_pair(row):
    result=[]
    for path in row['image']:
        with Image.open(path) as im:
            result.append(np.asarray(im.convert('RGB')).copy())
    return result


def same_pixels(left,right):
    return all(np.array_equal(a,b) for a,b in zip(left,right,strict=True))


def balanced_pairs(sources, assets, split):
    unique, pixels, excluded = [], [], []
    for source in sorted(sources,key=lambda r:r['task_id']):
        if source['relation']['kind']=='middle':
            excluded.append({'id':source['task_id'], 'reason':'three_subject_middle_is_not_a_two_crop_task'})
            continue
        row=_depth_pair(source,assets,random.Random(2028),invalid=False,
                        sample_id='depth-pair:'+source['task_id'],split=split,reverse=False)
        content=pixel_pair(row)
        duplicate=next((i for i,p in enumerate(pixels) if same_pixels(content,p)),None)
        if duplicate is not None:
            if unique[duplicate]['expected_answer']!=row['expected_answer']:
                raise ValueError('identical visible depth input has conflicting relation labels')
            unique[duplicate]['source_task_ids'].append(source['task_id'])
            continue
        levels=[float(np.median(x[:,:,0][x[:,:,0]>0])) if np.any(x[:,:,0]>0) else None for x in content]
        if None in levels or levels[0]==levels[1]:
            excluded.append({'id':source['task_id'],'reason':'displayed_depth_not_orderable'})
            continue
        visible='A_nearer' if levels[0]>levels[1] else 'B_nearer'
        if visible!=row['expected_answer']:
            raise ValueError('raw depth order contradicts displayed depth order')
        row.update({'source_task_ids':[source['task_id']], 'pair_id':f'{split}:pair:{len(unique)}',
                    'pair_orientation':'forward'})
        unique.append(row); pixels.append(content)
    records=[]
    for row in unique:
        records.extend([row,swapped(row)])
    return records,excluded


def build(release_dir, output):
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    if output.exists():raise FileExistsError('use a new depth-pair output directory')
    release=read(release_dir/'release.json')
    sources={r['task_id']:r for r in map(json.loads,(release_dir/'source_reviews.jsonl').read_text(encoding='utf-8-sig').splitlines())}
    train_ids={r['source_task_id'] for r in read(release['manifests']['M']) if r['task_category']=='depth_relation' or r['task_category']=='depth_joint'}
    # Joint rows retain source category "depth_relation"; ROI tasks have separate categories.
    diagnostic_ids={r['source_task_id'] for r in read(release['diagnostics']['normal']) if r['source_task_id'] in sources and sources[r['source_task_id']].get('relation')}
    if train_ids & diagnostic_ids:raise ValueError('training and held-out sources overlap')
    train, skipped_train=balanced_pairs([sources[i] for i in train_ids],output/'assets','train')
    diag, skipped_diag=balanced_pairs([sources[i] for i in diagnostic_ids],output/'assets','diagnostic')
    _write_json(output/'train.json',train);_write_json(output/'diagnostic.json',diag)
    summary={'scope':'oracle_region_reading_only','train_pairs':len(train)//2,
             'diagnostic_pairs':len(diag)//2,'train_presentations':len(train),'diagnostic_presentations':len(diag),
             'both_orders_per_pair':True,'skipped_train':skipped_train,'skipped_diagnostic':skipped_diag,
             'formal_middle_grounding_unchanged':True,'performance_verified':False}
    _write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.release_dir,args.output_dir),ensure_ascii=False,indent=2))
