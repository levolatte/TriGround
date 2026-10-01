"""Export visually reviewed RGB-sufficient cases that actually contain IR.

These controls measure harm from IR. They are never modality-benefit positives.
Reviews may correct a query while preserving the approved floating RGB box.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from tools.prepare_triground_tm_data import _bbox_task, _write_json


def build(candidates, reviews, excluded_rows, output):
    if output.exists():
        raise FileExistsError('use a new control output directory')
    sources={r['task_id']:r for r in candidates}
    excluded_ids={r.get('source_task_id') for r in excluded_rows}
    excluded_scenes={r.get('scene_id') for r in excluded_rows}
    selected=[]
    for review in reviews:
        if review['decision']!='accept':
            continue
        source=deepcopy(sources[review['task_id']])
        if source['task_id'] in excluded_ids or source['scene_id'] in excluded_scenes:
            raise ValueError(f"control scene used in training: {source['task_id']}")
        if not review['rgb_only_unique'] or 'infrared' not in source['images']:
            raise ValueError('IR harm controls require RGB-only uniqueness and real IR')
        source['query']=review['final_query']
        source['control_review']=review
        selected.append(source)
    if len({r['task_id'] for r in selected})!=len(selected):
        raise ValueError('duplicate control review')
    normal=[]; missing=[]; gt={}; scenes={}; classes={}
    for source in selected:
        sid='rgb-ir-control:'+source['task_id']
        normal.append(_bbox_task(source,sid,output/'assets',split='diagnostic'))
        missing.append(_bbox_task(source,sid,output/'assets',split='diagnostic',
                                  low='infrared',condition='ir_missing'))
        gt[sid]={'bbox':source['bbox']}
        scenes[sid]=source['scene_id'];classes[sid]='rgb_sufficient'
    _write_json(output/'normal.json',normal)
    _write_json(output/'ir_missing.json',missing)
    _write_json(output/'gt.json',gt)
    _write_json(output/'scene_map.json',scenes)
    _write_json(output/'class_map.json',classes)
    _write_json(output/'reviewed_sources.json',selected)
    summary={'purpose':'RGB-sufficient IR harm controls; not IR benefit evidence',
             'samples':len(normal),'scene_ids':len(set(scenes.values())),
             'known_sequence_groups':len({r['sequence_group'] for r in selected}),
             'query_corrections':sum(r['query']!=sources[r['task_id']]['query'] for r in selected),
             'gt_boxes_changed':0,'all_have_real_ir':True,'model_evaluated':False,
             'historical_independence':'no source/scene overlap in supplied B/M manifests; '
               'adjacent driving scenes and unknown older consumption are not proven independent'}
    _write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--reviews',type=Path,required=True)
    parser.add_argument('--exclude-manifest',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    excluded=[r for p in args.exclude_manifest for r in read(p)]
    print(json.dumps(build(read(args.candidates),read(args.reviews),excluded,args.output_dir),
                     ensure_ascii=False,indent=2))
