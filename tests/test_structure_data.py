from collections import Counter
from pathlib import Path
import json
import os
import pytest
from tools.cache_grounding_proposals import input_identity
from tools.evaluate_grounding_structure import summarize


ROOT=Path(os.environ.get('STRUCTURE_RELEASE',str(Path(__file__).resolve().parents[2]/'results/triground_structure_20260930')))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


@pytest.mark.skipif(not (ROOT/'data/train_inputs.json').exists(),reason='frozen release absent')
def test_frozen_schedule_supervision_and_holdout_boundaries():
    rows=read(ROOT/'data/train_inputs.json'); byid={r['id']:r for r in rows}
    diag=read(ROOT/'data/diagnostic_normal.json')
    assert len(byid)==1045 and len(diag)==119
    assert not {r['scene_id'] for r in rows}&{r['scene_id'] for r in diag}
    for steps in (200,400):
        schedule=read(ROOT/f'data/schedule_{steps}.json')
        assert len(schedule)==steps*8 and set(schedule)<=byid.keys()
        assert sum(byid[key]['task_category']=='old_city' for key in schedule)==(740 if steps==200 else 1480)
        exposure=Counter(byid[key]['mother_id'] for key in schedule if byid[key]['task_category']!='old_city')
        assert len(exposure)==215 and set(exposure.values())=={4 if steps==200 else 8}
    assert sum('ir' in r['supervision'].get('bindings',{}) for r in rows)==29
    for row in rows:
        identity=input_identity(row)
        assert 'rgb_gt_bbox' not in identity and 'supervision' not in identity
        assert row.get('degraded_modality') not in row['supervision'].get('bindings',{})


@pytest.mark.skipif(not (ROOT/'evaluation/inputs.json').exists(),reason='frozen release absent')
def test_actual_interventions_keep_rgb_query_gt_and_disable_raw_depth():
    rows=read(ROOT/'evaluation/inputs.json'); byid={r['id']:r for r in rows}
    assert Counter(r['condition'] for r in rows)=={'normal':531,'without_ir':87,'without_depth':55}
    for row in rows:
        if row['condition']=='normal':continue
        mother=byid[row['evaluation_mother_id']]
        assert row['rgb_gt_bbox']==mother['rgb_gt_bbox'] and row['prompt']==mother['prompt']
        assert row['image'][row['modalities'].index('rgb')]==mother['image'][mother['modalities'].index('rgb')]
        modality=row['condition'].removeprefix('without_')
        assert modality in mother['modalities'] and modality in row['missing_modalities_actual']


def test_summary_counts_failures_and_paired_modality_benefit():
    rows=[]
    for condition,hit in [('normal',True),('without_depth',False)]:
        rows.append({'id':condition,'condition':condition,'scene_id':'scene','task_category':'depth_relation',
                     'evaluation_mother_id':'mother','A':{'prediction':None,'hit':False,'iou':0},
                     'S':{'prediction':[0,0,1,1] if hit else None,'hit':hit,'iou':1 if hit else 0}})
    result=summarize(rows,['A','S'])
    assert result['S']['denominator']==2 and result['S']['parse_failures']==1
    assert result['S']['modality_effect']['depth']['net']==1
