import copy

import numpy as np
from PIL import Image

from tools.prepare_depth_reading_pairs import balanced_pairs
from tools.evaluate_aux_reading import load_reading_records, score_reading, summarize_reading
from tools.prepare_triground_tm_data import _write_json


def source(tmp_path):
    raw=np.full((10,30),1000,dtype=np.uint16)
    raw[:,10:20]=2000; raw[:,20:]=3000
    path=tmp_path/'depth.png';Image.fromarray(raw).save(path)
    evidence={key:{'region':region,'raw_median_mm':depth,'region_source':'manual_subject_interior'}
              for key,region,depth in [('one',[0,0,1/3,1],1000),('two',[1/3,0,2/3,1],2000),('three',[2/3,0,1,1],3000)]}
    return {'task_id':'q1','source':'city','scene_id':'one','query':'nearer than the reference',
            'location_group':'one','images':{'depth_raw':str(path)},
            'relation':{'kind':'nearer','target_id':'one','reference_id':'three','competitors':['two']},
            'depth_evidence':{'by_source_id':evidence}}


def test_reference_binding_dedup_and_both_orders(tmp_path):
    a=source(tmp_path);b=copy.deepcopy(a);b['task_id']='same-image-different-query-id'
    middle=copy.deepcopy(a);middle['task_id']='middle';middle['relation']['kind']='middle'
    rows,skipped=balanced_pairs([a,b,middle],tmp_path/'assets','train')
    assert len(rows)==2 and len(skipped)==1
    assert rows[0]['roi_source_ids']==['one','three']  # not the nearer competitor "two"
    assert rows[1]['roi_source_ids']==['three','one']
    assert rows[0]['expected_answer']=='A_nearer' and rows[1]['expected_answer']=='B_nearer'
    assert len(rows[0]['source_task_ids'])==2
    path=tmp_path/'pairs.json';_write_json(path,rows)
    loaded=load_reading_records(path)
    # Constant-position guessing gets 50% row accuracy and ZERO complete pairs.
    preds=[{**r,**score_reading(r,'A_nearer'),'latency_seconds':0} for r in loaded]
    summary=summarize_reading(preds)['depth_relation']
    assert summary['accuracy']==.5
    assert summary['paired_order_check']['both_correct']==0
    assert summary['paired_order_check']['order_consistent']==0
    assert summary['scope']=='oracle_region_reading_only'
