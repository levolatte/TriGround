import torch
import pytest
from tools.cache_grounding_proposals import load_cache,baseline_summary


def test_cache_rejects_changed_real_input_and_keeps_gt_out_of_metadata(tmp_path):
    row={'id':'x','prompt':'full query','image':['rgb.png'],'modalities':['rgb'],'rgb_gt_bbox':[0,0,.5,.5]}
    path=tmp_path/'A.pt'
    cache=load_cache(path,'base','A',[row]); torch.save(cache,path)
    assert 'rgb_gt_bbox' not in cache['metadata']['inputs'][0]
    load_cache(path,'base','A',[{**row,'rgb_gt_bbox':[.5,.5,1,1]}])
    for changed in [{**row,'prompt':'changed query'},{**row,'image':['blank.png']}]:
        with pytest.raises(ValueError):load_cache(path,'base','A',[changed])
    cache['metadata'].pop('retained_adapter_dtype')
    torch.save(cache,path)
    with pytest.raises(ValueError):load_cache(path,'base','A',[row])


def test_fresh_a_summary_uses_full_normal_denominator_and_original_gt():
    rows=[{'id':'city','task_category':'old_city','condition':'normal','rgb_gt_bbox':[0,0,.5,.5]},
          {'id':'diag','task_category':'ir_complement','condition':'normal','rgb_gt_bbox':[0,0,.5,.5]}]
    result=baseline_summary(rows,{'city':{'prediction':[0,0,.5,.5]},'diag':{'prediction':None}})
    assert result['city412']['correct']==1 and result['diagnostic119']['correct']==0
    assert result['all_conditions']['denominator']==2 and result['all_conditions']['parse_failures']==1
