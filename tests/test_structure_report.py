import pytest
from copy import deepcopy
from tools.report_structure_results import merge,paired


def test_cross_arm_report_uses_matched_a_and_isolates_s_against_r():
    a={'prediction':None,'raw_text':'bad','hit':False,'iou':0}
    good={'prediction':[0,0,1,1],'hit':True,'iou':1}
    left=[{'id':'x','scene_id':'scene','target':[0,0,1,1],'A':a,'R':good}]
    right=[{'id':'x','scene_id':'scene','target':[0,0,1,1],'A':a,'S':a,'S+G':good}]
    rows=merge(left,right)
    assert paired(rows,'S','R')['harm']==1
    assert paired(rows,'S+G','S')['rescue']==1
    changed=deepcopy(right);changed[0]['A']=good
    with pytest.raises(ValueError):merge(left,changed)
    changed=deepcopy(right);changed[0]['target']=[0,0,.5,.5]
    with pytest.raises(ValueError):merge(left,changed)
