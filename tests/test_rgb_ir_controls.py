import json
from pathlib import Path

from PIL import Image
import pytest

from tools.prepare_rgb_ir_controls import build


def test_controls_preserve_raw_gt_and_full_query_in_both_conditions(tmp_path):
    paths={}
    for modality in ('rgb','infrared'):
        path=tmp_path/f'{modality}.png'
        Image.new('RGB',(24,16),'white').save(path)
        paths[modality]=str(path)
    source={'task_id':'x','source_id':'x','source':'rgbt','scene_id':'scene',
            'category':'reliability','sequence_group':'sequence','depth_policy':'visual',
            'images':paths,'query':'wrong','bbox':[.11111,.22222,.55555,.66666]}
    review={'task_id':'x','decision':'accept','rgb_only_unique':True,
            'final_query':'The green bus beside the wall.'}
    out=tmp_path/'export'
    summary=build([source],[review],[],out)
    read=lambda f:json.loads((out/f).read_text(encoding='utf-8'))
    normal=read('normal.json')[0];blank=read('ir_missing.json')[0]
    assert normal['conversations']==blank['conversations']
    assert 'The green bus beside the wall.' in normal['conversations'][0]['value']
    assert '0.11111' not in normal['conversations'][0]['value']
    assert normal['image'][0]==blank['image'][0]
    assert normal['modalities']==['rgb','ir']
    assert Image.open(blank['image'][1]).getbbox() is None
    assert Image.open(blank['image'][1]).size==Image.open(normal['image'][1]).size
    assert read('gt.json')['rgb-ir-control:x']['bbox']==source['bbox']
    assert source['query']=='wrong' and summary['query_corrections']==1
    with pytest.raises(ValueError,match='training'):
        build([source],[review],[{'scene_id':'scene'}],tmp_path/'overlap')
    with pytest.raises(ValueError,match='require'):
        build([source],[dict(review,rgb_only_unique=False)],[],tmp_path/'not_reviewed')
