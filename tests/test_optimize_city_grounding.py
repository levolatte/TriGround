import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).parents[1]/'tools'/'optimize_city_grounding.py'; SPEC=importlib.util.spec_from_file_location('optimize_city_grounding',SCRIPT); MODULE=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)
def row(i,name,x1,x2): return {'sample_id':str(i),'class_name':name,'bbox':[x1,.2,x2,.5],'visible':'visible/a.png','infrared':'infrared/a.png','depth':'depth/a.png','stem':'a','source_group':'a'}
def test_crowded_class_caps_ordinals_at_three():
    selected=MODULE.optimize_scene([row(i,'person',i/10,i/10+.05) for i in range(7)],MODULE.Counter(person=7),12)
    assert len(selected)==4
    assert {x['query'] for x in selected}=={'The leftmost person','The second person from the left','The third person from the left','The rightmost person'}
def test_scale_uses_processed_short_side_not_only_area():
    scale,short_side,_=MODULE.scale_properties([.1,.1,.105,.8]); assert scale=='tiny'; assert short_side<8
