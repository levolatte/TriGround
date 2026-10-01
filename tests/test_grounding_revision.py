import json
from pathlib import Path

import pytest
import torch

from tools import evaluate_pretrained_grounder as grounder
from tools.grounding_revision import ground_then_verify, parse_revision, revision_prompt
from tools.prepare_grounding_revision import make_revision_example


def generator(text, calls):
    def run(prompt, images, placeholders, cap):
        calls.append((prompt, images, cap))
        return {"raw_text":text,"latency_seconds":.1,"generated_tokens":4,
                "input_tokens":10,"image_grid_thw":[[1,16,16]],"generation_cap_hit":False}
    return run


def test_keep_copies_exact_baseline_box_and_never_uses_gt():
    baseline='{"bbox_2d":[101.125,209.875,654.25,887.5]}'
    calls=[]
    r=ground_then_verify({"prompt_has_image_placeholders":True, "bbox":[.2,.3,.4,.5]},
                         '<image>Full original query', ['RGB'], generator(baseline,calls),
                         generator('{"action":"keep"}',calls), grounder.parse_generated_bbox,128)
    assert r['final']['raw_text']==baseline
    assert r['calls']==2
    assert 'Full original query' in calls[1][0]
    assert '101.125' in calls[1][0]
    assert '[200' not in calls[1][0]
    assert calls[0][1]==calls[1][1]==['RGB']


def test_revision_failure_is_not_a_silent_keep():
    for output in ['bad', '{"action":"replace","bbox_2d":[900,0,5,1]}']:
        r=ground_then_verify({"prompt_has_image_placeholders":False},'query',[],
            generator('{"bbox_2d":[10,20,30,40]}',[]),generator(output,[]),
            grounder.parse_generated_bbox,128)
        assert r['revision_action']=='parse_failure' and r['final']['raw_text']==''
    assert parse_revision('{"action":"keep","bbox_2d":[1,2,3,4]}')[0]=='parse_failure'


def test_replacement_and_invalid_baseline():
    r=ground_then_verify({"prompt_has_image_placeholders":False},'query',[],
        generator('invalid',[]),generator('{"action":"replace","bbox_2d":[10,20,30,40]}',[]),
        grounder.parse_generated_bbox,128)
    assert grounder.parse_generated_bbox(r['final']['raw_text'])==[.01,.02,.03,.04]
    r=ground_then_verify({"prompt_has_image_placeholders":False},'query',[],
        generator('invalid',[]),generator('{"action":"keep"}',[]),grounder.parse_generated_bbox,128)
    assert r['revision_action']=='parse_failure'


def test_auxiliary_and_grounder_are_separate_frozen_roles(monkeypatch):
    class TwoAdapters(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.base=torch.nn.Parameter(torch.ones(2));self.active='default'
            self.g=torch.nn.Parameter(torch.tensor([3.]));self.a=torch.nn.Parameter(torch.tensor([9.]))
        def set_adapter(self,name):
            self.active=name
            self.requires_grad_(True)  # PEFT's switching behavior must be neutralized.
        def load_adapter(self,*args,**kwargs):
            assert kwargs['is_trainable'] is False
    model=TwoAdapters();before={n:p.detach().clone() for n,p in model.named_parameters()}
    grounder.attach_frozen_ir_reader(model,Path('reader'))
    seen=[]
    def generate(model,*args):
        seen.append(model.active)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        return {'raw_text':'{"bbox_2d":null}' if model.active=='ir_reader' else '{"bbox_2d":[1,2,3,4]}',
                'latency_seconds':.1,'generated_tokens':4,'generation_cap_hit':False}
    monkeypatch.setattr(grounder,'generate_text',generate)
    r=grounder.ground_with_optional_ir(
        {'query':'full query','modalities':['rgb','ir'],'prompt_has_image_placeholders':False},
        'original',['RGB','IR'],'ir_then_ground',grounder.adapter_generator(model,None,'default'),128,
        read_ir=grounder.adapter_generator(model,None,'ir_reader'))
    assert seen==['ir_reader','default'] and r['final_prompt']=='original'
    assert all(torch.equal(before[n],p) for n,p in model.named_parameters())


def test_training_uses_actual_model_proposal_not_target_and_excludes_holdout():
    source={'task_id':'x','split':'train','selected_for_diagnostic':False,'bbox':[.4,.5,.6,.7]}
    original={'conversations':[{'from':'human','value':'<image>full query'}]}
    prediction={'prediction':[.1,.2,.3,.4],'target':source['bbox'],'prompt':'<image>full query',
                'raw_text':'{"bbox_2d":[100,200,300,400]}'}
    row=make_revision_example(source,prediction,original,'A')
    assert row['expected_answer']=={'action':'replace','bbox_2d':[400,500,600,700]}
    assert '[100.0,200.0,300.0,400.0]' in row['conversations'][0]['value']
    assert '[400,500,600,700]' not in row['conversations'][0]['value']
    with pytest.raises(ValueError,match='held-out'):
        make_revision_example(dict(source,selected_for_diagnostic=True),prediction,original,'A')
    with pytest.raises(ValueError,match='different full query'):
        make_revision_example(source,dict(prediction,prompt='modified'),original,'A')


def test_already_passing_proposal_is_keep_not_ground_truth_regression():
    source={'task_id':'x','split':'train','bbox':[.1,.2,.3,.4]}
    original={'conversations':[{'from':'human','value':'q'}]}
    p={'prediction':[.101,.2,.301,.4],'target':source['bbox'],'prompt':'q','raw_text':'box'}
    row=make_revision_example(source,p,original,'A')
    assert row['expected_answer']=={'action':'keep'}


def test_revision_scoring_reports_harm_and_retains_failure_denominator():
    correct=[.1,.2,.3,.4]
    wrong=[.6,.7,.8,.9]
    rows=[]
    for baseline, final, action in [(correct,correct,'keep'), (wrong,correct,'replace'),
                                   (correct,wrong,'replace'), (correct,None,'parse_failure'),
                                   (wrong,wrong,'keep')]:
        rows.append(dict(prediction=final,target=correct,baseline_prediction=baseline,
                         parsed=final is not None,revision_action=action,calls=2,
                         generated_tokens=4,first_generated_tokens=5,
                         latency_seconds=.2,total_latency_seconds=.5,generation_cap_hit=False))
    result=grounder.summarize_rows(rows)
    assert result['samples']==5 and result['hits']==2 and result['parse_failures']==1
    assert result['total_calls']==10 and result['total_call_generated_tokens']==45
    assert result['revision_retention']==dict(samples=5,baseline_hits=3,final_hits=2,
        rescued=1,harmed=2,net_hits=-1,correct_proposals_replaced=1,wrong_proposals_kept=1)


def test_resume_cannot_mix_direct_and_verifier_predictions():
    grounder.verify_run_config({'adapter':'A'}, {'adapter':'A'})
    with pytest.raises(ValueError,match='inference_mode'):
        grounder.verify_run_config({'adapter':'A','inference_mode':'ground_then_verify'}, {'adapter':'A'})
    with pytest.raises(ValueError,match='auxiliary_adapter'):
        grounder.verify_run_config({'adapter':'A','auxiliary_adapter':'old'}, {'adapter':'A'})
