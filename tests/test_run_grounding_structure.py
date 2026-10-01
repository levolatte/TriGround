import pytest
from tools.run_grounding_structure import run,forecast,train_command,warrants_repeat
from tools.compare_structure_resume import equal
from tools.train_grounding_structure import write
import numpy as np
import torch


def config(tmp_path):
    return {'python':'python','repo':'repo','model':'model','initial_adapter':'A',
            'data':str(tmp_path/'data'),'output':str(tmp_path),'budget_dir':str(tmp_path),'evaluation':'eval.json'}


def test_dry_plan_contains_full_two_arm_gates_and_shared_cache(tmp_path):
    stages=run(config(tmp_path))
    assert len(stages)==16
    assert [s['name'] for s in stages[:3]]==['zero_native_8b_bf16','cache_A_train_bf16','cache_A_evaluation_bf16']
    assert not (tmp_path/'gpu_budget.json').exists()
    assert [s['name'] for s in stages[-5:]]==['R_formal','R_evaluation','S_formal','S_evaluation','paired_report']
    for name in ['R','S']:
        split=next(s for s in stages if s['name']==name+'_resume4')
        assert '--resume' in split['command'] and split['command'][split['command'].index('--steps')+1]=='400'
    assert sum(s['gpu'] for s in stages)==13


def test_horizon_forecast_requires_full_evaluation_and_original_budget():
    long=forecast(35,2,400,30000); short=forecast(35,2,200,30000)
    assert not long['fits'] and short['fits']
    assert short['required_seconds']>35*400


def test_resume_comparator_checks_optimizer_rng_and_sample_position():
    state={'model':torch.ones(2),'rng':np.array([1,2]),'consumed':['a','b'],'adam':{'exp_avg':torch.zeros(2)}}
    equal(state,state)
    for changed in [{**state,'consumed':['b','a']},{**state,'rng':np.array([2,1])},
                    {**state,'adam':{'exp_avg':torch.ones(2)}}]:
        with pytest.raises(AssertionError):equal(state,changed)


def test_weak_positive_gain_does_not_start_another_seed():
    candidate={'city412':{'net':2,'scene_bootstrap_delta_95':[-.0047,.015]},
               'diagnostic119':{'net':3},'modality_effect':{'ir':{'net':3}}}
    assert not warrants_repeat(candidate)
    candidate['city412']['scene_bootstrap_delta_95']=[.002,.02]
    assert warrants_repeat(candidate)
    candidate['diagnostic119']['net']=-1
    assert not warrants_repeat(candidate)


def test_user_stopped_experiment_cannot_restart(tmp_path):
    write(tmp_path/'status.json',{'status':'stopped_no_clear_benefit'})
    with pytest.raises(RuntimeError,match='do not restart'):
        run(config(tmp_path),execute=True)
