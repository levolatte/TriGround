"""Serial structure experiment; dry-run by default, separate bounded GPU ledger."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from tools.train_grounding_structure import read, write
from tools.run_triground_abv import execute_stage, load_budget


def command(config,module,*arguments):
    return [config['python'],'-m','tools.'+module,*map(str,arguments)]


def train_command(config,out,variant,horizon,*,stop=None,resume=None,seed=2030):
    data=Path(config['data'])
    args=['--model',config['model'],'--initial-adapter',config['initial_adapter'],
          '--inputs',data/'train_inputs.cloud.json','--schedule',data/f'schedule_{horizon}.json',
          '--proposals',Path(config['output'])/'cache/train.pt','--output-dir',out,
          '--variant',variant,'--steps',horizon,'--seed',seed]
    if stop is not None: args+=['--stop-at',stop]
    if resume is not None: args+=['--resume',resume]
    return command(config,'train_grounding_structure',*args)


def forecast(step_seconds,query_seconds,horizon,remaining,*,training_updates=None,evaluation_calls=1346):
    # All mandatory final R/S evaluations plus process loading and 10% headroom.
    updates=2*horizon if training_updates is None else training_updates
    required=1.15*(step_seconds*updates+query_seconds*evaluation_calls)+6*90
    return {'horizon':horizon,'step_seconds':step_seconds,'query_seconds':query_seconds,
            'required_seconds':required,'remaining_seconds':remaining,'fits':required<=.9*remaining}


def warrants_repeat(candidate):
    return (candidate['city412']['net']>0
            and candidate['city412']['scene_bootstrap_delta_95'][0]>0
            and candidate['diagnostic119']['net']>=0
            and any(m['net']>0 for m in candidate['modality_effect'].values()))


def run(config,execute=False):
    out=Path(config['output']); out.mkdir(parents=True,exist_ok=True)
    if Path(config['budget_dir']).resolve()!=out.resolve(): raise ValueError('this stage needs its own output budget directory')
    state=load_budget(out)
    if execute and (out/'status.json').exists() and read(out/'status.json')['status']=='stopped_no_clear_benefit':
        raise RuntimeError('this experiment was stopped for insufficient benefit; do not restart its queue')
    completed={r['name'] for r in state['stages'] if r['status']=='complete'}
    stages=[]
    def stage(name,cmd,gpu=True):
        item={'name':name,'command':cmd,'gpu':gpu,'cwd':config['repo'],
              'env':{'TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8'}}
        stages.append(item)
        if execute and name not in completed:
            execute_stage(item,out,state,out); completed.add(name)
    if execute:
        if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable; CPU engineering is complete but experiment remains pending')
        for name in ('cpu_preprocess.json','cpu_checks.json'):
            if read(out/name)['status']!='pass': raise ValueError('CPU gate failed: '+name)
        audit=read(out/'cpu_preprocess.json')
        expected={str((Path(config['data'])/'train_inputs.cloud.json').resolve()):1045,
                  str(Path(config['evaluation']).resolve()):673}
        audited={str((Path(config['repo'])/path).resolve()):count for path,count in audit['by_manifest'].items()}
        if audited!=expected or audit['rows']!=1718: raise ValueError('CPU gate did not cover the complete frozen input release')
    stage('zero_native_8b_bf16',command(config,'preflight_structure_gpu','--model',config['model'],
          '--adapter',config['initial_adapter'],'--inputs',Path(config['data'])/'train_inputs.cloud.json','--output',out/'zero_native_8b.json'))
    for kind,input_path in [('train',Path(config['data'])/'train_inputs.cloud.json'),('evaluation',config['evaluation'])]:
        stage('cache_A_'+kind+'_bf16',command(config,'cache_grounding_proposals','--model',config['model'],'--adapter',config['initial_adapter'],
              '--inputs',input_path,'--output',out/f'cache/{kind}.pt'))
    if execute and 'retained_A_city_expected_correct' in config:
        current=read(out/'cache/evaluation.summary.json')['baseline_metrics']['city412']
        if current['denominator']!=412 or current['correct']!=config['retained_A_city_expected_correct']:
            raise RuntimeError('fresh retained-A baseline differs from the recorded historical City412 result; inspect before training')
        if config.get('retained_A_historical_predictions'):
            historical=[json.loads(line) for line in Path(config['retained_A_historical_predictions']).read_text(encoding='utf-8-sig').splitlines()]
            cache=torch.load(out/'cache/evaluation.pt',map_location='cpu',weights_only=False)
            differences=[{'id':row['id'],'field':key} for row in historical
                         for key in ('raw_text','input_tokens','image_grid_thw')
                         if row[key]!=cache['rows'][row['id']][key]]
            write(out/'A_historical_recheck.json',{'status':'fail' if differences else 'pass',
                  'rows':len(historical),'differences':differences,'retained_adapter_dtype':'bfloat16'})
            if differences: raise RuntimeError('retained A raw outputs differ from historical outputs; inspect before training')
    for arm,variant in [('R','plain'),('S','modal')]:
        root=out/'preflight'/arm
        stage(arm+'_continuous4',train_command(config,root/'continuous',variant,400,stop=4))
        if execute and arm=='S':
            outputs=read(root/'continuous/stage_summary.json')['output_parameter_max_abs']
            weights={name:value for name,value in outputs.items() if name.endswith('.weight')}
            if not weights or any(value<=0 for value in weights.values()):
                raise RuntimeError('a structural output remained zero after real 4-step training')
        stage(arm+'_split2',train_command(config,root/'split',variant,400,stop=2))
        stage(arm+'_resume4',train_command(config,root/'split',variant,400,stop=4,resume=root/'split/checkpoint-2'))
        stage(arm+'_compare',command(config,'compare_structure_resume','--left',root/'continuous/checkpoint-4',
              '--right',root/'split/checkpoint-4','--output',root/'comparison.json'),gpu=False)
    if execute:
        if (out/'formal_horizon.json').exists():
            horizon=read(out/'formal_horizon.json')['horizon']
        else:
            rate=max(read(out/f'preflight/{arm}/continuous/stage_summary.json')['elapsed_seconds']/4 for arm in ('R','S'))
            cache=torch.load(out/'cache/evaluation.pt',map_location='cpu',weights_only=False)
            query_seconds=max(1.,sum(r['wall_seconds'] for r in cache['rows'].values())/len(cache['rows']))
            estimates=[forecast(rate,query_seconds,h,state['limit_seconds']-state['spent_seconds']) for h in (400,200)]
            write(out/'budget_forecast.json',estimates)
            fitting=[e for e in estimates if e['fits']]
            if not fitting: raise RuntimeError('neither frozen horizon fits required full evaluation; no formal training started')
            horizon=fitting[0]['horizon']
            write(out/'formal_horizon.json',{'horizon':horizon,'reason':'measured R/S update time and A evaluation latency; chosen before formal training',
                    'step_seconds':rate,'query_seconds':query_seconds})
    else: horizon=400
    for arm,variant in [('R','plain'),('S','modal')]:
        root=out/'runs'/arm
        stage(arm+'_formal',train_command(config,root,variant,horizon))
        stage(arm+'_evaluation',command(config,'evaluate_grounding_structure','--model',config['model'],
              '--initial-adapter',config['initial_adapter'],'--checkpoint',root/f'checkpoint-{horizon}',
              '--inputs',config['evaluation'],'--proposals',out/'cache/evaluation.pt','--output',out/f'evaluation/{arm}'))
    stage('paired_report',command(config,'report_structure_results','--r',out/'evaluation/R','--s',out/'evaluation/S',
          '--inputs',config['evaluation'],'--output',out/'report'),gpu=False)
    write(out/'execution_plan.json',{'execute':execute,'stages':stages,'horizon':horizon,
              'second_seed':'only after positive City net with scene-bootstrap 95% lower bound above zero, nonnegative diagnostic net and positive actual modal benefit; best candidate only, subject to remaining original budget',
              'gpu_budget_seconds':43200,'max_model_calls_per_input':2})
    if execute:
        reports={arm:read(out/f'evaluation/{arm}/summary.json') for arm in ('R','S')}
        shortlist=[]
        for arm,report in reports.items():
            for name in (['R'] if arm=='R' else ['S','S+G']):
                candidate=report[name]
                if warrants_repeat(candidate):
                    shortlist.append(name)
        repeat_status='not_warranted'
        if shortlist:
            def score(name):
                candidate=reports['R' if name=='R' else 'S'][name]
                return candidate['city412']['net'],sum(m['net'] for m in candidate['modality_effect'].values())
            selected=max(shortlist,key=score)
            measured=read(out/'formal_horizon.json')
            estimate=forecast(measured['step_seconds'],measured['query_seconds'],horizon,
                state['limit_seconds']-state['spent_seconds'],training_updates=horizon+8,evaluation_calls=673)
            write(out/'second_seed_forecast.json',{'selected':selected,**estimate})
            repeat_status='insufficient_remaining_budget'
            if estimate['fits'] or 'seed2031_evaluation' in completed:
                arm='R' if selected=='R' else 'S'; variant='plain' if arm=='R' else 'modal'
                root=out/'seed2031'/arm
                stage('seed2031_continuous4',train_command(config,root/'continuous',variant,horizon,stop=4,seed=2031))
                stage('seed2031_split2',train_command(config,root/'split',variant,horizon,stop=2,seed=2031))
                stage('seed2031_resume4',train_command(config,root/'split',variant,horizon,stop=4,seed=2031,resume=root/'split/checkpoint-2'))
                stage('seed2031_compare',command(config,'compare_structure_resume','--left',root/'continuous/checkpoint-4',
                    '--right',root/'split/checkpoint-4','--output',root/'comparison.json'),gpu=False)
                stage('seed2031_formal',train_command(config,root/'main',variant,horizon,seed=2031))
                stage('seed2031_evaluation',command(config,'evaluate_grounding_structure','--model',config['model'],
                    '--initial-adapter',config['initial_adapter'],'--checkpoint',root/f'main/checkpoint-{horizon}',
                    '--inputs',config['evaluation'],'--proposals',out/'cache/evaluation.pt','--output',root/'evaluation'))
                write(out/'report/second_seed.json',{'selected':selected,'seed':2031,
                    'source':str(root/'evaluation'),'summary':read(root/'evaluation/summary.json')})
                repeat_status='complete'
        write(out/'status.json',{'status':'primary_complete','horizon':horizon,'shortlist':shortlist,
                'spent_seconds':state['spent_seconds'],'second_seed_status':repeat_status,'official_submission':False})
        plan=read(out/'execution_plan.json');plan['stages']=stages
        write(out/'execution_plan.json',plan)
    return stages


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(); stages=run(read(args.config),args.execute)
    print('\n'.join(s['name'] for s in stages))
