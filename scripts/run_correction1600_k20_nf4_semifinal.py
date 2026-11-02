"""Restore only the three user-authorized resolution settings for correction1600."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.predict_vllm_quant_submission import main as predict, build_llm_options
from tools.predict_quant_submission import package_predictions, load_queries
from tools.eval_quant_grounding import read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    ref=json.loads((args.reference_run/'full/run_config.json').read_text())
    prev=[r for r in ref['resume_history'] if r.get('engine')=='vllm'][-1]
    config={**ref,'max_pixels':8294400,'max_length':32768,'upscale':2.0}
    expected=deepcopy(prev['options'])
    expected['max_model_len']=32768
    expected['mm_processor_kwargs']['max_pixels']=8294400
    assert build_llm_options(config,8)==expected
    manifest={'reference_run':str(args.reference_run),'adapter':ref['adapter'],
              'changes':{'upscale':[1.0,2.0],'max_pixels':[ref['max_pixels'],8294400],
                         'max_length':[ref['max_length'],32768]},
              'llm_options':expected,'chunk_size':prev['chunk_size'],'max_tokens':ref['max_tokens'],
              'seed':ref['seed'],'temperature':0,'format_policy':ref['format_policy'],
              'resampling':'PIL LANCZOS','automatic_parameter_fallback':False}
    (args.output_dir/'parameter_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    state={'status':'running','stage':'loading','pid':os.getpid(),'expected_queries':5691}
    def save(**values):
        state.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        (args.output_dir/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        argv=['--dataset',str(args.reference_run/'full/unlabeled.jsonl'),
              '--output-dir',str(args.output_dir/'full'),'--adapter',ref['adapter'],
              '--max-num-seqs','8','--chunk-size',str(prev['chunk_size']),'--upscale','2.0']
        for key in ('model','data_root','min_pixels','max_pixels','max_length','max_tokens','seed'):
            argv+=['--'+key.replace('_','-'),str(config[key])]
        summary=predict(argv)
        full=args.output_dir/'full'
        actual=json.loads((full/'run_config.json').read_text())
        assert actual['llm_options']==expected and actual['upscale']==2.0
        save(stage='packaging')
        raw=read_jsonl(full/'predictions_raw.jsonl')
        source=load_queries(Path(ref['queries']),5691)
        assert [r['id'] for r in raw]==list(source)
        summary.update(package_predictions(source,raw,full,ref['format_policy']))
        summary['parameter_manifest']=str(args.output_dir/'parameter_manifest.json')
        (full/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        save(status='completed',stage='completed',completed_queries=len(raw),submission=str(full/'submission.zip'))
    except Exception as error:
        save(status='failed',stage='failed',error=f'{type(error).__name__}: {error}')
        raise

if __name__=='__main__':
    main()
