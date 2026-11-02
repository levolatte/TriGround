"""Retry explicitly selected invalid answers with a bbox JSON schema, then package."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import read_jsonl
from tools.predict_vllm_quant_submission import build_requests
from tools.predict_quant_submission import package_predictions
from tools.quant_grounding_postprocess import normalize_bbox_completion
from tools.quant_grounding_reward import parse_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    full = args.run_dir / 'full'
    out = args.run_dir / 'retry4_schema'
    out.mkdir(exist_ok=False)
    config = json.loads((full / 'run_config.json').read_text())
    dataset = read_jsonl(Path(config['dataset']))
    selected = [r for r in dataset if r['id'] in args.ids]
    assert len(selected) == len(args.ids) == 4
    raw = read_jsonl(full / 'predictions_raw.jsonl')
    assert len(raw) == len(dataset) == 5691
    schema = {'type': 'object', 'properties': {'bbox_2d': {'type': 'array',
              'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000},
              'minItems': 4, 'maxItems': 4}}, 'required': ['bbox_2d'], 'additionalProperties': False}
    report = dict(status='loading', ids=args.ids, llm_options=config['llm_options'],
                  schema=schema, max_tokens=config['max_tokens'], temperature=0,
                  changes='Only structured-output bbox schema; original prompts, pixels, weights and seed retained',
                  user_authorized=True)
    def save(**values):
        report.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        (out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoProcessor
        before = time.monotonic()
        llm = LLM(**config['llm_options'])
        processor = AutoProcessor.from_pretrained(config['model'])
        save(status='generating', loading_seconds=time.monotonic()-before)
        requests = build_requests(selected, Path(config['data_root']), processor)
        params = SamplingParams(temperature=0, max_tokens=config['max_tokens'], seed=config['seed'],
                                structured_outputs=StructuredOutputsParams(json=schema))
        before = time.monotonic()
        outputs = llm.generate(requests, params, lora_request=LoRARequest('grounding', 1, config['adapter']), use_tqdm=False)
        elapsed = time.monotonic()-before
        assert len(outputs) == 4
        retried=[]
        for row, output in zip(selected, outputs):
            assert len(output.outputs)==1
            answer=output.outputs[0]
            retried.append({'id':row['id'], 'samples':[{'text':answer.text,'tokens':len(answer.token_ids),
                           'finish_reason':answer.finish_reason,'seconds':elapsed/4}]})
        (out / 'predictions_raw.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in retried),encoding='utf-8')
        for row in retried:
            s=row['samples'][0]
            assert s['finish_reason']=='stop' and parse_bbox(s['text'],'short') is not None, row
        replacements={r['id']:r for r in retried}
        merged=[replacements.get(r['id'],r) for r in raw]
        normalized=[];wrapped=0
        for row in merged:
            s=row['samples'][0]
            if parse_bbox(normalize_bbox_completion(s['text'],'short','punctuation'),'short') is None:
                payload=json.loads(s['text'])
                assert isinstance(payload,list) and len(payload)==4
                text=json.dumps({'bbox_2d':payload})
                assert parse_bbox(text,'short') is not None
                s={**s,'text':text};wrapped+=1
            normalized.append({**row,'samples':[s]})
        reference=Path(json.loads((args.run_dir/'state.json').read_text())['reference_run'])
        original_config=json.loads((reference/'full/run_config.json').read_text())
        source=json.loads(Path(original_config['queries']).read_text())
        packaged=package_predictions(source,normalized,full,'punctuation')
        summary=json.loads((full/'summary.json').read_text())
        summary.update(packaged, retried_queries=4, retry_report=str(out/'report.json'),
                       first_pass_valid=5687, bare_coordinate_arrays_wrapped=wrapped,
                       comparison_note='4 invalid answers retried with JSON schema; correction package first 1720 uses HF.')
        (full/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        save(status='completed',generation_seconds=elapsed,valid_retries=4,**packaged)
        state_path=args.run_dir/'state.json'
        state=json.loads(state_path.read_text())
        (out/'original_failed_state.json').write_text(json.dumps(state,indent=2)+'\n')
        state.update(status='completed',stage='completed_after_retry4',completed_queries=5691,retried_queries=4,
                     original_packaging_error=state.pop('error',None),updated_at=datetime.now(timezone.utc).isoformat())
        state_path.write_text(json.dumps(state,indent=2)+'\n')
    except Exception as error:
        save(status='failed',error=f'{type(error).__name__}: {error}')
        raise

if __name__=='__main__':
    main()
