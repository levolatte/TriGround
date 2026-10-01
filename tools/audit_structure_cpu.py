"""Real processor audit on all frozen conditions without loading 8B weights."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
from transformers import AutoProcessor

from tools.structured_grounder import prepare_inputs, numeric_for_row
from tools.grounding_revision import revision_prompt
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def audit(args):
    torch.set_num_threads(1)
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=200704,max_pixels=602112,local_files_only=True)
    records=[]
    progress=args.output.with_suffix('.progress.json')
    identity=[str(path) for path in args.inputs]
    if args.resume:
        stored=read(progress)
        if stored['inputs']!=identity: raise ValueError('CPU audit resume uses different manifests')
        records=stored['records']
    offset=len(records)
    visited=0
    started=time.monotonic()
    for path in args.inputs:
        rows=read(path)
        for row in rows:
            if visited<offset:
                if records[visited]['id']!=row['id'] or records[visited]['manifest']!=str(path):
                    raise ValueError('CPU audit resume input prefix changed')
                visited+=1
                continue
            # Null is ONLY a processor preview. No real training proposal cache is written.
            prompt=revision_prompt(row['prompt'],None)
            answer=json.dumps({'action':'replace','bbox_2d':bbox_to_qwen1000(row['rgb_gt_bbox'])},separators=(',',':'))
            inputs,length=prepare_inputs(processor,row,prompt,answer,device='cpu')
            grids=inputs['image_grid_thw']
            if len(grids)!=len(row['modalities']): raise ValueError('image slot mismatch')
            if 'mm_token_type_ids' in inputs and inputs['mm_token_type_ids'].shape!=inputs['input_ids'].shape:
                raise ValueError('multimodal position types do not cover the full answer')
            numeric=numeric_for_row(row,grids,processor.image_processor.merge_size)
            tail=processor.decode(inputs['input_ids'][0,length:],skip_special_tokens=True).strip()
            if json.loads(tail)!=json.loads(answer): raise ValueError('supervision was altered or truncated')
            record={'id':row['id'],'manifest':str(path),'tokens':inputs['input_ids'].shape[1],
                    'prompt_tokens':length,'answer_tokens':inputs['input_ids'].shape[1]-length,
                    'grid':grids.tolist(),'numeric_tokens':len(numeric) if numeric is not None else 0}
            records.append(record)
            visited+=1
            if len(records)%50==0:
                write(progress,{'status':'running','inputs':identity,'records':records})
                print(f'CPU preprocess {len(records)}',flush=True)
    result={'status':'pass','rows':len(records),'max_tokens':max(r['tokens'] for r in records),
            'by_manifest':dict(Counter(r['manifest'] for r in records)),
            'numeric_enabled_rows':sum(r['numeric_tokens']>0 for r in records),
            'elapsed_seconds':time.monotonic()-started,'preview_only':True,
            'actual_A_proposals_generated':False,'records':records}
    write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    audit(parser.parse_args())
