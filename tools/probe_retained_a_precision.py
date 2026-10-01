"""Compare a small real-A probe with historical output before replacing caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from tools.train_grounding_structure import load_model, read, write
from tools.structured_grounder import QwenStructureRuntime, generate


def probe(args):
    historical={row['id']:row for row in map(json.loads,args.historical.read_text(encoding='utf-8-sig').splitlines())}
    rows=read(args.inputs)
    model,processor=load_model(args.model,args.adapter,trainable=False)
    model.eval().requires_grad_(False)
    runtime=QwenStructureRuntime(model)
    differences=[]
    for row in rows:
        actual=generate(model,processor,runtime,row)
        expected=historical[row['id']]
        fields=[key for key in ('raw_text','input_tokens','image_grid_thw') if actual[key]!=expected[key]]
        if fields: differences.append({'id':row['id'],'fields':fields,'old_raw':expected['raw_text'],'new_raw':actual['raw_text']})
    result={'status':'pass' if not differences else 'fail','rows':len(rows),'differences':differences,
            'retained_lora_dtypes':sorted({str(p.dtype) for name,p in model.named_parameters() if 'lora_' in name})}
    write(args.output,result)
    runtime.close()
    if differences: raise AssertionError('retained A differs from historical output; see precision probe')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','adapter','inputs','historical','output'):parser.add_argument('--'+name,type=Path,required=True)
    probe(parser.parse_args())
