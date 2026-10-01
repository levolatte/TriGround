"""Real 8B zero-effect and retained-A checks before any structural training."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from tools.train_grounding_structure import read, write, load_model
from tools.structured_grounder import QwenStructureRuntime, generate, predict_two_calls
from tools.grounding_structure import StructureConfig, StructureBundle


def preflight(args):
    if not torch.cuda.is_available(): raise RuntimeError('8B GPU checks pending')
    model,processor=load_model(args.model,args.adapter,trainable=False)
    model.eval().requires_grad_(False)
    rows=read(args.inputs)
    # Select actual modalities/categories; no GT participates in model inputs.
    chosen=[]; seen=set()
    for row in rows:
        key=(tuple(row['modalities']),row.get('task_category'),row.get('depth_encoding'))
        if key not in seen: chosen.append(row); seen.add(key)
        if len(chosen)==8: break
    runtime=QwenStructureRuntime(model)
    first=[generate(model,processor,runtime,row) for row in chosen]
    model.load_adapter(args.adapter,adapter_name='revision',is_trainable=False,local_files_only=True)
    for row,original in zip(chosen,first,strict=True):
        runtime.bundle=StructureBundle(StructureConfig(model.config.text_config.hidden_size)).to('cuda').float().eval()
        packet=predict_two_calls(model,processor,runtime,row)
        repeated,structured=packet['first'],packet['second']
        if original['raw_text']!=repeated['raw_text'] or not torch.equal(original['context'],repeated['context']):
            raise AssertionError('retained A output changed after attaching correction adapter')
        runtime.bundle=None
        baseline=generate(model,processor,runtime,row,context=original['context'],proposal=repeated['prediction'],revision=True)
        if baseline['raw_text']!=structured['raw_text'] or not torch.equal(baseline['context'],structured['context']):
            raise AssertionError('zero-initialized structures changed revision output')
    write(args.output,{'status':'pass','rows':[r['id'] for r in chosen],
          'retained_A_unchanged':True,'four_routes_and_evidence_zero_effect':True})
    runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('model','adapter','inputs','output'):parser.add_argument('--'+name,type=Path,required=True)
    preflight(parser.parse_args())
