"""Compare actual model, structure, Adam, scheduler, RNG and sample position."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tools.train_grounding_structure import read, write


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a,b): raise AssertionError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b): raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError(path+'.keys')
        for key in a: equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b): raise AssertionError(path+'.length')
        for index,(left,right) in enumerate(zip(a,b,strict=True)): equal(left,right,path+'.'+str(index))
    elif a!=b: raise AssertionError(path)


def comparable_config(config, name):
    # PEFT stores target_modules as a set and exports it in process-dependent order.
    if name == 'adapter_config.json' and isinstance(config.get('target_modules'), list):
        config = {**config, 'target_modules': sorted(config['target_modules'])}
    return config


def compare(left,right,output):
    for name in ('training_config.json','structure_config.json','adapter_config.json'):
        equal(comparable_config(read(left/name),name),comparable_config(read(right/name),name),name)
    equal(load_file(left/'adapter_model.safetensors'),load_file(right/'adapter_model.safetensors'),'LoRA')
    for name in ('structure.pt','training_state.pt'):
        equal(torch.load(left/name,map_location='cpu',weights_only=False),torch.load(right/name,map_location='cpu',weights_only=False),name)
    write(output,{'status':'pass','left':str(left),'right':str(right),
                  'checks':['all LoRA tensors','structure/G tensors','both Adam states','schedulers/LR','RNG','consumed sample order']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('left','right','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); compare(args.left,args.right,args.output)
