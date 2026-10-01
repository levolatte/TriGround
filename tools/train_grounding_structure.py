"""Finite native-Qwen R/S training with an independently optimized, detached G."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

from tools.grounding_structure import StructureBundle, StructureConfig
from tools.structured_grounder import QwenStructureRuntime, training_forward


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def optimizers(model, bundle):
    language = [p for p in model.parameters() if p.requires_grad]
    modal = list(bundle.modal.parameters()) if bundle.modal is not None else []
    geometry = list(bundle.geometry.parameters()) if bundle.geometry is not None else []
    groups = [{"params":language,"lr":5e-6,"group_name":"language"}]
    if modal:
        groups.append({"params":modal,"lr":1e-4,"group_name":"structure"})
    if any(p.dtype != torch.float32 for p in (*language,*modal,*geometry)):
        raise ValueError("trainable parameters must be FP32 before optimizer creation")
    body = torch.optim.AdamW(groups, betas=(.9,.999), eps=1e-8, weight_decay=0)
    head = torch.optim.AdamW(geometry, lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0) if geometry else None
    if len({id(p) for p in (*language,*modal,*geometry)}) != len(language)+len(modal)+len(geometry):
        raise ValueError("optimizer parameters overlap")
    return body, head, language+modal, geometry


def step_optimizers(body, head, body_params, head_params):
    # A shared clipping norm would change S even with all G inputs detached.
    torch.nn.utils.clip_grad_norm_(body_params, 1., error_if_nonfinite=True)
    body.step()
    body.zero_grad(set_to_none=True)
    if head is not None:
        torch.nn.utils.clip_grad_norm_(head_params, 1., error_if_nonfinite=True)
        head.step()
        head.zero_grad(set_to_none=True)


def optimizer_audit(optimizer):
    states = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.dtype != torch.float32 or parameter.grad is not None and parameter.grad.dtype != torch.float32:
                raise ValueError("parameter/gradient precision changed")
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg","exp_avg_sq"):
                if key in state:
                    if state[key].dtype != torch.float32 or not torch.isfinite(state[key]).all():
                        raise ValueError("non-FP32/nonfinite Adam moment")
                    states.append(key)
    return {"groups":[{"name":g.get("group_name","geometry"),"lr":g["lr"],"parameters":len(g["params"])} for g in optimizer.param_groups],
            "fp32_moment_tensors":len(states)}


def lora_products(model):
    return {name:(module.lora_A['default'].weight.detach().clone(),module.lora_B['default'].weight.detach().clone())
            for name,module in model.named_modules() if hasattr(module,'lora_A') and 'default' in module.lora_A}


def product_delta_norm(before, after):
    # ||B1 A1 - B0 A0||_F without allocating a dense D x D matrix.
    a0,b0=before; a1,b1=after
    u=torch.cat((b1.double()-b0.double(),b1.double()),dim=1)
    v=torch.cat((a0.double(),a1.double()-a0.double()),dim=0)
    return float(((u.T@u)*(v@v.T).T).sum().clamp_min(0).sqrt())


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(directory, model, bundle, body, head, schedulers, step, config, consumed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    bundle.save(directory)
    torch.save({"step":step,"body_optimizer":body.state_dict(),"head_optimizer":head.state_dict() if head else None,
                "schedulers":[s.state_dict() for s in schedulers],"rng":rng_state(),"consumed":consumed}, directory/"training_state.pt")
    write(directory/"training_config.json",config)


def load_model(model_path, adapter, device="cuda", trainable=True):
    from peft import PeftModel
    from tools.evaluate_pretrained_grounder import configure_torch_precision
    configure_torch_precision()
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=200704, max_pixels=602112, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16,
                    attn_implementation="sdpa", local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable, autocast_adapter_dtype=trainable, local_files_only=True)
    model.to(device)
    if not trainable:
        # Retained A must match its historical BF16 inference, including LoRA.
        model.to(dtype=torch.bfloat16)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, processor


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available; do CPU validation before launching this stage")
    set_seed(args.seed, deterministic=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows = {r["id"]:r for r in read(args.inputs)}
    schedule = read(args.schedule)
    if len(schedule) != args.steps*8:
        raise ValueError("schedule/horizon mismatch")
    proposals = torch.load(args.proposals, map_location="cpu", weights_only=False)
    if set(schedule)-set(proposals["rows"]):
        raise ValueError("real A predictions/context are missing from the frozen schedule")
    if proposals["adapter"] != str(args.initial_adapter):
        raise ValueError("A proposal cache and retained adapter differ")
    output = args.output_dir
    config = {"model":str(args.model),"initial_adapter":str(args.initial_adapter),"revision_structure":args.variant,
              "seed":args.seed,"steps":args.steps,"accumulation":8,"inputs":str(args.inputs),
              "schedule":str(args.schedule),"proposals":str(args.proposals),"loss":"sample_mean+.1competition+.1binding",
              "box_gradient":"detached_separate_optimizer_and_clip"}
    if args.resume:
        if read(args.resume/"training_config.json") != config:
            raise ValueError("resume differs from frozen run configuration")
    elif (output/"training_config.json").exists():
        raise FileExistsError("explicit resume is required for an existing run")
    adapter = args.resume or args.initial_adapter
    model, processor = load_model(args.model, adapter)
    bundle = StructureBundle.load(args.resume).to("cuda") if args.resume else StructureBundle(
        StructureConfig(model.config.text_config.hidden_size, revision_structure=args.variant,
                        box_decoder="continuous" if args.variant=="modal" else "text")).to("cuda")
    bundle.float().train()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    runtime = QwenStructureRuntime(model,bundle)
    body, head, body_params, head_params = optimizers(model,bundle)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda step: max(0.,1-step/args.steps)) for opt in (body,head) if opt]
    step, consumed = 0, []
    if args.resume:
        state = torch.load(args.resume/"training_state.pt", map_location="cpu", weights_only=False)
        step, consumed = state["step"], state["consumed"]
        if consumed != schedule[:step*8]:
            raise ValueError("checkpoint sample position does not match the frozen schedule")
        body.load_state_dict(state["body_optimizer"])
        if head: head.load_state_dict(state["head_optimizer"])
        for scheduler,saved in zip(schedulers,state["schedulers"],strict=True): scheduler.load_state_dict(saved)
        restore_rng(state["rng"])
    write(output/"training_config.json",config)
    before = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad} if step==0 else None
    products_before = lora_products(model) if step==0 else None
    started = time.monotonic()
    stop = args.stop_at or args.steps
    if not step <= stop <= args.steps: raise ValueError('stop-at must be between resumed step and frozen horizon')
    with (output/"trace.jsonl").open("a",encoding="utf-8") as log:
        for update in range(step,stop):
            stats=[]
            for key in schedule[update*8:(update+1)*8]:
                loss, box_loss, stat = training_forward(model,processor,bundle,runtime,rows[key],proposals["rows"][key],"cuda")
                (loss/8).backward()
                if box_loss is not None: (box_loss/8).backward()
                stats.append({"id":key,**stat})
                consumed.append(key)
                del loss, box_loss
            audit = optimizer_audit(body)
            if head: optimizer_audit(head)
            step_optimizers(body,head,body_params,head_params)
            for scheduler in schedulers: scheduler.step()
            current = update+1
            record = {"step":current,"samples":stats,"learning_rates":[g["lr"] for g in body.param_groups],
                      "elapsed_seconds":time.monotonic()-started,"optimizer_audit":audit}
            log.write(json.dumps(record)+"\n");log.flush()
            print(json.dumps({"step":current,"answer_loss":sum(s['answer_loss'] for s in stats)/8,
                              "elapsed_seconds":record['elapsed_seconds']}),flush=True)
            if before is not None:
                changes={n:float((p.detach()-before[n]).abs().max()) for n,p in model.named_parameters() if n in before}
                if not any(changes.values()): raise RuntimeError("no real language LoRA weight update")
                products_after=lora_products(model)
                movement={name:product_delta_norm(pair,products_after[name]) for name,pair in products_before.items()}
                if not movement or not any(movement.values()): raise RuntimeError('no real LoRA B@A movement')
                write(output/"first_update.json",{"language_max_abs_update":max(changes.values()),"lora_product_delta_frobenius":movement,"optimizer":optimizer_audit(body),
                    "structure_nonzero_output_parameters":{n:float(p.detach().abs().max()) for n,p in bundle.named_parameters()
                        if ".up.weight" in n or "numeric_out" in n or "evidence_out" in n or n.startswith("geometry.out")}})
                before=None
                products_before=None
            if current%100==0 or current==stop:
                save_checkpoint(output/f"checkpoint-{current}",model,bundle,body,head,schedulers,current,config,consumed)
    write(output/"stage_summary.json",{"step":stop,"elapsed_seconds":time.monotonic()-started,
          "consumed":len(consumed),"peak_cuda_bytes":torch.cuda.max_memory_allocated(),
          "output_parameter_max_abs":{n:float(p.detach().abs().max()) for n,p in bundle.named_parameters()
              if '.up.weight' in n or 'numeric_out' in n or 'evidence_out' in n or n.startswith('geometry.out')}})
    runtime.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for name in ("model","initial-adapter","inputs","schedule","proposals","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--variant",choices=["plain","modal"],required=True)
    parser.add_argument("--steps",type=int,choices=[400,200],default=400)
    parser.add_argument("--seed",type=int,default=2030)
    parser.add_argument("--stop-at",type=int)
    parser.add_argument("--resume",type=Path)
    train(parser.parse_args())
