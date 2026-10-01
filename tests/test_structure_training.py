"""Real CPU PEFT updates, optimizer isolation, and exact interrupted recovery."""
from copy import deepcopy
import random
import numpy as np
import pytest
import torch

from test_grounding_structure import tiny_native, config
from tools.grounding_structure import StructureBundle, geometry_loss
from tools.structured_grounder import QwenStructureRuntime, answer_loss
from tools.train_grounding_structure import (optimizers, step_optimizers, optimizer_audit,
    save_checkpoint, rng_state, restore_rng, lora_products, product_delta_norm)


def setup(seed=2030):
    peft=pytest.importorskip('peft')
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model,inputs=tiny_native()
    model=peft.get_peft_model(model,peft.LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,
                          target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    cfg=config(); cfg.dropout=.05
    bundle=StructureBundle(cfg).train()
    runtime=QwenStructureRuntime(model,bundle)
    body,head,bp,hp=optimizers(model,bundle)
    sched=[torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1-step/4) for opt in (body,head)]
    return model,inputs,bundle,runtime,body,head,bp,hp,sched


def update(state, geometry_multiplier=1):
    model,inputs,bundle,runtime,body,head,bp,hp,sched=state
    runtime.begin({'modalities':['rgb','ir']},inputs['image_grid_thw'],15,
                  revision=True,context=torch.arange(32).float()/32,proposal=[.1,.2,.6,.8])
    outputs=model(**inputs,use_cache=False)
    loss=answer_loss(outputs.logits,inputs['input_ids'],15)
    loss.backward()
    (geometry_loss(runtime.box,[.2,.2,.7,.9])*geometry_multiplier).backward()
    step_optimizers(body,head,bp,hp)
    for scheduler in sched: scheduler.step()


def test_geometry_optimizer_cannot_change_language_or_structure():
    a=setup(); update(a,1)
    rng_a=torch.get_rng_state().clone()
    b=setup(); update(b,10000)
    for left,right in zip(a[0].parameters(),b[0].parameters(),strict=True): assert torch.equal(left,right)
    for left,right in zip(a[2].modal.parameters(),b[2].modal.parameters(),strict=True): assert torch.equal(left,right)
    assert torch.equal(rng_a,torch.get_rng_state())
    a[3].close(); b[3].close()


def test_real_peft_updates_and_four_vs_two_plus_two(tmp_path):
    peft=pytest.importorskip('peft')
    continuous=setup()
    before=lora_products(continuous[0])
    for _ in range(4): update(continuous)
    assert all(layer.up.weight.abs().sum()>0 for layer in continuous[2].modal.adapters['ir'])
    assert continuous[2].modal.evidence_out['ir'].weight.abs().sum()>0
    assert continuous[2].geometry.out.weight.abs().sum()>0
    moments=optimizer_audit(continuous[4])
    assert moments['fp32_moment_tensors']>0
    assert any(product_delta_norm(pair,lora_products(continuous[0])[name])>0 for name,pair in before.items())
    expected_rng=deepcopy(rng_state())
    interrupted=setup()
    for _ in range(2): update(interrupted)
    save_checkpoint(tmp_path,interrupted[0],interrupted[2],interrupted[4],interrupted[5],interrupted[8],2,{'steps':4},['a','b'])
    interrupted[3].close()
    resumed=setup()
    # Reload actual PEFT tensors into an identical frozen tiny base.
    saved=peft.utils.save_and_load.load_peft_weights(tmp_path,device='cpu')
    peft.utils.save_and_load.set_peft_model_state_dict(resumed[0],saved)
    resumed[2].load_state_dict(StructureBundle.load(tmp_path).state_dict())
    checkpoint=torch.load(tmp_path/'training_state.pt',weights_only=False)
    resumed[4].load_state_dict(checkpoint['body_optimizer']); resumed[5].load_state_dict(checkpoint['head_optimizer'])
    for scheduler,stored in zip(resumed[8],checkpoint['schedulers'],strict=True): scheduler.load_state_dict(stored)
    restore_rng(checkpoint['rng'])
    for _ in range(2): update(resumed)
    for left,right in zip(continuous[0].parameters(),resumed[0].parameters(),strict=True): assert torch.equal(left,right)
    for left,right in zip(continuous[2].parameters(),resumed[2].parameters(),strict=True): assert torch.equal(left,right)
    assert torch.equal(expected_rng['torch'],rng_state()['torch'])
    for a,b in zip(continuous[4].state_dict()['state'].values(),resumed[4].state_dict()['state'].values(),strict=True):
        assert all(torch.equal(a[key],b[key]) for key in a)
    continuous[3].close(); resumed[3].close()


def test_all_four_numeric_depth_exits_connect_to_real_qwen_loss(tmp_path):
    from PIL import Image
    peft=pytest.importorskip('peft')
    model,inputs=tiny_native()
    model=peft.get_peft_model(model,peft.LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    ids=torch.cat((inputs['input_ids'][:,:14],torch.tensor([[3,102,100,100,100,100,103,20,21,22]])),1)
    inputs.update(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.randn(48,24),
                  image_grid_thw=torch.tensor([[1,4,4]]*3))
    if 'mm_token_type_ids' in inputs: inputs['mm_token_type_ids']=(ids==100).long()
    raw=tmp_path/'depth.png'; Image.fromarray(np.full((8,8),2000,dtype=np.uint16)).save(raw)
    bundle=StructureBundle(config()).train(); runtime=QwenStructureRuntime(model,bundle)
    row={'modalities':['rgb','ir','depth'],'depth_encoding':'millimeter','depth_raw':str(raw)}
    runtime.begin(row,inputs['image_grid_thw'],22,revision=True,context=torch.randn(32),proposal=[.1,.2,.6,.8])
    outputs=model(**inputs,use_cache=False)
    answer_loss(outputs.logits,ids,22).backward()
    assert all(layer.weight.grad.abs().sum()>0 for layer in bundle.modal.numeric_out)
    assert all(layer.up.weight.grad.abs().sum()>0 for layers in bundle.modal.adapters.values() for layer in layers)
    assert all(layer.weight.grad.abs().sum()>0 for layer in bundle.modal.evidence_out.values())
    runtime.close()


def test_low_rank_product_norm_matches_dense():
    a0,b0=torch.randn(4,16),torch.randn(12,4)
    a1,b1=a0+.01,b0-.01
    expected=torch.linalg.vector_norm(b1.double()@a1.double()-b0.double()@a0.double()).item()
    assert product_delta_norm((a0,b0),(a1,b1))==pytest.approx(expected,abs=1e-10)


def test_frozen_loader_preserves_bf16_a_and_fp32_revision(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tools.train_grounding_structure as training
    peft=pytest.importorskip('peft')
    torch.manual_seed(2030)
    base,inputs=tiny_native()
    frozen_base=deepcopy(base).to(dtype=torch.bfloat16)
    model=peft.get_peft_model(base,peft.LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    for name,parameter in model.named_parameters():
        if 'lora_B' in name:
            parameter.data.normal_(std=.03)
    model.save_pretrained(tmp_path)
    reference=peft.PeftModel.from_pretrained(deepcopy(frozen_base),tmp_path,
        is_trainable=False,autocast_adapter_dtype=False).to(dtype=torch.bfloat16).eval()
    monkeypatch.setattr(training.AutoProcessor,'from_pretrained',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(training.Qwen3VLForConditionalGeneration,'from_pretrained',lambda *a,**kw:deepcopy(frozen_base))
    actual,_=training.load_model('tiny',tmp_path,device='cpu',trainable=False)
    actual.eval()
    values={key:value.to(dtype=torch.bfloat16) if value.is_floating_point() else value for key,value in inputs.items()}
    with torch.no_grad():
        expected=reference(**values,use_cache=False).logits
        before=actual(**values,use_cache=False).logits
    assert torch.equal(expected,before)
    assert all(p.dtype==torch.bfloat16 for n,p in actual.named_parameters() if '.default.' in n)
    actual.load_adapter(tmp_path,adapter_name='revision',is_trainable=False)
    actual.set_adapter('default');actual.eval()
    with torch.no_grad():after=actual(**values,use_cache=False).logits
    assert torch.equal(before,after)
    assert all(p.dtype==torch.float32 for n,p in actual.named_parameters() if '.revision.' in n)
    trainable,_=training.load_model('tiny',tmp_path,device='cpu',trainable=True)
    assert all(p.dtype==torch.float32 for p in trainable.parameters() if p.requires_grad)


def test_live_inference_runs_native_qwen_twice_and_keeps_literal_a(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import tools.structured_grounder as integration
    peft=pytest.importorskip('peft')
    state=setup();model,inputs,bundle,runtime=state[:4]
    model.save_pretrained(tmp_path)
    model.load_adapter(tmp_path,adapter_name='revision',is_trainable=False)
    frozen={n:p.detach().clone() for n,p in model.named_parameters()}
    prompts=[];calls=[]
    def prepare(processor,row,prompt,answer=None,device='cpu'):
        prompts.append(prompt)
        prepared={**inputs,'input_ids':inputs['input_ids'][:,:15], 'attention_mask':inputs['attention_mask'][:,:15]}
        if 'mm_token_type_ids' in prepared:prepared['mm_token_type_ids']=prepared['mm_token_type_ids'][:,:15]
        return prepared,15
    monkeypatch.setattr(integration,'prepare_inputs',prepare)
    native_generate=model.generate
    def observed_generate(**kwargs):
        calls.append(model.active_adapter)
        kwargs['max_new_tokens']=2
        return native_generate(**kwargs)
    monkeypatch.setattr(model,'generate',observed_generate)
    raw='{ "bbox_2d": [100, 200, 600, 800] }'
    outputs=iter((raw,'{"action":"keep"}'))
    # Fixed decoded JSON isolates the protocol; both native visual/KV forwards run.
    processor=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=None),decode=lambda *a,**kw:next(outputs))
    row={'prompt':'<image> <image> complete query','modalities':['rgb','ir'],'rgb_gt_bbox':[.91,.92,.98,.99]}
    result=integration.predict_two_calls(model,processor,runtime,row)
    assert calls==['default','revision'] and result['calls']==2
    assert result['variants']['S']['raw_text']==result['variants']['S+G']['raw_text']==raw
    assert '910' not in prompts[1] and 'complete query' in prompts[1]
    assert all(not p.requires_grad and torch.equal(p,frozen[n]) for n,p in model.named_parameters())
    assert not bundle.training and all(not p.requires_grad for p in bundle.parameters())
    assert result['first']['context'].shape==result['second']['context'].shape==(32,)
    runtime.close()
