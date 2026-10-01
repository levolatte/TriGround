"""Tiny CPU PEFT integration: auxiliary training cannot update retained LoRA."""
import torch
import pytest

peft=pytest.importorskip('peft')
from transformers import LlamaConfig, LlamaForCausalLM
from tools.evaluate_pretrained_grounder import attach_frozen_ir_reader


def test_training_separate_lora_preserves_grounder_and_adapter_loading(tmp_path):
    torch.manual_seed(2028)
    base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
                                    num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    cfg=peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    model=peft.get_peft_model(base,cfg).eval()
    x=torch.tensor([[1,4,5,6]])
    with torch.no_grad(): before=model(input_ids=x).logits.clone()
    preserved={n:p.detach().clone() for n,p in model.named_parameters()}
    # A separately trained auxiliary adapter can share the immutable base.
    model.add_adapter('working',cfg);model.set_adapter('working');model.train()
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
    model(input_ids=x,labels=x).loss.backward();optimizer.step();optimizer.zero_grad()
    assert all(torch.equal(preserved[n],p) for n,p in model.named_parameters() if n in preserved)
    assert any(p.detach().abs().sum()>0 for n,p in model.named_parameters() if 'lora_B.working' in n)
    model.save_pretrained(tmp_path/'reader',selected_adapters=['working'],safe_serialization=True)
    attach_frozen_ir_reader(model,tmp_path/'reader/working')
    assert model.active_adapter=='default'
    assert not any(p.requires_grad for p in model.parameters())
    with torch.no_grad(): after=model(input_ids=x).logits
    assert torch.equal(before,after)
