import itertools
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from accelerate.utils import DistributedType
from transformers import Trainer


def actual_hf_update(G, m=1):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
    gas = 4*G//m
    class MiniDapoTrainer:
        model_accepts_loss_kwargs = False
        compute_loss_func = None
        use_apex = False
        def __init__(self):
            self.current_gradient_accumulation_steps = gas
            self.model=model
            self.optimizer=None
            self.state=SimpleNamespace(global_step=0)
            self.args=SimpleNamespace(optim='adamw_torch',n_gpu=1,torch_empty_cache_steps=None,
                                      device=torch.device('cpu'),gradient_accumulation_steps=gas)
            self.accelerator=SimpleNamespace(distributed_type=DistributedType.NO,
                                            backward=lambda loss,**_:loss.backward())
            self._prepare_context_parallel_inputs=lambda _model,inputs:(nullcontext,inputs)
            self.compute_loss_context_manager=nullcontext
            self._prepare_inputs=lambda inputs:inputs
        def compute_loss(self,_model,inputs,**_):
            signs=inputs['signs']
            logp = model.weight*signs
            old_logp = .3*signs
            ratio = (logp-old_logp).exp()
            reward=signs
            kl=(-logp).exp()+logp-1
            per_token_loss=-(ratio*reward)+.01*kl
            # G responses x 1 token per group. The local microbatch compensation
            # multiplies Swift's G-token denominator by m.
            return per_token_loss.sum()/(G*m)
    trainer=MiniDapoTrainer()
    losses=[]
    for _ in range(4):
        signs=torch.tensor([1.,-1.]*(G//2),dtype=torch.float64)
        for start in range(0,G,m):
            losses.append(Trainer.training_step(trainer,model,dict(signs=signs[start:start+m])).item())
    return dict(G=G,microbatch=m,gradient_accumulation_steps=gas,loss=sum(losses),gradient=model.weight.grad.item())


def bernoulli_expectation(G,p=.7):
    expected=0.
    for xs in itertools.product([0,1],repeat=G):
        probability=p**sum(xs)*(1-p)**(G-sum(xs))
        mean=sum(xs)/G
        grad=sum((x-mean)*(x-p) for x in xs)/G
        expected += probability*grad
    return dict(G=G,true_pg=p*(1-p),mean_centered_pg=expected,rloo_pg=expected*G/(G-1),
                native_extra_G_scaling_pg=expected/G)


result=dict(transformers=transformers.__version__,torch=torch.__version__,
            same_repeated_distribution_updates=[actual_hf_update(G,m) for G,m in [(8,1),(8,2),(8,8),(2,1),(2,2)]],
            exact_bernoulli_enumeration=[bernoulli_expectation(G) for G in [2,8]])
Path('F:/AIC/code/docs/research/2026-10-28-rl-reassessment/evidence/reward-loss-scale-cpu.json').write_text(
    json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
