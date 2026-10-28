"""Check Qwen's actual selected causal convolution on BF16 forward/backward."""
import inspect
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

sys.path.insert(0, '/root/autodl-tmp/quant_rl_20261026/code')
from tools.train_quant_grounding import selected_kernel

implementation = selected_kernel(qwen.causal_conv1d_fn)
assert implementation['module'].startswith('causal_conv1d'), implementation
update_implementation = selected_kernel(qwen.causal_conv1d_update)
assert update_implementation['module'].startswith('causal_conv1d'), update_implementation
torch.manual_seed(2026)
x = torch.randn(2, 96, 129, device='cuda', dtype=torch.bfloat16, requires_grad=True)
w = torch.randn(96, 4, device='cuda', dtype=torch.bfloat16, requires_grad=True)
reference_x, reference_w = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
y = qwen.causal_conv1d_fn(x, w, activation='silu')
expected = F.silu(F.conv1d(reference_x, reference_w.unsqueeze(1), padding=3, groups=96)[..., :129])
signal = torch.randn_like(y)
(y * signal).float().mean().backward()
(expected * signal).float().mean().backward()


def compare(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return {'max_abs': (a - b).abs().max().item(),
            'relative_l2': ((a - b).norm() / b.norm()).item(),
            'cosine': F.cosine_similarity(a, b, dim=0).item(),
            'finite': bool(torch.isfinite(a).all().item())}


report = {'selected_implementation': implementation, 'dtype': str(x.dtype),
          'forward': compare(y, expected), 'input_gradient': compare(x.grad, reference_x.grad),
          'weight_gradient': compare(w.grad, reference_w.grad)}
with torch.inference_mode():
    state = torch.randn(2, 96, 4, device='cuda', dtype=torch.bfloat16)
    token = torch.randn(2, 96, 1, device='cuda', dtype=torch.bfloat16)
    expected_state = torch.cat([state[:, :, 1:], token], dim=-1)
    expected_update = F.silu(F.conv1d(expected_state, w.detach().unsqueeze(1), groups=96))
    actual_update = qwen.causal_conv1d_update(token, state, w.detach(), activation='silu')
    assert torch.equal(state, expected_state)
    report['decode_update'] = compare(actual_update, expected_update)
    report['selected_update_implementation'] = update_implementation
for key in ['forward', 'input_gradient', 'weight_gradient', 'decode_update']:
    assert report[key]['finite'] and report[key]['cosine'] >= 0.999 and report[key]['relative_l2'] <= 0.01, report
Path('/root/autodl-tmp/quant_rl_20261026/runs/causal_conv1d_check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
