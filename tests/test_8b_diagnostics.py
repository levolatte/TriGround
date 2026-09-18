from types import SimpleNamespace

import torch
from torch import nn

from mm_grounding.adapters import JointQueryAwareStageFusion
from tools.diagnose_8b_fusion import FusionProbe
from tools.stress_8b_training import select_representative_indices


def test_pressure_selection_includes_combined_large_image_long_query():
    profiles = [
        {"index": i, "id": str(i), "pixels": 1000 + 200 * i,
         "query_tokens": 10 + 20 * i}
        for i in range(20)
    ]
    profiles[0]["pixels"] = 9999
    profiles[1]["query_tokens"] = 999
    selected = select_representative_indices(profiles)
    assert len(selected) == len(set(selected)) == 8
    assert {0, 1}.issubset(selected)
    assert selected == select_representative_indices(profiles)


def test_fusion_hooks_preserve_output_and_measure_actual_bf16_writeback():
    torch.manual_seed(2026)
    module = JointQueryAwareStageFusion(
        token_dim=16, hidden_dim=8, query_attention_heads=2,
        modality_dropout=0.0, residual_scale_init=1.0, zero_init_restore=False,
    ).eval()
    model = SimpleNamespace(fusion=SimpleNamespace(
        stage_fusions=nn.ModuleDict(), joint_stage_fusions=nn.ModuleDict({"8": module}),
    ))
    rgb, ir, depth = (torch.randn(5, 16).bfloat16() for _ in range(3))
    queries = [torch.randn(2, 4, 8) for _ in range(2)]
    inputs = (rgb, ir, depth, *queries, torch.ones(2, 4, dtype=torch.long), [2, 3])
    state = {k: v.clone() for k, v in module.state_dict().items()}
    with torch.no_grad():
        expected = module(*inputs)
        probe = FusionProbe(model)
        actual = module(*inputs)
        probe.close()
    assert torch.equal(expected, actual)
    assert all(torch.equal(v, state[k]) for k, v in module.state_dict().items())
    stats = probe.report()["joint.8"]
    assert stats["bf16_changed_fraction"]["mean"] == float((actual != rgb).float().mean())
    assert stats["token_gate"]["count"] > 0
