import pytest
import torch

from mm_grounding.adapters import RDTDeepFusion, RDTStylePatchFusion, SafePostEmbedFusion


def test_fusion_is_rgb_safe_at_initialization_and_learns():
    module = RDTStylePatchFusion(24, hidden_dim=8, orthogonal_dim=4)
    rgb, ir, depth = (torch.randn(4, 24) for _ in range(3))
    output = module(rgb, ir, depth, [2, 2])
    assert torch.equal(output, rgb)
    output.sum().backward()
    assert module.prompt_restore.weight.grad is not None


def test_fusion_validates_patch_boundaries():
    module = RDTStylePatchFusion(24, hidden_dim=8, orthogonal_dim=4)
    values = torch.randn(4, 24)
    with pytest.raises(ValueError):
        module(values, values, values, [3])


def test_safe_fusion_is_exactly_rgb_at_initialization_and_rezero_learns():
    module = SafePostEmbedFusion(
        raw_patch_dim=24,
        rgb_token_dim=16,
        hidden_dim=8,
        orthogonal_dim=4,
        modality_dropout=0.0,
    )
    rgb = torch.randn(4, 16)
    thermal, depth = (torch.randn(4, 24) for _ in range(2))
    output = module(rgb, thermal, depth, [2, 2])
    assert torch.equal(output, rgb)
    output.sum().backward()
    assert module.residual_scale.grad is not None
    assert module.residual_scale.grad.abs() > 0


def test_true_projection_removes_overlap_when_fully_enabled():
    torch.manual_seed(3)
    depth = torch.randn(7, 5)
    thermal = torch.randn(7, 5)
    depth_orthogonal, thermal_orthogonal = SafePostEmbedFusion._remove_overlap(
        depth,
        thermal,
        torch.tensor(1.0),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        (depth_orthogonal * thermal).sum(dim=-1),
        torch.zeros(7),
        atol=1e-5,
    )
    assert torch.allclose(
        (thermal_orthogonal * depth).sum(dim=-1),
        torch.zeros(7),
        atol=1e-5,
    )


def test_safe_fusion_computes_in_parameter_dtype_and_returns_rgb_dtype():
    module = SafePostEmbedFusion(
        raw_patch_dim=24,
        rgb_token_dim=16,
        hidden_dim=8,
        orthogonal_dim=4,
        modality_dropout=0.0,
    )
    rgb = torch.randn(4, 16, dtype=torch.bfloat16)
    thermal = torch.randn(4, 24)
    depth = torch.randn(4, 24)
    output = module(rgb, thermal, depth, [2, 2])
    assert output.dtype == torch.bfloat16
    assert torch.equal(output, rgb)


def test_zero_prompt_with_small_gamma_starts_exactly_rgb_and_trains_restore_first():
    module = SafePostEmbedFusion(
        raw_patch_dim=24,
        rgb_token_dim=16,
        hidden_dim=8,
        orthogonal_dim=4,
        modality_dropout=0.0,
        residual_scale_init=0.001,
        zero_init_prompt_restore=True,
    )
    rgb = torch.randn(4, 16)
    thermal, depth = (torch.randn(4, 24) for _ in range(2))
    output = module(rgb, thermal, depth, [2, 2])
    assert torch.equal(output, rgb)
    output.sum().backward()
    assert module.prompt_restore.weight.grad is not None
    assert torch.count_nonzero(module.prompt_restore.weight.grad) > 0
    assert module.depth_encoder[1].weight.grad is not None
    assert torch.count_nonzero(module.depth_encoder[1].weight.grad) == 0


def test_deep_prompts_are_rgb_safe_and_every_layer_receives_gradient():
    module = RDTDeepFusion(
        raw_patch_dim=24,
        token_dim=16,
        num_layers=3,
        hidden_dim=8,
        orthogonal_dim=4,
        modality_dropout=0.0,
        residual_scale_init=0.001,
        zero_init_prompt_restore=True,
    )
    rgb = torch.randn(4, 16)
    thermal, depth = (torch.randn(4, 24) for _ in range(2))
    prompt = module.initial_prompt(thermal, depth, [2, 2])
    hidden = rgb
    for index in range(3):
        prompt, injected = module.prompt_for_layer(index, hidden, prompt, [2, 2])
        assert torch.equal(injected, hidden)
        hidden = injected
    hidden.sum().backward()
    for block in module.prompt_blocks:
        assert block.prompt_restore.weight.grad is not None
        assert torch.count_nonzero(block.prompt_restore.weight.grad) > 0
