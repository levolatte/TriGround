import pytest
import torch

from mm_grounding.adapters import (
    JointQueryAwareStageFusion,
    QueryTokenEncoder,
    RDTDeepFusion,
    RDTStylePatchFusion,
    SafePostEmbedFusion,
)


def test_query_position_encoding_breaks_order_invariance_without_new_parameters():
    torch.manual_seed(2026)
    unordered = QueryTokenEncoder(12, 8, 1, 2, 0.0, "none").eval()
    positional = QueryTokenEncoder(12, 8, 1, 2, 0.0, "sinusoidal").eval()
    positional.load_state_dict(unordered.state_dict())
    embeddings = torch.randn(1, 4, 12)
    mask = torch.ones(1, 4, dtype=torch.long)
    permutation = torch.tensor([2, 0, 3, 1])
    unordered_original = unordered(embeddings, mask)
    unordered_permuted = unordered(embeddings[:, permutation], mask[:, permutation])
    positional_original = positional(embeddings, mask)
    positional_permuted = positional(
        embeddings[:, permutation], mask[:, permutation]
    )
    assert torch.allclose(
        unordered_permuted, unordered_original[:, permutation], atol=1e-5
    )
    assert not torch.allclose(
        positional_permuted, positional_original[:, permutation], atol=1e-5
    )
    assert set(unordered.state_dict()) == set(positional.state_dict())


def test_query_position_encoding_handles_padding_and_variable_lengths():
    encoder = QueryTokenEncoder(12, 8, 1, 2, 0.0, "sinusoidal").eval()
    output = encoder(
        torch.randn(2, 5, 12),
        torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]),
    )
    assert output.shape == (2, 5, 8)
    assert torch.isfinite(output).all()


def test_query_position_encoding_preserves_bfloat16_output_dtype():
    encoder = QueryTokenEncoder(12, 8, 1, 2, 0.0, "sinusoidal").to(
        dtype=torch.bfloat16
    ).eval()
    output = encoder(
        torch.randn(2, 5, 12, dtype=torch.bfloat16),
        torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
    )
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()


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


def test_joint_query_fusion_is_exactly_rgb_safe_and_learns_one_residual():
    module = JointQueryAwareStageFusion(
        token_dim=16,
        hidden_dim=8,
        query_attention_heads=2,
        modality_dropout=0.0,
        residual_scale_init=0.001,
        zero_init_restore=True,
    )
    rgb, ir, depth = (torch.randn(5, 16) for _ in range(3))
    ir_query, depth_query = (torch.randn(2, 4, 8) for _ in range(2))
    mask = torch.ones(2, 4, dtype=torch.long)
    output = module(rgb, ir, depth, ir_query, depth_query, mask, [2, 3])
    assert torch.equal(output, rgb)
    output.sum().backward()
    assert module.restore.weight.grad is not None
    assert torch.count_nonzero(module.restore.weight.grad) > 0
    # Zero initialization deliberately delays gradients to upstream fusion
    # layers until the restore projection has moved away from zero.
    assert module.modality_attention.in_proj_weight.grad is not None
    assert torch.count_nonzero(module.modality_attention.in_proj_weight.grad) == 0


def test_joint_query_fusion_supports_either_auxiliary_and_rgb_only():
    module = JointQueryAwareStageFusion(
        token_dim=16,
        hidden_dim=8,
        query_attention_heads=2,
        modality_dropout=0.0,
        residual_scale_init=0.001,
        zero_init_restore=False,
    )
    rgb, ir, depth = (torch.randn(5, 16) for _ in range(3))
    query = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.long)
    ir_output = module(rgb, ir, None, query, None, mask, [2, 3])
    depth_output = module(rgb, None, depth, None, query, mask, [2, 3])
    rgb_output = module(rgb, None, None, None, None, mask, [2, 3])
    assert ir_output.shape == depth_output.shape == rgb.shape
    assert torch.equal(rgb_output, rgb)


def test_joint_query_fusion_scales_target_the_active_joint_path():
    torch.manual_seed(2026)
    module = JointQueryAwareStageFusion(
        token_dim=16,
        hidden_dim=8,
        query_attention_heads=2,
        modality_dropout=0.0,
        residual_scale_init=0.5,
        zero_init_restore=False,
    ).eval()
    rgb, ir, depth = (torch.randn(5, 16) for _ in range(3))
    ir_query, depth_query = (torch.randn(2, 4, 8) for _ in range(2))
    mask = torch.ones(2, 4, dtype=torch.long)
    arguments = (rgb, ir, depth, ir_query, depth_query, mask, [2, 3])
    default = module(*arguments)
    explicit_one = module(*arguments, ir_fusion_scale=1.0, depth_fusion_scale=1.0)
    zero = module(*arguments, ir_fusion_scale=0.0, depth_fusion_scale=0.0)
    depth_only_by_scale = module(
        *arguments, ir_fusion_scale=0.0, depth_fusion_scale=1.0
    )
    depth_only_by_omission = module(
        rgb, None, depth, None, depth_query, mask, [2, 3]
    )
    half = module(*arguments, ir_fusion_scale=0.5, depth_fusion_scale=1.0)
    assert torch.equal(default, explicit_one)
    assert torch.equal(zero, rgb)
    assert torch.allclose(depth_only_by_scale, depth_only_by_omission, atol=1e-6)
    assert not torch.allclose(half, default)
