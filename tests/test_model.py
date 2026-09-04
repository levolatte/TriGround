from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from mm_grounding.engine import _gradient_diagnostics
from mm_grounding.model import MultiModalGrounder


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=2, temporal_patch_size=2, in_channels=3)
        )

    def forward(self, pixel_values, labels, **kwargs):
        return SimpleNamespace(loss=pixel_values.mean() * self.anchor, logits=pixel_values)


class DummyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 16
        self.proj = nn.Linear(24, 16)

    def forward(self, values):
        return self.proj(values)


class DummyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = DummyPatchEmbed()


class DummyDeepBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, **_):
        self.calls += 1
        return hidden_states + 0.01


class DummyMerger(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class DummyDeepVision(DummyVision):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([DummyDeepBlock(), DummyDeepBlock()])
        self.deepstack_visual_indexes = []
        self.deepstack_merger_list = nn.ModuleList()
        self.merger = DummyMerger()

    def fast_pos_embed_interpolate(self, grid_thw):
        count = int(grid_thw.prod(dim=-1).sum()) // 2
        return torch.zeros(count, 16)

    def rot_pos_emb(self, grid_thw):
        count = int(grid_thw.prod(dim=-1).sum()) // 2
        return torch.zeros(count, 8)

    def forward(self, hidden_states, grid_thw, **kwargs):
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
        for block in self.blocks:
            hidden_states = block(hidden_states, **kwargs)
        return self.merger(hidden_states), []


class DummyPostEmbedBackbone(DummyBackbone):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = DummyVision()

    def forward(self, pixel_values, labels, **kwargs):
        tokens = self.model.visual.patch_embed(pixel_values)
        return SimpleNamespace(loss=tokens.mean() * self.anchor, logits=tokens)


class DummyAuxBackbone(DummyPostEmbedBackbone):
    def __init__(self):
        super().__init__()
        self.config.text_config = SimpleNamespace(hidden_size=12)
        self.token_embed = nn.Embedding(32, 12)
        self.visual_projection = nn.Linear(16, 12)
        self.lm_head = nn.Linear(12, 32)

    def forward(self, pixel_values, input_ids, labels, **kwargs):
        tokens = self.model.visual.patch_embed(pixel_values).reshape(2, 2, 16).mean(dim=1)
        hidden = self.token_embed(input_ids) + self.visual_projection(tokens).unsqueeze(1)
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=(hidden,))


class DummyDeepBackbone(DummyBackbone):
    def __init__(self):
        super().__init__()
        self.config.text_config = SimpleNamespace(hidden_size=12)
        self.token_embed = nn.Embedding(32, 12)
        self.model = nn.Module()
        self.model.visual = DummyDeepVision()

    def get_input_embeddings(self):
        return self.token_embed

    def forward(self, pixel_values, image_grid_thw, labels, **kwargs):
        tokens, _ = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        return SimpleNamespace(loss=tokens.mean() * self.anchor, logits=tokens)


def _batch():
    return {
        "pixel_values": torch.randn(4, 24),
        "ir_pixel_values": torch.randn(4, 24),
        "depth_pixel_values": torch.randn(4, 24),
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
        "labels": torch.ones(2, 3, dtype=torch.long),
        "query_input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "query_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }


def test_phase_a_has_only_fusion_and_no_loc_decoder():
    model = MultiModalGrounder(DummyBackbone(), adapter_channels=8, orthogonal_channels=4)
    model.set_phase_a_trainable()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith("fusion.") for name in trainable)
    assert not hasattr(model, "grounding")
    assert not hasattr(model, "loc_token_id")


def test_native_qwen_loss_reaches_fusion():
    model = MultiModalGrounder(DummyBackbone(), adapter_channels=8, orthogonal_channels=4)
    model.set_phase_a_trainable()
    model(**_batch())["loss"].backward()
    assert model.fusion.prompt_restore.weight.grad is not None


def test_post_embed_fusion_preserves_rgb_and_clears_context():
    backbone = DummyPostEmbedBackbone()
    model = MultiModalGrounder(
        backbone,
        adapter_channels=8,
        orthogonal_channels=4,
        fusion_type="safe_post_embed",
        modality_dropout=0.0,
    )
    model.set_phase_a_trainable()
    batch = _batch()
    wrapper = model._post_embed_wrapper()
    assert wrapper is not None
    assert not any(name.startswith("_fused_patch_embed") for name in model.state_dict())
    expected = wrapper.base(batch["pixel_values"])
    result = model(**batch)
    assert model.bbox_head is None
    assert "bbox_l1_loss" not in result
    assert torch.equal(result["loss"], result["token_loss"])
    assert torch.equal(result["logits"], expected)
    assert wrapper._thermal_patches is None
    result["loss"].backward()
    assert model.fusion.residual_scale.grad is not None


def test_rdt_deep_updates_every_layer_and_preserves_rgb_at_initialization():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        orthogonal_channels=4,
        fusion_type="rdt_deep",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.001,
        fusion_zero_init_prompt_restore=True,
    )
    model.set_phase_a_trainable()
    batch = _batch()
    assert len(model.fusion.prompt_blocks) == len(model.backbone.model.visual.blocks) == 2
    assert not any("_rdt_" in name for name in model.state_dict())
    joint = model(**batch)
    rgb = model(**batch, rgb_only=True)
    assert torch.equal(joint["logits"], rgb["logits"])
    assert model._deep_prompt_vision()._rdt_prompt_context is None
    joint["loss"].backward()
    for block in model.fusion.prompt_blocks:
        assert block.prompt_restore.weight.grad is not None
        assert torch.count_nonzero(block.prompt_restore.weight.grad) > 0


def test_parallel_backbone_runs_three_streams_and_trains_adapters_and_fusion():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.001,
        parallel_fusion_stages=2,
        parallel_adapter_scale_init=0.01,
    )
    model.set_phase_a_trainable()
    batch = _batch()
    output = model(**batch)
    assert model.fusion.fusion_layer_indices == (0, 1)
    assert all(block.calls == 3 for block in model.backbone.model.visual.blocks)
    assert model._parallel_backbone_vision()._parallel_backbone_context is None
    output["loss"].backward()
    assert model.fusion.ir_adapters[0].up.weight.grad is not None
    assert model.fusion.depth_adapters[0].up.weight.grad is not None
    for stage in model.fusion.stage_fusions.values():
        assert stage.ir.restore.weight.grad is not None
        assert stage.depth.restore.weight.grad is not None
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all(name.startswith("fusion.") for name in trainable)


def test_parallel_backbone_rgb_only_uses_one_stream_and_clears_context():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.001,
        parallel_fusion_stages=2,
    )
    model.set_phase_a_trainable()
    model(**_batch(), rgb_only=True)
    assert all(block.calls == 1 for block in model.backbone.model.visual.blocks)
    assert model._parallel_backbone_vision()._parallel_backbone_context is None


def test_parallel_backbone_ir_stage_skips_depth_stream_and_parameters():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.001,
        parallel_fusion_stages=1,
    )
    model.set_phase_a_trainable("ir")
    assert any(parameter.requires_grad for parameter in model.fusion.ir_adapters.parameters())
    assert not any(parameter.requires_grad for parameter in model.fusion.depth_adapters.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.ir_query_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.fusion.depth_query_encoder.parameters())
    batch = _batch()
    batch.pop("depth_pixel_values")
    output = model(**batch)
    assert all(block.calls == 2 for block in model.backbone.model.visual.blocks)
    output["loss"].backward()
    assert model.fusion.ir_adapters[0].up.weight.grad is not None
    assert model.fusion.depth_adapters[0].up.weight.grad is None
    stage = next(iter(model.fusion.stage_fusions.values()))
    assert stage.ir.restore.weight.grad is not None
    assert stage.depth.restore.weight.grad is None


def test_joint_calibration_can_freeze_both_modality_adapters():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        parallel_fusion_stages=1,
    )
    model.set_phase_a_trainable("joint", freeze_parallel_adapters=True)
    assert not any(parameter.requires_grad for parameter in model.fusion.ir_adapters.parameters())
    assert not any(parameter.requires_grad for parameter in model.fusion.depth_adapters.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.stage_fusions.parameters())


def test_joint_fusion_replaces_independent_injection_and_preserves_rgb_initially():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.001,
        fusion_zero_init_prompt_restore=True,
        parallel_fusion_stages=1,
        parallel_joint_fusion=True,
    )
    model.set_phase_a_trainable("joint", freeze_parallel_adapters=True)
    assert not any(parameter.requires_grad for parameter in model.fusion.stage_fusions.parameters())
    assert any(
        parameter.requires_grad for parameter in model.fusion.joint_stage_fusions.parameters()
    )
    batch = _batch()
    rgb = model(**batch, rgb_only=True)
    joint = model(**batch)
    assert torch.equal(joint["logits"], rgb["logits"])
    joint["loss"].backward()
    stage = next(iter(model.fusion.joint_stage_fusions.values()))
    assert stage.restore.weight.grad is not None
    assert torch.count_nonzero(stage.restore.weight.grad) > 0


def test_final_joint_model_applies_inference_scales_to_active_fusion():
    torch.manual_seed(2026)
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        modality_dropout=0.0,
        fusion_residual_scale_init=0.5,
        fusion_zero_init_prompt_restore=False,
        parallel_fusion_stages=1,
        parallel_joint_fusion=True,
    ).eval()
    batch = _batch()
    default = model(**batch)["logits"]
    explicit_one = model(
        **batch, ir_fusion_scale=1.0, depth_fusion_scale=1.0
    )["logits"]
    zero = model(**batch, ir_fusion_scale=0.0, depth_fusion_scale=0.0)["logits"]
    rgb = model(**batch, rgb_only=True)["logits"]
    half = model(**batch, ir_fusion_scale=0.5, depth_fusion_scale=1.0)[
        "logits"
    ]
    assert torch.equal(default, explicit_one)
    assert torch.equal(zero, rgb)
    assert not torch.allclose(half, default)


def test_joint_fusion_warm_start_reuses_alignment_but_not_legacy_restore():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        fusion_zero_init_prompt_restore=True,
        parallel_fusion_stages=1,
        parallel_joint_fusion=True,
    )
    legacy = next(iter(model.fusion.stage_fusions.values()))
    joint = next(iter(model.fusion.joint_stage_fusions.values()))
    with torch.no_grad():
        legacy.ir.aux_projection[1].weight.fill_(2.0)
        legacy.depth.aux_projection[1].weight.fill_(3.0)
        legacy.ir.rgb_projection[1].weight.fill_(4.0)
        legacy.depth.rgb_projection[1].weight.fill_(6.0)
        legacy.ir.language_attention.in_proj_weight.fill_(8.0)
        legacy.depth.language_attention.in_proj_weight.fill_(12.0)
        legacy.ir.restore.weight.fill_(9.0)
    model.fusion.warm_start_joint_from_legacy()
    assert torch.all(joint.ir_projection[1].weight == 2.0)
    assert torch.all(joint.depth_projection[1].weight == 3.0)
    assert torch.all(joint.rgb_projection[1].weight == 5.0)
    assert torch.all(joint.language_attention.in_proj_weight == 10.0)
    assert torch.count_nonzero(joint.restore.weight) == 0


def test_parallel_backbone_can_train_only_last_ir_adapter():
    model = MultiModalGrounder(
        DummyDeepBackbone(),
        adapter_channels=8,
        fusion_type="parallel_backbone",
        parallel_fusion_stages=1,
    )
    model.set_phase_a_trainable("ir", parallel_adapter_train_last_n=1)
    assert not any(parameter.requires_grad for parameter in model.fusion.ir_adapters[0].parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.ir_adapters[-1].parameters())
    assert not any(parameter.requires_grad for parameter in model.fusion.depth_adapters.parameters())
    assert any(parameter.requires_grad for parameter in model.fusion.ir_query_encoder.parameters())
    stage = next(iter(model.fusion.stage_fusions.values()))
    assert any(parameter.requires_grad for parameter in stage.ir.parameters())
    assert not any(parameter.requires_grad for parameter in stage.depth.parameters())


def test_auxiliary_bbox_uses_pre_answer_feature_and_has_legal_output():
    model = MultiModalGrounder(
        DummyAuxBackbone(),
        adapter_channels=8,
        orthogonal_channels=4,
        fusion_type="safe_post_embed",
        modality_dropout=0.0,
        auxiliary_bbox_enabled=True,
    )
    model.set_phase_a_trainable()
    batch = _batch()
    batch["labels"] = torch.tensor([[-100, -100, 1], [-100, -100, 1]])
    batch["coordinate_mask"] = batch["labels"].ne(-100)
    batch["bbox"] = torch.tensor([[0.1, 0.2, 0.7, 0.8], [0.2, 0.1, 0.9, 0.6]])
    output = model(**batch, geometry_gradient_scale=1.0)
    assert isinstance(model.bbox_head, nn.Linear)
    assert torch.isfinite(output["loss"])
    assert torch.isfinite(output["coordinate_token_loss"])
    assert (output["auxiliary_bbox"] >= 0).all()
    assert (output["auxiliary_bbox"] <= 1).all()
    assert (output["auxiliary_bbox"][:, 2:] > output["auxiliary_bbox"][:, :2]).all()
    diagnostic = _gradient_diagnostics(model, output)
    assert diagnostic["token_gradient_norm"] >= 0
    assert diagnostic["weighted_geometry_gradient_norm"] >= 0
    output["loss"].backward()
    assert model.bbox_head.weight.grad is not None


def test_auxiliary_bbox_rejects_answer_token_leakage():
    hidden = torch.randn(1, 3, 4)
    labels = torch.tensor([[1, 2, 3]])
    try:
        MultiModalGrounder._prompt_features_from_labels(hidden, labels)
    except ValueError as error:
        assert "preceding prompt" in str(error)
    else:
        raise AssertionError("answer without a prompt should be rejected")
