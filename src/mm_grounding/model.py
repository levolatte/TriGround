from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .adapters import (
    ParallelBackboneFusion,
    RDTDeepFusion,
    RDTStylePatchFusion,
    SafePostEmbedFusion,
)
from .boxes import cxcywh_to_xyxy, generalized_iou_aligned, xyxy_to_cxcywh
from .lora import inject_vision_lora, vision_lora_parameters


def _raw_patch_counts(image_grid_thw: torch.Tensor, total_patches: int) -> list[int]:
    counts = image_grid_thw.prod(dim=-1).detach().cpu().tolist()
    described = sum(counts)
    if described == total_patches:
        return counts
    if described > 0 and described % total_patches == 0:
        factor = described // total_patches
        if all(count % factor == 0 for count in counts):
            return [count // factor for count in counts]
    raise ValueError(
        f"image_grid_thw describes {described} patches, but tensors contain {total_patches}"
    )


def _patch_dim(backbone: nn.Module) -> int:
    vision = backbone.config.vision_config
    patch_size = int(vision.patch_size)
    channels = int(getattr(vision, "in_channels", getattr(vision, "in_chans", 3)))
    return channels * int(vision.temporal_patch_size) * patch_size**2


def _vision_model(backbone: nn.Module) -> nn.Module:
    for path in ("model.visual", "visual"):
        try:
            vision = backbone.get_submodule(path)
        except AttributeError:
            continue
        if hasattr(vision, "patch_embed"):
            return vision
    raise RuntimeError("Could not find the Qwen3-VL vision model Patch Embed")


def _language_hidden_size(backbone: nn.Module) -> int:
    text_config = getattr(backbone.config, "text_config", backbone.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Could not determine the Qwen language hidden size")
    return int(hidden_size)


def _rdt_deep_prompt_vision_forward(
    vision: nn.Module,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Qwen3-VL vision forward with an RDT prompt update before every block."""
    context = getattr(vision, "_rdt_prompt_context", None)
    if context is None:
        return vision._rdt_original_forward(hidden_states, grid_thw, **kwargs)
    fusion_ref = getattr(vision, "_rdt_fusion_ref", None)
    fusion = fusion_ref() if fusion_ref is not None else None
    if fusion is None:
        raise RuntimeError("RDT deep-prompt fusion module is no longer available")
    thermal_patches, depth_patches, patch_counts = context

    hidden_states = vision.patch_embed(hidden_states)
    hidden_states = hidden_states + vision.fast_pos_embed_interpolate(grid_thw)
    previous_prompt = fusion.initial_prompt(thermal_patches, depth_patches, patch_counts)
    if previous_prompt.shape != hidden_states.shape:
        raise RuntimeError(
            "initial D-TIR prompt shape does not match Qwen vision tokens: "
            f"{tuple(previous_prompt.shape)} vs {tuple(hidden_states.shape)}"
        )

    rotary_pos_emb = vision.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32, dim=0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    for layer_num, block in enumerate(vision.blocks):
        previous_prompt, injected = fusion.prompt_for_layer(
            layer_num, hidden_states, previous_prompt, patch_counts
        )
        hidden_states = block(
            injected,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in vision.deepstack_visual_indexes:
            merger_index = vision.deepstack_visual_indexes.index(layer_num)
            deepstack_feature_lists.append(vision.deepstack_merger_list[merger_index](hidden_states))

    return vision.merger(hidden_states), deepstack_feature_lists


def _install_rdt_deep_prompt_forward(vision: nn.Module, fusion: RDTDeepFusion) -> None:
    required = (
        "blocks",
        "patch_embed",
        "fast_pos_embed_interpolate",
        "rot_pos_emb",
        "deepstack_visual_indexes",
        "deepstack_merger_list",
        "merger",
    )
    missing = [name for name in required if not hasattr(vision, name)]
    if missing:
        raise RuntimeError(f"Qwen vision model cannot support deep prompting; missing {missing}")
    if len(vision.blocks) != len(fusion.prompt_blocks):
        raise ValueError("the number of prompt blocks must equal the number of vision blocks")
    object.__setattr__(vision, "_rdt_original_forward", vision.forward)
    object.__setattr__(vision, "_rdt_fusion_ref", weakref.ref(fusion))
    object.__setattr__(vision, "_rdt_prompt_context", None)
    object.__setattr__(vision, "forward", MethodType(_rdt_deep_prompt_vision_forward, vision))


def _parallel_backbone_vision_forward(
    vision: nn.Module,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run registered RGB/IR/depth through the shared frozen vision blocks."""
    context = getattr(vision, "_parallel_backbone_context", None)
    if context is None:
        return vision._parallel_original_forward(hidden_states, grid_thw, **kwargs)
    fusion_ref = getattr(vision, "_parallel_fusion_ref", None)
    fusion = fusion_ref() if fusion_ref is not None else None
    if fusion is None:
        raise RuntimeError("parallel-backbone fusion module is no longer available")
    (
        ir_patches,
        depth_patches,
        ir_query_tokens,
        depth_query_tokens,
        query_attention_mask,
    ) = context

    rgb_tokens = vision.patch_embed(hidden_states)
    ir_tokens = (
        vision.patch_embed(
            ir_patches.to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        if ir_patches is not None
        else None
    )
    depth_tokens = (
        vision.patch_embed(
            depth_patches.to(device=hidden_states.device, dtype=hidden_states.dtype)
        )
        if depth_patches is not None
        else None
    )
    for name, tokens in (("IR", ir_tokens), ("depth", depth_tokens)):
        if tokens is not None and rgb_tokens.shape != tokens.shape:
            raise ValueError(f"parallel RGB/{name} streams produced different token shapes")
    position = vision.fast_pos_embed_interpolate(grid_thw)
    rgb_tokens = rgb_tokens + position
    if ir_tokens is not None:
        ir_tokens = ir_tokens + position
    if depth_tokens is not None:
        depth_tokens = depth_tokens + position
    patch_counts = _raw_patch_counts(grid_thw, rgb_tokens.shape[0])

    rotary_pos_emb = vision.rot_pos_emb(grid_thw)
    seq_len, _ = rgb_tokens.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32, dim=0)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    for layer_num, block in enumerate(vision.blocks):
        block_kwargs = {
            "cu_seqlens": cu_seqlens,
            "position_embeddings": position_embeddings,
            **kwargs,
        }
        rgb_tokens = block(rgb_tokens, **block_kwargs)
        if ir_tokens is not None:
            ir_tokens = fusion.adapt_ir(layer_num, block(ir_tokens, **block_kwargs))
        if depth_tokens is not None:
            depth_tokens = fusion.adapt_depth(
                layer_num, block(depth_tokens, **block_kwargs)
            )
        rgb_tokens = fusion.fuse(
            layer_num,
            rgb_tokens,
            ir_tokens,
            depth_tokens,
            ir_query_tokens,
            depth_query_tokens,
            query_attention_mask,
            patch_counts,
        )
        if layer_num in vision.deepstack_visual_indexes:
            merger_index = vision.deepstack_visual_indexes.index(layer_num)
            deepstack_feature_lists.append(
                vision.deepstack_merger_list[merger_index](rgb_tokens)
            )
    return vision.merger(rgb_tokens), deepstack_feature_lists


def _install_parallel_backbone_forward(
    vision: nn.Module, fusion: ParallelBackboneFusion
) -> None:
    required = (
        "blocks",
        "patch_embed",
        "fast_pos_embed_interpolate",
        "rot_pos_emb",
        "deepstack_visual_indexes",
        "deepstack_merger_list",
        "merger",
    )
    missing = [name for name in required if not hasattr(vision, name)]
    if missing:
        raise RuntimeError(f"Qwen vision model cannot support parallel streams; missing {missing}")
    if len(vision.blocks) != len(fusion.ir_adapters):
        raise ValueError("parallel adapter count must equal the Qwen vision block count")
    object.__setattr__(vision, "_parallel_original_forward", vision.forward)
    object.__setattr__(vision, "_parallel_fusion_ref", weakref.ref(fusion))
    object.__setattr__(vision, "_parallel_backbone_context", None)
    object.__setattr__(vision, "forward", MethodType(_parallel_backbone_vision_forward, vision))


class _ContextualFusedPatchEmbed(nn.Module):
    """Wrap Qwen's frozen Patch Embed and apply an externally owned fusion module."""

    def __init__(self, base: nn.Module, fusion: SafePostEmbedFusion) -> None:
        super().__init__()
        self.base = base
        object.__setattr__(self, "_fusion_ref", weakref.ref(fusion))
        self._thermal_patches: torch.Tensor | None = None
        self._depth_patches: torch.Tensor | None = None
        self._patch_counts: list[int] | None = None

    def set_auxiliary(
        self,
        thermal_patches: torch.Tensor,
        depth_patches: torch.Tensor,
        patch_counts: list[int],
    ) -> None:
        self._thermal_patches = thermal_patches
        self._depth_patches = depth_patches
        self._patch_counts = patch_counts

    def clear_auxiliary(self) -> None:
        self._thermal_patches = None
        self._depth_patches = None
        self._patch_counts = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        rgb_tokens = self.base(pixel_values)
        if self._thermal_patches is None:
            return rgb_tokens
        fusion = self._fusion_ref()
        if fusion is None or self._depth_patches is None or self._patch_counts is None:
            raise RuntimeError("post-embed fusion context is incomplete")
        return fusion(
            rgb_tokens,
            self._thermal_patches,
            self._depth_patches,
            self._patch_counts,
        )


class MultiModalGrounder(nn.Module):
    """The project's only model: RGB/IR/depth fusion plus native Qwen grounding."""

    model_type = "multimodal_qwen"

    def __init__(
        self,
        backbone: nn.Module,
        adapter_channels: int = 128,
        orthogonal_channels: int = 8,
        prompt_gate_init: float = -3.0,
        fusion_type: str = "legacy_patch",
        modality_dropout: float = 0.1,
        fusion_residual_scale_init: float = 0.0,
        fusion_zero_init_prompt_restore: bool = False,
        parallel_fusion_stages: int = 4,
        parallel_adapter_scale_init: float = 0.01,
        query_encoder_layers: int = 1,
        query_attention_heads: int = 4,
        query_dropout: float = 0.0,
        auxiliary_bbox_enabled: bool = False,
        auxiliary_bbox_l1_weight: float = 2.0,
        auxiliary_bbox_giou_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.fusion_type = fusion_type
        raw_patch_dim = _patch_dim(backbone)
        object.__setattr__(self, "_fused_patch_embed_ref", None)
        object.__setattr__(self, "_deep_prompt_vision_ref", None)
        object.__setattr__(self, "_parallel_backbone_vision_ref", None)
        if fusion_type == "legacy_patch":
            self.fusion = RDTStylePatchFusion(
                raw_patch_dim, adapter_channels, orthogonal_channels, prompt_gate_init
            )
        elif fusion_type == "safe_post_embed":
            vision = _vision_model(backbone)
            token_dim = int(getattr(vision.patch_embed, "embed_dim"))
            self.fusion = SafePostEmbedFusion(
                raw_patch_dim,
                token_dim,
                adapter_channels,
                orthogonal_channels,
                modality_dropout,
                fusion_residual_scale_init,
                fusion_zero_init_prompt_restore,
            )
            wrapper = _ContextualFusedPatchEmbed(vision.patch_embed, self.fusion)
            vision.patch_embed = wrapper
            object.__setattr__(self, "_fused_patch_embed_ref", weakref.ref(wrapper))
        elif fusion_type == "rdt_deep":
            vision = _vision_model(backbone)
            token_dim = int(getattr(vision.patch_embed, "embed_dim"))
            self.fusion = RDTDeepFusion(
                raw_patch_dim,
                token_dim,
                len(vision.blocks),
                adapter_channels,
                orthogonal_channels,
                modality_dropout,
                fusion_residual_scale_init,
                fusion_zero_init_prompt_restore,
            )
            _install_rdt_deep_prompt_forward(vision, self.fusion)
            object.__setattr__(self, "_deep_prompt_vision_ref", weakref.ref(vision))
        elif fusion_type == "parallel_backbone":
            vision = _vision_model(backbone)
            token_dim = int(getattr(vision.patch_embed, "embed_dim"))
            self.fusion = ParallelBackboneFusion(
                token_dim=token_dim,
                num_layers=len(vision.blocks),
                hidden_dim=adapter_channels,
                language_dim=_language_hidden_size(backbone),
                num_fusion_stages=parallel_fusion_stages,
                query_encoder_layers=query_encoder_layers,
                query_attention_heads=query_attention_heads,
                query_dropout=query_dropout,
                modality_dropout=modality_dropout,
                adapter_scale_init=parallel_adapter_scale_init,
                fusion_scale_init=fusion_residual_scale_init,
                zero_init_restore=fusion_zero_init_prompt_restore,
            )
            _install_parallel_backbone_forward(vision, self.fusion)
            object.__setattr__(
                self, "_parallel_backbone_vision_ref", weakref.ref(vision)
            )
        else:
            raise ValueError(f"unsupported fusion_type: {fusion_type}")
        self.auxiliary_bbox_enabled = auxiliary_bbox_enabled
        self.auxiliary_bbox_l1_weight = auxiliary_bbox_l1_weight
        self.auxiliary_bbox_giou_weight = auxiliary_bbox_giou_weight
        self.bbox_head = (
            nn.Linear(_language_hidden_size(backbone), 4)
            if auxiliary_bbox_enabled
            else None
        )
        if self.bbox_head is not None:
            nn.init.normal_(self.bbox_head.weight, std=0.01)
            with torch.no_grad():
                self.bbox_head.bias.copy_(torch.tensor([0.0, 0.0, -1.1, -1.1]))

    def vision_lora_parameters(self) -> list[nn.Parameter]:
        return list(vision_lora_parameters(self.backbone))

    def enable_vision_lora(self) -> None:
        parameters = self.vision_lora_parameters()
        if not parameters:
            raise RuntimeError("No Vision LoRA parameters were injected")
        for parameter in parameters:
            parameter.requires_grad = True

    def set_phase_a_trainable(
        self, stage: str = "joint", freeze_parallel_adapters: bool = False
    ) -> None:
        if stage not in {"ir", "depth", "joint"}:
            raise ValueError("stage must be 'ir', 'depth', or 'joint'")
        for parameter in self.parameters():
            parameter.requires_grad = False
        if self.fusion_type == "parallel_backbone" and stage != "joint":
            prefixes = (f"{stage}_adapters.", f"{stage}_query_encoder.")
            for name, parameter in self.fusion.named_parameters():
                if name.startswith(prefixes) or f".{stage}." in name:
                    parameter.requires_grad = True
        else:
            for parameter in self.fusion.parameters():
                parameter.requires_grad = True
        if self.fusion_type == "parallel_backbone" and freeze_parallel_adapters:
            for name, parameter in self.fusion.named_parameters():
                if name.startswith(("ir_adapters.", "depth_adapters.")):
                    parameter.requires_grad = False
        if self.bbox_head is not None:
            for parameter in self.bbox_head.parameters():
                parameter.requires_grad = True

    def task_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.fusion.parameters())
        if self.bbox_head is not None:
            parameters.extend(self.bbox_head.parameters())
        return [parameter for parameter in parameters if parameter.requires_grad]

    def _post_embed_wrapper(self) -> _ContextualFusedPatchEmbed | None:
        reference = self._fused_patch_embed_ref
        return reference() if reference is not None else None

    def _deep_prompt_vision(self) -> nn.Module | None:
        reference = self._deep_prompt_vision_ref
        return reference() if reference is not None else None

    def _parallel_backbone_vision(self) -> nn.Module | None:
        reference = self._parallel_backbone_vision_ref
        return reference() if reference is not None else None

    def _fused_pixels(
        self,
        pixel_values: torch.Tensor,
        ir_pixel_values: torch.Tensor,
        depth_pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        rgb_only: bool = False,
    ) -> torch.Tensor:
        if rgb_only:
            return pixel_values
        if self.fusion_type in {"safe_post_embed", "rdt_deep", "parallel_backbone"}:
            return pixel_values
        if ir_pixel_values is None or depth_pixel_values is None:
            raise ValueError("RGB, IR and depth are all required")
        counts = _raw_patch_counts(image_grid_thw, pixel_values.shape[0])
        return self.fusion(pixel_values, ir_pixel_values, depth_pixel_values, counts)

    @contextmanager
    def _fusion_context(
        self,
        pixel_values: torch.Tensor,
        ir_pixel_values: torch.Tensor | None,
        depth_pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor,
        rgb_only: bool,
        query_input_ids: torch.Tensor | None = None,
        query_attention_mask: torch.Tensor | None = None,
    ):
        if self.fusion_type not in {
            "safe_post_embed",
            "rdt_deep",
            "parallel_backbone",
        } or rgb_only:
            yield
            return
        if self.fusion_type == "parallel_backbone":
            vision = self._parallel_backbone_vision()
            if ir_pixel_values is None and depth_pixel_values is None:
                raise ValueError("at least one auxiliary modality is required")
            if vision is None or query_input_ids is None or query_attention_mask is None:
                raise ValueError("parallel fusion requires query tokens and a vision model")
            embedding = self.backbone.get_input_embeddings()
            query_embeddings = embedding(query_input_ids).detach()
            ir_query, depth_query = self.fusion.encode_queries(
                query_embeddings,
                query_attention_mask,
                use_ir=ir_pixel_values is not None,
                use_depth=depth_pixel_values is not None,
            )
            object.__setattr__(
                vision,
                "_parallel_backbone_context",
                (
                    ir_pixel_values,
                    depth_pixel_values,
                    ir_query,
                    depth_query,
                    query_attention_mask,
                ),
            )
            try:
                yield
            finally:
                object.__setattr__(vision, "_parallel_backbone_context", None)
            return
        if self.fusion_type == "rdt_deep":
            vision = self._deep_prompt_vision()
            if ir_pixel_values is None or depth_pixel_values is None or vision is None:
                raise ValueError("RGB, IR and depth are all required")
            counts = _raw_patch_counts(image_grid_thw, pixel_values.shape[0])
            object.__setattr__(
                vision,
                "_rdt_prompt_context",
                (ir_pixel_values, depth_pixel_values, counts),
            )
            try:
                yield
            finally:
                object.__setattr__(vision, "_rdt_prompt_context", None)
            return
        wrapper = self._post_embed_wrapper()
        if ir_pixel_values is None or depth_pixel_values is None or wrapper is None:
            raise ValueError("RGB, IR and depth are all required")
        counts = _raw_patch_counts(image_grid_thw, pixel_values.shape[0])
        wrapper.set_auxiliary(ir_pixel_values, depth_pixel_values, counts)
        try:
            yield
        finally:
            wrapper.clear_auxiliary()

    def _qwen_inputs(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        image_grid_thw,
        ir_pixel_values=None,
        depth_pixel_values=None,
        query_input_ids=None,
        query_attention_mask=None,
        rgb_only=False,
    ) -> dict[str, torch.Tensor]:
        fused = self._fused_pixels(
            pixel_values, ir_pixel_values, depth_pixel_values, image_grid_thw, rgb_only
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": fused.to(next(self.backbone.parameters()).dtype),
            "image_grid_thw": image_grid_thw,
        }

    @staticmethod
    def _prompt_features_from_labels(
        hidden_states: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        supervised = labels.ne(-100)
        if not supervised.any(dim=1).all():
            raise ValueError("each sample must contain at least one supervised answer token")
        first_answer = supervised.to(torch.int64).argmax(dim=1)
        if (first_answer <= 0).any():
            raise ValueError("answer must have a preceding prompt token")
        prompt_positions = first_answer - 1
        rows = torch.arange(labels.shape[0], device=labels.device)
        if labels[rows, prompt_positions].ne(-100).any():
            raise RuntimeError("auxiliary bbox feature leaks a ground-truth answer token")
        if labels[rows, prompt_positions + 1].eq(-100).any():
            raise RuntimeError("auxiliary bbox feature is not immediately before the answer")
        return hidden_states[rows, prompt_positions]

    def _predict_bbox_from_features(
        self, features: torch.Tensor, gradient_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bbox_head is None:
            raise RuntimeError("auxiliary bbox head is disabled")
        scaled = features.detach() + gradient_scale * (features - features.detach())
        normalized = F.layer_norm(scaled.float(), (scaled.shape[-1],))
        raw = self.bbox_head(normalized.to(self.bbox_head.weight.dtype))
        predicted_cxcywh = raw.sigmoid()
        predicted_cxcywh = torch.cat(
            (predicted_cxcywh[..., :2], predicted_cxcywh[..., 2:].clamp_min(1e-4)),
            dim=-1,
        )
        return predicted_cxcywh, cxcywh_to_xyxy(predicted_cxcywh, clamp=False)

    @staticmethod
    def _coordinate_token_loss(
        logits: torch.Tensor, labels: torch.Tensor, coordinate_mask: torch.Tensor
    ) -> torch.Tensor:
        selected = coordinate_mask[:, 1:] & labels[:, 1:].ne(-100)
        if not selected.any():
            raise ValueError("coordinate token mask is empty after causal shift")
        return F.cross_entropy(logits[:, :-1][selected].float(), labels[:, 1:][selected])

    def forward(
        self,
        labels: torch.Tensor,
        bbox: torch.Tensor | None = None,
        coordinate_mask: torch.Tensor | None = None,
        geometry_gradient_scale: float = 1.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        with self._fusion_context(
            kwargs["pixel_values"],
            kwargs.get("ir_pixel_values"),
            kwargs.get("depth_pixel_values"),
            kwargs["image_grid_thw"],
            kwargs.get("rgb_only", False),
            kwargs.get("query_input_ids"),
            kwargs.get("query_attention_mask"),
        ):
            output = self.backbone(
                **self._qwen_inputs(**kwargs),
                labels=labels,
                use_cache=False,
                return_dict=True,
                output_hidden_states=self.auxiliary_bbox_enabled,
            )
        result = {"loss": output.loss, "token_loss": output.loss, "logits": output.logits}
        if not self.auxiliary_bbox_enabled:
            return result
        if bbox is None or coordinate_mask is None:
            raise ValueError("bbox and coordinate_mask are required for auxiliary supervision")
        features = self._prompt_features_from_labels(output.hidden_states[-1], labels)
        predicted_cxcywh, predicted_xyxy = self._predict_bbox_from_features(
            features, geometry_gradient_scale
        )
        target_cxcywh = xyxy_to_cxcywh(bbox).to(predicted_cxcywh.dtype)
        target_xyxy = bbox.to(predicted_xyxy.dtype)
        bbox_l1_loss = F.smooth_l1_loss(predicted_cxcywh, target_cxcywh)
        bbox_giou_loss = 1.0 - generalized_iou_aligned(
            predicted_xyxy, target_xyxy
        ).mean()
        coordinate_token_loss = self._coordinate_token_loss(
            output.logits, labels, coordinate_mask
        )
        total = (
            output.loss
            + self.auxiliary_bbox_l1_weight * bbox_l1_loss
            + self.auxiliary_bbox_giou_weight * bbox_giou_loss
        )
        result.update(
            {
                "loss": total,
                "coordinate_token_loss": coordinate_token_loss,
                "bbox_l1_loss": bbox_l1_loss,
                "bbox_giou_loss": bbox_giou_loss,
                "auxiliary_bbox": cxcywh_to_xyxy(predicted_cxcywh),
            }
        )
        return result

    @torch.no_grad()
    def predict_auxiliary_bbox(self, **kwargs) -> torch.Tensor:
        if self.bbox_head is None:
            raise RuntimeError("auxiliary bbox head is disabled")
        with self._fusion_context(
            kwargs["pixel_values"],
            kwargs.get("ir_pixel_values"),
            kwargs.get("depth_pixel_values"),
            kwargs["image_grid_thw"],
            kwargs.get("rgb_only", False),
            kwargs.get("query_input_ids"),
            kwargs.get("query_attention_mask"),
        ):
            output = self.backbone(
                **self._qwen_inputs(**kwargs),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )
        attention_mask = kwargs["attention_mask"]
        positions = attention_mask.sum(dim=1).to(torch.int64) - 1
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        features = output.hidden_states[-1][rows, positions]
        predicted_cxcywh, _ = self._predict_bbox_from_features(features)
        return cxcywh_to_xyxy(predicted_cxcywh)

    @torch.no_grad()
    def generate(self, max_new_tokens: int = 96, **kwargs) -> torch.Tensor:
        with self._fusion_context(
            kwargs["pixel_values"],
            kwargs.get("ir_pixel_values"),
            kwargs.get("depth_pixel_values"),
            kwargs["image_grid_thw"],
            kwargs.get("rgb_only", False),
            kwargs.get("query_input_ids"),
            kwargs.get("query_attention_mask"),
        ):
            return self.backbone.generate(
                **self._qwen_inputs(**kwargs), max_new_tokens=max_new_tokens, do_sample=False
            )


def build_grounder(model_config, processor=None) -> MultiModalGrounder:
    """Build the single supported RGB+IR+depth architecture."""
    from transformers import Qwen3VLForConditionalGeneration

    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        model_config.backbone, dtype="auto"
    )
    if model_config.vision_lora_enabled:
        inject_vision_lora(
            backbone,
            rank=model_config.vision_lora_rank,
            alpha=model_config.vision_lora_alpha,
            dropout=model_config.vision_lora_dropout,
            last_n_blocks=model_config.vision_lora_last_n_blocks,
        )
    return MultiModalGrounder(
        backbone,
        adapter_channels=model_config.adapter_channels,
        orthogonal_channels=model_config.orthogonal_channels,
        prompt_gate_init=model_config.prompt_gate_init,
        fusion_type=model_config.fusion_type,
        modality_dropout=model_config.modality_dropout,
        fusion_residual_scale_init=model_config.fusion_residual_scale_init,
        fusion_zero_init_prompt_restore=model_config.fusion_zero_init_prompt_restore,
        parallel_fusion_stages=model_config.parallel_fusion_stages,
        parallel_adapter_scale_init=model_config.parallel_adapter_scale_init,
        query_encoder_layers=model_config.query_encoder_layers,
        query_attention_heads=model_config.query_attention_heads,
        query_dropout=model_config.query_dropout,
        auxiliary_bbox_enabled=model_config.auxiliary_bbox_enabled,
        auxiliary_bbox_l1_weight=model_config.auxiliary_bbox_l1_weight,
        auxiliary_bbox_giou_weight=model_config.auxiliary_bbox_giou_weight,
    )
