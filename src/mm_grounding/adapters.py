from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RDTStylePatchFusion(nn.Module):
    """Fuse aligned depth and thermal patches into an RGB-guided Qwen prompt."""

    def __init__(self, patch_dim: int, hidden_dim: int = 128, orthogonal_dim: int = 8, gate_init: float = -3.0):
        super().__init__()
        self.shared_aux_encoder = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden_dim), nn.GELU()
        )
        self.depth_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.thermal_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.dt_restore = nn.Sequential(nn.Linear(orthogonal_dim * 2, hidden_dim), nn.GELU())
        self.rgb_prompt_projection = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, orthogonal_dim))
        self.dt_prompt_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.prompt_restore = nn.Linear(orthogonal_dim, patch_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(0.5))
        self.beta_logit = nn.Parameter(torch.tensor(0.5))
        self.fovea_scale = nn.Parameter(torch.tensor(10.0))
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))
        nn.init.zeros_(self.prompt_restore.weight)
        nn.init.zeros_(self.prompt_restore.bias)

    @staticmethod
    def _check_shape(name, patches, rgb):
        if patches.shape != rgb.shape:
            raise ValueError(f"RGB/{name} patch shapes differ: {rgb.shape} vs {patches.shape}")

    @staticmethod
    def _foveate(features, patch_counts, scale):
        if sum(patch_counts) != features.shape[0]:
            raise ValueError("patch counts do not match the flattened image patches")
        return torch.cat([
            chunk * torch.softmax(chunk * scale, dim=0)
            for chunk in torch.split(features, patch_counts, dim=0)
        ], dim=0)

    def forward(self, rgb_patches, thermal_patches, depth_patches, patch_counts):
        self._check_shape("thermal", thermal_patches, rgb_patches)
        self._check_shape("depth", depth_patches, rgb_patches)
        depth = self.depth_projection(self.shared_aux_encoder(depth_patches))
        thermal = self.thermal_projection(self.shared_aux_encoder(thermal_patches))
        product = depth * thermal
        depth_overlap = product / (thermal.norm(dim=-1, keepdim=True) + 1e-6) * thermal
        thermal_overlap = product / (depth.norm(dim=-1, keepdim=True) + 1e-6) * depth
        depth = depth - torch.sigmoid(self.alpha_logit) * depth_overlap
        thermal = thermal - torch.sigmoid(self.beta_logit) * thermal_overlap
        dt = self.dt_restore(torch.cat((depth, thermal), dim=-1))
        rgb_prompt = self._foveate(
            self.rgb_prompt_projection(rgb_patches), patch_counts, self.fovea_scale
        )
        prompt = rgb_prompt + self.dt_prompt_projection(dt)
        return rgb_patches + torch.sigmoid(self.gate_logit) * self.prompt_restore(prompt)


class SafePostEmbedFusion(nn.Module):
    """Identity-preserving auxiliary fusion for post-Patch-Embed RGB tokens.

    Depth and thermal inputs remain raw flattened patches and use independent
    encoders.  The resulting dense auxiliary prompt is added only after Qwen's
    frozen RGB Patch Embed, so the pretrained RGB input path remains intact.
    The ReZero-style residual scale is initialized to zero, making a newly
    constructed module exactly equivalent to RGB-only inference.
    """

    def __init__(
        self,
        raw_patch_dim: int,
        rgb_token_dim: int,
        hidden_dim: int = 128,
        orthogonal_dim: int = 8,
        modality_dropout: float = 0.1,
        residual_scale_init: float = 0.0,
        zero_init_prompt_restore: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.raw_patch_dim = raw_patch_dim
        self.rgb_token_dim = rgb_token_dim
        self.modality_dropout = modality_dropout
        self.depth_encoder = nn.Sequential(
            nn.LayerNorm(raw_patch_dim), nn.Linear(raw_patch_dim, hidden_dim), nn.GELU()
        )
        self.thermal_encoder = nn.Sequential(
            nn.LayerNorm(raw_patch_dim), nn.Linear(raw_patch_dim, hidden_dim), nn.GELU()
        )
        self.depth_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.thermal_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.aux_restore = nn.Sequential(
            nn.Linear(orthogonal_dim * 2, hidden_dim), nn.GELU()
        )
        self.rgb_context = nn.Sequential(
            nn.LayerNorm(rgb_token_dim), nn.Linear(rgb_token_dim, hidden_dim), nn.GELU()
        )
        self.reliability = nn.Linear(hidden_dim * 2, 1)
        self.prompt_restore = nn.Linear(hidden_dim, rgb_token_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(0.5))
        self.beta_logit = nn.Parameter(torch.tensor(0.5))
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        nn.init.constant_(self.reliability.bias, -2.0)
        if zero_init_prompt_restore:
            nn.init.zeros_(self.prompt_restore.weight)
            nn.init.zeros_(self.prompt_restore.bias)

    @staticmethod
    def _check_patch_counts(patch_counts: list[int], total_patches: int) -> None:
        if any(count <= 0 for count in patch_counts) or sum(patch_counts) != total_patches:
            raise ValueError("patch counts do not match the flattened image patches")

    @staticmethod
    def _remove_overlap(
        depth: torch.Tensor,
        thermal: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remove the vector projection of each modality onto the other."""
        dot = (depth * thermal).sum(dim=-1, keepdim=True)
        depth_overlap = dot / thermal.square().sum(dim=-1, keepdim=True).clamp_min(eps) * thermal
        thermal_overlap = dot / depth.square().sum(dim=-1, keepdim=True).clamp_min(eps) * depth
        return depth - alpha * depth_overlap, thermal - beta * thermal_overlap

    def _drop_modalities(
        self,
        depth: torch.Tensor,
        thermal: torch.Tensor,
        patch_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout == 0.0:
            return depth, thermal
        repeats = torch.tensor(patch_counts, device=depth.device)
        depth_keep = torch.rand(len(patch_counts), 1, device=depth.device) >= self.modality_dropout
        thermal_keep = torch.rand(len(patch_counts), 1, device=thermal.device) >= self.modality_dropout
        depth_mask = torch.repeat_interleave(depth_keep, repeats, dim=0).to(depth.dtype)
        thermal_mask = torch.repeat_interleave(thermal_keep, repeats, dim=0).to(thermal.dtype)
        return depth * depth_mask, thermal * thermal_mask

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        thermal_patches: torch.Tensor,
        depth_patches: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        if thermal_patches.shape != depth_patches.shape:
            raise ValueError(
                "Depth and thermal patch shapes differ: "
                f"{depth_patches.shape} vs {thermal_patches.shape}"
            )
        if thermal_patches.ndim != 2 or thermal_patches.shape[-1] != self.raw_patch_dim:
            raise ValueError(
                f"expected auxiliary patches [N, {self.raw_patch_dim}], "
                f"got {tuple(thermal_patches.shape)}"
            )
        if rgb_tokens.ndim != 2 or rgb_tokens.shape != (
            thermal_patches.shape[0], self.rgb_token_dim
        ):
            raise ValueError(
                f"expected RGB tokens [{thermal_patches.shape[0]}, {self.rgb_token_dim}], "
                f"got {tuple(rgb_tokens.shape)}"
            )
        self._check_patch_counts(patch_counts, rgb_tokens.shape[0])
        work_dtype = self.depth_encoder[1].weight.dtype
        depth = self.depth_projection(self.depth_encoder(depth_patches.to(work_dtype)))
        thermal = self.thermal_projection(self.thermal_encoder(thermal_patches.to(work_dtype)))
        depth, thermal = self._drop_modalities(depth, thermal, patch_counts)
        depth, thermal = self._remove_overlap(
            depth,
            thermal,
            torch.sigmoid(self.alpha_logit),
            torch.sigmoid(self.beta_logit),
        )
        auxiliary = self.aux_restore(torch.cat((depth, thermal), dim=-1))
        context = self.rgb_context(rgb_tokens.to(work_dtype))
        reliability = torch.sigmoid(self.reliability(torch.cat((context, auxiliary), dim=-1)))
        prompt = reliability * self.prompt_restore(auxiliary)
        return rgb_tokens + self.residual_scale.to(rgb_tokens.dtype) * prompt.to(rgb_tokens.dtype)


class RDTRecurrentPromptBlock(nn.Module):
    """Regenerate a visual prompt from current RGB tokens and the previous prompt."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        residual_scale_init: float,
        zero_init_prompt_restore: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rgb_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.prompt_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.prompt_restore = nn.Linear(hidden_dim, token_dim)
        self.fovea_scale = nn.Parameter(torch.tensor(10.0))
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        if zero_init_prompt_restore:
            nn.init.zeros_(self.prompt_restore.weight)
            nn.init.zeros_(self.prompt_restore.bias)

    def _foveate(self, rgb: torch.Tensor, patch_counts: list[int]) -> torch.Tensor:
        return torch.cat(
            [
                chunk * torch.softmax(chunk * self.fovea_scale, dim=0)
                for chunk in torch.split(rgb, patch_counts, dim=0)
            ],
            dim=0,
        )

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        previous_prompt: torch.Tensor,
        patch_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb_tokens.shape != previous_prompt.shape:
            raise ValueError("RGB tokens and previous prompt must have identical shapes")
        work_dtype = self.prompt_restore.weight.dtype
        rgb = self.rgb_projection(rgb_tokens.to(work_dtype))
        previous = self.prompt_projection(previous_prompt.to(work_dtype))
        enhanced_rgb = self._foveate(rgb, patch_counts)
        prompt = self.prompt_restore(F.gelu(enhanced_rgb + previous))
        injected = rgb_tokens + self.residual_scale.to(rgb_tokens.dtype) * prompt.to(
            rgb_tokens.dtype
        )
        return prompt, injected


class RDTDeepFusion(nn.Module):
    """Depth/TIR fusion followed by the paper's recurrent RGB prompt updates."""

    def __init__(
        self,
        raw_patch_dim: int,
        token_dim: int,
        num_layers: int,
        hidden_dim: int = 128,
        orthogonal_dim: int = 8,
        modality_dropout: float = 0.1,
        residual_scale_init: float = 0.0,
        zero_init_prompt_restore: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("RDT deep prompting requires at least one vision layer")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.raw_patch_dim = raw_patch_dim
        self.token_dim = token_dim
        self.modality_dropout = modality_dropout
        self.depth_encoder = nn.Sequential(
            nn.LayerNorm(raw_patch_dim), nn.Linear(raw_patch_dim, hidden_dim), nn.GELU()
        )
        self.thermal_encoder = nn.Sequential(
            nn.LayerNorm(raw_patch_dim), nn.Linear(raw_patch_dim, hidden_dim), nn.GELU()
        )
        self.depth_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.thermal_projection = nn.Linear(hidden_dim, orthogonal_dim)
        self.aux_restore = nn.Sequential(
            nn.Linear(orthogonal_dim * 2, hidden_dim), nn.GELU()
        )
        self.initial_prompt_restore = nn.Linear(hidden_dim, token_dim)
        self.alpha_logit = nn.Parameter(torch.tensor(0.5))
        self.beta_logit = nn.Parameter(torch.tensor(0.5))
        self.prompt_blocks = nn.ModuleList(
            [
                RDTRecurrentPromptBlock(
                    token_dim,
                    hidden_dim,
                    residual_scale_init,
                    zero_init_prompt_restore,
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _check_patch_counts(patch_counts: list[int], total_patches: int) -> None:
        if any(count <= 0 for count in patch_counts) or sum(patch_counts) != total_patches:
            raise ValueError("patch counts do not match the flattened image patches")

    def initial_prompt(
        self,
        thermal_patches: torch.Tensor,
        depth_patches: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        if thermal_patches.shape != depth_patches.shape:
            raise ValueError("Depth and thermal patch shapes differ")
        if thermal_patches.ndim != 2 or thermal_patches.shape[-1] != self.raw_patch_dim:
            raise ValueError(f"expected auxiliary patches [N, {self.raw_patch_dim}]")
        self._check_patch_counts(patch_counts, thermal_patches.shape[0])
        work_dtype = self.depth_encoder[1].weight.dtype
        depth = self.depth_projection(self.depth_encoder(depth_patches.to(work_dtype)))
        thermal = self.thermal_projection(self.thermal_encoder(thermal_patches.to(work_dtype)))
        if self.training and self.modality_dropout:
            repeats = torch.tensor(patch_counts, device=depth.device)
            depth_keep = torch.rand(len(patch_counts), 1, device=depth.device)
            thermal_keep = torch.rand(len(patch_counts), 1, device=thermal.device)
            depth_mask = torch.repeat_interleave(
                depth_keep >= self.modality_dropout, repeats, dim=0
            ).to(depth.dtype)
            thermal_mask = torch.repeat_interleave(
                thermal_keep >= self.modality_dropout, repeats, dim=0
            ).to(thermal.dtype)
            depth, thermal = depth * depth_mask, thermal * thermal_mask
        depth, thermal = SafePostEmbedFusion._remove_overlap(
            depth,
            thermal,
            torch.sigmoid(self.alpha_logit),
            torch.sigmoid(self.beta_logit),
        )
        auxiliary = self.aux_restore(torch.cat((depth, thermal), dim=-1))
        return self.initial_prompt_restore(auxiliary)

    def prompt_for_layer(
        self,
        layer_index: int,
        rgb_tokens: torch.Tensor,
        previous_prompt: torch.Tensor,
        patch_counts: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= layer_index < len(self.prompt_blocks):
            raise IndexError("prompt layer index is out of range")
        return self.prompt_blocks[layer_index](rgb_tokens, previous_prompt, patch_counts)


class ModalityBackboneAdapter(nn.Module):
    """A small residual adapter for one frozen vision-backbone stream."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.down = nn.Linear(token_dim, hidden_dim)
        self.up = nn.Linear(hidden_dim, token_dim)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        # Preserve the frozen stream exactly at initialization while allowing
        # the up projection to receive gradients on the first update.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        work_dtype = self.down.weight.dtype
        residual = self.up(F.gelu(self.down(self.norm(tokens.to(work_dtype)))))
        return tokens + self.residual_scale.to(tokens.dtype) * residual.to(tokens.dtype)


class QueryTokenEncoder(nn.Module):
    """Contextualize frozen Qwen word embeddings without a second LM pass."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or attention_mask.shape != embeddings.shape[:2]:
            raise ValueError("query embeddings and attention mask shapes are incompatible")
        work_dtype = self.input_projection[1].weight.dtype
        tokens = self.input_projection(embeddings.to(work_dtype))
        tokens = self.encoder(tokens, src_key_padding_mask=attention_mask.eq(0))
        return self.output_norm(tokens)


class ModalityStageFusion(nn.Module):
    """Query-conditioned residual from one auxiliary modality into RGB."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        query_attention_heads: int,
        modality_dropout: float,
        residual_scale_init: float,
        zero_init_restore: bool,
    ) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.rgb_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.aux_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim), nn.GELU()
        )
        self.language_attention = nn.MultiheadAttention(
            hidden_dim, query_attention_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.restore = nn.Linear(hidden_dim, token_dim)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        nn.init.zeros_(self.restore.bias)
        if zero_init_restore:
            nn.init.zeros_(self.restore.weight)

    def _language_context(
        self,
        rgb: torch.Tensor,
        query_tokens: torch.Tensor,
        query_attention_mask: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        if query_tokens.shape[:2] != query_attention_mask.shape:
            raise ValueError("query tokens and attention mask shapes differ")
        if query_tokens.shape[0] != len(patch_counts):
            raise ValueError("query batch size does not match image patch groups")
        contexts = []
        for index, rgb_chunk in enumerate(torch.split(rgb, patch_counts, dim=0)):
            context, _ = self.language_attention(
                rgb_chunk.unsqueeze(0),
                query_tokens[index : index + 1],
                query_tokens[index : index + 1],
                key_padding_mask=query_attention_mask[index : index + 1].eq(0),
                need_weights=False,
            )
            contexts.append(context.squeeze(0))
        return torch.cat(contexts, dim=0)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        auxiliary_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        query_attention_mask: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        if rgb_tokens.shape != auxiliary_tokens.shape:
            raise ValueError("parallel RGB/auxiliary token shapes must match")
        if sum(patch_counts) != rgb_tokens.shape[0]:
            raise ValueError("patch counts do not match parallel vision tokens")
        work_dtype = self.restore.weight.dtype
        rgb = self.rgb_projection(rgb_tokens.to(work_dtype))
        auxiliary = self.aux_projection(auxiliary_tokens.to(work_dtype))
        if self.training and self.modality_dropout:
            keep = torch.rand(len(patch_counts), 1, device=auxiliary.device)
            keep = keep >= self.modality_dropout
            repeats = torch.tensor(patch_counts, device=auxiliary.device)
            auxiliary = auxiliary * torch.repeat_interleave(
                keep, repeats, dim=0
            ).to(auxiliary.dtype)
        language = self._language_context(
            rgb, query_tokens.to(work_dtype), query_attention_mask, patch_counts
        )
        gate = torch.sigmoid(self.gate(torch.cat((rgb, auxiliary, language), dim=-1)))
        residual = self.restore(F.gelu(auxiliary + language))
        return self.residual_scale.to(rgb_tokens.dtype) * gate.to(
            rgb_tokens.dtype
        ) * residual.to(rgb_tokens.dtype)


class QueryAwareStageFusion(nn.Module):
    """Independent IR and depth residuals at one sparse fusion boundary."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        query_attention_heads: int,
        modality_dropout: float,
        residual_scale_init: float,
        zero_init_restore: bool,
    ) -> None:
        super().__init__()
        arguments = (
            token_dim,
            hidden_dim,
            query_attention_heads,
            modality_dropout,
            residual_scale_init,
            zero_init_restore,
        )
        self.ir = ModalityStageFusion(*arguments)
        self.depth = ModalityStageFusion(*arguments)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        ir_tokens: torch.Tensor | None,
        depth_tokens: torch.Tensor | None,
        ir_query_tokens: torch.Tensor | None,
        depth_query_tokens: torch.Tensor | None,
        query_attention_mask: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        output = rgb_tokens
        if ir_tokens is not None:
            if ir_query_tokens is None:
                raise ValueError("IR query tokens are required for IR fusion")
            output = output + self.ir(
                rgb_tokens, ir_tokens, ir_query_tokens, query_attention_mask, patch_counts
            )
        if depth_tokens is not None:
            if depth_query_tokens is None:
                raise ValueError("depth query tokens are required for depth fusion")
            output = output + self.depth(
                rgb_tokens,
                depth_tokens,
                depth_query_tokens,
                query_attention_mask,
                patch_counts,
            )
        return output


class ParallelBackboneFusion(nn.Module):
    """IR/depth backbone adapters plus sparse stage-boundary fusion modules."""

    def __init__(
        self,
        token_dim: int,
        num_layers: int,
        hidden_dim: int = 128,
        language_dim: int = 2048,
        num_fusion_stages: int = 4,
        query_encoder_layers: int = 1,
        query_attention_heads: int = 4,
        query_dropout: float = 0.0,
        modality_dropout: float = 0.1,
        adapter_scale_init: float = 0.01,
        fusion_scale_init: float = 0.001,
        zero_init_restore: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("parallel-backbone fusion requires at least one vision layer")
        if not 1 <= num_fusion_stages <= num_layers:
            raise ValueError("fusion stages must be between one and the vision layer count")
        self.ir_adapters = nn.ModuleList(
            [
                ModalityBackboneAdapter(token_dim, hidden_dim, adapter_scale_init)
                for _ in range(num_layers)
            ]
        )
        self.depth_adapters = nn.ModuleList(
            [
                ModalityBackboneAdapter(token_dim, hidden_dim, adapter_scale_init)
                for _ in range(num_layers)
            ]
        )
        query_arguments = (
            language_dim,
            hidden_dim,
            query_encoder_layers,
            query_attention_heads,
            query_dropout,
        )
        self.ir_query_encoder = QueryTokenEncoder(*query_arguments)
        self.depth_query_encoder = QueryTokenEncoder(*query_arguments)
        # Use approximately equally spaced stage ends and always include the
        # final vision block.  A set removes duplicate rounded positions.
        indices = {
            max(0, min(num_layers - 1, round((stage + 1) * num_layers / num_fusion_stages) - 1))
            for stage in range(num_fusion_stages)
        }
        self.fusion_layer_indices = tuple(sorted(indices))
        self.stage_fusions = nn.ModuleDict(
            {
                str(index): QueryAwareStageFusion(
                    token_dim,
                    hidden_dim,
                    query_attention_heads,
                    modality_dropout,
                    fusion_scale_init,
                    zero_init_restore,
                )
                for index in self.fusion_layer_indices
            }
        )

    def adapt_ir(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.ir_adapters[layer_index](tokens)

    def adapt_depth(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.depth_adapters[layer_index](tokens)

    def encode_queries(
        self,
        query_embeddings: torch.Tensor,
        query_attention_mask: torch.Tensor,
        use_ir: bool,
        use_depth: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ir = (
            self.ir_query_encoder(query_embeddings, query_attention_mask)
            if use_ir
            else None
        )
        depth = (
            self.depth_query_encoder(query_embeddings, query_attention_mask)
            if use_depth
            else None
        )
        return ir, depth

    def fuse(
        self,
        layer_index: int,
        rgb_tokens: torch.Tensor,
        ir_tokens: torch.Tensor | None,
        depth_tokens: torch.Tensor | None,
        ir_query_tokens: torch.Tensor | None,
        depth_query_tokens: torch.Tensor | None,
        query_attention_mask: torch.Tensor,
        patch_counts: list[int],
    ) -> torch.Tensor:
        key = str(layer_index)
        if key not in self.stage_fusions:
            return rgb_tokens
        return self.stage_fusions[key](
            rgb_tokens,
            ir_tokens,
            depth_tokens,
            ir_query_tokens,
            depth_query_tokens,
            query_attention_mask,
            patch_counts,
        )
