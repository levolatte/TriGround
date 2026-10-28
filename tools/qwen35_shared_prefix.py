# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Explicit, differentiable, one-question shared-prefix Qwen3.5 training.

The two-round GDN layout adapts VeOmni's Apache-2.0 shared_prefix.py:
https://github.com/ByteDance-Seed/VeOmni/tree/main/veomni/models/transformers/qwen3_5
Mixer equations adapt Transformers 5.16.1 modeling_qwen3_5.py (Apache-2.0).
Local changes: explicit question identity, real multimodal positions, batched
SDPA/conv, existing HF/PEFT projection modules, and an opt-in functional entry.
The full license is in docs/research/2026-10-28-training-speed/evidence/Apache-2.0.txt.
Nothing is patched globally. This module never uses an inference Cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


GDN_CHUNK_SIZE = 64
TraceCallback = Callable[[str, torch.Tensor, dict], None]


def _emit(trace: TraceCallback | None, name: str, value: torch.Tensor, *, seq_dim: int = 1):
    if trace is not None:
        # The callback owns selection/copying. No hidden .cpu(), clone, detach or
        # full-activation retention is imposed on the normal/diagnostic caller.
        trace(name, value, {"layout": "compact", "seq_dim": seq_dim})


@dataclass
class SharedPrefixPlan:
    batch_size: int
    sequence_length: int
    prefix_length: int
    suffix_lengths: tuple[int, ...]
    starts: tuple[int, ...]
    compact_src: torch.Tensor
    expand_index: torch.Tensor
    valid_mask: torch.Tensor
    suffix_rows: tuple[slice, ...]
    suffix_attention_mask: torch.Tensor
    gdn_b_index: torch.Tensor
    gdn_b_cu: torch.Tensor
    gdn_b_lengths: tuple[int, ...]
    gdn_b_keep: torch.Tensor

    @property
    def compact_length(self) -> int:
        return self.prefix_length + sum(self.suffix_lengths)

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.prefix_length + sum(self.suffix_lengths)

    @property
    def gdn_boundary(self) -> int:
        return self.prefix_length // GDN_CHUNK_SIZE * GDN_CHUNK_SIZE

    @property
    def gdn_replay_tokens(self) -> int:
        return self.batch_size * (self.prefix_length - self.gdn_boundary)

    def compact(self, value: torch.Tensor) -> torch.Tensor:
        """[G,L,...] -> [1,C,...], preserving autograd scatter-add semantics."""
        flat = value.reshape(self.batch_size * self.sequence_length, *value.shape[2:])
        return flat.index_select(0, self.compact_src).unsqueeze(0)

    def compact_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        pos = position_ids.unsqueeze(0) if position_ids.ndim == 2 else position_ids
        # TextModel strips the first (text-only) coordinate of four-axis input.
        pos = pos[1:] if pos.shape[0] == 4 else pos
        if pos.shape[0] == 1:
            pos = pos.expand(3, -1, -1)
        return pos.reshape(3, -1).index_select(-1, self.compact_src).unsqueeze(1)

    def expand(self, value: torch.Tensor, logits_to_keep: int = 0) -> torch.Tensor:
        """[1,C,H] -> [G,K,H]. Padding rows are zero, and never supervised."""
        index = self.expand_index[:, -logits_to_keep:] if logits_to_keep else self.expand_index
        mask = self.valid_mask[:, -logits_to_keep:] if logits_to_keep else self.valid_mask
        out = value[0].index_select(0, index.reshape(-1)).reshape(*index.shape, -1)
        return out * mask.unsqueeze(-1)

    def causal_conv1d(self, x: torch.Tensor, conv, activation: str) -> torch.Tensor:
        """Project once; replay only the last kernel_size-1 *pre-conv* rows."""
        p = self.prefix_length
        history = min(conv.kernel_size[0] - 1, p)

        def run(rows):
            length = rows.shape[1]
            out = F.conv1d(rows.transpose(1, 2).to(conv.weight.dtype), conv.weight,
                           conv.bias, padding=conv.kernel_size[0] - 1,
                           groups=conv.groups)[..., :length]
            if activation not in ("silu", "swish"):
                raise NotImplementedError(f"Qwen3.5 conv activation {activation!r}")
            return F.silu(out).transpose(1, 2).to(x.dtype)

        prefix = run(x[:, :p])
        branches = [torch.cat((x[0, p - history:p], x[0, rows])) for rows in self.suffix_rows]
        suffix = run(pad_sequence(branches, batch_first=True))
        return torch.cat([prefix, *[suffix[i:i + 1, history:history + n]
                                   for i, n in enumerate(self.suffix_lengths)]], dim=1)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  scaling: float, diagnostic_attention: str = "current") -> torch.Tensor:
        """Two SDPA calls: one causal prefix, one batch of independent suffixes."""
        p = self.prefix_length
        groups = query.shape[1] // key.shape[1]
        repeated_key = key.repeat_interleave(groups, dim=1)
        repeated_value = value.repeat_interleave(groups, dim=1)
        if diagnostic_attention == "native_gqa":
            # HF 5.16.1's no-mask native prefix dispatch. This is an explicit
            # diagnostic candidate, NOT a silent change to the failed baseline.
            from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
            use_gqa = groups > 1 and use_gqa_in_sdpa(None, key, value)
            prefix_k = key[:, :, :p] if use_gqa else repeated_key[:, :, :p]
            prefix_v = value[:, :, :p] if use_gqa else repeated_value[:, :, :p]
            prefix = F.scaled_dot_product_attention(query[:, :, :p], prefix_k, prefix_v,
                                                    is_causal=True, scale=scaling, enable_gqa=use_gqa)
        elif diagnostic_attention == "current":
            prefix = F.scaled_dot_product_attention(query[:, :, :p],
                                                    repeated_key[:, :, :p], repeated_value[:, :, :p],
                                                    is_causal=True, scale=scaling)
        else:
            raise ValueError(f"unknown diagnostic attention mode: {diagnostic_attention!r}")
        key, value = repeated_key, repeated_value

        def suffix_batch(t):
            # [1,H,C,D] -> [G,H,max(R),D]
            return pad_sequence([t[0, :, rows].transpose(0, 1) for rows in self.suffix_rows],
                                batch_first=True).transpose(1, 2)

        sq, sk, sv = [suffix_batch(t) for t in (query, key, value)]
        sk = torch.cat((key[:, :, :p].expand(self.batch_size, -1, -1, -1), sk), dim=2)
        sv = torch.cat((value[:, :, :p].expand(self.batch_size, -1, -1, -1), sv), dim=2)
        suffix = F.scaled_dot_product_attention(sq, sk, sv,
                                                attn_mask=self.suffix_attention_mask, scale=scaling)
        return torch.cat([prefix, *[suffix[i:i + 1, :, :n]
                                   for i, n in enumerate(self.suffix_lengths)]], dim=2)

    def gated_delta_rule(self, kernel: Callable, query, key, value, g, beta):
        """FLA varlen initial-state gradient; fork at a 64-token chunk boundary."""
        tensors = (query, key, value, g, beta)
        boundary = self.gdn_boundary
        if boundary:
            prefix, state = kernel(*(t[:, :boundary].contiguous() for t in tensors),
                                   initial_state=None, output_final_state=True,
                                   use_qk_l2norm_in_kernel=True)
            # index_select backward SUMS every branch's boundary-state gradient.
            initial = state.index_select(0, torch.zeros(len(self.gdn_b_lengths),
                                                        dtype=torch.long, device=state.device))
        else:
            prefix = value[:, :0]
            initial = torch.zeros((len(self.gdn_b_lengths), key.shape[2], key.shape[-1], value.shape[-1]),
                                  dtype=torch.float32, device=key.device)
        tail, _ = kernel(*(t.index_select(1, self.gdn_b_index).contiguous() for t in tensors),
                         initial_state=initial, output_final_state=False,
                         cu_seqlens=self.gdn_b_cu, use_qk_l2norm_in_kernel=True)
        return torch.cat((prefix, tail.index_select(1, self.gdn_b_keep)), dim=1)


def build_shared_prefix_plan(input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                             position_ids: torch.Tensor, prefix_length: int) -> SharedPrefixPlan:
    """Build ONE explicit same-question group, with no token-based cross-media grouping.

    input_ids/attention_mask: [G,L], positions: [3,G,L], [4,G,L] or text [G,L].
    prefix_length counts valid prompt tokens, excluding left padding. Valid
    regions must be contiguous; each member must have at least one suffix token.
    The caller supplies exact original multimodal positions; this function never
    derives them from flattened image placeholder IDs.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] < 2:
        raise ValueError("shared prefix requires input_ids [G,L], G >= 2")
    g, length = input_ids.shape
    if position_ids is None or position_ids.shape[-2:] != (g, length):
        raise ValueError("explicit original position_ids must end in [G,L]")
    if position_ids.ndim not in (2, 3) or (position_ids.ndim == 3 and position_ids.shape[0] not in (3, 4)):
        raise ValueError("position_ids must be [G,L], [3,G,L], or [4,G,L]")
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if valid.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    ids_cpu, mask_cpu = input_ids.detach().cpu(), valid.cpu()
    positions = position_ids.detach().cpu().reshape(-1, g, length)
    starts, suffix_lengths, sources, suffix_rows = [], [], [], []
    expand = torch.zeros((g, length), dtype=torch.long)
    for i in range(g):
        nonzero = mask_cpu[i].nonzero().flatten()
        if nonzero.numel() == 0:
            raise ValueError("empty member sequence")
        start, end = int(nonzero[0]), int(nonzero[-1]) + 1
        if not mask_cpu[i, start:end].all() or not 0 < prefix_length < end - start:
            raise ValueError("require contiguous valid tokens and 0 < prefix_length < valid length")
        starts.append(start)
        suffix_lengths.append(end - start - prefix_length)
        if i == 0:
            sources.extend(range(start, start + prefix_length))
        else:
            if not torch.equal(ids_cpu[0, starts[0]:starts[0] + prefix_length], ids_cpu[i, start:start + prefix_length]):
                raise ValueError("explicit question group has unequal prompt token IDs")
            if not torch.equal(positions[:, 0, starts[0]:starts[0] + prefix_length], positions[:, i, start:start + prefix_length]):
                raise ValueError("explicit question group has unequal prompt multimodal positions")
        expand[i, start:start + prefix_length] = torch.arange(prefix_length)
        row0 = len(sources)
        sources.extend(range(i * length + start + prefix_length, i * length + end))
        suffix_rows.append(slice(row0, len(sources)))
        expand[i, start + prefix_length:end] = torch.arange(row0, len(sources))
    device = input_ids.device
    def tensor(values):
        return torch.tensor(values, dtype=torch.long, device=device)
    max_r = max(suffix_lengths)
    qi = torch.arange(max_r, device=device)[:, None]
    ki = torch.arange(prefix_length + max_r, device=device)[None, :]
    suffix_mask = (ki <= prefix_length + qi)[None, None]
    # Padded suffix queries are discarded; future padded keys are already causal-masked.
    boundary = prefix_length // GDN_CHUNK_SIZE * GDN_CHUNK_SIZE
    tail = list(range(boundary, prefix_length))
    b_index, b_lengths, b_keep = [], [], []
    if tail:
        b_index.extend(tail)
        b_lengths.append(len(tail))
        b_keep.extend(range(len(tail)))
    for rows in suffix_rows:
        offset = len(b_index) + len(tail)
        branch = list(range(rows.start, rows.stop))
        b_index.extend(tail + branch)
        b_lengths.append(len(tail) + len(branch))
        b_keep.extend(range(offset, offset + len(branch)))
    cu = [0]
    for n in b_lengths:
        cu.append(cu[-1] + n)
    return SharedPrefixPlan(g, length, prefix_length, tuple(suffix_lengths), tuple(starts),
                            tensor(sources), expand.to(device), valid.to(device), tuple(suffix_rows),
                            suffix_mask, tensor(b_index), torch.tensor(cu, dtype=torch.int32, device=device),
                            tuple(b_lengths), tensor(b_keep))


def torch_reference_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                                     output_final_state=False, cu_seqlens=None,
                                     use_qk_l2norm_in_kernel=True):
    """Slow differentiable CPU oracle, selected explicitly; never a production fallback."""
    if cu_seqlens is not None:
        cu = cu_seqlens.cpu().tolist()
        outputs, states = [], []
        for i, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
            out, state = torch_reference_gated_delta_rule(
                *(t[:, start:end] for t in (query, key, value, g, beta)),
                initial_state=None if initial_state is None else initial_state[i:i + 1],
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
            outputs.append(out)
            if state is not None:
                states.append(state)
        return torch.cat(outputs, dim=1), torch.cat(states) if states else None
    dtype = query.dtype
    query, key, value, g, beta = [t.float() for t in (query, key, value, g, beta)]
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    query = query * query.shape[-1] ** -0.5
    state = (torch.zeros((key.shape[0], key.shape[2], key.shape[-1], value.shape[-1]),
                         dtype=torch.float32, device=key.device) if initial_state is None else initial_state.float())
    outputs = []
    for i in range(query.shape[1]):
        state = state * g[:, i].exp()[..., None, None]
        memory = (state * key[:, i, ..., None]).sum(-2)
        delta = (value[:, i] - memory) * beta[:, i, ..., None]
        state = state + key[:, i, ..., None] * delta[..., None, :]
        outputs.append((state * query[:, i, ..., None]).sum(-2))
    out = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return out.to(dtype), state if output_final_state else None


def _resolve_gdn_kernel(backend: str) -> Callable:
    if backend == "torch_reference":
        return torch_reference_gated_delta_rule
    if backend != "fla":
        raise ValueError(f"unknown shared-prefix GDN backend: {backend!r}")
    # Deliberately bypass HF's fallback dispatcher: missing FLA is a hard error.
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _gdn_forward(module, hidden_states, plan, kernel, trace=None, trace_prefix=""):
    batch, length, _ = hidden_states.shape
    qkv = module.in_proj_qkv(hidden_states)
    _emit(trace, trace_prefix + ".in_proj_qkv", qkv)
    mixed = plan.causal_conv1d(qkv, module.conv1d, module.activation)
    _emit(trace, trace_prefix + ".conv", mixed)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    z_projection = module.in_proj_z(hidden_states)
    _emit(trace, trace_prefix + ".in_proj_z", z_projection)
    z = z_projection.reshape(batch, length, -1, module.head_v_dim)
    b = module.in_proj_b(hidden_states)
    _emit(trace, trace_prefix + ".in_proj_b", b)
    beta = b.sigmoid()
    a = module.in_proj_a(hidden_states)
    _emit(trace, trace_prefix + ".in_proj_a", a)
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    _emit(trace, trace_prefix + ".g", g)
    _emit(trace, trace_prefix + ".beta", beta)
    groups = module.num_v_heads // module.num_k_heads
    query, key = [t.repeat_interleave(groups, dim=2) for t in (query, key)]
    out = plan.gated_delta_rule(kernel, query, key, value, g, beta)
    _emit(trace, trace_prefix + ".core", out)
    out = module.norm(out.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    out = out.reshape(batch, length, -1)
    _emit(trace, trace_prefix + ".norm", out)
    out = module.out_proj(out)
    _emit(trace, trace_prefix + ".out_proj", out)
    return out


def _attention_forward(module, hidden_states, position_embeddings, plan, trace=None,
                       trace_prefix="", diagnostic_attention="current"):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    shape = hidden_states.shape[:-1]
    q_projection = module.q_proj(hidden_states)
    _emit(trace, trace_prefix + ".q_proj", q_projection)
    query, gate = torch.chunk(q_projection.view(*shape, -1, 2 * module.head_dim), 2, dim=-1)
    query = module.q_norm(query.reshape(*shape, -1, module.head_dim)).transpose(1, 2)
    _emit(trace, trace_prefix + ".q_norm", query.transpose(1, 2))
    k_projection = module.k_proj(hidden_states)
    _emit(trace, trace_prefix + ".k_proj", k_projection)
    key = module.k_norm(k_projection.view(*shape, -1, module.head_dim)).transpose(1, 2)
    _emit(trace, trace_prefix + ".k_norm", key.transpose(1, 2))
    v_projection = module.v_proj(hidden_states)
    _emit(trace, trace_prefix + ".v_proj", v_projection)
    value = v_projection.view(*shape, -1, module.head_dim).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    _emit(trace, trace_prefix + ".rotary_query", query, seq_dim=2)
    _emit(trace, trace_prefix + ".rotary_key", key, seq_dim=2)
    out = plan.attention(query, key, value, module.scaling, diagnostic_attention).transpose(1, 2).reshape(*shape, -1)
    _emit(trace, trace_prefix + ".core", out)
    out = module.o_proj(out * gate.reshape(*shape, -1).sigmoid())
    _emit(trace, trace_prefix + ".o_proj", out)
    return out


def _decoder_forward(hidden_states, layer, positions, plan, kernel, trace=None,
                     trace_prefix="", diagnostic_attention="current"):
    _emit(trace, trace_prefix + ".input", hidden_states)
    normed = layer.input_layernorm(hidden_states)
    _emit(trace, trace_prefix + ".input_layernorm", normed)
    if layer.block_type == "linear_attention":
        mixed = _gdn_forward(layer.linear_attn, normed, plan, kernel, trace, trace_prefix + ".linear_attn")
    elif layer.block_type == "full_attention":
        if layer.self_attn.attention_dropout:
            raise ValueError("shared-prefix training requires attention_dropout=0")
        mixed = _attention_forward(layer.self_attn, normed, positions, plan, trace,
                                    trace_prefix + ".self_attn", diagnostic_attention)
    else:
        raise NotImplementedError(layer.block_type)
    _emit(trace, trace_prefix + ".mixer", mixed)
    hidden_states = hidden_states + mixed
    _emit(trace, trace_prefix + ".after_mixer", hidden_states)
    post_norm = layer.post_attention_layernorm(hidden_states)
    _emit(trace, trace_prefix + ".post_attention_layernorm", post_norm)
    mlp = layer.mlp(post_norm)
    _emit(trace, trace_prefix + ".mlp", mlp)
    hidden_states = hidden_states + mlp
    _emit(trace, trace_prefix + ".output", hidden_states)
    return hidden_states


def native_sequence_mixer_forward(layer, *, plan: SharedPrefixPlan, hidden_states: torch.Tensor,
                                   position_embeddings: tuple[torch.Tensor, torch.Tensor],
                                   native_sequence_kwargs: list[dict] | None = None) -> torch.Tensor:
    """Diagnostic same-input mixer replay, sequential native mb1, compact output.

    Pass a captured compact input_layernorm output. This isolates local mixer
    numerical error from error already accumulated in previous layers. No full
    G-by-P activation batch is materialized. Projection and sequence kernels in
    each native mixer retain their original full-sequence shapes.
    """
    if torch.is_grad_enabled():
        raise ValueError("native mixer replay is diagnostic only; call inside torch.no_grad()")
    pieces = []
    for i, rows in enumerate(plan.suffix_rows):
        sequence_kwargs = {} if native_sequence_kwargs is None else native_sequence_kwargs[i]
        index = torch.cat((torch.arange(plan.prefix_length, device=hidden_states.device),
                           torch.arange(rows.start, rows.stop, device=hidden_states.device)))
        branch = hidden_states.index_select(1, index)
        if layer.block_type == "linear_attention":
            out = layer.linear_attn(branch, cache_params=None, attention_mask=None, **sequence_kwargs)
        elif layer.block_type == "full_attention":
            positions = tuple(t.index_select(1, index) for t in position_embeddings)
            out = layer.self_attn(branch, position_embeddings=positions,
                                  attention_mask=None, past_key_values=None, **sequence_kwargs)[0]
        else:
            raise NotImplementedError(layer.block_type)
        if i == 0:
            pieces.append(out[:, :plan.prefix_length])
        pieces.append(out[:, plan.prefix_length:])
    return torch.cat(pieces, dim=1)


def _unwrap_model(model):
    # PEFT get_base_model retains the LoRA/NF4 modules in the returned backbone.
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base.config, "model_type", None) != "qwen3_5":
        raise ValueError("shared-prefix entry requires Qwen3_5ForConditionalGeneration")
    return base


def shared_prefix_forward(model, *, plan: SharedPrefixPlan, input_ids: torch.Tensor,
                          position_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                          pixel_values: torch.Tensor | None = None,
                          image_grid_thw: torch.Tensor | None = None,
                          frozen_image_features: torch.Tensor | None = None,
                          logits_to_keep: int = 0, checkpoint_layers: bool = True,
                          gdn_backend: str = "fla", trace: TraceCallback | None = None,
                          diagnostic_attention: str = "current"):
    """Functional opt-in HF/PEFT forward, returning logits [G,K,V].

    Pixels/grid/features belong to ONE question (e.g. its three original images),
    not G repeated copies. Precomputed features are final visual+merger outputs
    [N_image_tokens,H], with the original placeholder order. Only valid rows are
    equivalent to native forward; padding logits must be masked by the caller.
    Explicit positions are mandatory. Cache/generation/distributed/offload hooks
    are intentionally outside this single-device training entry.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast
    base = _unwrap_model(model)
    if trace is not None and checkpoint_layers and torch.is_grad_enabled() and base.training:
        raise ValueError("diagnostic trace requires checkpoint_layers=False or torch.no_grad()")
    if diagnostic_attention != "current" and torch.is_grad_enabled():
        raise ValueError("diagnostic attention candidate is no_grad only")
    if diagnostic_attention not in ("current", "native_gqa"):
        raise ValueError(f"unknown diagnostic attention mode: {diagnostic_attention!r}")
    multimodal, language = base.model, base.model.language_model
    if tuple(input_ids.shape) != (plan.batch_size, plan.sequence_length):
        raise ValueError("input_ids shape differs from the plan")
    if not 0 <= logits_to_keep <= plan.sequence_length:
        raise ValueError("logits_to_keep must be between 0 and sequence length")
    if attention_mask is not None and not torch.equal(attention_mask.bool(), plan.valid_mask):
        raise ValueError("attention_mask differs from the plan")
    if pixel_values is not None and frozen_image_features is not None:
        raise ValueError("pass unique pixels or frozen_image_features, not both")
    if any(p.requires_grad for p in multimodal.visual.parameters()):
        raise ValueError("shared-prefix visual features require a fully frozen visual+merger")
    if base.training and any(getattr(m, "p", 0) for m in language.modules() if isinstance(m, torch.nn.Dropout)):
        raise ValueError("shared-prefix equivalence requires language/LoRA dropout=0")
    kernel = _resolve_gdn_kernel(gdn_backend)
    if gdn_backend == "fla" and input_ids.device.type != "cuda":
        raise ValueError("FLA production backend requires CUDA; use explicit torch_reference only for CPU tests")
    compact_ids = plan.compact(input_ids)
    hidden_states = language.embed_tokens(compact_ids)
    if pixel_values is not None:
        with torch.no_grad():
            image_output = multimodal.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            frozen_image_features = torch.cat(image_output.pooler_output, dim=0)
    if frozen_image_features is not None:
        features = frozen_image_features.detach()
        if features.is_inference():
            features = features.clone()
        features = features.to(device=hidden_states.device, dtype=hidden_states.dtype)
        image_mask, _ = multimodal.get_placeholder_mask(compact_ids, inputs_embeds=hidden_states, image_features=features)
        hidden_states = hidden_states.masked_scatter(image_mask, features)
    elif (compact_ids == base.config.image_token_id).any():
        raise ValueError("image placeholders require unique pixels or frozen_image_features")
    if (compact_ids == base.config.video_token_id).any():
        raise NotImplementedError("shared-prefix first version supports image questions, not video")
    _emit(trace, "compact.input_embeds", hidden_states)
    compact_positions = plan.compact_positions(position_ids)
    _emit(trace, "compact.position_ids", compact_positions, seq_dim=2)
    positions = language.rotary_emb(hidden_states, compact_positions)
    _emit(trace, "rope.cos", positions[0])
    _emit(trace, "rope.sin", positions[1])
    for i, layer in enumerate(language.layers[:language.config.num_hidden_layers]):
        forward = partial(_decoder_forward, layer=layer, positions=positions, plan=plan, kernel=kernel,
                           trace=trace, trace_prefix=f"layers.{i}", diagnostic_attention=diagnostic_attention)
        if checkpoint_layers and torch.is_grad_enabled() and base.training:
            hidden_states = checkpoint(forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = forward(hidden_states)
    hidden_states = language.norm(hidden_states)
    _emit(trace, "final.norm", hidden_states)
    # The last prefix hidden row is gathered G times: each answer's first label
    # receives its own probability, and their gradients sum into the same row.
    selected = plan.expand(hidden_states, logits_to_keep)
    logits = base.lm_head(selected)
    if trace is not None:
        metadata = {"layout": "expanded", "seq_dim": 1, "logits_to_keep": selected.shape[1],
                    "sequence_offset": plan.sequence_length - selected.shape[1]}
        trace("final.selected", selected, metadata)
        trace("final.logits", logits, metadata)
    return Qwen3_5CausalLMOutputWithPast(logits=logits, past_key_values=None,
                                        rope_deltas=None)


def shared_prefix_answer_logps(model, *, completion_ids: torch.Tensor, **forward_kwargs) -> torch.Tensor:
    """Return [G,R] aligned to the original padded completion IDs (mask externally)."""
    if "logits_to_keep" in forward_kwargs:
        raise ValueError("answer_logps selects logits_to_keep itself")
    logits = shared_prefix_forward(model, logits_to_keep=completion_ids.shape[1] + 1,
                                   **forward_kwargs).logits[:, :-1].float()
    return logits.log_softmax(-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
