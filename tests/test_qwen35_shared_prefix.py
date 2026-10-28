"""CPU gradient checks against independent dense branches and real HF Qwen3.5."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen35_shared_prefix import (
    build_shared_prefix_plan,
    native_sequence_mixer_forward,
    shared_prefix_forward,
    torch_reference_gated_delta_rule,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _batch(prefix=67, suffix_lengths=(3,) * 8, left=0):
    length = prefix + max(suffix_lengths) + left
    ids = torch.zeros((len(suffix_lengths), length), dtype=torch.long)
    mask = torch.zeros_like(ids)
    positions = torch.zeros((4, len(suffix_lengths), length), dtype=torch.long)
    for i, n in enumerate(suffix_lengths):
        ids[i, left:left + prefix] = torch.arange(prefix) % 40 + 1
        ids[i, left + prefix:left + prefix + n] = (torch.arange(n) + i * 3) % 20 + 41
        mask[i, left:left + prefix + n] = 1
        positions[:, i, left:left + prefix + n] = torch.arange(prefix + n)
        # Nontrivial, shared multimodal H/W coordinates, not a flat-text proxy.
        positions[2, i, left:left + prefix] //= 3
        positions[3, i, left:left + prefix] %= 5
    return ids, mask, positions


def test_explicit_group_plan_and_first_answer_scatter():
    ids, mask, positions = _batch(suffix_lengths=(2, 4, 3), left=2)
    plan = build_shared_prefix_plan(ids, mask, positions, 67)
    assert plan.compact_length == 76
    assert plan.total_tokens == 210
    assert plan.gdn_boundary == 64
    assert plan.gdn_replay_tokens == 9
    x = torch.randn(1, plan.compact_length, 2, requires_grad=True)
    expanded = plan.expand(x)
    torch.testing.assert_close(expanded[:, 2:69], x[:, :67].expand(3, -1, -1))
    # Every member predicts its first answer from this exact shared hidden row.
    (expanded[0, 68].sum() + 2 * expanded[1, 68].sum() + 4 * expanded[2, 68].sum()).backward()
    torch.testing.assert_close(x.grad[0, 66], torch.full((2,), 7.0))
    assert not expanded[:, :2].any()


def test_reject_different_media_positions_and_interior_mask():
    ids, mask, positions = _batch(suffix_lengths=(2, 3))
    positions[3, 1, 10] += 1
    with pytest.raises(ValueError, match="multimodal positions"):
        build_shared_prefix_plan(ids, mask, positions, 67)
    positions[3, 1, 10] -= 1
    mask[0, 10] = 0
    with pytest.raises(ValueError, match="contiguous"):
        build_shared_prefix_plan(ids, mask, positions, 67)


@pytest.mark.parametrize("prefix", [5, 64, 67])
def test_mixers_match_independent_branch_outputs_and_gradients(prefix):
    torch.manual_seed(6)
    ids, mask, pos = _batch(prefix=prefix, suffix_lengths=(2, 4, 3))
    plan = build_shared_prefix_plan(ids, mask, pos, prefix)
    x = torch.randn(1, plan.compact_length, 10, requires_grad=True)
    dense = x.detach().clone().requires_grad_()
    conv = torch.nn.Conv1d(10, 10, 4, groups=10, padding=3, bias=True)
    conv_ref = copy.deepcopy(conv)
    actual = plan.causal_conv1d(x, conv, "silu")
    prefix_out = F.silu(conv_ref(dense[:, :prefix].transpose(1, 2))[:, :, :prefix]).transpose(1, 2)
    suffixes = []
    for rows in plan.suffix_rows:
        full = torch.cat((dense[:, :prefix], dense[:, rows]), dim=1)
        out = F.silu(conv_ref(full.transpose(1, 2))[:, :, :full.shape[1]]).transpose(1, 2)
        suffixes.append(out[:, prefix:])
    expected = torch.cat((prefix_out, *suffixes), dim=1)
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(x.grad, dense.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(conv.weight.grad, conv_ref.weight.grad, atol=2e-6, rtol=2e-5)

    inputs = [torch.randn(1, plan.compact_length, 2, d, requires_grad=True) for d in (4, 4, 3)]
    inputs += [(-torch.rand(1, plan.compact_length, 2)).requires_grad_(),
               torch.rand(1, plan.compact_length, 2, requires_grad=True)]
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    actual = plan.gated_delta_rule(torch_reference_gated_delta_rule, *inputs)
    prefix_out, _ = torch_reference_gated_delta_rule(*(t[:, :prefix] for t in refs))
    suffixes = []
    for rows in plan.suffix_rows:
        out, _ = torch_reference_gated_delta_rule(*(torch.cat((t[:, :prefix], t[:, rows]), dim=1) for t in refs))
        suffixes.append(out[:, prefix:])
    expected = torch.cat((prefix_out, *suffixes), dim=1)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    for actual_input, ref_input in zip(inputs, refs):
        torch.testing.assert_close(actual_input.grad, ref_input.grad, atol=3e-6, rtol=3e-4)
    # A loss exclusively on answer rows must backpropagate through shared prefix state.
    assert inputs[1].grad[:, :prefix].abs().sum() > 0


def test_sdpa_branches_and_prefix_kv_gradients():
    torch.manual_seed(7)
    ids, mask, pos = _batch(prefix=9, suffix_lengths=(2, 4, 3))
    plan = build_shared_prefix_plan(ids, mask, pos, 9)
    inputs = [torch.randn(1, h, plan.compact_length, 4, requires_grad=True) for h in (4, 2, 2)]
    refs = [t.detach().clone().requires_grad_() for t in inputs]
    actual = plan.attention(*inputs, scaling=0.5)
    q, k, v = refs
    k, v = [t.repeat_interleave(2, dim=1) for t in (k, v)]
    expected = [F.scaled_dot_product_attention(q[:, :, :9], k[:, :, :9], v[:, :, :9], is_causal=True)]
    for rows in plan.suffix_rows:
        branch = [torch.cat((t[:, :, :9], t[:, :, rows]), dim=2) for t in (q, k, v)]
        expected.append(F.scaled_dot_product_attention(*branch, is_causal=True)[:, :, 9:])
    expected = torch.cat(expected, dim=2)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(inputs, refs):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5)


def _tiny_model(cuda_probe=False):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    hidden = 128 if cuda_probe else 32
    head_dim = 64 if cuda_probe else 16
    linear_dim = 32 if cuda_probe else 4
    config = modeling.Qwen3_5Config(
        image_token_id=62, video_token_id=63, vision_start_token_id=60, vision_end_token_id=61,
        text_config=dict(vocab_size=64, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=head_dim,
                         linear_num_key_heads=1, linear_num_value_heads=2,
                         linear_key_head_dim=linear_dim, linear_value_head_dim=linear_dim,
                         layer_types=["linear_attention", "full_attention"],
                         rope_parameters=dict(rope_type="default", rope_theta=10000,
                                              partial_rotary_factor=1.0,
                                              mrope_section=[11, 11, 10] if cuda_probe else [3, 3, 2])),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
                           out_hidden_size=hidden, num_position_embeddings=16,
                           patch_size=2, temporal_patch_size=1, spatial_merge_size=1))
    config._attn_implementation = "sdpa"
    model = modeling.Qwen3_5ForConditionalGeneration(config)
    model.model.visual.requires_grad_(False)
    return model.train()


def _native_logits(model, ids, mask, positions, features, logits_to_keep=4):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    outputs = []
    for i in range(len(ids)):
        embeds = base.model.language_model.embed_tokens(ids[i:i + 1])
        image_mask, _ = base.model.get_placeholder_mask(ids[i:i + 1], inputs_embeds=embeds, image_features=features)
        embeds = embeds.masked_scatter(image_mask, features)
        outputs.append(model(inputs_embeds=embeds, attention_mask=mask[i:i + 1],
                             position_ids=positions[:, i:i + 1], use_cache=False,
                             logits_to_keep=logits_to_keep).logits)
    return torch.cat(outputs)


@pytest.mark.parametrize("with_lora,checkpoint_layers,padded", [(False, False, False), (True, True, False),
                                                              (True, True, True)])
def test_real_hf_hybrid_multimodal_logps_gradients_and_update(with_lora, checkpoint_layers, padded):
    torch.manual_seed(8)
    native = _tiny_model()
    if with_lora:
        peft = pytest.importorskip("peft")
        native = peft.get_peft_model(native, peft.LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0,
            target_modules=["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        # Nonzero adapter B exercises A as well as B gradient paths.
        for name, param in native.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.01)
    shared = copy.deepcopy(native)
    ids, mask, positions = (_batch(suffix_lengths=(1, 2, 3, 1, 3, 2, 3, 2), left=2) if padded else _batch())
    ids[:, [8, 18, 28]] = 62  # three images' final visual feature placeholders
    features = torch.randn(3, 32)
    plan = build_shared_prefix_plan(ids, mask, positions, 67)
    expected = _native_logits(native, ids, mask, positions, features)
    actual = shared_prefix_forward(shared, plan=plan, input_ids=ids, position_ids=positions,
                                   attention_mask=mask, frozen_image_features=features,
                                   logits_to_keep=4, gdn_backend="torch_reference",
                                   checkpoint_layers=checkpoint_layers).logits
    valid_logits = mask[:, -4:].bool()
    torch.testing.assert_close(actual[valid_logits], expected[valid_logits], atol=3e-6, rtol=5e-4)
    answers = ids[:, -3:]
    expected_logps = expected[:, :-1].float().log_softmax(-1).gather(-1, answers[..., None]).squeeze(-1)
    actual_logps = actual[:, :-1].float().log_softmax(-1).gather(-1, answers[..., None]).squeeze(-1)
    # Different first-answer labels and per-answer weights exercise boundary scatter-add.
    weights = torch.linspace(0.3, 2, 24).reshape(8, 3) * mask[:, -3:]
    expected_loss = -(expected_logps * weights).mean()
    actual_loss = -(actual_logps * weights).mean()
    expected_loss.backward()
    actual_loss.backward()
    nonzero = 0
    for (name, a), (_, b) in zip(native.named_parameters(), shared.named_parameters()):
        if a.requires_grad:
            assert a.grad is not None and b.grad is not None, name
            torch.testing.assert_close(b.grad, a.grad, atol=3e-6, rtol=8e-4, msg=name)
            nonzero += int(a.grad.abs().sum() > 0)
    assert nonzero > 10
    if not with_lora:
        assert shared.model.language_model.embed_tokens.weight.grad[1:40].abs().sum() > 0
    torch.optim.SGD((p for p in native.parameters() if p.requires_grad), lr=0.03).step()
    torch.optim.SGD((p for p in shared.parameters() if p.requires_grad), lr=0.03).step()
    for (name, a), (_, b) in zip(native.named_parameters(), shared.named_parameters()):
        torch.testing.assert_close(b, a, atol=3e-7, rtol=2e-5, msg=name)


def test_missing_backend_and_unfrozen_vision_fail_explicitly():
    model = _tiny_model()
    ids, mask, positions = _batch(prefix=5)
    plan = build_shared_prefix_plan(ids, mask, positions, 5)
    with pytest.raises(ValueError, match="unknown shared-prefix GDN backend"):
        shared_prefix_forward(model, plan=plan, input_ids=ids, position_ids=positions, gdn_backend="guess")
    model.model.visual.requires_grad_(True)
    with pytest.raises(ValueError, match="fully frozen"):
        shared_prefix_forward(model, plan=plan, input_ids=ids, position_ids=positions, gdn_backend="torch_reference")


def test_unique_three_image_encoding_and_inference_feature_tensor():
    torch.manual_seed(9)
    model = _tiny_model()
    ids, mask, positions = _batch(prefix=5)
    ids[:, [0, 2, 4]] = 62
    plan = build_shared_prefix_plan(ids, mask, positions, 5)
    pixels = torch.randn(3, 12)
    grid = torch.tensor([[1, 1, 1]] * 3)
    with torch.inference_mode():
        features = torch.cat(model.model.get_image_features(pixels, grid, return_dict=True).pooler_output)
    assert features.is_inference()
    calls = []
    handle = model.model.visual.register_forward_hook(lambda *args: calls.append(1))
    try:
        from_pixels = shared_prefix_forward(model, plan=plan, input_ids=ids, position_ids=positions,
                                            pixel_values=pixels, image_grid_thw=grid, logits_to_keep=4,
                                            gdn_backend="torch_reference", checkpoint_layers=False).logits
        from_features = shared_prefix_forward(model, plan=plan, input_ids=ids, position_ids=positions,
                                              frozen_image_features=features, logits_to_keep=4,
                                              gdn_backend="torch_reference", checkpoint_layers=False).logits
        assert calls == [1]
        torch.testing.assert_close(from_pixels, from_features)
        from_features.sum().backward()
        assert model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.model.visual.parameters())
    finally:
        handle.remove()


def test_trace_keeps_default_path_and_native_same_input_mixer_replay():
    torch.manual_seed(11)
    model = _tiny_model()
    ids, mask, positions = _batch(prefix=67)
    plan = build_shared_prefix_plan(ids, mask, positions, 67)
    captures = {}

    def trace(name, value, metadata):
        captures[name] = (value.detach().clone(), dict(metadata))

    arguments = dict(plan=plan, input_ids=ids, position_ids=positions, attention_mask=mask,
                     logits_to_keep=4, gdn_backend="torch_reference", checkpoint_layers=False)
    with torch.no_grad():
        plain = shared_prefix_forward(model, **arguments).logits
        traced = shared_prefix_forward(model, **arguments, trace=trace).logits
        candidate = shared_prefix_forward(model, **arguments, diagnostic_attention="native_gqa").logits
        assert torch.equal(traced, plain)
        torch.testing.assert_close(candidate, plain, atol=3e-6, rtol=5e-5)
        assert captures["compact.input_embeds"][0].shape == (1, plan.compact_length, 32)
        assert captures["compact.position_ids"][1]["seq_dim"] == 2
        assert captures["layers.1.self_attn.rotary_query"][1]["seq_dim"] == 2
        assert captures["final.selected"][1] == {
            "layout": "expanded", "seq_dim": 1, "logits_to_keep": 4,
            "sequence_offset": ids.shape[1] - 4}
        rope = (captures["rope.cos"][0], captures["rope.sin"][0])
        for i, layer in enumerate(model.model.language_model.layers):
            replay = native_sequence_mixer_forward(
                layer, plan=plan, hidden_states=captures[f"layers.{i}.input_layernorm"][0],
                position_embeddings=rope)
            torch.testing.assert_close(replay, captures[f"layers.{i}.mixer"][0], atol=3e-6, rtol=5e-4)
    with pytest.raises(ValueError, match="no_grad only"):
        shared_prefix_forward(model, **arguments, diagnostic_attention="native_gqa")
    with pytest.raises(ValueError, match="diagnostic only"):
        native_sequence_mixer_forward(model.model.language_model.layers[0], plan=plan,
                                     hidden_states=captures["layers.0.input_layernorm"][0],
                                     position_embeddings=rope)


@pytest.mark.skipif(os.environ.get("AIC_RUN_CUDA_SHARED_PREFIX") != "1" or not torch.cuda.is_available(),
                    reason="explicit AIC_RUN_CUDA_SHARED_PREFIX=1 and CUDA required")
def test_cuda_fla_long_prefix_lora_parity(monkeypatch):
    """Independent GPU probe: opt in explicitly; BF16 + FLA + real LoRA, P3500/R23/G8.

    AIC_RUN_CUDA_SHARED_PREFIX=1 python -m pytest -s -q \
        tests/test_qwen35_shared_prefix.py -k cuda_fla_long
    This intentionally does not load the 27B model or quantize a random tiny model.
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    import peft

    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    # Force the native oracle to use exactly the installed production FLA kernel,
    # so an HF dispatcher fallback cannot turn the test into another CPU oracle.
    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", chunk_gated_delta_rule)
    native = _tiny_model(cuda_probe=True).cuda().bfloat16()
    native = peft.get_peft_model(native, peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for name, parameter in native.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
    shared = copy.deepcopy(native)
    ids, mask, positions = [t.cuda() for t in _batch(prefix=3500, suffix_lengths=(23,) * 8)]
    ids[:, [8, 18, 28]] = 62
    features = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    plan = build_shared_prefix_plan(ids, mask, positions, 3500)
    assert plan.gdn_boundary == 3456 and plan.gdn_replay_tokens == 352

    # This small direct kernel check makes the initial/final-state gradient
    # requirement observable independently of the model's output tolerances.
    q, k = [torch.randn(1, 64, 2, 32, device="cuda", dtype=torch.bfloat16,
                         requires_grad=True) for _ in range(2)]
    v = torch.randn_like(q, requires_grad=True)
    g = -torch.rand(1, 64, 2, device="cuda")
    beta = torch.rand(1, 64, 2, device="cuda", dtype=torch.bfloat16)
    _, state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True,
                                      use_qk_l2norm_in_kernel=True)
    state.retain_grad()
    initial = state.index_select(0, torch.tensor([0, 0], device="cuda"))
    short = [t[:, :23].repeat(1, 2, *([1] * (t.ndim - 2))) for t in (q, k, v, g, beta)]
    cu = torch.tensor([0, 23, 46], dtype=torch.int32, device="cuda")
    branch, _ = chunk_gated_delta_rule(*short, initial_state=initial, cu_seqlens=cu,
                                       use_qk_l2norm_in_kernel=True)
    branch.float().square().sum().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0

    torch.cuda.synchronize()
    expected = _native_logits(native, ids, mask, positions, features, logits_to_keep=24)
    actual = shared_prefix_forward(shared, plan=plan, input_ids=ids, position_ids=positions,
                                   attention_mask=mask, frozen_image_features=features,
                                   logits_to_keep=24, gdn_backend="fla", checkpoint_layers=True).logits
    labels = ids[:, -23:, None]
    expected_logps = expected[:, :-1].float().log_softmax(-1).gather(-1, labels).squeeze(-1)
    actual_logps = actual[:, :-1].float().log_softmax(-1).gather(-1, labels).squeeze(-1)
    weights = torch.linspace(0.3, 2, 8 * 23, device="cuda").reshape(8, 23)
    expected_loss = -(expected_logps * weights).mean()
    actual_loss = -(actual_logps * weights).mean()
    expected_loss.backward()
    actual_loss.backward()
    grad_squared, diff_squared, max_abs_grad = 0.0, 0.0, 0.0
    nonzero = 0
    for (name, reference), (_, candidate) in zip(native.named_parameters(), shared.named_parameters()):
        if reference.requires_grad:
            assert reference.grad is not None and candidate.grad is not None, name
            a, b = reference.grad.float(), candidate.grad.float()
            assert torch.isfinite(b).all(), name
            grad_squared += a.square().sum().item()
            diff_squared += (b - a).square().sum().item()
            max_abs_grad = max(max_abs_grad, (b - a).abs().max().item())
            nonzero += int(a.abs().sum() > 0)
    relative_l2 = (diff_squared / grad_squared) ** 0.5
    max_abs_logp = (actual_logps - expected_logps).abs().max().item()
    report = {"kind": "small_qwen35_cuda_fla_bf16_lora", "prefix": 3500, "answer": 23,
              "group": 8, "compact_tokens": plan.compact_length, "expanded_tokens": plan.total_tokens,
              "gdn_boundary": plan.gdn_boundary, "gdn_replay_tokens": plan.gdn_replay_tokens,
              "max_abs_logp": max_abs_logp, "gradient_relative_l2": relative_l2,
              "gradient_max_abs": max_abs_grad, "nonzero_gradient_parameters": nonzero,
              "native_loss": expected_loss.item(), "shared_loss": actual_loss.item(),
              "fla_initial_state_gradient_abs_sum": state.grad.abs().sum().item()}
    print(json.dumps(report, sort_keys=True))
    assert nonzero > 10
    # Explicit BF16 smoke thresholds, not the 27B NF4 equivalence acceptance gate.
    assert max_abs_logp < 0.02, report
    assert relative_l2 < 0.05, report
