"""CPU checks for complete inputs, causal alignment and original DAPO scaling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.benchmark_shared_prefix import (
    VectorStatistics,
    answer_logps,
    collate_shared_inputs,
    compare_trace_records,
    compare_expanded_inputs,
    compare_operator_rows,
    diagnose_layer0_operators,
    diagnostic_token_positions,
    normalized_group_loss,
    sample_trace_rows,
)


def fixture_inputs():
    batches, positions = [], []
    for answer in ([7, 8], [6, 7, 8]):
        ids = torch.tensor([[1, 2, 3, *answer]])
        positions.append(torch.arange(ids.shape[1]).reshape(1, 1, -1).expand(3, 1, -1).clone())
        batches.append({"model_inputs": {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * 3),
        }, "grpo_batch": SimpleNamespace(logits_to_keep=len(answer),
            completion_mask=torch.ones((1, len(answer)), dtype=torch.bool))})
    return batches, positions


def test_input_expansion_preserves_pixels_grid_positions_and_variable_answers():
    batches, positions = fixture_inputs()
    packed = collate_shared_inputs(batches, positions, pad_token_id=0)
    assert packed["input_ids"].tolist() == [[0, 1, 2, 3, 7, 8], [1, 2, 3, 6, 7, 8]]
    assert packed["attention_mask"].tolist() == [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    assert packed["position_ids"][:, 0, 1:].equal(positions[0][:, 0])
    assert packed["prefix_length"] == 3
    assert packed["completion_width"] == 3
    result = compare_expanded_inputs(batches, positions, packed)
    assert result["all_equal"]
    assert set(result["checked_elements"]) == {
        "input_ids", "attention_mask", "position_ids", "pixel_values", "image_grid_thw"}
    # An error in one pixel must fail, even when token ids and grids match.
    changed = {**packed, "pixel_values": packed["pixel_values"].clone()}
    changed["pixel_values"][0, 0] += 1
    with pytest.raises(RuntimeError, match="pixel_values"):
        compare_expanded_inputs(batches, positions, changed)


@pytest.mark.parametrize("field", ["pixel_values", "image_grid_thw", "input_ids"])
def test_different_images_or_prompt_are_not_shared(field):
    batches, positions = fixture_inputs()
    batches[1]["model_inputs"][field].flatten()[0] += 1
    with pytest.raises(ValueError, match="不同"):
        collate_shared_inputs(batches, positions, 0)


def test_positions_are_compared_and_four_axis_positions_are_preserved():
    batches, positions = fixture_inputs()
    four_axis = [torch.cat((value[:1], value), 0) for value in positions]
    packed = collate_shared_inputs(batches, four_axis, 0)
    assert packed["position_ids"].shape == (4, 2, 6)
    four_axis[1][2, 0, 0] += 1
    with pytest.raises(ValueError, match="三维位置不同"):
        collate_shared_inputs(batches, four_axis, 0)


def test_answer_first_token_uses_prompt_last_logit_and_retains_its_gradient():
    logits = torch.zeros((1, 6, 10), requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 6, 7, 8]])
    # P=3; logit at position P-1 predicts answer[0]=6.
    with torch.no_grad():
        logits[0, 2, 6] = 3
        logits[0, 3, 7] = 4
        logits[0, 4, 8] = 5
        logits[0, 5, 9] = 99  # Last row predicts the next, unscored token.
    values = answer_logps(logits, ids, 3)
    expected = torch.stack([torch.log_softmax(logits[0, index].float(), -1)[target]
                            for index, target in ((2, 6), (3, 7), (4, 8))])
    torch.testing.assert_close(values[0], expected)
    values.sum().backward()
    assert logits.grad[0, 2].count_nonzero() > 0
    assert logits.grad[0, 5].count_nonzero() == 0
    with pytest.raises(ValueError, match="P-1"):
        answer_logps(logits[:, -3:], ids, 3)


def test_group_forward_uses_original_group_token_denominator_and_gas32():
    torch.manual_seed(11)
    features = torch.randn(8, 4) * 0.1
    mask = torch.ones(8, 4)
    mask[1::2, -1] = 0
    advantages = torch.linspace(-0.6, 0.6, 8).unsqueeze(1)
    denominator = mask.sum()

    def compute(shared):
        parameter = torch.nn.Parameter(torch.tensor(0.3))
        losses = []
        for index in range(8):
            # A nonzero old/current difference exercises the differentiable ratio.
            ratio = (parameter * features[index]).exp()
            loss = -(ratio * advantages[index] * mask[index]).sum() / denominator
            losses.append(loss)
            if not shared:
                (loss / 32).backward()
        if shared:
            normalized_group_loss(losses, 32).backward()
        return normalized_group_loss([loss.detach() for loss in losses], 32), parameter.grad

    serial_loss, serial_grad = compute(False)
    shared_loss, shared_grad = compute(True)
    torch.testing.assert_close(shared_loss, serial_loss)
    torch.testing.assert_close(shared_grad, serial_grad)
    # Dividing by G=8 instead would silently amplify the original group contribution fourfold.
    assert normalized_group_loss([torch.tensor(1.0)] * 8, 32).item() == 0.25


def test_vector_statistics_streams_gradients_without_storing_candidate():
    stats = VectorStatistics()
    stats.add(torch.tensor([3.0, 4.0]), torch.tensor([3.0, 4.0]))
    stats.add(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    result = stats.report()
    assert result["relative_l2"] == 0
    assert result["cosine"] == pytest.approx(1)
    assert result["reference_l2"] == pytest.approx(26**0.5)
    assert result["elements"] == 4
    assert result["finite"]


def test_diagnostic_sampling_covers_gdn_conv_and_first_answer_boundaries():
    positions = diagnostic_token_positions(3501, 22)
    assert {0, 3455, 3456, 3497, 3500} <= set(positions)
    assert all(position in positions for position in range(3501, 3523))
    assert positions == sorted(set(positions))


def test_diagnostic_native_and_compact_layouts_select_identical_token_features():
    native = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
    expected = native[1, [1, 4]].clone()
    torch.testing.assert_close(sample_trace_rows(native, [1, 4], seq_dim=1, batch_index=1), expected)
    rotary = native.permute(2, 0, 1)
    torch.testing.assert_close(sample_trace_rows(rotary, [1, 4], seq_dim=2, batch_dim=1, batch_index=1), expected)
    heads = native.reshape(2, 6, 2, 2).transpose(1, 2)
    torch.testing.assert_close(sample_trace_rows(heads, [1, 4], seq_dim=2, batch_index=1), expected)


def test_diagnostic_trace_reports_first_observed_difference_and_unpaired_events():
    def record(values):
        return {"positions": [3, 4], "values": torch.tensor(values),
                "source_shape": [1, 6, 2], "source_dtype": "torch.float32"}

    reference = {"layers.0.input": record([[1.0], [2.0]]),
                 "layers.0.output": record([[2.0], [3.0]])}
    candidate = {"layers.0.input": record([[1.0], [2.0]]),
                 "layers.0.output": record([[2.0], [3.25]]),
                 "layers.0.extra": record([[1.0], [1.0]])}
    comparison = compare_trace_records(reference, candidate)
    assert comparison["first_observed_nonzero_event"] == "layers.0.output"
    assert comparison["events"]["layers.0.output"]["max_abs"] == 0.25
    assert comparison["candidate_only_events"] == ["layers.0.extra"]
    assert comparison["reference_only_events"] == []
    candidate["layers.0.output"]["positions"] = [4, 5]
    with pytest.raises(RuntimeError, match="未对齐"):
        compare_trace_records(reference, candidate)


def test_operator_statistics_locate_full_rows_and_each_branch():
    from tools.qwen35_shared_prefix import build_shared_prefix_plan
    batches, positions = fixture_inputs()
    packed = collate_shared_inputs(batches, positions, 0)
    plan = build_shared_prefix_plan(packed["input_ids"], packed["attention_mask"],
                                    packed["position_ids"], packed["prefix_length"])
    left = torch.ones(1, plan.compact_length, 2)
    right = left.clone()
    right[0, plan.suffix_rows[1].start + 1, 1] += 0.25
    result = compare_operator_rows(left, right, plan)
    assert result["all_rows"]["nonzero_count"] == 1
    assert result["all_rows"]["max_abs"] == 0.25
    assert result["all_rows"]["first_nonzero_token"] == {
        "compact_row": 6, "branch": 1, "unpadded_position": 4}
    assert result["regions"]["prefix_before_boundary"]["elements"] == 0
    assert result["regions"]["prefix_tail"]["nonzero_count"] == 0
    assert result["regions"]["suffix.1"]["nonzero_token_count"] == 1


@pytest.mark.parametrize("prefix", [5, 64, 67])
def test_operator_ladder_uses_identical_inputs_and_separates_metadata_controls(prefix):
    from tools.qwen35_shared_prefix import build_shared_prefix_plan, torch_reference_gated_delta_rule
    import torch.nn.functional as F

    torch.manual_seed(19)
    ids = torch.ones(2, prefix + 3, dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    positions = torch.arange(ids.shape[1]).reshape(1, 1, -1).expand(3, 2, -1)
    plan = build_shared_prefix_plan(ids, mask, positions, prefix)
    mixer = SimpleNamespace(
        in_proj_qkv=torch.nn.Linear(8, 8, bias=False), in_proj_z=torch.nn.Linear(8, 4, bias=False),
        in_proj_b=torch.nn.Linear(8, 2, bias=False), in_proj_a=torch.nn.Linear(8, 2, bias=False),
        conv1d=torch.nn.Conv1d(8, 8, 4, padding=3, groups=8, bias=False), activation="silu",
        key_dim=2, value_dim=4, head_k_dim=2, head_v_dim=2, num_k_heads=1, num_v_heads=2,
        A_log=torch.zeros(2), dt_bias=torch.zeros(2))
    normed = torch.randn(1, plan.compact_length, 8)
    calls = []
    def kernel(*args, **kwargs):
        calls.append((args[0].shape[1], kwargs.get("cu_seqlens") is not None,
                      kwargs["output_final_state"], kwargs["initial_state"] is not None))
        return torch_reference_gated_delta_rule(*args, **kwargs)
    def conv(value, weight, bias, activation, **kwargs):
        assert int(kwargs["cu_seq_lens_q"][-1]) == value.shape[-1]
        return F.silu(F.conv1d(value, weight.unsqueeze(1), bias, padding=3,
                               groups=weight.shape[0])[:, :, :value.shape[-1]])
    with torch.no_grad():
        result = diagnose_layer0_operators(mixer, normed, plan, kernel=kernel, native_conv=conv,
            native_sequence_kwargs={"cu_seq_lens_q": torch.tensor([0, prefix + 2], dtype=torch.int32),
                                    "max_length_q": prefix + 2})
    assert result["convolution_same_qkv"]["all_rows"]["max_abs"] < 1e-6
    for metric in result["gdn_same_inputs"].values():
        assert metric["all_rows"]["max_abs"] < 1e-6
        assert set(metric["regions"]) == {"prefix_before_boundary", "prefix_tail", "suffix.0", "suffix.1"}
    assert (prefix + 2, True, False, False) in calls
    assert (prefix + 2, False, False, False) in calls
    assert (prefix + 2, True, True, False) in calls
    if prefix >= 64:
        assert (64, True, True, False) in calls
        assert (64, False, True, False) in calls
    with pytest.raises(ValueError, match="no_grad"):
        diagnose_layer0_operators(mixer, normed, plan, kernel=kernel, native_conv=conv, native_sequence_kwargs={})


def test_operator_capture_calls_native_once_and_reconstructs_only_text_suffixes(monkeypatch):
    from contextlib import nullcontext
    from tools import benchmark_shared_prefix as benchmark
    pytest.importorskip("swift.trainers")
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    class Language(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 4)
            layer = torch.nn.Module()
            layer.block_type = "linear_attention"
            layer.input_layernorm = torch.nn.LayerNorm(4)
            layer.linear_attn = torch.nn.Identity()
            self.layers = torch.nn.ModuleList([layer])

        def forward(self, input_ids, **kwargs):
            return self.layers[0].input_layernorm(self.embed_tokens(input_ids))

    language = Language()
    model = SimpleNamespace(get_base_model=lambda: SimpleNamespace(model=SimpleNamespace(language_model=language)))
    calls = []
    def native(model, inputs, signal):
        calls.append(1)
        length = inputs["input_ids"].shape[1]
        return language(inputs["input_ids"], cu_seq_lens_q=torch.tensor([0, length], dtype=torch.int32),
                        max_length_q=length), None
    trainer = SimpleNamespace(model=model, args=SimpleNamespace(gradient_checkpointing_kwargs=None),
        accelerator=SimpleNamespace(autocast=nullcontext), template=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        _get_per_token_logps_and_entropies=native)
    batches, positions = fixture_inputs()
    for batch, position in zip(batches, positions):
        batch["model_inputs"]["position_ids"] = position
    def operators(mixer, normed, plan, **kwargs):
        assert not torch.is_grad_enabled()
        assert normed.shape == (1, 8, 4)
        assert kwargs["native_sequence_kwargs"]["cu_seq_lens_q"].tolist() == [0, 5]
        return {"capture_checked": True}
    monkeypatch.setattr(benchmark, "diagnose_layer0_operators", operators)
    report = benchmark.diagnose_operators(trainer, batches, gdn_backend="torch_reference")
    assert calls == [1]
    assert report["native_forward_calls"] == 1
    assert report["first_answer_suffix_reconstruction"]["max_abs"] == 0
    assert report["operators"] == {"capture_checked": True}
    assert not language._forward_pre_hooks
    assert not language.layers[0].input_layernorm._forward_hooks
