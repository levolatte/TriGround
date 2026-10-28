"""CPU checks for fixed-answer selection and target replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.benchmark_quant_grounding import (
    SELECTED_IDS,
    _masked_rows,
    _replay_targets,
    _vector_comparison,
    compare,
    install_preclip_snapshot,
    prepare,
)


def test_prepare_selects_real_three_image_groups(tmp_path):
    root = Path(__file__).resolve().parents[2]
    evidence = root / "code/docs/research/2026-10-26-quant-rl/evidence"
    manifest = prepare(
        root / "data/quant_grounding_20261026/schedule.jsonl",
        root / "data",
        evidence / "warmup50-diagnostic32-full-vocab-raw.jsonl",
        evidence / "warmup50-diagnostic32-full-vocab-scored.jsonl",
        tmp_path,
    )
    assert tuple(row["id"] for row in manifest["selected"]) == SELECTED_IDS
    assert sum(row["mixed_reward"] for row in manifest["selected"]) >= 2
    assert all(len(row["images"]) == 3 and len(row["responses"]) == 8
               for row in manifest["selected"])
    dataset = [json.loads(line) for line in (tmp_path / "swift_dataset.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(dataset) == 4
    assert all(all(Path(image).is_file() for image in row["images"]) for row in dataset)


def test_masked_target_replay_preserves_baseline_logps():
    def batch(old, ref):
        return {"grpo_batch": SimpleNamespace(
            completion_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
            truncated_mask=torch.tensor([False, False]),
            old_per_token_logps=torch.tensor(old, dtype=torch.float32),
            ref_per_token_logps=torch.tensor(ref, dtype=torch.float32),
            advantages=torch.tensor([[0.5, 0.5, 0.0], [-0.5, 0.0, 0.0]]),
        )}

    baseline = batch([[-1.0, -2.0, 0.0], [-3.0, 0.0, 0.0]],
                     [[-1.5, -2.5, 0.0], [-3.5, 0.0, 0.0]])
    target = _masked_rows([baseline], [1.0, 0.0])
    for index, (row, sample_id) in enumerate(zip(target, ("a", "b"))):
        row.update({"sample_id": sample_id, "token_ids": [1, 2], "response_index": index})
    candidate = batch([[-1.01, -2.01, 0.0], [-3.01, 0.0, 0.0]],
                      [[-1.49, -2.49, 0.0], [-3.49, 0.0, 0.0]])
    native = _masked_rows([candidate], [1.0, 0.0])
    for index, (row, sample_id) in enumerate(zip(native, ("a", "b"))):
        row.update({"sample_id": sample_id, "token_ids": [1, 2], "response_index": index})
    drift = _replay_targets([candidate], native, target)
    assert drift["old_max_abs"] == pytest.approx(0.01)
    assert candidate["grpo_batch"].old_per_token_logps[0, :2].tolist() == [-1.0, -2.0]
    assert candidate["grpo_batch"].ref_per_token_logps[1, 0].item() == -3.5
    native[1]["token_ids"] = [9]
    with pytest.raises(RuntimeError, match="固定目标"):
        _replay_targets([candidate], native, target)


def test_chunkwise_vector_comparison():
    reference = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 1.0])}
    metrics = _vector_comparison(reference, {key: value.clone() for key, value in reference.items()})
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == 0.0


def test_fp32_toy_dapo_loss_microbatch_equivalence():
    """GAS scaling needs a 2x DAPO denominator for mb2/GAS16."""
    torch.manual_seed(2026)
    old = torch.randn(32, 4, dtype=torch.float32) * 0.1 - 2
    ref = old - 0.08
    features = torch.randn(32, 4, dtype=torch.float32) * 0.03
    advantages = torch.linspace(-0.7, 0.8, 32).unsqueeze(1).expand(-1, 4)
    mask = torch.ones(32, 4, dtype=torch.float32)
    mask[::3, -1] = 0

    def one_update(microbatch, compensate):
        parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        loss_total = 0.0
        gas = 32 // microbatch
        for group_start in range(0, 32, 8):
            group_tokens = mask[group_start:group_start + 8].sum()
            for start in range(group_start, group_start + 8, microbatch):
                sl = slice(start, start + microbatch)
                current = old[sl] + parameter * features[sl]
                ratio = torch.exp(current - old[sl])
                clipped = ratio.clamp(0.8, 1.28)
                policy = -torch.minimum(ratio * advantages[sl], clipped * advantages[sl])
                safe_ratio = (ref[sl] - current).clamp(-20, 20)
                kl = (safe_ratio.exp() - safe_ratio - 1).clamp(-10, 10)
                denominator = group_tokens * (microbatch if compensate else 1)
                loss = ((policy + 0.01 * kl) * mask[sl]).sum() / denominator / gas
                loss_total += float(loss.detach())
                loss.backward()
        gradient = float(parameter.grad)
        optimizer.step()
        return loss_total, gradient, float(parameter.detach())

    loss1, grad1, weight1 = one_update(1, compensate=True)
    bad_loss, bad_grad, _ = one_update(2, compensate=False)
    assert bad_loss == pytest.approx(2 * loss1, rel=1e-5)
    assert bad_grad == pytest.approx(2 * grad1, rel=1e-5)
    loss2, grad2, weight2 = one_update(2, compensate=True)
    assert abs(loss1 - loss2) / abs(loss1) <= 1e-5
    assert abs(grad1 - grad2) / abs(grad1) <= 1e-5
    assert abs(weight1 - weight2) <= 1e-6


def test_compare_marks_unrun_variants_without_claiming_pass(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "result.json").write_text('{"variant":"baseline"}', encoding="utf-8")
    report = compare(tmp_path)
    assert report["candidates"] == {}
    assert "gc2" in report["not_run"]
    assert "mb2_conv" in report["not_run"]


def test_injected_assistant_content_uses_swift_reward_prefix():
    from tools.swift_quant_grounding_reward import QuantBBoxContrastReward, SHORT_RESPONSE_PREFIX

    text = '{"bbox_2d":[100,200,300,400]}'
    token_ids = [1, 2, 3]

    class Template:
        def _get_response_prefix(self, template_inputs):
            assert template_inputs.chat_template_kwargs == {}
            return SHORT_RESPONSE_PREFIX

        def decode_generate_ids(self, ids, **kwargs):
            assert ids == token_ids
            assert kwargs["first_token"] is False
            return text

    reward = QuantBBoxContrastReward()
    reward.bind_template(Template())
    values = reward(
        [SHORT_RESPONSE_PREFIX + text], [[0.1, 0.2, 0.3, 0.4]],
        response_token_ids=[token_ids], finish_reason=["stop"],
        chat_template_kwargs=[{}], grounding_negative_boxes=[[]],
    )
    assert values == pytest.approx([1.1])


def test_gradient_snapshot_precedes_actual_clip(tmp_path):
    model = torch.nn.Linear(1, 2, bias=False)
    model.weight.grad = torch.tensor([[3.0], [4.0]])

    class Trainer:
        def _clip_grad_norm(self, model_arg):
            assert model_arg is model
            return torch.nn.utils.clip_grad_norm_(model_arg.parameters(), max_norm=1.0)

    trainer = Trainer()
    original = trainer._clip_grad_norm
    state = {}
    destination = tmp_path / "gradients.pt"
    restore = install_preclip_snapshot(trainer, model, destination, state, torch)
    trainer._clip_grad_norm(model)
    saved = torch.load(destination, weights_only=True)
    assert saved["weight"].flatten().tolist() == [3.0, 4.0]
    assert torch.linalg.vector_norm(model.weight.grad).item() == pytest.approx(1.0, rel=1e-5)
    assert state["preclip_calls"] == 1
    assert state["gradient_snapshot_seconds"] >= 0
    restore()
    assert trainer._clip_grad_norm == original
