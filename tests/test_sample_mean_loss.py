"""CPU checks for the ms-swift sample-mean SFT loss."""

import runpy
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


PLUGIN = Path(__file__).resolve().parents[1] / "tools" / "swift_sample_mean_loss.py"
HELPER = runpy.run_path(str(PLUGIN), run_name="__main__")["sample_mean_from_token_losses"]


def _token_losses(logits, labels):
    if hasattr(logits, "logits"):
        logits = logits.logits
    targets = torch.roll(labels, shifts=-1, dims=-1)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction="none")


def _examples(count=8):
    torch.manual_seed(2026)
    features = torch.randn(count, 16, 5, dtype=torch.float64)
    labels = torch.full((count, 16), -100, dtype=torch.long)
    for i in range(count):
        begin = 1 + i % 4
        answer_length = 2 + i
        labels[i, begin:begin + answer_length - 1] = torch.arange(3, 3 + answer_length - 1)
        labels[i, begin + answer_length - 1] = 2  # EOS is supervised.
    return features, labels


def _reference_loss(logits, labels):
    losses = []
    for row_logits, row_labels in zip(logits, labels):
        losses.append(F.cross_entropy(row_logits[:-1].float(), row_labels[1:],
                                      ignore_index=-100, reduction="sum") / row_labels[1:].ne(-100).sum())
    return torch.stack(losses).mean()


def _plugin(monkeypatch, gradient_accumulation_steps, training):
    swift = types.ModuleType("swift")
    swift_loss = types.ModuleType("swift.loss")
    swift_trainers = types.ModuleType("swift.trainers")

    class BaseLoss:
        def __init__(self, args, trainer):
            self.trainer = trainer

    swift_loss.BaseLoss = BaseLoss
    swift_loss.loss_map = {}
    swift_trainers.per_token_loss_func = _token_losses
    monkeypatch.setitem(sys.modules, "swift", swift)
    monkeypatch.setitem(sys.modules, "swift.loss", swift_loss)
    monkeypatch.setitem(sys.modules, "swift.trainers", swift_trainers)
    runpy.run_path(str(PLUGIN), run_name="swift_sample_mean_loss")
    trainer = types.SimpleNamespace(model=types.SimpleNamespace(training=training),
                                    current_gradient_accumulation_steps=gradient_accumulation_steps)
    return swift_loss.loss_map["grounding_sample_mean"](None, trainer)


def test_eight_variable_answers_match_per_example_reference_and_gradients(monkeypatch):
    features, labels = _examples()
    weight = torch.randn(5, 19, dtype=torch.float64, requires_grad=True)
    reference = _reference_loss(features @ weight, labels)
    expected_gradient, = torch.autograd.grad(reference, weight)

    plugin = _plugin(monkeypatch, gradient_accumulation_steps=8, training=True)
    for i in range(8):
        outputs = types.SimpleNamespace(logits=(features[i:i + 1] @ weight))
        plugin(outputs, labels[i:i + 1]).backward()
    torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(HELPER(_token_losses(features @ weight, labels), labels), reference)


def test_short_final_group_and_eval_scaling(monkeypatch):
    features, labels = _examples(3)
    weight = torch.randn(5, 19, dtype=torch.float64, requires_grad=True)
    expected = _reference_loss(features @ weight, labels)
    expected_gradient, = torch.autograd.grad(expected, weight)

    train_loss = _plugin(monkeypatch, gradient_accumulation_steps=3, training=True)
    for i in range(3):
        train_loss(types.SimpleNamespace(logits=features[i:i + 1] @ weight), labels[i:i + 1]).backward()
    torch.testing.assert_close(weight.grad, expected_gradient, rtol=1e-6, atol=1e-7)

    eval_loss = _plugin(monkeypatch, gradient_accumulation_steps=8, training=False)
    torch.testing.assert_close(eval_loss(types.SimpleNamespace(logits=features @ weight), labels), expected)


def test_logits_to_keep_compact_shape_keeps_eos_and_ignores_prefix(monkeypatch):
    features, labels = _examples(1)
    logits = features @ torch.randn(5, 19, dtype=torch.float64)
    full = _reference_loss(logits, labels)
    supervised = labels[0].ne(-100)
    logits_to_keep = F.pad(supervised[1:], (0, 1), value=True)
    compact_logits = logits[:, logits_to_keep]
    compact_labels = F.pad(labels[:, supervised], (1, 0), value=-100)
    assert compact_logits.shape[:2] == compact_labels.shape
    assert compact_labels[0, -1] == 2
    plugin = _plugin(monkeypatch, gradient_accumulation_steps=8, training=False)
    torch.testing.assert_close(plugin(types.SimpleNamespace(logits=compact_logits), compact_labels), full)


def test_empty_supervision_fails_fast():
    labels = torch.full((1, 4), -100, dtype=torch.long)
    losses = torch.zeros(4)
    try:
        HELPER(losses, labels)
    except ValueError as exc:
        assert "supervised answer token" in str(exc)
    else:
        raise AssertionError("empty answer should fail")


def test_nonbinary_loss_scale_is_rejected(monkeypatch):
    plugin = _plugin(monkeypatch, gradient_accumulation_steps=1, training=False)
    logits = torch.randn(1, 3, 7)
    labels = torch.tensor([[-100, 3, 2]])
    try:
        plugin(types.SimpleNamespace(logits=logits), labels, loss_scale=torch.tensor([[0., 0.5, 1.]]))
    except ValueError as exc:
        assert "binary loss mask" in str(exc)
    else:
        raise AssertionError("nonbinary token weights should fail")
