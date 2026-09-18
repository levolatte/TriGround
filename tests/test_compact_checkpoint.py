from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

from mm_grounding.checkpoint import (
    load_initialization_checkpoints,
    load_model_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
)


METRICS = {
    "acc_0.5": 0.5,
    "mean_iou": 0.6,
    "acc_0.7": 0.4,
    "parse_rate": 1.0,
}
SELECTION_ORDER = ("acc_0.5", "mean_iou", "acc_0.7", "parse_rate")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.fusion = nn.Linear(3, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False


@dataclass
class CompactTrainConfig:
    compact_checkpoints: bool = True


@dataclass
class CompactConfig:
    stage: str | None = None
    train: CompactTrainConfig = field(default_factory=CompactTrainConfig)


@dataclass
class LegacyConfig:
    stage: str | None = None


def _optimizers(model):
    optimizer = torch.optim.AdamW(model.fusion.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return optimizer, scheduler, scaler


def _save(path, model, config, *, compact):
    optimizer, scheduler, scaler = _optimizers(model)
    model.fusion.weight.sum().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch=2,
        metrics=METRICS,
        global_step=12,
        selection_order=SELECTION_ORDER,
        compact=compact,
    )


def test_compact_checkpoint_is_loadable_for_inference_and_initialization(tmp_path):
    source = TinyModel()
    source.fusion.bias.data.fill_(2.5)
    path = tmp_path / "best.pt"
    _save(path, source, CompactConfig(), compact=True)

    payload = torch.load(path, weights_only=False)
    assert payload["compact"] is True
    assert payload["config"]["train"]["compact_checkpoints"] is True
    assert payload["epoch"] == 2
    assert payload["metrics"] == METRICS
    assert payload["selection_order"] == list(SELECTION_ORDER)
    assert all(name not in payload for name in ("optimizer", "scheduler", "scaler", "global_step"))

    target = TinyModel()
    loaded = load_model_checkpoint(path, target)
    assert loaded["compact"] is True
    assert torch.allclose(target.fusion.bias, torch.full((1,), 2.5))
    initialized = TinyModel()
    load_initialization_checkpoints([path], initialized)
    assert torch.equal(initialized.fusion.weight, source.fusion.weight)


def test_compact_checkpoint_cannot_be_used_to_resume(tmp_path):
    model = TinyModel()
    path = tmp_path / "best.pt"
    _save(path, model, CompactConfig(), compact=True)
    optimizer, scheduler, scaler = _optimizers(model)

    with pytest.raises(RuntimeError, match="cannot resume from compact checkpoint"):
        load_training_checkpoint(path, model, optimizer, scheduler, scaler)


def test_compact_run_keeps_last_checkpoint_full_and_resumable(tmp_path):
    source = TinyModel()
    path = tmp_path / "last.pt"
    _save(path, source, CompactConfig(), compact=False)

    payload = torch.load(path, weights_only=False)
    assert payload.get("compact", False) is False
    assert {"optimizer", "scheduler", "scaler", "global_step"} <= set(payload)

    target = TinyModel()
    optimizer, scheduler, scaler = _optimizers(target)
    restored = load_training_checkpoint(path, target, optimizer, scheduler, scaler)
    assert restored["restored"] == {"optimizer": True, "scheduler": True, "scaler": True}
    state = optimizer.state_dict()["state"]
    assert state
    for key, values in payload["optimizer"]["state"].items():
        assert torch.equal(state[key]["exp_avg"], values["exp_avg"])
        assert torch.equal(state[key]["exp_avg_sq"], values["exp_avg_sq"])


def test_default_save_remains_full_without_train_config(tmp_path):
    source = TinyModel()
    path = tmp_path / "legacy.pt"
    _save(path, source, LegacyConfig(), compact=False)
    payload = torch.load(path, weights_only=False)
    assert payload.get("compact", False) is False
    assert {"optimizer", "scheduler", "scaler", "global_step"} <= set(payload)
