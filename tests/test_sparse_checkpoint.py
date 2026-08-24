from dataclasses import dataclass

import torch
from torch import nn

from mm_grounding.checkpoint import load_model_checkpoint, save_checkpoint


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.fusion = nn.Linear(3, 1)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False


@dataclass
class TinyConfig:
    name: str = "test"
    stage: str | None = None


def test_checkpoint_omits_frozen_backbone_and_restores_trainable_weights(tmp_path):
    source = TinyModel()
    source.fusion.bias.data.fill_(2.5)
    optimizer = torch.optim.AdamW(source.fusion.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "sparse.pt"
    save_checkpoint(
        path,
        source,
        optimizer,
        scheduler,
        scaler,
        TinyConfig(),
        epoch=1,
        score=0.5,
        global_step=10,
    )

    payload = torch.load(path, weights_only=False)
    assert payload["trainable_only"] is True
    assert set(payload["model"]) == {"fusion.weight", "fusion.bias"}

    target = TinyModel()
    load_model_checkpoint(path, target)
    assert torch.allclose(target.fusion.bias, torch.full((1,), 2.5))


def test_joint_checkpoint_keeps_initialized_fusion_even_when_frozen(tmp_path):
    source = TinyModel()
    source.fusion.bias.data.fill_(3.5)
    for parameter in source.fusion.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(source.fusion.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    path = tmp_path / "joint.pt"
    save_checkpoint(
        path,
        source,
        optimizer,
        scheduler,
        scaler,
        TinyConfig(stage="joint"),
        epoch=1,
        score=0.5,
        global_step=10,
    )

    payload = torch.load(path, weights_only=False)
    assert set(payload["model"]) == {"fusion.weight", "fusion.bias"}
    target = TinyModel()
    load_model_checkpoint(path, target)
    assert torch.allclose(target.fusion.bias, torch.full((1,), 3.5))
