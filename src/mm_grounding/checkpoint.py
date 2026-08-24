from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, config, epoch: int, score: float,
    global_step: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
        "trainable_only": True,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "config": asdict(config),
        "epoch": epoch,
        "score": score,
    }, output)


def load_model_checkpoint(path, model) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    bad_missing = [name for name in missing if not name.startswith("backbone.")]
    if bad_missing or unexpected:
        raise RuntimeError(f"incompatible checkpoint: missing={bad_missing}, unexpected={unexpected}")
    return checkpoint


def load_training_checkpoint(path, model, optimizer, scheduler, scaler) -> dict:
    checkpoint = load_model_checkpoint(path, model)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restored = {"optimizer": True, "scheduler": False, "scaler": False}
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        restored["scheduler"] = True
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
        restored["scaler"] = True
    checkpoint["restored"] = restored
    return checkpoint
