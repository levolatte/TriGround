from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


CHECKPOINT_FORMAT_VERSION = 2


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def save_checkpoint(
    path, model, optimizer, scheduler, scaler, config, epoch: int, score: float,
    global_step: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = getattr(config, "stage", None)
    parameters = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or (stage == "joint" and name.startswith("fusion."))
    }
    trainable_parameter_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    backbone = getattr(model, "backbone", None)
    backbone_config = getattr(backbone, "config", None)
    torch.save({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": parameters,
        "trainable_only": True,
        "saved_parameter_names": sorted(parameters),
        "trainable_parameter_names": trainable_parameter_names,
        "runtime": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "backbone_commit": getattr(backbone_config, "_commit_hash", None),
            "attention_implementation": getattr(
                backbone_config, "_attn_implementation", None
            ),
        },
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
    bad_missing = [] if checkpoint.get("trainable_only") else missing
    if bad_missing or unexpected:
        raise RuntimeError(f"incompatible checkpoint: missing={bad_missing}, unexpected={unexpected}")
    return checkpoint


def load_initialization_checkpoints(paths, model) -> list[dict]:
    """Merge sparse initialization checkpoints without silent key overwrites."""
    checkpoints = []
    merged = {}
    sources = {}
    expected = model.state_dict()
    for raw_path in paths:
        path = Path(raw_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint has no model state: {path}")
        for name, value in state.items():
            if name not in expected:
                raise RuntimeError(f"unexpected checkpoint key {name!r} in {path}")
            if expected[name].shape != value.shape:
                raise RuntimeError(
                    f"shape mismatch for {name!r} in {path}: "
                    f"checkpoint={tuple(value.shape)}, model={tuple(expected[name].shape)}"
                )
            if name in merged and not torch.equal(merged[name], value):
                raise RuntimeError(
                    f"conflicting checkpoint key {name!r}: {sources[name]} and {path}"
                )
            merged[name] = value
            sources[name] = path
        checkpoints.append(checkpoint)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected merged checkpoint keys: {unexpected}")
    # Missing frozen backbone and not-yet-trained project modules are expected
    # when independent Stage 1A/1B sparse checkpoints initialize Stage 2.
    _ = missing
    return checkpoints


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
