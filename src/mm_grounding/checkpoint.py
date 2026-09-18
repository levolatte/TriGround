from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch


CHECKPOINT_FORMAT_VERSION = 2
MIN_FREE_CHECKPOINT_BYTES = 3 * 1024**3


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch: int,
    metrics: dict[str, float],
    global_step: int,
    selection_order: tuple[str, ...],
    compact: bool = False,
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
    payload = {
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
        "config": asdict(config),
        "epoch": epoch,
        "score": float(metrics["acc_0.5"]),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "selection_order": list(selection_order),
    }
    if compact:
        payload["compact"] = True
    if not compact:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": global_step,
            }
        )
    preserve_free_space = compact or bool(
        getattr(getattr(config, "train", None), "compact_checkpoints", False)
    )
    if preserve_free_space:
        _save_with_free_space_guard(payload, output)
    else:
        torch.save(payload, output)


def _save_with_free_space_guard(payload: dict, output: Path) -> None:
    """Write a checkpoint while retaining the configured free space.

    The temporary file is created next to the destination so the free-space
    check accounts for the complete serialized payload.  The old checkpoint
    remains in place until the new file has passed the check.
    """
    estimated_bytes = _payload_tensor_bytes(payload) + 64 * 1024**2
    free_before = shutil.disk_usage(output.parent).free
    if free_before - estimated_bytes < MIN_FREE_CHECKPOINT_BYTES:
        raise RuntimeError(
            "insufficient free space for checkpoint estimate: "
            f"{free_before / 1024**3:.2f} GiB available, "
            f"{estimated_bytes / 1024**2:.1f} MiB estimated, "
            f"{MIN_FREE_CHECKPOINT_BYTES / 1024**3:.2f} GiB required"
        )
    temporary = tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        torch.save(payload, temporary_path)
        free_bytes = shutil.disk_usage(output.parent).free
        if free_bytes < MIN_FREE_CHECKPOINT_BYTES:
            raise RuntimeError(
                "insufficient free space after checkpoint save: "
                f"{free_bytes / 1024**3:.2f} GiB available, "
                f"{MIN_FREE_CHECKPOINT_BYTES / 1024**3:.2f} GiB required"
            )
        os.replace(temporary_path, output)
        print(
            json.dumps(
                {
                    "event": "checkpoint_saved",
                    "path": str(output),
                    "size_bytes": output.stat().st_size,
                    "free_gib": free_bytes / 1024**3,
                    "compact": bool(payload.get("compact", False)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _payload_tensor_bytes(value) -> int:
    """Conservatively estimate serialized tensor storage for a checkpoint."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_payload_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_tensor_bytes(item) for item in value)
    return 0


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
    if "optimizer" not in checkpoint:
        if checkpoint.get("compact"):
            raise RuntimeError(
                f"cannot resume from compact checkpoint {path}; use the matching last checkpoint"
            )
        raise RuntimeError(f"checkpoint has no optimizer state and cannot be resumed: {path}")
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
