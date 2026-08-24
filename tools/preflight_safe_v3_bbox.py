from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _gradient_diagnostics, _optimizer
from mm_grounding.model import build_grounder


def _move(batch):
    return {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _inputs(batch):
    names = (
        "pixel_values",
        "ir_pixel_values",
        "depth_pixel_values",
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "labels",
        "bbox",
        "coordinate_mask",
    )
    return {name: batch[name] for name in names}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_safe_v3_bbox.yaml")
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        local_files_only=True,
    )
    model = build_grounder(config.model, processor).to("cuda")
    if config.train.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    model.set_phase_a_trainable()
    optimizer = _optimizer(model, config, "a")
    dataset = GroundingDataset(
        (root / config.data.train_manifest).resolve(),
        "joint",
        config.data.depth_scale,
        config.data.depth_clip,
    )
    batch = _move(
        next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=1,
                    collate_fn=NativeGroundingCollator(processor, "joint"),
                )
            )
        )
    )

    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        head_only = model(**_inputs(batch), geometry_gradient_scale=0.0)
    optimizer.zero_grad(set_to_none=True)
    head_only["bbox_l1_loss"].backward()
    fusion_geometry_gradients = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.fusion.parameters()
    )
    head_gradients = sum(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.bbox_head.parameters()
    )
    if fusion_geometry_gradients != 0 or head_gradients == 0:
        raise RuntimeError("head-only warmup does not isolate shared geometry gradients")

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        shared = model(**_inputs(batch), geometry_gradient_scale=1.0)
    diagnostics = _gradient_diagnostics(model, shared)
    shared["loss"].backward()
    optimizer.step()

    model.eval()
    generation_inputs = {
        "pixel_values": batch["pixel_values"],
        "ir_pixel_values": batch["ir_pixel_values"],
        "depth_pixel_values": batch["depth_pixel_values"],
        "input_ids": batch["generation_input_ids"],
        "attention_mask": batch["generation_attention_mask"],
        "image_grid_thw": batch["image_grid_thw"],
    }
    auxiliary = model.predict_auxiliary_bbox(**generation_inputs)
    if not torch.isfinite(auxiliary).all():
        raise RuntimeError("auxiliary prediction is non-finite")

    print(
        json.dumps(
            {
                "loss": float(shared["loss"].detach()),
                "token_ce": float(shared["token_loss"].detach()),
                "coordinate_token_ce": float(shared["coordinate_token_loss"].detach()),
                "aux_l1": float(shared["bbox_l1_loss"].detach()),
                "aux_giou_loss": float(shared["bbox_giou_loss"].detach()),
                "coordinate_token_count": int(batch["coordinate_mask"].sum()),
                "head_only_fusion_geometry_gradients": fusion_geometry_gradients,
                "head_only_head_gradients": head_gradients,
                "auxiliary_bbox": auxiliary.detach().cpu().tolist(),
                **diagnostics,
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "gpu": torch.cuda.get_device_name(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
