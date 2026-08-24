from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.model import build_grounder


def main() -> None:
    root = Path.cwd()
    config = load_config(root / "configs/multimodal_safe_v2.yaml")
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        local_files_only=True,
    )
    model = build_grounder(config.model, processor).to("cuda")
    checkpoint = load_model_checkpoint(
        root / "runs/multimodal_safe_v2/best_phase_a.pt", model
    )
    dataset = GroundingDataset(
        (root / config.data.val_manifest).resolve(),
        "joint",
        config.data.depth_scale,
        config.data.depth_clip,
    )
    loader = DataLoader(
        Subset(dataset, range(8)),
        batch_size=1,
        collate_fn=NativeGroundingCollator(processor, "joint"),
    )
    drift_ratios = []
    cosines = []
    reliabilities = []

    def fusion_hook(_module, inputs, output):
        rgb = inputs[0].float()
        fused = output.float()
        drift = (fused - rgb).norm(dim=-1) / rgb.norm(dim=-1).clamp_min(1e-8)
        cosine = torch.nn.functional.cosine_similarity(fused, rgb, dim=-1)
        drift_ratios.extend(drift.detach().cpu().tolist())
        cosines.extend(cosine.detach().cpu().tolist())

    def reliability_hook(_module, _inputs, output):
        reliabilities.extend(output.sigmoid().detach().float().cpu().flatten().tolist())

    fusion_handle = model.fusion.register_forward_hook(fusion_hook)
    reliability_handle = model.fusion.reliability.register_forward_hook(reliability_hook)
    model.eval()
    tri_losses = []
    rgb_losses = []
    names = (
        "pixel_values",
        "ir_pixel_values",
        "depth_pixel_values",
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "labels",
    )
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to("cuda") if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            inputs = {name: batch[name] for name in names}
            tri_losses.append(float(model(**inputs)["loss"]))
            rgb_losses.append(float(model(**inputs, rgb_only=True)["loss"]))
    fusion_handle.remove()
    reliability_handle.remove()

    def summary(values):
        tensor = torch.tensor(values)
        return {
            "mean": float(tensor.mean()),
            "p50": float(tensor.quantile(0.5)),
            "p95": float(tensor.quantile(0.95)),
            "max": float(tensor.max()),
        }

    print(
        json.dumps(
            {
                "epoch": checkpoint["epoch"],
                "checkpoint_score": checkpoint["score"],
                "residual_scale": float(model.fusion.residual_scale.detach()),
                "token_drift_ratio": summary(drift_ratios),
                "fused_rgb_cosine": summary(cosines),
                "reliability": summary(reliabilities),
                "val_teacher_forced_ce_tri": sum(tri_losses) / len(tri_losses),
                "val_teacher_forced_ce_rgb": sum(rgb_losses) / len(rgb_losses),
                "ce_tri_minus_rgb": (
                    sum(tri_losses) / len(tri_losses)
                    - sum(rgb_losses) / len(rgb_losses)
                ),
                "gpu": torch.cuda.get_device_name(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
