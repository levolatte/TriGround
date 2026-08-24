from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.model import _raw_patch_counts, build_grounder


def _nonzero_gradients(module: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in module.named_parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
    ]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_safe_v2.yaml")
    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        local_files_only=True,
    )
    model = build_grounder(config.model, processor).to("cuda")
    model.set_phase_a_trainable()
    manifest = (root / config.data.train_manifest).resolve()
    loader = DataLoader(
        GroundingDataset(manifest, "joint", config.data.depth_scale, config.data.depth_clip),
        batch_size=1,
        collate_fn=NativeGroundingCollator(processor, "joint"),
    )
    batch = next(iter(loader))
    batch = {
        key: value.to("cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    names = (
        "pixel_values",
        "ir_pixel_values",
        "depth_pixel_values",
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "labels",
    )
    inputs = {name: batch[name] for name in names}

    wrapper = model._post_embed_wrapper()
    if wrapper is None:
        raise RuntimeError("safe post-embed wrapper was not installed")
    pixel_values = batch["pixel_values"].to(next(model.backbone.parameters()).dtype)
    counts = _raw_patch_counts(batch["image_grid_thw"], pixel_values.shape[0])
    model.eval()
    with torch.no_grad():
        rgb_tokens = wrapper.base(pixel_values)
        wrapper.set_auxiliary(batch["ir_pixel_values"], batch["depth_pixel_values"], counts)
        fused_at_init = wrapper(pixel_values)
        wrapper.clear_auxiliary()
    if not torch.equal(rgb_tokens, fused_at_init):
        raise RuntimeError("new fusion does not preserve the pretrained RGB path at initialization")

    model.train()
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=config.train.fusion_lr_a)
    losses = []
    gradients = []
    scales = [float(model.fusion.residual_scale.detach())]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**inputs)["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append(_nonzero_gradients(model.fusion))
        optimizer.step()
        scales.append(float(model.fusion.residual_scale.detach()))

    branch_prefixes = ("depth_encoder.", "thermal_encoder.", "depth_projection.", "thermal_projection.")
    if not any(name.startswith(branch_prefixes) for name in gradients[1]):
        raise RuntimeError("auxiliary encoders received no nonzero gradient after ReZero opened")
    if any(parameter.requires_grad for parameter in model.vision_lora_parameters()):
        raise RuntimeError("Vision LoRA must remain frozen during Phase A")

    print(
        json.dumps(
            {
                "rgb_identity_exact": True,
                "losses": losses,
                "residual_scales": scales,
                "first_step_nonzero_gradients": gradients[0],
                "second_step_branch_gradient_count": sum(
                    name.startswith(branch_prefixes) for name in gradients[1]
                ),
                "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "gpu": torch.cuda.get_device_name(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
