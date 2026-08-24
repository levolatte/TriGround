from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.model import build_grounder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--max-samples", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() else root / config_path)
    report = {
        "architecture": "RGB + IR + Depth -> RDT fusion -> Qwen native bbox_2d",
        "fusion_type": config.model.fusion_type,
    }
    manifests = {}
    for split in ("train", "val"):
        manifest = Path(getattr(config.data, f"{split}_manifest"))
        manifest = manifest if manifest.is_absolute() else (root / manifest).resolve()
        dataset = GroundingDataset(manifest, "joint", config.data.depth_scale, config.data.depth_clip)
        for index in range(min(len(dataset), args.max_samples)):
            dataset[index]
        report[split] = {"samples": len(dataset), "manifest": str(manifest)}
        manifests[split] = manifest
    if not args.offline:
        processor = AutoProcessor.from_pretrained(
            config.model.backbone, min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels, local_files_only=True,
        )
        model = build_grounder(config.model, processor).to(args.device)
        if config.train.init_checkpoint:
            checkpoint = Path(config.train.init_checkpoint)
            checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
            load_model_checkpoint(checkpoint, model)
            report["init_checkpoint"] = str(checkpoint)
        if config.train.gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()
            model.backbone.enable_input_require_grads()
        model.set_phase_a_trainable()
        if config.train.phase_a_epochs == 0:
            model.enable_vision_lora()
        loader = DataLoader(
            GroundingDataset(manifests["train"], "joint", config.data.depth_scale, config.data.depth_clip),
            batch_size=config.train.batch_size, collate_fn=NativeGroundingCollator(processor, "joint"),
        )
        batch = next(iter(loader))
        batch = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        names = (
            "pixel_values", "ir_pixel_values", "depth_pixel_values", "input_ids",
            "attention_mask", "image_grid_thw", "labels", "bbox", "coordinate_mask",
        )
        loss = model(**{name: batch[name] for name in names})["loss"]
        if args.backward:
            loss.backward()
            if not any(parameter.grad is not None for parameter in model.fusion.parameters()):
                raise RuntimeError("Qwen bbox loss did not reach multimodal fusion")
            if config.train.phase_a_epochs == 0 and not any(
                parameter.grad is not None for parameter in model.vision_lora_parameters()
            ):
                raise RuntimeError("Qwen bbox loss did not reach Vision LoRA")
        report["real_model"] = {
            "loss": float(loss.detach()), "gpu": torch.cuda.get_device_name(),
            "batch_size": config.train.batch_size,
            "joint_from_start": config.train.phase_a_epochs == 0,
            "deep_prompt_layers": len(getattr(model.fusion, "prompt_blocks", [])),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
