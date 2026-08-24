from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import evaluate
from mm_grounding.model import build_grounder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        config.model.backbone, min_pixels=config.data.min_pixels, max_pixels=config.data.max_pixels
    )
    model = build_grounder(config.model, processor).to(device)
    load_model_checkpoint(args.checkpoint, model)
    dataset = GroundingDataset(
        config.data.val_manifest, stage="joint", depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    loader = DataLoader(
        dataset, batch_size=config.train.batch_size, shuffle=False,
        num_workers=config.data.workers, collate_fn=NativeGroundingCollator(processor, "joint"),
    )
    report = {
        "joint": evaluate(model, loader, processor, device),
        "rgb_baseline": evaluate(model, loader, processor, device, rgb_only=True),
    }
    report["joint_gain_over_rgb"] = {
        key: report["joint"][key] - report["rgb_baseline"][key]
        for key in ("mean_iou", "acc_0.5", "acc_0.7")
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
