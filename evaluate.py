from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import evaluate
from mm_grounding.model import build_grounder


def _stratified_subset(dataset, limit: int, seed: int):
    """Deterministic proportional sample across class and target scale."""
    if limit >= len(dataset):
        return dataset
    groups = {}
    for index, record in enumerate(dataset.records):
        key = (record.get("class_name", ""), record.get("scale_bin", ""))
        groups.setdefault(key, []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[i] for i in order]
    exact = {key: len(indices) * limit / len(dataset) for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key])),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:quotas[key]])
    return Subset(dataset, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help=(
            "Checkpoint to load. Repeat the option to compose independently trained "
            "modality checkpoints; omit it for the native frozen backbone baseline."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--manifest",
        help="Optional validation manifest override for paired/gated evaluation",
    )
    parser.add_argument("--output", help="Optional path for the JSON evaluation report")
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="Evaluate only the RGB baseline (useful for the native Qwen baseline).",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Evaluate a deterministic stratified subset of at most this many records.",
    )
    parser.add_argument("--subset-seed", type=int, default=2026)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(
        config.model.backbone, min_pixels=config.data.min_pixels, max_pixels=config.data.max_pixels
    )
    model = build_grounder(config.model, processor).to(device)
    loaded_checkpoints = []
    for checkpoint_path in args.checkpoint:
        load_model_checkpoint(checkpoint_path, model)
        loaded_checkpoints.append(checkpoint_path)
    manifest = args.manifest or config.data.val_manifest
    dataset = GroundingDataset(
        manifest, stage=config.stage, depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    eval_dataset = (
        _stratified_subset(dataset, args.subset_size, args.subset_seed)
        if args.subset_size is not None
        else dataset
    )
    loader = DataLoader(
        eval_dataset, batch_size=config.train.val_batch_size, shuffle=False,
        num_workers=config.data.workers,
        collate_fn=NativeGroundingCollator(processor, config.stage),
    )
    modes = {
        "ir": {"rgb_ir": {"ir"}},
        "depth": {"rgb_depth": {"depth"}},
        "joint": {
            "rgb_ir_depth": {"ir", "depth"},
            "rgb_ir": {"ir"},
            "rgb_depth": {"depth"},
        },
    }[config.stage]
    if config.stage == "joint" and config.model.fusion_type != "parallel_backbone":
        # Legacy/post-embed/deep fusion variants were defined for a complete
        # RGB+IR+Depth tuple and cannot perform auxiliary-modality ablations.
        modes = {"rgb_ir_depth": {"ir", "depth"}}
    report = {}
    if not args.rgb_only:
        report.update({
            name: evaluate(
                model, loader, processor, device, config.train.max_new_tokens,
                modalities=modalities,
            )
            for name, modalities in modes.items()
        })
    report["rgb_baseline"] = evaluate(
        model, loader, processor, device, config.train.max_new_tokens, rgb_only=True
    )
    if not args.rgb_only:
        primary = next(iter(modes))
        report[f"{primary}_gain_over_rgb"] = {
            key: report[primary][key] - report["rgb_baseline"][key]
            for key in ("mean_iou", "acc_0.5", "acc_0.7")
        }
    report["diagnostic_metadata"] = {
        "config": args.config,
        "manifest": manifest,
        "checkpoints": loaded_checkpoints,
        "rgb_only": args.rgb_only,
        "samples": len(eval_dataset),
        "source_samples": len(dataset),
        "subset_seed": args.subset_seed if args.subset_size is not None else None,
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
