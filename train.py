from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import seed_everything, train
from mm_grounding.model import build_grounder


def _stratified_eval_subset(dataset, limit: int, seed: int):
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
    total = len(dataset)
    exact = {key: len(indices) * limit / total for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(groups, key=lambda key: (exact[key] - quotas[key], len(groups[key])), reverse=True)
    for key in priority[:remaining]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:quotas[key]])
    return Subset(dataset, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.train.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    processor = AutoProcessor.from_pretrained(
        config.model.backbone, min_pixels=config.data.min_pixels, max_pixels=config.data.max_pixels
    )
    model = build_grounder(config.model, processor)
    for checkpoint in config.train.initialization_checkpoints:
        load_model_checkpoint(checkpoint, model)
    if config.train.init_checkpoint:
        load_model_checkpoint(config.train.init_checkpoint, model)
    if config.train.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    common = dict(
        stage=config.stage,
        depth_scale=config.data.depth_scale,
        depth_clip=config.data.depth_clip,
    )
    collator = NativeGroundingCollator(processor, config.stage)
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(
        GroundingDataset(config.data.train_manifest, **common), batch_size=config.train.batch_size,
        shuffle=True, generator=generator, num_workers=config.data.workers, collate_fn=collator,
        pin_memory=torch.cuda.is_available(), persistent_workers=config.data.workers > 0,
    )
    val_dataset = GroundingDataset(config.data.val_manifest, **common)
    fast_val_dataset = _stratified_eval_subset(
        val_dataset, config.train.eval_subset_size, config.train.seed
    )
    val_loader = DataLoader(
        fast_val_dataset, batch_size=config.train.val_batch_size,
        shuffle=False, num_workers=config.data.workers, collate_fn=collator,
        pin_memory=torch.cuda.is_available(), persistent_workers=config.data.workers > 0,
    )
    full_val_loader = DataLoader(
        val_dataset, batch_size=config.train.val_batch_size,
        shuffle=False, num_workers=config.data.workers, collate_fn=collator,
        pin_memory=torch.cuda.is_available(), persistent_workers=config.data.workers > 0,
    )
    probe_val_loader = None
    if config.train.early_probe_step or config.train.early_probe_steps:
        probe_dataset = _stratified_eval_subset(
            val_dataset, config.train.early_probe_subset_size, config.train.seed + 1
        )
        probe_val_loader = DataLoader(
            probe_dataset,
            batch_size=config.train.val_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
    train(
        model,
        train_loader,
        val_loader,
        full_val_loader,
        probe_val_loader,
        processor,
        config,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
