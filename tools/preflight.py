from __future__ import annotations

import argparse
from dataclasses import replace
import json
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_initialization_checkpoints, load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _optimizer
from mm_grounding.model import build_grounder


def _to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _model_inputs(batch, stage: str):
    names = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "query_input_ids",
        "query_attention_mask",
        "labels",
        "bbox",
        "coordinate_mask",
    ]
    if stage in {"ir", "joint"}:
        names.append("ir_pixel_values")
    if stage in {"depth", "joint"}:
        names.append("depth_pixel_values")
    return {name: batch[name] for name in names}


def _largest_visual_batch(dataset, collator, scan_samples: int):
    best = None
    best_tokens = -1
    for index in range(min(len(dataset), scan_samples)):
        batch = collator([dataset[index]])
        tokens = int(batch["pixel_values"].shape[0])
        if tokens > best_tokens:
            best = batch
            best_tokens = tokens
    if best is None:
        raise RuntimeError("training dataset is empty")
    return best, best_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=0,
        help="run complete AdamW steps; use 2 to verify delayed zero-init gradients",
    )
    parser.add_argument("--expected-val-samples", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() else root / config_path)
    if args.optimizer_steps < 0 or args.max_samples < 1:
        raise ValueError("optimizer-steps must be non-negative and max-samples must be positive")
    report = {
        "architecture": f"query-aware RGB + {config.stage} residual fusion -> Qwen bbox_2d",
        "fusion_type": config.model.fusion_type,
    }
    manifests = {}
    for split in ("train", "val"):
        manifest = Path(getattr(config.data, f"{split}_manifest"))
        manifest = manifest if manifest.is_absolute() else (root / manifest).resolve()
        dataset = GroundingDataset(
            manifest, config.stage, config.data.depth_scale, config.data.depth_clip
        )
        for index in range(min(len(dataset), args.max_samples)):
            dataset[index]
        report[split] = {"samples": len(dataset), "manifest": str(manifest)}
        if (
            split == "val"
            and args.expected_val_samples is not None
            and len(dataset) != args.expected_val_samples
        ):
            raise ValueError(
                f"expected {args.expected_val_samples} validation samples, "
                f"found {len(dataset)}"
            )
        manifests[split] = manifest
    if not args.offline:
        # Gradient reachability needs both modalities on every diagnostic step.
        # This process never saves weights; train.py still uses the source dropout.
        report["training_modality_dropout"] = config.model.modality_dropout
        config = replace(config, model=replace(config.model, modality_dropout=0.0))
        report["preflight_modality_dropout"] = 0.0
        processor = AutoProcessor.from_pretrained(
            config.model.backbone,
            revision=config.model.backbone_revision,
            min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels,
            local_files_only=True,
        )
        model = build_grounder(config.model, processor).to(args.device)
        initialization_checkpoints = []
        for value in config.train.initialization_checkpoints:
            checkpoint = Path(value)
            initialization_checkpoints.append(
                checkpoint if checkpoint.is_absolute() else root / checkpoint
            )
        if initialization_checkpoints:
            load_initialization_checkpoints(initialization_checkpoints, model)
        if config.train.warm_start_joint_fusion_from_legacy:
            model.fusion.warm_start_joint_from_legacy()
        if config.train.init_checkpoint:
            checkpoint = Path(config.train.init_checkpoint)
            checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
            load_model_checkpoint(checkpoint, model)
            report["init_checkpoint"] = str(checkpoint)
        if config.train.gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()
            model.backbone.enable_input_require_grads()
        model.set_phase_a_trainable(
            config.stage,
            freeze_parallel_adapters=config.model.freeze_parallel_adapters,
            parallel_adapter_train_last_n=config.model.parallel_adapter_train_last_n,
        )
        if config.train.phase_a_epochs == 0:
            model.enable_vision_lora()
        # from_pretrained leaves the backbone in eval mode. Checkpointing is
        # active only in training mode, which must match the real train loop.
        model.train()
        model.backbone.config.use_cache = False
        train_dataset = GroundingDataset(
            manifests["train"],
            config.stage,
            config.data.depth_scale,
            config.data.depth_clip,
        )
        collator = NativeGroundingCollator(processor, config.stage)
        batch, visual_tokens = _largest_visual_batch(
            train_dataset, collator, args.max_samples
        )
        batch = _to_device(batch, args.device)
        inputs = _model_inputs(batch, config.stage)
        needs_two_step_check = (
            (args.backward or args.optimizer_steps >= 2)
            and config.model.fusion_type == "parallel_backbone"
            and config.model.parallel_joint_fusion
        )
        run_steps = args.optimizer_steps or (2 if needs_two_step_check else 1)
        optimizer_phase = "b" if config.train.phase_a_epochs == 0 else "a"
        optimizer = (
            _optimizer(model, config, optimizer_phase)
            if args.optimizer_steps or needs_two_step_check
            else None
        )
        gradient_reports = []
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        started = time.perf_counter()
        loss = None
        for step in range(run_steps):
            loss = model(**inputs)["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite preflight loss at step {step + 1}")
            if args.backward or args.optimizer_steps:
                loss.backward()
                fusion_gradients = [
                    parameter.grad
                    for parameter in model.fusion.parameters()
                    if parameter.requires_grad
                ]
                present = sum(gradient is not None for gradient in fusion_gradients)
                nonzero = sum(
                    gradient is not None and bool(torch.count_nonzero(gradient))
                    for gradient in fusion_gradients
                )
                nonfinite = sum(
                    gradient is not None and not bool(torch.isfinite(gradient).all())
                    for gradient in fusion_gradients
                )
                gradient_reports.append(
                    {
                        "step": step + 1,
                        "present": present,
                        "nonzero": nonzero,
                        "nonfinite": nonfinite,
                    }
                )
                if not present:
                    raise RuntimeError("Qwen bbox loss did not reach multimodal fusion")
                if nonfinite:
                    raise RuntimeError(
                        f"multimodal fusion received {nonfinite} non-finite gradients"
                    )
                frozen_with_grad = [
                    name
                    for name, parameter in model.backbone.named_parameters()
                    if not parameter.requires_grad and parameter.grad is not None
                ]
                if frozen_with_grad:
                    raise RuntimeError(
                        f"frozen backbone parameters received gradients: {frozen_with_grad[:5]}"
                    )
                if config.train.phase_a_epochs == 0:
                    lora_gradients = [
                        parameter.grad for parameter in model.vision_lora_parameters()
                    ]
                    if not any(gradient is not None for gradient in lora_gradients):
                        raise RuntimeError("Qwen bbox loss did not reach Vision LoRA")
                    if any(
                        gradient is not None and not bool(torch.isfinite(gradient).all())
                        for gradient in lora_gradients
                    ):
                        raise RuntimeError("Vision LoRA received non-finite gradients")
                if step == 1 and needs_two_step_check:
                    query_parameters = [
                        *model.fusion.ir_query_encoder.parameters(),
                        *model.fusion.depth_query_encoder.parameters(),
                    ]
                    joint_parameters = list(model.fusion.joint_stage_fusions.parameters())
                    adapter_parameters = [
                        *model.fusion.ir_adapters.parameters(),
                        *model.fusion.depth_adapters.parameters(),
                    ]
                    query_nonzero = sum(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        and bool(torch.count_nonzero(parameter.grad))
                        for parameter in query_parameters
                    )
                    joint_nonzero = sum(
                        parameter.grad is not None
                        and bool(torch.isfinite(parameter.grad).all())
                        and bool(torch.count_nonzero(parameter.grad))
                        for parameter in joint_parameters
                    )
                    if query_nonzero == 0 or joint_nonzero == 0:
                        raise RuntimeError(
                            "two-step preflight did not produce finite nonzero gradients "
                            "for query encoder and joint fusion"
                        )
                    if config.model.freeze_parallel_adapters and any(
                        parameter.grad is not None for parameter in adapter_parameters
                    ):
                        raise RuntimeError("frozen parallel adapters received gradients")
                    gradient_reports[-1].update(
                        {
                            "query_nonzero": query_nonzero,
                            "joint_nonzero": joint_nonzero,
                        }
                    )
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        elapsed = time.perf_counter() - started
        vision = model.backbone.model.visual
        actual_deepstack = list(getattr(vision, "deepstack_visual_indexes", []))
        actual_fusion = list(getattr(model.fusion, "fusion_layer_indices", []))
        if config.model.parallel_fusion_layer_indices:
            expected = list(config.model.parallel_fusion_layer_indices)
            if actual_fusion != expected:
                raise RuntimeError(
                    f"fusion layer mismatch: configured={expected}, actual={actual_fusion}"
                )
            required = [*actual_deepstack, len(vision.blocks) - 1]
            if any(index not in actual_fusion for index in required):
                raise RuntimeError(
                    f"fusion layers {actual_fusion} do not cover DeepStack+final {required}"
                )
        cuda_active = torch.cuda.is_available() and str(args.device).startswith("cuda")
        report["real_model"] = {
            "loss": float(loss.detach()),
            "backbone_training": model.backbone.training,
            "gpu": torch.cuda.get_device_name(args.device) if cuda_active else None,
            "batch_size": config.train.batch_size,
            "joint_from_start": config.train.phase_a_epochs == 0,
            "deep_prompt_layers": len(getattr(model.fusion, "prompt_blocks", [])),
            "vision_depth": len(vision.blocks),
            "vision_hidden_size": int(vision.config.hidden_size),
            "deepstack_visual_indexes": actual_deepstack,
            "fusion_layer_indices": actual_fusion,
            "scanned_samples": min(len(train_dataset), args.max_samples),
            "selected_visual_tokens": visual_tokens,
            "optimizer_steps": run_steps if optimizer is not None else 0,
            "requested_optimizer_steps": args.optimizer_steps,
            "optimizer_phase": optimizer_phase if optimizer is not None else None,
            "gradient_reports": gradient_reports,
            "elapsed_seconds": round(elapsed, 3),
            "peak_memory_allocated_gib": (
                round(torch.cuda.max_memory_allocated(args.device) / 2**30, 2)
                if cuda_active
                else None
            ),
            "peak_memory_reserved_gib": (
                round(torch.cuda.max_memory_reserved(args.device) / 2**30, 2)
                if cuda_active
                else None
            ),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
