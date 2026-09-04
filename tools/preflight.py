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
    parser.add_argument("--expected-val-samples", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    config = load_config(config_path if config_path.is_absolute() else root / config_path)
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
        processor = AutoProcessor.from_pretrained(
            config.model.backbone, min_pixels=config.data.min_pixels,
            max_pixels=config.data.max_pixels, local_files_only=True,
        )
        model = build_grounder(config.model, processor).to(args.device)
        for value in config.train.initialization_checkpoints:
            checkpoint = Path(value)
            checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
            load_model_checkpoint(checkpoint, model)
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
        loader = DataLoader(
            GroundingDataset(
                manifests["train"], config.stage,
                config.data.depth_scale, config.data.depth_clip,
            ),
            batch_size=config.train.batch_size,
            collate_fn=NativeGroundingCollator(processor, config.stage),
        )
        batch = next(iter(loader))
        batch = {key: value.to(args.device) if torch.is_tensor(value) else value for key, value in batch.items()}
        names = [
            "pixel_values", "input_ids", "attention_mask", "image_grid_thw",
            "query_input_ids", "query_attention_mask", "labels", "bbox", "coordinate_mask",
        ]
        if config.stage in {"ir", "joint"}:
            names.append("ir_pixel_values")
        if config.stage in {"depth", "joint"}:
            names.append("depth_pixel_values")
        model_inputs = {name: batch[name] for name in names}
        loss = model(**model_inputs)["loss"]
        gradient_report = None
        if args.backward:
            loss.backward()
            if not any(parameter.grad is not None for parameter in model.fusion.parameters()):
                raise RuntimeError("Qwen bbox loss did not reach multimodal fusion")
            if config.train.phase_a_epochs == 0 and not any(
                parameter.grad is not None for parameter in model.vision_lora_parameters()
            ):
                raise RuntimeError("Qwen bbox loss did not reach Vision LoRA")
            if (
                config.model.fusion_type == "parallel_backbone"
                and config.model.parallel_joint_fusion
            ):
                optimizer = torch.optim.SGD(model.task_parameters(), lr=1e-3)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model(**model_inputs)["loss"].backward()
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
                    and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter in query_parameters
                )
                joint_nonzero = sum(
                    parameter.grad is not None
                    and torch.count_nonzero(parameter.grad).item() > 0
                    for parameter in joint_parameters
                )
                adapter_gradients = sum(
                    parameter.grad is not None for parameter in adapter_parameters
                )
                if query_nonzero == 0 or joint_nonzero == 0:
                    raise RuntimeError(
                        "two-step preflight did not reach query encoder and joint fusion"
                    )
                if config.model.freeze_parallel_adapters and adapter_gradients:
                    raise RuntimeError("frozen parallel adapters received gradients")
                gradient_report = {
                    "query_parameters_with_nonzero_gradient": query_nonzero,
                    "joint_parameters_with_nonzero_gradient": joint_nonzero,
                    "adapter_parameters_with_gradient": adapter_gradients,
                }
        report["real_model"] = {
            "loss": float(loss.detach()), "gpu": torch.cuda.get_device_name(),
            "batch_size": config.train.batch_size,
            "joint_from_start": config.train.phase_a_epochs == 0,
            "deep_prompt_layers": len(getattr(model.fusion, "prompt_blocks", [])),
            "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        if gradient_report is not None:
            report["real_model"]["gradient_report"] = gradient_report
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
