from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model_checkpoint, load_training_checkpoint, save_checkpoint
from .boxes import generalized_iou_aligned
from .metrics import grounding_metrics, merge_metric_sums


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _training_inputs(batch):
    names = (
        "pixel_values", "ir_pixel_values", "depth_pixel_values", "input_ids",
        "attention_mask", "image_grid_thw", "labels", "bbox", "coordinate_mask",
        "query_input_ids", "query_attention_mask",
    )
    return {name: batch[name] for name in names if name in batch}


def _generation_inputs(batch, rgb_only=False, modalities: set[str] | None = None):
    output = {
        "pixel_values": batch["pixel_values"],
        "input_ids": batch["generation_input_ids"],
        "attention_mask": batch["generation_attention_mask"],
        "image_grid_thw": batch["image_grid_thw"],
        "query_input_ids": batch["query_input_ids"],
        "query_attention_mask": batch["query_attention_mask"],
        "rgb_only": rgb_only,
    }
    for modality in ("ir", "depth"):
        name = f"{modality}_pixel_values"
        if name in batch and (modalities is None or modality in modalities):
            output[name] = batch[name]
    return output


def parse_bbox(text: str) -> list[float] | None:
    match = re.search(r'"?bbox_2d"?\s*:\s*\[([^\]]+)\]', text, flags=re.I)
    if not match:
        return None
    values = re.findall(r"-?\d+(?:\.\d+)?", match.group(1))
    if len(values) != 4:
        return None
    box = [float(value) / 1000.0 for value in values]
    if not all(0 <= value <= 1 for value in box) or box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


@torch.no_grad()
def evaluate(
    model,
    loader,
    processor,
    device: torch.device,
    max_new_tokens: int,
    rgb_only: bool = False,
    modalities: set[str] | None = None,
):
    model.eval()
    rows = []
    parsed = 0
    total = 0
    cap_hits = 0
    cap_parse_failures = 0
    auxiliary_rows = []
    auxiliary_giou_sum = 0.0
    eos_ids = processor.tokenizer.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    for batch in loader:
        batch = _move(batch, device)
        if model.auxiliary_bbox_enabled:
            auxiliary = model.predict_auxiliary_bbox(
                **_generation_inputs(batch, rgb_only=rgb_only, modalities=modalities)
            )
            auxiliary_rows.append(
                (auxiliary.shape[0], grounding_metrics(auxiliary, batch["bbox"]))
            )
            auxiliary_giou_sum += float(
                generalized_iou_aligned(auxiliary, batch["bbox"]).sum().item()
            )
        generated = model.generate(
            **_generation_inputs(batch, rgb_only=rgb_only, modalities=modalities),
            max_new_tokens=max_new_tokens,
        )
        prompt_length = batch["generation_input_ids"].shape[1]
        new_tokens = generated[:, prompt_length:]
        capped = [
            new_tokens.shape[1] >= max_new_tokens
            and not any(int(token) in eos_ids for token in tokens)
            for tokens in new_tokens
        ]
        cap_hits += sum(capped)
        answers = processor.batch_decode(new_tokens, skip_special_tokens=True)
        predictions = []
        for answer, hit_cap in zip(answers, capped):
            box = parse_bbox(answer)
            parsed += box is not None
            cap_parse_failures += hit_cap and box is None
            predictions.append(box or [0.0, 0.0, 0.0, 0.0])
        predicted = torch.tensor(predictions, device=device, dtype=batch["bbox"].dtype)
        size = predicted.shape[0]
        total += size
        rows.append((size, grounding_metrics(predicted, batch["bbox"])))
    metrics = merge_metric_sums(rows)
    metrics["parse_rate"] = parsed / max(total, 1)
    metrics["generation_cap_hit_rate"] = cap_hits / max(total, 1)
    metrics["cap_parse_failure_rate"] = cap_parse_failures / max(total, 1)
    if auxiliary_rows:
        auxiliary_metrics = merge_metric_sums(auxiliary_rows)
        metrics.update({f"aux_{name}": value for name, value in auxiliary_metrics.items()})
        metrics["aux_mean_giou"] = auxiliary_giou_sum / max(total, 1)
    return metrics


def _optimizer(model, config, phase: str):
    if phase == "a":
        adapter_parameters = (
            model.parallel_adapter_parameters()
            if config.train.parallel_adapter_lr is not None
            else []
        )
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        fusion_parameters = [
            parameter
            for parameter in model.task_parameters()
            if id(parameter) not in adapter_ids
        ]
        groups = [{"params": fusion_parameters, "lr": config.train.fusion_lr_a}]
        if adapter_parameters:
            groups.append(
                {
                    "params": adapter_parameters,
                    "lr": config.train.parallel_adapter_lr,
                }
            )
    else:
        groups = [
            {"params": model.task_parameters(), "lr": config.train.fusion_lr_b},
            {"params": model.vision_lora_parameters(), "lr": config.train.vision_lora_lr},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.train.weight_decay)
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if trainable != optimized:
        raise RuntimeError(
            f"optimizer/trainable mismatch: missing={len(trainable - optimized)}, "
            f"extra={len(optimized - trainable)}"
        )
    return optimizer


def _scheduler(optimizer, config, phase: str):
    warmup = config.train.phase_b_warmup_steps if phase == "b" else 0
    if warmup == 0:
        def schedule(_):
            return 1.0
    else:
        def schedule(step):
            return min(1.0, (step + 1) / warmup)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _override_optimizer_scheduler_lrs(optimizer, scheduler, learning_rates) -> None:
    learning_rates = list(learning_rates)
    if len(optimizer.param_groups) != len(learning_rates):
        raise RuntimeError(
            "resume learning-rate group mismatch: "
            f"optimizer={len(optimizer.param_groups)}, configured={len(learning_rates)}"
        )
    for group, learning_rate in zip(optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    scheduler.base_lrs = learning_rates.copy()
    scheduler._last_lr = learning_rates.copy()


def _geometry_gradient_scale(config, phase: str, global_step: int) -> float:
    if not config.model.auxiliary_bbox_enabled or phase != "a":
        return 1.0
    head_only = config.train.auxiliary_head_only_steps
    warmup = config.train.auxiliary_geometry_warmup_steps
    if global_step < head_only:
        return 0.0
    if warmup == 0:
        return 1.0
    return min(1.0, (global_step - head_only + 1) / warmup)


def _loss_gradients(loss, model, parameters):
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _gradient_diagnostics(model, output) -> dict[str, float]:
    parameters = [parameter for parameter in model.fusion.parameters() if parameter.requires_grad]
    token = _loss_gradients(output["token_loss"], model, parameters)
    l1 = _loss_gradients(output["bbox_l1_loss"], model, parameters)
    giou = _loss_gradients(output["bbox_giou_loss"], model, parameters)
    model.zero_grad(set_to_none=True)
    token_sq = geometry_sq = l1_sq = giou_sq = dot = l1_giou_dot = 0.0
    for token_grad, l1_grad, giou_grad in zip(token, l1, giou):
        if token_grad is None:
            continue
        token_grad = token_grad.float()
        l1_grad = torch.zeros_like(token_grad) if l1_grad is None else l1_grad.float()
        giou_grad = torch.zeros_like(token_grad) if giou_grad is None else giou_grad.float()
        geometry_grad = (
            model.auxiliary_bbox_l1_weight * l1_grad
            + model.auxiliary_bbox_giou_weight * giou_grad
        )
        token_sq += float(token_grad.square().sum())
        l1_sq += float(l1_grad.square().sum())
        giou_sq += float(giou_grad.square().sum())
        geometry_sq += float(geometry_grad.square().sum())
        dot += float((token_grad * geometry_grad).sum())
        l1_giou_dot += float((l1_grad * giou_grad).sum())
    token_norm = math.sqrt(token_sq)
    geometry_norm = math.sqrt(geometry_sq)
    return {
        "token_gradient_norm": token_norm,
        "bbox_l1_gradient_norm": math.sqrt(l1_sq),
        "bbox_giou_gradient_norm": math.sqrt(giou_sq),
        "weighted_geometry_gradient_norm": geometry_norm,
        "geometry_to_token_gradient_ratio": geometry_norm / max(token_norm, 1e-12),
        "geometry_token_gradient_cosine": dot / max(token_norm * geometry_norm, 1e-12),
        "bbox_l1_giou_gradient_cosine": (
            l1_giou_dot / max(math.sqrt(l1_sq * giou_sq), 1e-12)
        ),
    }


def _write_metric(output_dir: Path, record: dict) -> None:
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _run_phase(
    model, train_loader, val_loader, full_val_loader, probe_val_loader,
    processor, config, device, scaler, amp_dtype, phase, epochs, offset,
    start_phase_epoch=1, resume_checkpoint=None,
):
    optimizer = _optimizer(model, config, phase)
    scheduler = _scheduler(optimizer, config, phase)
    output_dir = Path(config.output_dir)
    updates_per_epoch = (len(train_loader) + config.train.grad_accumulation - 1) // config.train.grad_accumulation
    global_step = (start_phase_epoch - 1) * updates_per_epoch
    if resume_checkpoint:
        checkpoint = load_training_checkpoint(
            resume_checkpoint, model, optimizer, scheduler, scaler
        )
        global_step = int(checkpoint.get("global_step", global_step))
        resumed_learning_rates = [group["lr"] for group in optimizer.param_groups]
        if config.train.override_resume_learning_rates:
            configured_learning_rates = (
                [
                    config.train.fusion_lr_a,
                    *(
                        [config.train.parallel_adapter_lr]
                        if config.train.parallel_adapter_lr is not None
                        else []
                    ),
                ]
                if phase == "a"
                else [config.train.fusion_lr_b, config.train.vision_lora_lr]
            )
            _override_optimizer_scheduler_lrs(
                optimizer, scheduler, configured_learning_rates
            )
            resumed_learning_rates = configured_learning_rates
        print(json.dumps({
            "event": "resume", "phase": phase.upper(), "epoch": checkpoint.get("epoch"),
            "global_step": global_step,
            "learning_rates": resumed_learning_rates,
            "learning_rates_overridden": config.train.override_resume_learning_rates,
            **checkpoint["restored"],
        }), flush=True)
    best_score = -1.0
    best_name = "best_phase_a.pt" if phase == "a" else "best.pt"
    best_path = output_dir / best_name
    if start_phase_epoch > 1 and best_path.exists():
        baseline = evaluate(model, val_loader, processor, device, config.train.max_new_tokens)
        best_score = baseline["mean_iou"]
        print(json.dumps({
            "epoch": offset + start_phase_epoch - 1, "phase": phase.upper(),
            "eval_scope": "resume_baseline_subset", "eval_samples": len(val_loader.dataset),
            **baseline,
        }, ensure_ascii=False), flush=True)
    probe_rgb_score = None
    initial_phase = phase == "a" or (phase == "b" and config.train.phase_a_epochs == 0)
    probe_steps = tuple(config.train.early_probe_steps)
    if not probe_steps and config.train.early_probe_step:
        probe_steps = (config.train.early_probe_step,)
    if initial_phase and probe_val_loader is not None and probe_steps:
        baseline = evaluate(
            model, probe_val_loader, processor, device,
            config.train.max_new_tokens, rgb_only=True,
        )
        probe_rgb_score = baseline["mean_iou"]
        _write_metric(output_dir, {
            "event": "early_probe_rgb_baseline", "phase": phase.upper(),
            "global_step": global_step, "eval_scope": "early_probe_rgb",
            "eval_samples": len(probe_val_loader.dataset), **baseline,
        })
    stale_evals = 0
    diagnostics_written = False
    for phase_epoch in range(start_phase_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = defaultdict(float)
        for step, batch in enumerate(train_loader, 1):
            batch = _move(batch, device)
            geometry_scale = _geometry_gradient_scale(config, phase, global_step)
            with torch.autocast(
                device_type=device.type,
                enabled=config.train.amp and device.type == "cuda",
                dtype=amp_dtype,
            ):
                output = model(
                    **_training_inputs(batch), geometry_gradient_scale=geometry_scale
                )
                loss = output["loss"]
                scaled_loss = loss / config.train.grad_accumulation
            if (
                config.train.auxiliary_gradient_diagnostics
                and config.model.auxiliary_bbox_enabled
                and phase == "a"
                and geometry_scale >= 1.0
                and not diagnostics_written
                and (step - 1) % config.train.grad_accumulation == 0
            ):
                diagnostic = {
                    "event": "auxiliary_gradient_diagnostics",
                    "phase": phase.upper(),
                    "global_step": global_step,
                    **_gradient_diagnostics(model, output),
                }
                print(json.dumps(diagnostic), flush=True)
                diagnostics_written = True
            scaler.scale(scaled_loss).backward()
            for name in (
                "loss", "token_loss", "coordinate_token_loss",
                "bbox_l1_loss", "bbox_giou_loss",
            ):
                if name in output:
                    running[name] += float(output[name].detach())
            running["geometry_gradient_scale"] += geometry_scale
            if step % config.train.grad_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                if (
                    initial_phase
                    and probe_val_loader is not None
                    and global_step in probe_steps
                ):
                    probe_metrics = evaluate(
                        model, probe_val_loader, processor, device,
                        config.train.max_new_tokens,
                    )
                    ratio = probe_metrics["mean_iou"] / max(probe_rgb_score or 0.0, 1e-12)
                    probe_record = {
                        "event": "early_probe", "epoch": offset + phase_epoch,
                        "phase": phase.upper(), "global_step": global_step,
                        "eval_scope": "early_probe",
                        "eval_samples": len(probe_val_loader.dataset),
                        "rgb_mean_iou": probe_rgb_score,
                        "mean_iou_ratio_to_rgb": ratio,
                        "train_loss_so_far": running["loss"] / step,
                        **probe_metrics,
                    }
                    _write_metric(output_dir, probe_record)
                    save_checkpoint(
                        output_dir / f"probe_phase_{phase}_step_{global_step:04d}.pt",
                        model, optimizer, scheduler, scaler, config,
                        offset + phase_epoch, probe_metrics["mean_iou"], global_step,
                    )
                    if (
                        config.train.early_probe_abort_ratio
                        and global_step >= config.train.early_probe_abort_from_step
                        and ratio < config.train.early_probe_abort_ratio
                    ):
                        print(json.dumps({
                            "event": "early_probe_abort", "global_step": global_step,
                            "mean_iou_ratio_to_rgb": ratio,
                            "threshold": config.train.early_probe_abort_ratio,
                            "abort_from_step": config.train.early_probe_abort_from_step,
                        }), flush=True)
                        return False
                    if config.train.stop_after_last_probe and global_step == probe_steps[-1]:
                        print(json.dumps({
                            "event": "controlled_probe_stop",
                            "phase": phase.upper(),
                            "global_step": global_step,
                        }), flush=True)
                        return False
                    model.train()
        epoch = offset + phase_epoch
        training_metrics = {
            "train_loss": running["loss"] / len(train_loader),
            "train_token_ce": running["token_loss"] / len(train_loader),
            "train_geometry_gradient_scale": (
                running["geometry_gradient_scale"] / len(train_loader)
            ),
        }
        if config.model.auxiliary_bbox_enabled:
            training_metrics.update(
                {
                    "train_coordinate_token_ce": (
                        running["coordinate_token_loss"] / len(train_loader)
                    ),
                    "train_aux_l1": running["bbox_l1_loss"] / len(train_loader),
                    "train_aux_giou_loss": (
                        running["bbox_giou_loss"] / len(train_loader)
                    ),
                }
            )
        should_eval = phase_epoch % config.train.eval_every_n_epochs == 0 or phase_epoch == epochs
        if not should_eval:
            print(json.dumps({"epoch": epoch, "phase": phase.upper(), **training_metrics}), flush=True)
            continue
        metrics = evaluate(model, val_loader, processor, device, config.train.max_new_tokens)
        record = {"epoch": epoch, "phase": phase.upper(), "eval_scope": "subset", "eval_samples": len(val_loader.dataset), **training_metrics, **metrics}
        _write_metric(output_dir, record)
        last_name = "last_phase_a.pt" if phase == "a" else "last.pt"
        save_checkpoint(
            output_dir / last_name, model, optimizer, scheduler, scaler, config,
            epoch, metrics["mean_iou"], global_step,
        )
        if metrics["mean_iou"] > best_score + config.train.early_stopping_min_delta:
            best_score = metrics["mean_iou"]
            stale_evals = 0
            save_checkpoint(
                output_dir / best_name, model, optimizer, scheduler, scaler, config,
                epoch, best_score, global_step,
            )
        else:
            stale_evals += 1
        if phase_epoch == epochs or stale_evals >= config.train.early_stopping_patience:
            load_model_checkpoint(best_path, model)
            # When the configured validation cap already contains the complete
            # validation set (as in lightweight pipeline checks), reuse the
            # metrics computed immediately above instead of generating every
            # prediction twice.
            if full_val_loader.dataset is val_loader.dataset:
                full_metrics = metrics
            else:
                full_metrics = evaluate(
                    model, full_val_loader, processor, device, config.train.max_new_tokens
                )
            full_record = {"epoch": epoch, "phase": phase.upper(), "eval_scope": "full", "eval_samples": len(full_val_loader.dataset), **full_metrics}
            _write_metric(output_dir, full_record)
        if stale_evals >= config.train.early_stopping_patience:
            print(json.dumps({"event": "early_stop", "phase": phase.upper(), "epoch": epoch, "patience": stale_evals}), flush=True)
            break
    return True


def train(
    model, train_loader, val_loader, full_val_loader, probe_val_loader,
    processor, config, device,
):
    model.to(device)
    model.set_phase_a_trainable(
        config.stage,
        freeze_parallel_adapters=config.model.freeze_parallel_adapters,
        parallel_adapter_train_last_n=config.model.parallel_adapter_train_last_n,
    )
    backbone_dtype = next(model.backbone.parameters()).dtype
    amp_dtype = backbone_dtype if backbone_dtype in (torch.float16, torch.bfloat16) else torch.float16
    # BF16 has FP32-like exponent range and must not use CUDA GradScaler; PyTorch
    # does not implement BF16 foreach unscale for these LoRA gradients.
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=config.train.amp and device.type == "cuda" and amp_dtype == torch.float16,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_epoch = config.train.resume_epoch
    if resume_epoch < config.train.phase_a_epochs:
        phase_a_completed = _run_phase(
            model, train_loader, val_loader, full_val_loader, probe_val_loader,
            processor, config, device, scaler, amp_dtype,
            "a", config.train.phase_a_epochs, 0, resume_epoch + 1,
            config.train.init_checkpoint,
        )
        if not phase_a_completed:
            return
    if config.train.phase_b_epochs:
        if config.train.phase_a_epochs:
            load_model_checkpoint(output_dir / "best_phase_a.pt", model)
        model.enable_vision_lora()
        phase_b_done = max(0, resume_epoch - config.train.phase_a_epochs)
        _run_phase(
            model, train_loader, val_loader, full_val_loader,
            probe_val_loader if config.train.phase_a_epochs == 0 else None,
            processor, config, device, scaler, amp_dtype,
            "b", config.train.phase_b_epochs, config.train.phase_a_epochs, phase_b_done + 1,
        )
