"""Small, real-model 8B pressure test for the three training phases.

The pressure test deliberately uses the regular processor, collator, model and
optimizer.  It does not edit a manifest or change the image-size budget on a
per-sample basis.  A caller can rerun it with a uniformly smaller
``data.max_pixels`` after an exit status of 42 (CUDA out of memory).

The selected eight records are a deterministic coverage sample over the actual
RGB image area and tokenized query length. Each record is presented four times,
covering two complete 16-microbatch accumulation groups at batch size one.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint, save_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _generation_inputs, _optimizer, _training_inputs, seed_everything
from mm_grounding.model import build_grounder


OOM_EXIT_CODE = 42
MICROBATCHES_PER_GROUP = 16
OPTIMIZER_UPDATES = 2
REPRESENTATIVE_COUNT = 8


def _resolve_path(value: str, config_path: Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Existing configs use paths relative to the repository root, while a few
    # older local configs used paths relative to configs/.  Prefer the path that
    # is already valid and only then try the config directory.
    candidates = (root / path, config_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _profile_record(
    dataset: GroundingDataset, index: int, tokenizer, max_pixels: int
) -> dict:
    record = dataset.records[index]
    image_path = dataset._path(dataset._field(record, "rgb", "visible"))
    with Image.open(image_path) as image:
        width, height = image.size
    encoded = tokenizer(
        str(record.get("query", "")), add_special_tokens=True, return_attention_mask=False
    )
    return {
        "index": index,
        "id": str(record.get("id", index)),
        "width": int(width),
        "height": int(height),
        "pixels": int(width * height),
        "effective_pixels": int(min(width * height, max_pixels)),
        "query_tokens": int(len(encoded["input_ids"])),
        "estimated_combined": float(min(width * height, max_pixels) / 256 + len(encoded["input_ids"])),
    }


def select_representative_indices(profiles: list[dict], count: int = REPRESENTATIVE_COUNT) -> list[int]:
    """Choose deterministic area/query coverage points from record profiles.

    Always include the largest image, longest query, largest capped visual plus
    query budget, and the longest query among the large-image quartile.  This
    avoids missing a coincident large-image/long-query worst case.  Remaining
    slots use anti-diagonal quantiles to cover the rest of the range.
    """
    if not profiles:
        raise ValueError("cannot select representatives from an empty dataset")
    count = min(int(count), len(profiles))
    if count < 1:
        raise ValueError("representative count must be positive")
    area_values = sorted(profile.get("effective_pixels", profile["pixels"]) for profile in profiles)
    query_values = sorted(profile["query_tokens"] for profile in profiles)
    area_min, area_max = min(area_values), max(area_values)
    query_min, query_max = min(query_values), max(query_values)

    def normalized(value: float, low: int, high: int) -> float:
        return 0.0 if high == low else (value - low) / (high - low)

    ordered = sorted(profiles, key=lambda item: (item["index"], item["id"]))
    selected: list[int] = []
    used: set[int] = set()

    def append(profile: dict) -> None:
        if len(selected) < count and profile["index"] not in used:
            selected.append(profile["index"])
            used.add(profile["index"])

    large_cut = sorted(area_values)[max(0, len(area_values) * 3 // 4 - 1)]
    forced = [
        max(ordered, key=lambda item: (item.get("effective_pixels", item["pixels"]), -item["index"])),
        max(ordered, key=lambda item: (item["query_tokens"], -item["index"])),
        max(
            ordered,
            key=lambda item: (item.get("estimated_combined", item["pixels"] + item["query_tokens"]), -item["index"]),
        ),
        max(
            (item for item in ordered if item.get("effective_pixels", item["pixels"]) >= large_cut),
            key=lambda item: (item["query_tokens"], -item["index"]),
        ),
    ]
    for profile in forced:
        append(profile)

    for slot in range(count * 2):
        if len(selected) >= count:
            break
        quantile = slot / max(count - 1, 1)
        target_area = area_min + quantile * (area_max - area_min)
        target_query = query_max - quantile * (query_max - query_min)
        candidates = [profile for profile in ordered if profile["index"] not in used]
        chosen = min(
            candidates,
            key=lambda profile: (
                abs(normalized(profile.get("effective_pixels", profile["pixels"]), area_min, area_max)
                    - normalized(target_area, area_min, area_max))
                + abs(normalized(profile["query_tokens"], query_min, query_max)
                      - normalized(target_query, query_min, query_max)),
                profile["index"],
            ),
        )
        append(chosen)
    if len(selected) < count:
        for profile in ordered:
            append(profile)
            if len(selected) >= count:
                break
    return selected


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _finite_gradients(model) -> tuple[int, int, int, list[str]]:
    present = nonzero = nonfinite = 0
    frozen_with_grad: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if not parameter.requires_grad:
            if gradient is not None:
                frozen_with_grad.append(name)
            continue
        if gradient is None:
            continue
        present += 1
        if not bool(torch.isfinite(gradient).all()):
            nonfinite += 1
        elif bool(torch.count_nonzero(gradient)):
            nonzero += 1
    return present, nonzero, nonfinite, frozen_with_grad


def _group_gradient_report(model) -> dict[str, dict[str, int]]:
    groups: dict[str, Iterable[tuple[str, torch.nn.Parameter]]] = {
        "ir_adapters": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.ir_adapters.")
        ),
        "depth_adapters": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.depth_adapters.")
        ),
        "ir_query_encoder": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.ir_query_encoder.")
        ),
        "depth_query_encoder": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.depth_query_encoder.")
        ),
        "stage_fusions": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.stage_fusions.")
        ),
        "joint_stage_fusions": (
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name.startswith("fusion.joint_stage_fusions.")
        ),
    }
    report = {}
    for group, parameters in groups.items():
        values = [parameter for _, parameter in parameters if parameter.requires_grad]
        report[group] = {
            "parameters": len(values),
            "grad_present": sum(parameter.grad is not None for parameter in values),
            "grad_nonzero": sum(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
                for parameter in values
            ),
        }
    return report


def _save_reload_prediction_check(model, batch: dict, config, device: torch.device, output_dir: Path) -> dict:
    """Check deterministic generation after a sparse trainable-state reload.

    Only trainable parameters are written, and the temporary file is removed
    immediately.  This exercises the repository's normal ``load_model_checkpoint``
    helper without making another persistent experiment checkpoint.
    """
    model.eval()
    generation_inputs = _generation_inputs(batch)
    with torch.no_grad():
        before = model.generate(
            **generation_inputs, max_new_tokens=config.train.max_new_tokens
        ).detach().cpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix="stress_reload_", suffix=".pt", dir=output_dir, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        save_checkpoint(
            temporary_path, model, None, None, None, config, 0,
            {"acc_0.5": 0.0}, OPTIMIZER_UPDATES, ("acc_0.5",), compact=True,
        )
        with torch.no_grad():
            next(parameter for parameter in model.parameters() if parameter.requires_grad).add_(1)
        load_model_checkpoint(temporary_path, model)
        with torch.no_grad():
            after = model.generate(
                **generation_inputs, max_new_tokens=config.train.max_new_tokens
            ).detach().cpu()
    finally:
        temporary_path.unlink(missing_ok=True)
    identical = bool(torch.equal(before, after))
    if not identical:
        raise RuntimeError("fixed-sample prediction changed after sparse checkpoint reload")
    return {
        "identical": identical,
        "tokens": int(before.shape[-1]),
        "checkpoint": "temporary_compact_checkpoint",
    }


def run_stress(config_path: str | Path, output_path: str | Path, device_name: str = "cuda") -> dict:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path)
    config_path = config_path if config_path.is_absolute() else root / config_path
    config = load_config(config_path)
    seed_everything(config.train.seed)
    training_dropout = config.model.modality_dropout
    train_manifest = _resolve_path(config.data.train_manifest, config_path, root)
    val_manifest = _resolve_path(config.data.val_manifest, config_path, root)
    config = replace(
        config,
        model=replace(config.model, modality_dropout=0.0),
        data=replace(
            config.data,
            train_manifest=str(train_manifest),
            val_manifest=str(val_manifest),
        ),
        # Stress validates the architecture and data path.  Initialization
        # checkpoints are intentionally not required for this isolated probe.
        train=replace(
            config.train,
            initialization_checkpoints=(),
            init_checkpoint=None,
            warm_start_joint_fusion_from_legacy=False,
        ),
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA stress requested but CUDA is unavailable")

    processor = AutoProcessor.from_pretrained(
        config.model.backbone,
        revision=config.model.backbone_revision,
        min_pixels=config.data.min_pixels,
        max_pixels=config.data.max_pixels,
        local_files_only=True,
    )
    dataset = GroundingDataset(
        config.data.train_manifest,
        config.stage,
        config.data.depth_scale,
        config.data.depth_clip,
    )
    profiles = [
        _profile_record(dataset, index, processor.tokenizer, config.data.max_pixels)
        for index in range(len(dataset))
    ]
    selected_indices = select_representative_indices(profiles)
    selected_profiles = [profiles[index] for index in selected_indices]
    collator = NativeGroundingCollator(processor, config.stage)

    # No initialization is loaded above.  The actual training runner supplies
    # the shared Stage-1 checkpoint; pressure testing an absent checkpoint would
    # obscure the memory and gradient properties we need here.
    model = build_grounder(config.model, processor).to(device)
    if config.train.gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable()
        model.backbone.enable_input_require_grads()
    model.set_phase_a_trainable(
        config.stage,
        freeze_parallel_adapters=config.model.freeze_parallel_adapters,
        parallel_adapter_train_last_n=config.model.parallel_adapter_train_last_n,
    )
    model.train()
    model.backbone.config.use_cache = False
    optimizer = _optimizer(model, config, "a")
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = next(model.backbone.parameters()).dtype

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    microbatch_reports = []
    final_batch = None
    gradient_updates = []
    for update_index in range(OPTIMIZER_UPDATES):
        for microbatch_index in range(MICROBATCHES_PER_GROUP):
            record_index = selected_indices[microbatch_index % len(selected_indices)]
            batch = _to_device(collator([dataset[record_index]]), device)
            final_batch = batch
            print(json.dumps({
                "event": "stress_microbatch_start", "update": update_index + 1,
                "microbatch": microbatch_index + 1, "sample_id": batch["sample_id"][0],
                "input_tokens": int(batch["input_ids"].shape[-1]),
            }), flush=True)
            with torch.autocast(
                device_type=device.type, enabled=config.train.amp and device.type == "cuda",
                dtype=amp_dtype,
            ):
                output = model(**_training_inputs(batch))
            loss = output["loss"]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite stress loss at microbatch {microbatch_index + 1}")
            (loss / MICROBATCHES_PER_GROUP).backward()
            present, nonzero, nonfinite, frozen_with_grad = _finite_gradients(model)
            if present == 0:
                raise RuntimeError("loss did not reach any trainable fusion parameter")
            if nonfinite:
                raise RuntimeError(
                    f"non-finite gradient in stress group at microbatch {microbatch_index + 1}"
                )
            if frozen_with_grad:
                raise RuntimeError(
                    "frozen parameters received gradients: "
                    + ", ".join(frozen_with_grad[:5])
                )
            microbatch_reports.append(
                {
                    "microbatch": microbatch_index + 1,
                    "optimizer_update": update_index + 1,
                    "record_index": record_index,
                    "sample_id": str(batch["sample_id"][0]),
                    "loss": float(loss.detach()),
                    "grad_present": present,
                    "grad_nonzero": nonzero,
                    "grad_nonfinite": nonfinite,
                }
            )
            del output, loss
        before_step_groups = _group_gradient_report(model)
        gradient_updates.append(before_step_groups)
        if update_index == 1:
            expected = (
                ["ir_query_encoder", "depth_query_encoder", "joint_stage_fusions"]
                if config.stage == "joint"
                else [f"{config.stage}_query_encoder", "stage_fusions"]
            )
            for group in expected:
                if before_step_groups[group]["grad_nonzero"] == 0:
                    raise RuntimeError(f"second optimizer update has no nonzero gradient in {group}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if final_batch is None:
        raise RuntimeError("stress group did not produce a batch")
    reload_report = _save_reload_prediction_check(
        model, final_batch, config, device, Path(output_path).resolve().parent
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - step_started
    cuda_active = device.type == "cuda"
    report = {
        "event": "8b_training_stress",
        "stage": config.stage,
        "config": str(config_path.resolve()),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "max_pixels": int(config.data.max_pixels),
        "min_pixels": int(config.data.min_pixels),
        "model_backbone": config.model.backbone,
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "training_modality_dropout": training_dropout,
        "stress_modality_dropout": 0.0,
        "backbone_dtype": str(next(model.backbone.parameters()).dtype),
        "device": str(device),
        "representative_count": len(selected_indices),
        "representatives": selected_profiles,
        "gradient_accumulation": MICROBATCHES_PER_GROUP,
        "microbatches": len(microbatch_reports),
        "optimizer_steps": OPTIMIZER_UPDATES,
        "microbatch_reports": microbatch_reports,
        "gradient_groups_before_step": before_step_groups,
        "gradient_groups_each_update": gradient_updates,
        "reload_prediction": reload_report,
        "elapsed_seconds": round(elapsed, 3),
        "peak_memory_allocated_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if cuda_active
            else None
        ),
        "peak_memory_reserved_gib": (
            round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            if cuda_active
            else None
        ),
        "total_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        report = run_stress(args.config, args.output, args.device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error).lower() and not isinstance(error, torch.cuda.OutOfMemoryError):
            raise
        payload = {
            "event": "8b_training_stress",
            "status": "oom",
            "config": str(Path(args.config).resolve()),
            "max_pixels": load_config(args.config).data.max_pixels,
            "error": str(error),
            "oom_exit_code": OOM_EXIT_CODE,
            "traceback": traceback.format_exc(),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return OOM_EXIT_CODE
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
