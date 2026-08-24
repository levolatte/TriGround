from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _move, _optimizer, _scheduler, _training_inputs
from mm_grounding.model import build_grounder


config = load_config("configs/multimodal_from_a3.yaml")
device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(
    config.model.backbone,
    min_pixels=config.data.min_pixels,
    max_pixels=config.data.max_pixels,
)
model = build_grounder(config.model, processor)
model.to(device)
model.set_phase_a_trainable()
checkpoint = load_model_checkpoint("runs/multimodal_from_a3/best_phase_a.pt", model)
model.enable_vision_lora()
model.backbone.gradient_checkpointing_enable()
model.backbone.enable_input_require_grads()
optimizer = _optimizer(model, config, "b")
scheduler = _scheduler(optimizer, config, "b")
dataset = GroundingDataset(
    config.data.train_manifest,
    stage="joint",
    depth_scale=config.data.depth_scale,
    depth_clip=config.data.depth_clip,
)
loader = DataLoader(
    dataset,
    batch_size=config.train.batch_size,
    collate_fn=NativeGroundingCollator(processor, "joint"),
)
batch = _move(next(iter(loader)), device)
amp_dtype = next(model.backbone.parameters()).dtype
with torch.autocast("cuda", dtype=amp_dtype):
    loss = model(**_training_inputs(batch))["loss"]
if not math.isfinite(float(loss.detach())):
    raise RuntimeError(f"non-finite preflight loss: {loss}")
loss.backward()
lora_grads = [p.grad for p in model.vision_lora_parameters() if p.grad is not None]
fusion_grads = [p.grad for p in model.fusion.parameters() if p.grad is not None]
if not lora_grads or not fusion_grads:
    raise RuntimeError("missing LoRA or fusion gradients")
if not all(torch.isfinite(grad).all() for grad in lora_grads + fusion_grads):
    raise RuntimeError("non-finite gradients")
optimizer.step()
scheduler.step()
print({
    "source_epoch": checkpoint.get("epoch"),
    "source_global_step": checkpoint.get("global_step"),
    "source_score": checkpoint.get("score"),
    "loss": float(loss.detach()),
    "lora_params_with_grad": len(lora_grads),
    "fusion_params_with_grad": len(fusion_grads),
    "optimizer_groups": len(optimizer.param_groups),
    "first_step_lrs": scheduler.get_last_lr(),
    "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
})
