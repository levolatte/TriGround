import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mm_grounding.checkpoint import load_model_checkpoint
from mm_grounding.config import load_config
from mm_grounding.data import GroundingDataset, NativeGroundingCollator
from mm_grounding.engine import _move, _optimizer, _scheduler, _training_inputs
from mm_grounding.model import build_grounder


config = load_config("configs/multimodal.yaml")
device = torch.device("cuda")
processor = AutoProcessor.from_pretrained(
    config.model.backbone,
    min_pixels=config.data.min_pixels,
    max_pixels=config.data.max_pixels,
)
model = build_grounder(config.model, processor)
load_model_checkpoint("runs/multimodal/best_phase_a.pt", model)
model.to(device)
model.set_phase_a_trainable()
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
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(**_training_inputs(batch))["loss"]
loss.backward()
optimizer.step()
scheduler.step()
lora_with_grad = sum(parameter.grad is not None for parameter in model.vision_lora_parameters())
print({
    "loss": float(loss.detach()),
    "lora_params_with_grad": lora_with_grad,
    "optimizer_groups": len(optimizer.param_groups),
    "lrs": scheduler.get_last_lr(),
    "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
})
