from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Linear layer with a frozen-compatible low-rank residual.

    Subclassing Linear preserves the original ``weight``/``bias`` checkpoint
    keys, so an RGB checkpoint can initialize a model before LoRA is trained.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        self.bias = base.bias
        self.lora_a = nn.Linear(
            base.in_features, rank, bias=False,
            device=base.weight.device, dtype=base.weight.dtype,
        )
        self.lora_b = nn.Linear(
            rank, base.out_features, bias=False,
            device=base.weight.device, dtype=base.weight.dtype,
        )
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        update = self.lora_b(self.lora_a(self.lora_dropout(inputs)))
        return base + update * self.lora_scale


def _vision_blocks(backbone: nn.Module) -> nn.ModuleList | list[nn.Module]:
    candidates = (
        "model.visual.blocks",
        "visual.blocks",
        "model.visual.encoder.layers",
        "visual.encoder.layers",
    )
    for path in candidates:
        try:
            blocks = backbone.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            return blocks
    raise RuntimeError(
        "找不到 Qwen3-VL vision blocks；当前 transformers 模型结构可能已变化"
    )


def inject_vision_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    last_n_blocks: int,
    target_names: tuple[str, ...] = ("qkv", "q_proj", "k_proj", "v_proj", "o_proj", "proj"),
) -> int:
    """Inject LoRA into attention projections of the last vision blocks."""
    blocks = _vision_blocks(backbone)
    if not 0 < last_n_blocks <= len(blocks):
        raise ValueError(
            f"vision_lora_last_n_blocks 必须在 [1, {len(blocks)}]，实际为 {last_n_blocks}"
        )
    replaced = 0
    for block in list(blocks)[-last_n_blocks:]:
        matches = [
            name
            for name, module in block.named_modules()
            if name and name.rsplit(".", 1)[-1] in target_names
            and isinstance(module, nn.Linear)
            and not isinstance(module, LoRALinear)
        ]
        for name in matches:
            parent_name, _, child_name = name.rpartition(".")
            parent = block.get_submodule(parent_name) if parent_name else block
            base = getattr(parent, child_name)
            setattr(parent, child_name, LoRALinear(base, rank, alpha, dropout))
            replaced += 1
    if replaced == 0:
        raise RuntimeError("选定的 Qwen vision blocks 中没有找到 attention projection Linear")
    return replaced


def vision_lora_parameters(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield from child.lora_a.parameters()
            yield from child.lora_b.parameters()
