from __future__ import annotations

import math
from pathlib import Path

import torch


root = Path("runs/multimodal")
for name in ("best_phase_a.pt", "last_phase_a.pt", "best.pt"):
    checkpoint = torch.load(root / name, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    gate = state.get("fusion.gate_logit")
    alpha = state.get("fusion.alpha_logit")
    beta = state.get("fusion.beta_logit")
    prompt_weight = state.get("fusion.prompt_restore.weight")
    prompt_bias = state.get("fusion.prompt_restore.bias")
    lora = {key: value for key, value in state.items() if "lora" in key.lower()}
    print({
        "checkpoint": name,
        "epoch": checkpoint.get("epoch"),
        "score": checkpoint.get("score"),
        "gate_logit": None if gate is None else float(gate),
        "gate_sigmoid": None if gate is None else float(torch.sigmoid(gate)),
        "alpha_sigmoid": None if alpha is None else float(torch.sigmoid(alpha)),
        "beta_sigmoid": None if beta is None else float(torch.sigmoid(beta)),
        "prompt_weight_norm": None if prompt_weight is None else float(prompt_weight.float().norm()),
        "prompt_bias_norm": None if prompt_bias is None else float(prompt_bias.float().norm()),
        "lora_tensors": len(lora),
        "lora_total_norm": math.sqrt(sum(float(value.float().square().sum()) for value in lora.values())),
    })
