from pathlib import Path

import torch


for name in ("best.pt", "last.pt"):
    path = Path("runs/multimodal_from_a3") / name
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print(name, {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "score": checkpoint.get("score"),
        "optimizer": checkpoint.get("optimizer") is not None,
        "scheduler": checkpoint.get("scheduler") is not None,
        "scaler": checkpoint.get("scaler") is not None,
    })
