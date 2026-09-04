from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip training state from a project checkpoint")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"unsupported checkpoint: {args.input}")

    artifact = {
        "model": checkpoint["model"],
        "trainable_only": bool(checkpoint.get("trainable_only", True)),
        "config": checkpoint.get("config"),
        "epoch": checkpoint.get("epoch"),
        "score": checkpoint.get("score"),
        "metrics": checkpoint.get("metrics"),
        "selection_order": checkpoint.get("selection_order"),
        "global_step": checkpoint.get("global_step"),
        "artifact_metadata": {
            "name": args.name,
            "tag": args.tag,
            "format": "mm-grounding-adapters inference checkpoint v1",
            "training_state_removed": ["optimizer", "scheduler", "scaler"],
            "source_filename": args.input.name,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(args.output)

    tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in artifact["model"].values()
        if isinstance(tensor, torch.Tensor)
    )
    print(json.dumps({
        "name": args.name,
        "tag": args.tag,
        "output": str(args.output.resolve()),
        "file_bytes": args.output.stat().st_size,
        "tensor_bytes": tensor_bytes,
        "tensors": len(artifact["model"]),
        "sha256": sha256(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

