"""Compare saved D LoRA tensors with the shared warmup on CPU, without model loading."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    changed = []
    parameter_count = 0
    with safe_open(str(args.warmup), framework="pt", device="cpu") as before, safe_open(
        str(args.checkpoint), framework="pt", device="cpu"
    ) as after:
        names = sorted(before.keys())
        assert names == sorted(after.keys()), "Adapter tensor keys differ"
        for name in names:
            a, b = before.get_tensor(name), after.get_tensor(name)
            assert a.shape == b.shape and a.dtype == b.dtype, name
            parameter_count += a.numel()
            if not torch.equal(a, b):
                difference = b.to(torch.float64) - a.to(torch.float64)
                changed.append({"name": name, "changed_elements": int(torch.count_nonzero(difference)),
                                "max_abs_difference": float(difference.abs().max()),
                                "l2_difference": float(torch.linalg.vector_norm(difference))})
    result = {"device": "cpu", "warmup": str(args.warmup), "checkpoint": str(args.checkpoint),
              "tensor_count": len(names), "parameter_count": parameter_count,
              "identical_tensor_count": len(names) - len(changed), "changed_tensors": changed,
              "all_adapter_tensors_exactly_equal": not changed}
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
