from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ALLOWED_DIFFERENCES = {
    "output_dir",
    "model.query_position_encoding",
}


def differences(left, right, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                output.append(path)
            else:
                output.extend(differences(left[key], right[key], path))
        return output
    return [] if left == right else [prefix]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the Query-position A/B configs")
    parser.add_argument(
        "--control",
        type=Path,
        default=Path("configs/stage2_joint_fusion_v3_control.yaml"),
    )
    parser.add_argument(
        "--treatment",
        type=Path,
        default=Path("configs/stage2_joint_fusion_v3_positional.yaml"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    control = yaml.safe_load(args.control.read_text(encoding="utf-8"))
    treatment = yaml.safe_load(args.treatment.read_text(encoding="utf-8"))
    found = set(differences(control, treatment))
    if found != ALLOWED_DIFFERENCES:
        raise ValueError(
            f"unexpected A/B differences: found={sorted(found)}, "
            f"expected={sorted(ALLOWED_DIFFERENCES)}"
        )
    if control["model"]["query_position_encoding"] != "none":
        raise ValueError("control must use query_position_encoding=none")
    if treatment["model"]["query_position_encoding"] != "sinusoidal":
        raise ValueError("treatment must use query_position_encoding=sinusoidal")
    if control["train"]["seed"] != treatment["train"]["seed"]:
        raise ValueError("A/B seeds differ")
    if (
        control["train"]["initialization_checkpoints"]
        != treatment["train"]["initialization_checkpoints"]
    ):
        raise ValueError("A/B initialization checkpoints differ")
    report = {
        "valid": True,
        "allowed_differences": sorted(found),
        "seed": control["train"]["seed"],
        "initialization_checkpoints": control["train"][
            "initialization_checkpoints"
        ],
        "run_order": ["treatment", "control"],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
