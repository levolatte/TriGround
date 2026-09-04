from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import torch


RELEASE_URL = (
    "https://github.com/levolatte/TriGround/releases/download/models-v1.0.0/"
    "triground-parallel-a-v1.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve one shared initialization checkpoint for the Query-position A/B"
    )
    parser.add_argument(
        "--pre-joint",
        type=Path,
        default=Path("runs/stage2_clean_after_weak1024/best_phase_a.pt"),
    )
    parser.add_argument(
        "--release-cache",
        type=Path,
        default=Path("runs/stage2_query_position_ab/triground-parallel-a-v1.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stage2_query_position_ab/shared_init.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/stage2_query_position_ab/shared_init.source.json"),
    )
    args = parser.parse_args()

    if args.pre_joint.exists():
        source = args.pre_joint
        source_kind = "pre_joint_retrain"
    else:
        source = args.release_cache
        source_kind = "release_continuation"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            temporary = source.with_suffix(source.suffix + ".download")
            urllib.request.urlretrieve(RELEASE_URL, temporary)
            temporary.replace(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != args.output.resolve():
        shutil.copy2(source, args.output)
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError(f"unsupported checkpoint: {args.output}")
    report = {
        "source_kind": source_kind,
        "source": str(source.resolve()),
        "shared_initialization": str(args.output.resolve()),
        "release_url": RELEASE_URL if source_kind == "release_continuation" else None,
        "epoch": checkpoint.get("epoch"),
        "score": checkpoint.get("score"),
        "metrics": checkpoint.get("metrics"),
        "model_tensors": len(model_state),
        "has_joint_stage_fusion": any(
            name.startswith("fusion.joint_stage_fusions.") for name in model_state
        ),
        "checkpoint_config": checkpoint.get("config"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
