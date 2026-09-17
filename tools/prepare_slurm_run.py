"""Materialize the five 8B configs for one Slurm run without editing source YAMLs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from mm_grounding.config import load_config


STAGES = ("stage1a_ir", "stage1b_depth", "stage2_joint", "stage2_weak", "stage2_clean")
DEPENDENCIES = {
    "stage1a_ir": (), "stage1b_depth": (),
    "stage2_joint": ("stage1a_ir", "stage1b_depth"),
    "stage2_weak": ("stage2_joint",), "stage2_clean": ("stage2_weak",),
}


def prepare_configs(config_dir: Path, run_dir: Path, *, smoke=False, overrides=None):
    overrides = overrides or {}
    run_dir = run_dir.resolve()
    # A new run must not replace an earlier run's weights or metrics.
    run_dir.mkdir(parents=True, exist_ok=False)
    destination = run_dir / "configs"
    destination.mkdir()
    configs = {}
    for stage in STAGES:
        source = config_dir / f"qwen3_vl_8b_{stage}.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["output_dir"] = os.path.relpath(run_dir / stage)
        if overrides.get("BACKBONE"):
            raw["model"]["backbone"] = overrides["BACKBONE"]
        data = raw["data"]
        if stage == "stage1a_ir" and overrides.get("RGBT_ROOT"):
            root = Path(overrides["RGBT_ROOT"]).resolve()
            data["train_manifest"] = str(root / "extracted/subsets/train_50.jsonl")
            data["val_manifest"] = str(root / "extracted/manifests/val.jsonl")
        if stage == "stage1b_depth" and overrides.get("ROBOREFIT_ROOT"):
            root = Path(overrides["ROBOREFIT_ROOT"]).resolve()
            data["train_manifest"] = str(root / "manifests/formal_subsets/train_50.jsonl")
            data["val_manifest"] = str(root / "manifests/testA.jsonl")
        if stage.startswith("stage2") and overrides.get("CITY_ROOT"):
            root = Path(overrides["CITY_ROOT"]).resolve() / "train/target_v2"
            data["train_manifest"] = str(
                root / ("train_weak_scene_safe.json" if stage == "stage2_weak"
                        else "manual_split/train_100.json")
            )
            data["val_manifest"] = str(root / "manual_split/val.json")
        for key in ("train_manifest", "val_manifest"):
            data[key] = os.path.relpath(Path(data[key]).resolve())
            if not Path(data[key]).is_file():
                raise FileNotFoundError(f"{stage}: {key} does not exist: {data[key]}")
        raw["train"]["initialization_checkpoints"] = [] if smoke else [
            os.path.relpath(run_dir / parent / "best_phase_a.pt") for parent in DEPENDENCIES[stage]
        ]
        # These configs describe fresh jobs, never implicit training resumes.
        raw["train"]["init_checkpoint"] = None
        raw["train"]["resume_epoch"] = 0
        if smoke:
            raw["train"]["warm_start_joint_fusion_from_legacy"] = False
            data["workers"] = 0
            data["min_pixels"] = data["max_pixels"]
            raw["model"]["modality_dropout"] = 0.0
        path = destination / f"qwen3_vl_8b_{stage}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        load_config(path)
        configs[stage] = raw
    return configs


def audit_splits(configs, run_dir: Path, eval_manifest: Path | None):
    script = Path(__file__).with_name("audit_manifest_overlap.py")
    train_manifests = list(dict.fromkeys(
        configs[stage]["data"]["train_manifest"] for stage in STAGES
    ))
    validation = configs["stage2_clean"]["data"]["val_manifest"]
    targets = [("reviewed_val", validation, train_manifests)]
    for stage in STAGES[:2]:
        data = configs[stage]["data"]
        targets.append((stage + "_val", data["val_manifest"], [data["train_manifest"]]))
    if eval_manifest is not None and str(eval_manifest) != validation:
        targets.append(("heldout_eval", str(eval_manifest), train_manifests))
    for name, target, training in targets:
        command = [sys.executable, str(script), training[0], target]
        for path in training[1:]:
            command += ["--additional-train", path]
        command += ["--fail-on-overlap", "--output", str(run_dir / f"audit_{name}.json")]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configs = prepare_configs(args.config_dir, args.run_dir, smoke=args.smoke, overrides=os.environ)
    eval_manifest = Path(os.environ["EVAL_MANIFEST"]).resolve() if os.environ.get("EVAL_MANIFEST") else None
    if eval_manifest is not None and not eval_manifest.is_file():
        raise FileNotFoundError(f"EVAL_MANIFEST does not exist: {eval_manifest}")
    audit_splits(configs, args.run_dir, eval_manifest)
    plan = {
        "smoke": args.smoke, "stages": list(STAGES),
        "eval_manifest": str(eval_manifest) if eval_manifest else configs["stage2_clean"]["data"]["val_manifest"],
        "eval_scope": "explicit_manifest" if eval_manifest else "reviewed_validation",
        "source_config_dir": str(args.config_dir.resolve()),
    }
    (args.run_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
