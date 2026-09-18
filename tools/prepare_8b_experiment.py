"""Prepare the fixed 8B tri-modal experiment without touching source manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from mm_grounding.config import load_config


SEED = 2026
OPTIONAL_SENTINEL = "__SELECTED_CANDIDATE_CHECKPOINT__"
IMAGE_FIELDS = ("rgb", "visible", "ir", "infrared", "thermal", "depth", "aux")


def _read_manifest(path: Path) -> tuple[list[tuple[str, dict]], bool]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"JSON manifest must be object-keyed: {path}")
        return [(str(key), value) for key, value in payload.items()], True
    return [
        (str(record.get("id", index)), record)
        for index, line in enumerate(text.splitlines())
        if line.strip()
        for record in [json.loads(line)]
    ], False


def _absolute_record(record: dict, source: Path) -> dict:
    copied = dict(record)
    for field in IMAGE_FIELDS:
        value = copied.get(field)
        if value:
            path = Path(str(value))
            copied[field] = str(path if path.is_absolute() else (source.parent / path).resolve())
    return copied


def _write_jsonl(path: Path, rows: list[tuple[str, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, record in rows:
            copied = dict(record)
            copied["id"] = sample_id
            handle.write(json.dumps(copied, ensure_ascii=False) + "\n")


def stratified_indices(rows: list[tuple[str, dict]], limit: int, seed: int) -> list[int]:
    if limit >= len(rows):
        return list(range(len(rows)))
    groups: dict[tuple[str, str], list[int]] = {}
    for index, (_, record) in enumerate(rows):
        key = (str(record.get("class_name", "")), str(record.get("scale_bin", "")))
        groups.setdefault(key, []).append(index)
    generator = torch.Generator().manual_seed(seed)
    for indices in groups.values():
        order = torch.randperm(len(indices), generator=generator).tolist()
        indices[:] = [indices[index] for index in order]
    exact = {key: len(indices) * limit / len(rows) for key, indices in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = limit - sum(quotas.values())
    priority = sorted(
        groups,
        key=lambda key: (exact[key] - quotas[key], len(groups[key])),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    selected: list[int] = []
    for key in sorted(groups):
        selected.extend(groups[key][: quotas[key]])
    return selected


def materialize_subset(source: Path, destination: Path, limit: int = 512) -> int:
    rows, _ = _read_manifest(source)
    selected = stratified_indices(rows, limit, SEED)
    absolute = [(rows[index][0], _absolute_record(rows[index][1], source)) for index in selected]
    _write_jsonl(destination, absolute)
    return len(absolute)


def materialize_control(clean_source: Path, destination: Path) -> int:
    rows, object_keyed = _read_manifest(clean_source)
    if not object_keyed or len(rows) != 923:
        raise ValueError(f"expected object-keyed 923-row reviewed training set: {clean_source}")
    generator = torch.Generator().manual_seed(SEED)
    sampled = torch.randperm(len(rows), generator=generator)[:69].tolist()
    payload = {}
    for sample_id, record in rows:
        copied = _absolute_record(record, clean_source)
        copied.pop("id", None)
        payload[sample_id] = copied
    for repeat_index, row_index in enumerate(sampled):
        sample_id, record = rows[row_index]
        copied = _absolute_record(record, clean_source)
        copied.pop("id", None)
        payload[f"{sample_id}__fixed_repeat_{repeat_index:02d}"] = copied
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(payload)


def _required_fields(stage: str) -> tuple[tuple[str, ...], ...]:
    fields: list[tuple[str, ...]] = [("rgb", "visible")]
    if stage in {"ir", "joint"}:
        fields.append(("ir", "infrared", "thermal", "aux"))
    if stage in {"depth", "joint"}:
        fields.append(("depth", "aux"))
    return tuple(fields)


def audit_paths(path: Path, stage: str) -> dict:
    rows, _ = _read_manifest(path)
    missing: list[str] = []
    for sample_id, record in rows:
        for aliases in _required_fields(stage):
            value = next((record.get(field) for field in aliases if record.get(field)), None)
            if aliases[-1] == "aux" and value == record.get("aux"):
                required_aux_type = (
                    "ir" if any(field in aliases for field in ("ir", "infrared", "thermal"))
                    else "depth"
                )
                if str(record.get("aux_type", "")).lower() != required_aux_type:
                    value = None
            if value is None:
                missing.append(f"{sample_id}:missing:{'/'.join(aliases)}")
                continue
            candidate = Path(str(value))
            candidate = candidate if candidate.is_absolute() else path.parent / candidate
            if not candidate.is_file():
                missing.append(f"{sample_id}:{candidate}")
    return {"manifest": str(path), "stage": stage, "samples": len(rows), "missing": missing}


def _source_config(config_dir: Path, name: str) -> dict:
    return yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))


def _fresh_config(
    source: dict,
    *,
    output_dir: Path,
    backbone: Path,
    train_manifest: Path,
    val_manifest: Path,
    epochs: int,
    lr: float,
    seed: int,
    max_pixels: int,
    initialization: list[str] | None = None,
    query_position: str = "none",
    warm_start: bool = False,
) -> dict:
    raw = json.loads(json.dumps(source))
    raw["output_dir"] = str(output_dir.resolve())
    raw["model"]["backbone"] = str(backbone.resolve())
    raw["model"]["query_position_encoding"] = query_position
    raw["data"]["train_manifest"] = str(train_manifest.resolve())
    raw["data"]["val_manifest"] = str(val_manifest.resolve())
    raw["data"]["max_pixels"] = max_pixels
    raw["train"].update(
        {
            "epochs": epochs,
            "phase_a_epochs": epochs,
            "phase_b_epochs": 0,
            "fusion_lr_a": lr,
            "batch_size": 1,
            "grad_accumulation": 16,
            "seed": seed,
            "compact_checkpoints": True,
            "initialization_checkpoints": initialization or [],
            "init_checkpoint": None,
            "resume_epoch": 0,
            "warm_start_joint_fusion_from_legacy": warm_start,
        }
    )
    raw["train"].pop("early_probe_step", None)
    raw["train"].pop("early_probe_steps", None)
    raw["train"]["stop_after_last_probe"] = False
    if raw["data"]["stage"] == "joint":
        raw["train"]["eval_subset_size"] = 119
        raw["train"]["early_stopping_patience"] = 5
    else:
        raw["train"]["eval_subset_size"] = 512
    return raw


def _write_config(path: Path, raw: dict, validate: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    if validate:
        load_config(path)


def _audit_overlap(script: Path, train: Path, val: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(train),
            str(val),
            "--fail-on-overlap",
            "--output",
            str(output),
        ],
        check=True,
    )


def prepare(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    config_dir = args.config_dir.resolve()
    manifests = run_dir / "manifests"
    configs = run_dir / "configs"
    outputs = run_dir / "outputs"
    results = run_dir / "results"
    for directory in (manifests, configs, outputs, results):
        directory.mkdir()

    rgbt_train = args.rgbt_root.resolve() / "extracted/subsets/train_50.jsonl"
    rgbt_val_source = args.rgbt_root.resolve() / "extracted/manifests/val.jsonl"
    depth_train = args.roborefit_root.resolve() / "manifests/formal_subsets/train_50.jsonl"
    depth_val_source = args.roborefit_root.resolve() / "manifests/testA.jsonl"
    target = args.city_root.resolve() / "train/target_v2"
    clean_train = target / "manual_split/train_100.json"
    target_val = target / "manual_split/val.json"
    weak_train = target / "train_weak_scene_safe.json"
    expected = {
        rgbt_train: 13302,
        depth_train: 18458,
        clean_train: 923,
        target_val: 119,
        weak_train: 992,
    }
    for path, count in expected.items():
        rows, _ = _read_manifest(path)
        if len(rows) != count:
            raise ValueError(f"unexpected sample count for {path}: {len(rows)} != {count}")

    ir_val = manifests / "rgbt_val_512_seed2026.jsonl"
    depth_val = manifests / "depth_val_512_seed2026.jsonl"
    control_train = manifests / "reviewed_control_992_seed2026.json"
    materialize_subset(rgbt_val_source, ir_val)
    materialize_subset(depth_val_source, depth_val)
    materialize_control(clean_train, control_train)

    path_audits = [
        audit_paths(rgbt_train, "ir"),
        audit_paths(ir_val, "ir"),
        audit_paths(depth_train, "depth"),
        audit_paths(depth_val, "depth"),
    ]
    missing = [item for audit in path_audits for item in audit["missing"]]
    (run_dir / "external_path_audit.json").write_text(
        json.dumps(path_audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if missing:
        raise FileNotFoundError(f"external manifest path audit found {len(missing)} missing files")
    overlap_script = Path(__file__).with_name("audit_manifest_overlap.py")
    _audit_overlap(overlap_script, rgbt_train, ir_val, run_dir / "audit_ir_val.json")
    _audit_overlap(overlap_script, depth_train, depth_val, run_dir / "audit_depth_val.json")
    _audit_overlap(
        overlap_script, rgbt_train, target_val, run_dir / "audit_ir_target_val.json"
    )
    _audit_overlap(
        overlap_script, depth_train, target_val, run_dir / "audit_depth_target_val.json"
    )

    ir_source = _source_config(config_dir, "qwen3_vl_8b_stage1a_ir.yaml")
    depth_source = _source_config(config_dir, "qwen3_vl_8b_stage1b_depth.yaml")
    joint_source = _source_config(config_dir, "qwen3_vl_8b_stage2_joint.yaml")
    stage1_ir = _fresh_config(
        ir_source, output_dir=outputs / "stage1_ir", backbone=args.backbone,
        train_manifest=rgbt_train, val_manifest=ir_val, epochs=1, lr=3e-5,
        seed=SEED, max_pixels=args.max_pixels,
    )
    stage1_depth = _fresh_config(
        depth_source, output_dir=outputs / "stage1_depth", backbone=args.backbone,
        train_manifest=depth_train, val_manifest=depth_val, epochs=1, lr=3e-5,
        seed=SEED, max_pixels=args.max_pixels,
    )
    _write_config(configs / "stage1_ir.yaml", stage1_ir)
    _write_config(configs / "stage1_depth.yaml", stage1_depth)
    shared = [
        str((outputs / "stage1_ir/best_phase_a.pt").resolve()),
        str((outputs / "stage1_depth/best_phase_a.pt").resolve()),
    ]
    native = _fresh_config(
        joint_source, output_dir=outputs / "native_rgb", backbone=args.backbone,
        train_manifest=clean_train, val_manifest=target_val, epochs=1, lr=1e-5,
        seed=SEED, max_pixels=args.max_pixels,
    )
    _write_config(configs / "native_rgb.yaml", native)

    generated: dict[str, Path] = {}
    for name, position, warm, seed in (
        ("c4", "none", False, 2026),
        ("t4", "sinusoidal", True, 2026),
        ("w4", "none", True, 2026),
        ("c4_seed2027", "none", False, 2027),
        ("t4_seed2027", "sinusoidal", True, 2027),
        ("w4_seed2027", "none", True, 2027),
    ):
        raw = _fresh_config(
            joint_source, output_dir=outputs / name, backbone=args.backbone,
            train_manifest=clean_train, val_manifest=target_val, epochs=4, lr=1e-5,
            seed=seed, max_pixels=args.max_pixels, initialization=shared,
            query_position=position, warm_start=warm,
        )
        path = configs / f"{name}.yaml"
        _write_config(path, raw)
        generated[name] = path

    weak_source = _source_config(config_dir, "qwen3_vl_8b_stage2_weak.yaml")
    clean_source = _source_config(config_dir, "qwen3_vl_8b_stage2_clean.yaml")
    optional_specs = (
        ("weak", weak_source, weak_train, 1e-5, OPTIONAL_SENTINEL),
        ("weak_clean", clean_source, clean_train, 3e-6, str((outputs / "weak/best_phase_a.pt").resolve())),
        ("control", weak_source, control_train, 1e-5, OPTIONAL_SENTINEL),
        ("control_clean", clean_source, clean_train, 3e-6, str((outputs / "control/best_phase_a.pt").resolve())),
    )
    for name, source, train_manifest, lr, initialization in optional_specs:
        raw = _fresh_config(
            source, output_dir=outputs / name, backbone=args.backbone,
            train_manifest=train_manifest, val_manifest=target_val, epochs=1, lr=lr,
            seed=SEED, max_pixels=args.max_pixels, initialization=[initialization],
        )
        _write_config(configs / f"{name}.yaml", raw, validate=OPTIONAL_SENTINEL not in initialization)

    c4 = yaml.safe_load((configs / "c4.yaml").read_text(encoding="utf-8"))
    t4 = yaml.safe_load((configs / "t4.yaml").read_text(encoding="utf-8"))
    differing = []
    for section in ("model", "data", "train"):
        keys = set(c4[section]) | set(t4[section])
        differing.extend(f"{section}.{key}" for key in keys if c4[section].get(key) != t4[section].get(key))
    if set(differing) != {
        "model.query_position_encoding", "train.warm_start_joint_fusion_from_legacy"
    }:
        raise AssertionError(f"unexpected C4/T4 scientific differences: {differing}")

    plan = {
        "schema_version": 1,
        "seed": SEED,
        "max_pixels": args.max_pixels,
        "hard_budget_hours": 48,
        "training_cutoff_hours": 44,
        "minimum_free_gib": 3,
        "manifests": {
            "ir_train": str(rgbt_train), "ir_val_512": str(ir_val),
            "depth_train": str(depth_train), "depth_val_512": str(depth_val),
            "reviewed_train": str(clean_train), "reviewed_val": str(target_val),
            "weak_train": str(weak_train), "control_train_992": str(control_train),
        },
        "counts": {"ir_train": 13302, "ir_val": 512, "depth_train": 18458,
                   "depth_val": 512, "reviewed_train": 923, "reviewed_val": 119,
                   "weak_train": 992, "control_train": 992},
        "mandatory": ["pressure", "native_rgb", "stage1_ir", "stage1_depth", "c4", "t4"],
        "optional": ["w4", "c4_seed2027", "t4_seed2027", "w4_seed2027",
                     "weak", "weak_clean", "control", "control_clean"],
        "optional_candidate_sentinel": OPTIONAL_SENTINEL,
        "c4_t4_differences": sorted(differing),
    }
    (run_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan


def materialize_optional(run_dir: Path, candidate: Path) -> None:
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
    candidate_config = checkpoint.get("config")
    if not isinstance(candidate_config, dict) or not isinstance(candidate_config.get("model"), dict):
        raise ValueError(f"candidate checkpoint does not contain a model config: {candidate}")
    candidate_model = candidate_config["model"]
    for name in ("weak", "control", "weak_clean", "control_clean"):
        path = run_dir.resolve() / "configs" / f"{name}.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["model"] = json.loads(json.dumps(candidate_model))
        raw["train"]["warm_start_joint_fusion_from_legacy"] = False
        if name in {"weak", "control"}:
            values = raw["train"]["initialization_checkpoints"]
            if values != [OPTIONAL_SENTINEL]:
                raise ValueError(f"optional config already materialized: {path}")
            raw["train"]["initialization_checkpoints"] = [str(candidate.resolve())]
        _write_config(path, raw)


def set_max_pixels(run_dir: Path, max_pixels: int) -> None:
    run_dir = run_dir.resolve()
    for path in sorted((run_dir / "configs").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw["data"]["max_pixels"] = max_pixels
        _write_config(path, raw, validate=OPTIONAL_SENTINEL not in str(raw))
    plan_path = run_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["max_pixels"] = max_pixels
    plan["max_pixels_adjustment"] = "pressure_test_oom"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--rgbt-root", type=Path)
    parser.add_argument("--roborefit-root", type=Path)
    parser.add_argument("--city-root", type=Path)
    parser.add_argument("--max-pixels", type=int, default=802816)
    parser.add_argument("--materialize-optional-candidate", type=Path)
    parser.add_argument("--set-max-pixels", type=int)
    args = parser.parse_args()
    if args.set_max_pixels is not None:
        set_max_pixels(args.run_dir, args.set_max_pixels)
        print(json.dumps({"max_pixels": args.set_max_pixels, "run_dir": str(args.run_dir.resolve())}))
        return
    if args.materialize_optional_candidate:
        materialize_optional(args.run_dir, args.materialize_optional_candidate.resolve())
        print(json.dumps({"optional_candidate": str(args.materialize_optional_candidate.resolve())}))
        return
    required = (args.backbone, args.rgbt_root, args.roborefit_root, args.city_root)
    if any(value is None for value in required):
        parser.error("preparation requires --backbone, --rgbt-root, --roborefit-root and --city-root")
    plan = prepare(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
