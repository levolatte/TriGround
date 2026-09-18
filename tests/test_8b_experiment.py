import json
from pathlib import Path

import yaml
import torch

from tools.prepare_8b_experiment import (
    OPTIONAL_SENTINEL,
    _fresh_config,
    materialize_control,
    materialize_optional,
    materialize_subset,
    set_max_pixels,
    stratified_indices,
)


def test_stratified_subset_is_deterministic_and_exact():
    rows = [
        (str(index), {"class_name": f"c{index % 3}", "scale_bin": f"s{index % 2}"})
        for index in range(900)
    ]
    first = stratified_indices(rows, 512, 2026)
    second = stratified_indices(rows, 512, 2026)
    assert first == second
    assert len(first) == len(set(first)) == 512


def test_materialized_external_subset_uses_absolute_image_paths(tmp_path):
    source = tmp_path / "source" / "val.jsonl"
    source.parent.mkdir()
    source.write_text(
        "".join(
            json.dumps({
                "id": str(index), "rgb": "rgb.png", "infrared": "ir.png",
                "query": "x", "bbox": [0, 0, 1, 1], "class_name": str(index % 2),
            }) + "\n"
            for index in range(600)
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "val512.jsonl"
    assert materialize_subset(source, destination) == 512
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 512
    assert all(Path(row["rgb"]).is_absolute() and Path(row["infrared"]).is_absolute() for row in rows)


def test_control_has_923_plus_fixed_69_rows(tmp_path):
    source = tmp_path / "clean.json"
    source.write_text(json.dumps({str(i): {"rgb": f"{i}.png"} for i in range(923)}))
    first = tmp_path / "control1.json"
    second = tmp_path / "control2.json"
    assert materialize_control(source, first) == 992
    assert materialize_control(source, second) == 992
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text())
    assert sum("__fixed_repeat_" in key for key in payload) == 69


def test_optional_candidate_is_materialized_in_both_first_stages(tmp_path):
    run = tmp_path / "run"
    configs = run / "configs"
    configs.mkdir(parents=True)
    source_raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen3_vl_8b_stage2_weak.yaml")
        .read_text(encoding="utf-8")
    )
    source_raw["model"]["query_position_encoding"] = "none"
    for name in ("weak", "control", "weak_clean", "control_clean"):
        raw = json.loads(json.dumps(source_raw))
        raw["train"]["initialization_checkpoints"] = (
            [OPTIONAL_SENTINEL] if name in {"weak", "control"} else ["previous.pt"]
        )
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    candidate = tmp_path / "candidate.pt"
    candidate_model = json.loads(json.dumps(source_raw["model"]))
    candidate_model["query_position_encoding"] = "sinusoidal"
    torch.save({"config": {"model": candidate_model}}, candidate)
    materialize_optional(run, candidate)
    for name in ("weak", "control", "weak_clean", "control_clean"):
        raw = yaml.safe_load((configs / f"{name}.yaml").read_text())
        assert raw["model"]["query_position_encoding"] == "sinusoidal"
        assert not raw["train"]["warm_start_joint_fusion_from_legacy"]
        if name in {"weak", "control"}:
            assert raw["train"]["initialization_checkpoints"] == [str(candidate.resolve())]


def test_oom_pixel_adjustment_updates_every_config_and_plan(tmp_path):
    run = tmp_path / "run"
    configs = run / "configs"
    configs.mkdir(parents=True)
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen3_vl_8b_stage1a_ir.yaml")
        .read_text(encoding="utf-8")
    )
    for name in ("a", "b"):
        (configs / f"{name}.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")
    (run / "plan.json").write_text(json.dumps({"max_pixels": 802816}), encoding="utf-8")
    set_max_pixels(run, 602112)
    assert all(
        yaml.safe_load(path.read_text())["data"]["max_pixels"] == 602112
        for path in configs.glob("*.yaml")
    )
    assert json.loads((run / "plan.json").read_text())["max_pixels"] == 602112


def test_c4_t4_complete_configs_have_only_three_intended_differences(tmp_path):
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/qwen3_vl_8b_stage2_joint.yaml")
        .read_text(encoding="utf-8")
    )
    common = dict(
        source=source,
        backbone=tmp_path / "model",
        train_manifest=tmp_path / "train.json",
        val_manifest=tmp_path / "val.json",
        epochs=4,
        lr=1e-5,
        seed=2026,
        max_pixels=802816,
        initialization=["ir.pt", "depth.pt"],
    )
    control = _fresh_config(
        output_dir=tmp_path / "c4", query_position="none", warm_start=False, **common
    )
    treatment = _fresh_config(
        output_dir=tmp_path / "t4", query_position="sinusoidal", warm_start=True, **common
    )

    def flatten(value, prefix=""):
        if not isinstance(value, dict):
            return {prefix: value}
        return {
            key: item
            for name, child in value.items()
            for key, item in flatten(child, f"{prefix}.{name}" if prefix else name).items()
        }

    left, right = flatten(control), flatten(treatment)
    differences = {key for key in left if left[key] != right[key]}
    assert differences == {
        "output_dir",
        "model.query_position_encoding",
        "train.warm_start_joint_fusion_from_legacy",
    }
