from pathlib import Path
import json
import subprocess

import pytest
import yaml

from tools.prepare_slurm_run import STAGES, audit_splits, prepare_configs


def make_data_roots(tmp_path):
    paths = {
        "RGBT_ROOT": ("rgbt", ["extracted/subsets/train_50.jsonl", "extracted/manifests/val.jsonl"]),
        "ROBOREFIT_ROOT": ("robo", ["manifests/formal_subsets/train_50.jsonl", "manifests/testA.jsonl"]),
        "CITY_ROOT": ("city", ["train/target_v2/manual_split/train_100.json",
                              "train/target_v2/manual_split/val.json",
                              "train/target_v2/train_weak_scene_safe.json"]),
    }
    overrides = {"BACKBONE": "/models/Qwen3-VL-8B-Instruct"}
    for variable, (name, files) in paths.items():
        root = tmp_path / name
        for filename in files:
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        overrides[variable] = str(root)
    return overrides


def test_formal_configs_keep_hyperparameters_and_isolate_checkpoint_chain(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs"
    original = {stage: (source / f"qwen3_vl_8b_{stage}.yaml").read_bytes() for stage in STAGES}
    run = tmp_path / "run with spaces"
    configs = prepare_configs(source, run, overrides=make_data_roots(tmp_path))
    for stage, config in configs.items():
        previous = yaml.safe_load(original[stage])
        assert config["train"]["epochs"] == previous["train"]["epochs"]
        assert config["train"]["grad_accumulation"] == 16
        assert config["model"]["parallel_fusion_layer_indices"] == [8, 16, 24, 26]
        assert config["output_dir"] == str(run / stage)
        assert config["data"]["max_pixels"] == previous["data"]["max_pixels"]
        assert (source / f"qwen3_vl_8b_{stage}.yaml").read_bytes() == original[stage]
    assert configs["stage2_joint"]["train"]["initialization_checkpoints"] == [
        str(run / stage / "best_phase_a.pt") for stage in STAGES[:2]
    ]
    assert configs["stage2_clean"]["train"]["initialization_checkpoints"] == [
        str(run / "stage2_weak/best_phase_a.pt")
    ]


def test_smoke_has_no_trained_checkpoint_dependency_and_uses_max_budget(tmp_path):
    configs = prepare_configs(Path(__file__).resolve().parents[1] / "configs", tmp_path / "smoke",
                              smoke=True, overrides=make_data_roots(tmp_path))
    for config in configs.values():
        assert config["train"]["initialization_checkpoints"] == []
        assert config["train"]["init_checkpoint"] is None
        assert config["data"]["min_pixels"] == config["data"]["max_pixels"] == 802816
        assert config["model"]["modality_dropout"] == 0


def test_existing_run_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        prepare_configs(Path("configs"), tmp_path)


def test_missing_manifest_stops_before_gpu_work(tmp_path):
    overrides = make_data_roots(tmp_path)
    (Path(overrides["RGBT_ROOT"]) / "extracted/subsets/train_50.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="stage1a_ir"):
        prepare_configs(Path(__file__).resolve().parents[1] / "configs", tmp_path / "missing",
                        overrides=overrides)


@pytest.mark.parametrize("overlap", [False, True])
def test_audit_includes_weak_training_and_explicit_holdout(tmp_path, overlap):
    def manifest(name, scene):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({name: {"rgb": f"{name}.png", "scene_id": scene}}))
        return str(path)

    validation = manifest("validation", "val_scene")
    holdout = Path(manifest("holdout", "holdout_scene"))
    configs = {}
    for stage in STAGES:
        scene = "val_scene" if overlap and stage == "stage2_weak" else stage
        configs[stage] = {"data": {
            "train_manifest": manifest(stage, scene),
            "val_manifest": (manifest(stage + "_val", stage + "_val")
                             if stage in STAGES[:2] else validation),
        }}
    if overlap:
        with pytest.raises(subprocess.CalledProcessError):
            audit_splits(configs, tmp_path, holdout)
        report = json.loads((tmp_path / "audit_reviewed_val.json").read_text())
        assert report["shared_sequence_count"] == 1
    else:
        audit_splits(configs, tmp_path, holdout)
        report = json.loads((tmp_path / "audit_heldout_eval.json").read_text())
        assert len(report["train_manifests"]) == 5
        assert report["shared_sequence_count"] == 0
