from pathlib import Path

from mm_grounding.config import load_config


def test_joint_config_skips_phase_a_and_loads_probe_checkpoint():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_safe_joint_from_probe.yaml")
    assert config.train.phase_a_epochs == 0
    assert config.train.phase_b_epochs == config.train.epochs == 5
    assert config.train.init_checkpoint.endswith("probe_phase_a_step_0100.pt")
    assert not config.model.auxiliary_bbox_enabled


def test_rdt_deep_config_freezes_backbone_and_trains_prompts_only():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_rdt_deep.yaml")
    assert config.model.fusion_type == "rdt_deep"
    assert not config.model.vision_lora_enabled
    assert config.train.phase_a_epochs == config.train.epochs == 5
    assert config.train.phase_b_epochs == 0
    assert config.train.init_checkpoint is None


def test_rdt_deep_two_epoch_run_has_staged_checks_without_premature_stop():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_rdt_deep_2epoch.yaml")
    assert config.train.early_probe_step == 0
    assert config.train.early_probe_steps == [50, 100, 300]
    assert not config.train.stop_after_last_probe
    assert config.train.epochs == config.train.phase_a_epochs == 2


def test_rdt_deep_reviewed_run_uses_clean_validation_without_resume():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_rdt_deep_reviewed.yaml")
    assert config.data.train_manifest.endswith("grounding_final_train.json")
    assert config.data.val_manifest.endswith("grounding_final_val_reviewed.json")
    assert config.model.fusion_type == "rdt_deep"
    assert not config.model.auxiliary_bbox_enabled
    assert not config.model.vision_lora_enabled
    assert config.train.resume_epoch == 0
    assert config.train.early_probe_steps == [50, 100, 300]


def test_parallel_backbone_probe_runs_adapter_only_with_four_fusion_stages():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/multimodal_parallel_backbone_probe.yaml")
    assert config.model.fusion_type == "parallel_backbone"
    assert config.model.parallel_fusion_stages == 4
    assert not config.model.vision_lora_enabled
    assert config.train.phase_a_epochs == config.train.epochs == 1
    assert config.train.early_probe_steps == [50, 100, 300]


def test_new_stages_use_independent_auxiliary_paths_and_joint_checkpoint_merge():
    root = Path(__file__).resolve().parents[1]
    ir = load_config(root / "configs/stage1a_ir.yaml")
    depth = load_config(root / "configs/stage1b_depth.yaml")
    joint = load_config(root / "configs/stage2_joint_calibration.yaml")
    stage3 = load_config(root / "configs/stage3_city_finetune.yaml")
    assert ir.stage == "ir"
    assert depth.stage == "depth"
    assert joint.stage == stage3.stage == "joint"
    assert joint.model.freeze_parallel_adapters
    assert len(joint.train.initialization_checkpoints) == 2
    assert "city_detection_prepared" in stage3.data.train_manifest
