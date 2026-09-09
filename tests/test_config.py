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


def test_qwen8b_recommended_configs_align_fusion_with_deepstack_and_disable_lora():
    root = Path(__file__).resolve().parents[1]
    names = (
        "qwen3_vl_8b_stage1a_ir.yaml",
        "qwen3_vl_8b_stage1b_depth.yaml",
        "qwen3_vl_8b_stage2_joint.yaml",
    )
    configs = [load_config(root / "configs" / name) for name in names]
    for config in configs:
        assert config.model.backbone == "Qwen/Qwen3-VL-8B-Instruct"
        assert config.model.dtype == "bfloat16"
        assert config.model.adapter_channels == 256
        assert config.model.fusion_dim == 512
        assert config.model.fusion_attention_heads == 8
        assert config.model.fusion_mixer_dim == 2048
        assert config.model.query_dim == 128
        assert config.model.query_attention_heads == 4
        assert config.model.parallel_fusion_layer_indices == [8, 16, 24, 26]
        assert not config.model.vision_lora_enabled
        assert config.train.phase_b_epochs == 0
    assert not configs[0].model.parallel_joint_fusion
    assert not configs[1].model.parallel_joint_fusion
    assert configs[2].model.parallel_joint_fusion
    assert configs[2].model.freeze_parallel_adapters


def test_joint_fusion_v2_is_rgb_safe_and_loads_existing_target_checkpoint():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/stage2_joint_fusion_v2.yaml")
    assert config.stage == "joint"
    assert config.model.parallel_joint_fusion
    assert config.model.freeze_parallel_adapters
    assert config.model.fusion_zero_init_prompt_restore
    assert config.train.initialization_checkpoints == [
        "runs/stage2_clean_after_weak1024/best_phase_a.pt"
    ]


def test_target_v2_probes_observe_without_aborting():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "stage2_joint_fusion_target_v2_weak.yaml",
        "stage2_joint_fusion_target_v2_manual.yaml",
        "stage2_joint_unfreeze_last2_target_v2.yaml",
    ):
        config = load_config(root / "configs" / name)
        assert config.train.early_probe_steps[0] == 25
        assert config.train.early_probe_abort_ratio == 0.0
        assert config.train.early_probe_abort_from_step == 50
    weak = load_config(root / "configs" / "stage2_joint_fusion_target_v2_weak.yaml")
    assert not weak.train.stop_after_last_probe


def test_target_v2_fusion_extension_is_manual_only_then_unfreezes_last_two():
    root = Path(__file__).resolve().parents[1]
    fusion = load_config(
        root / "configs" / "stage2_joint_fusion_target_v2_manual_extend2.yaml"
    )
    unfreeze = load_config(
        root / "configs" / "stage2_joint_unfreeze_last2_target_v2_after_extend.yaml"
    )
    assert fusion.train.epochs == fusion.train.phase_a_epochs == 2
    assert fusion.model.freeze_parallel_adapters
    assert "manual_split/train_100.json" in fusion.data.train_manifest
    assert "weak" not in fusion.data.train_manifest
    assert unfreeze.model.parallel_adapter_train_last_n == 2
    assert unfreeze.train.initialization_checkpoints == [
        "runs/stage2_joint_fusion_target_v2_manual_extend2/selected_best_phase_a.pt"
    ]


def test_joint_fusion_projection_warm_start_is_an_isolated_ablation():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/stage2_joint_fusion_v2_warm_projection.yaml")
    assert config.model.parallel_joint_fusion
    assert config.train.warm_start_joint_fusion_from_legacy
    assert config.train.epochs == 2
    assert config.train.early_stopping_min_delta == 0.0


def test_target_v2_unfreeze_uses_manual_data_and_separate_adapter_lr():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/stage2_joint_unfreeze_last2_target_v2.yaml")
    assert config.model.parallel_adapter_train_last_n == 2
    assert not config.model.freeze_parallel_adapters
    assert config.train.parallel_adapter_lr == 3e-7
    assert "manual_split" in config.data.train_manifest
    assert "weak" not in config.data.train_manifest
