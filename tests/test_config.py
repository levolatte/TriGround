from dataclasses import asdict
from pathlib import Path

from mm_grounding.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_released_configs_use_portable_backbone_identifier():
    for path in (ROOT / "configs").glob("*.yaml"):
        config = load_config(path)
        assert config.model.backbone == "Qwen/Qwen3-VL-2B-Instruct"


def test_rdt_release_route_keeps_backbone_frozen():
    initial = load_config(ROOT / "configs/multimodal_rdt_deep_reviewed.yaml")
    extension = load_config(
        ROOT / "configs/multimodal_rdt_deep_reviewed_extend_e5.yaml"
    )
    manual = load_config(ROOT / "configs/triground_rdt_ws_v1_manual_ft1.yaml")

    assert initial.model.fusion_type == extension.model.fusion_type == "rdt_deep"
    assert not initial.model.vision_lora_enabled
    assert not extension.model.vision_lora_enabled
    assert extension.train.resume_epoch == 4
    assert manual.train.initialization_checkpoints == [
        "runs/multimodal_rdt_deep_reviewed/best_phase_a.pt"
    ]
    assert manual.data.train_manifest.endswith("manual_split/train_100.json")


def test_parallel_release_route_is_staged_and_rgb_safe():
    ir = load_config(ROOT / "configs/stage1a_ir.yaml")
    depth = load_config(ROOT / "configs/stage1b_depth.yaml")
    calibration = load_config(ROOT / "configs/stage2_joint_calibration.yaml")
    fusion = load_config(ROOT / "configs/stage2_joint_fusion_v2.yaml")

    assert ir.stage == "ir"
    assert depth.stage == "depth"
    assert calibration.stage == fusion.stage == "joint"
    assert calibration.model.freeze_parallel_adapters
    assert len(calibration.train.initialization_checkpoints) == 2
    assert fusion.model.parallel_joint_fusion
    assert fusion.model.fusion_zero_init_prompt_restore
    assert fusion.train.initialization_checkpoints == [
        "runs/stage2_clean_after_weak1024/best_phase_a.pt"
    ]


def test_query_position_ab_configs_have_exactly_one_experimental_variable():
    control = asdict(
        load_config(ROOT / "configs/stage2_joint_fusion_v3_control.yaml")
    )
    positional = asdict(
        load_config(ROOT / "configs/stage2_joint_fusion_v3_positional.yaml")
    )
    assert control["model"]["query_position_encoding"] == "none"
    assert positional["model"]["query_position_encoding"] == "sinusoidal"
    assert control["train"]["seed"] == positional["train"]["seed"] == 2026
    assert control["train"]["initialization_checkpoints"] == positional["train"][
        "initialization_checkpoints"
    ]
    control.pop("output_dir")
    positional.pop("output_dir")
    control["model"].pop("query_position_encoding")
    positional["model"].pop("query_position_encoding")
    assert control == positional
