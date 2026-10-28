"""CPU contracts for independent G2 candidates; no model loading or rollout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.quant_grounding_config import load_config, swift_arguments


CONFIGS = Path(__file__).resolve().parents[1] / "configs/quant_grounding"


@pytest.mark.parametrize("filename,reward", [
    ("arm_b_g2.json", "quant_bbox_acc05"),
    ("arm_b_g2_continuous_iou.json", "quant_bbox_acc05_continuous_iou"),
])
@pytest.mark.parametrize("micro_batch", [1, 2])
def test_g2_candidates_keep_four_questions_per_update(filename, reward, micro_batch) -> None:
    config = load_config(CONFIGS / filename)
    config["train_micro_batch_size"] = micro_batch
    args = swift_arguments(config, Path("prepared.jsonl"))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--num_generations"] == pairs["--generation_batch_size"] == "2"
    assert int(pairs["--per_device_train_batch_size"]) * int(pairs["--gradient_accumulation_steps"]) == 8
    assert pairs["--gradient_accumulation_steps"] == str(8 // micro_batch)
    assert pairs["--reward_funcs"] == reward
    assert pairs["--scale_rewards"] == "none"
    assert pairs["--loss_type"] == "dapo"
    assert pairs["--beta"] == "0.01"
    assert pairs["--temperature"] == pairs["--top_p"] == "1"
    assert pairs["--max_completion_length"] == "64"
    assert pairs["--max_steps"] == "550"
    assert config["matched_query_exposures"] == 4 * int(pairs["--max_steps"])
    assert config["max_gpu_hours"] is None and config["save_hours"] == []
    assert pairs["--adapters"] == pairs["--ref_adapters"]
    assert "--resume_from_checkpoint" not in args


def test_g2_candidates_differ_only_in_reward_and_output() -> None:
    original = json.loads((CONFIGS / "arm_b_g2.json").read_text(encoding="utf-8"))
    continuous = json.loads((CONFIGS / "arm_b_g2_continuous_iou.json").read_text(encoding="utf-8"))
    assert {key for key in original if original[key] != continuous[key]} == {"geometry_reward", "output_dir"}
    assert original["geometry_reward"] == "capped_iou"
    assert continuous["geometry_reward"] == "continuous_iou"
