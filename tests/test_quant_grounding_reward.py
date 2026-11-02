from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.quant_grounding_reward import (  # noqa: E402
    grounding_rewards,
    parse_short_bbox,
    parse_thinking_bbox,
    score_completion,
    score_completions,
)
from tools.verify_acc05_dev import iou_xyxy  # noqa: E402


FULL = [0.0, 0.0, 1.0, 1.0]


def answer(box: list[int]) -> str:
    return '{"bbox_2d":[' + ",".join(map(str, box)) + "]}"


@pytest.mark.parametrize("reward_mode", ["acc05", "contrast"])
@pytest.mark.parametrize("width", [0, 400, 500, 750, 1000])
def test_default_reward_keeps_original_capped_formula(reward_mode, width) -> None:
    completion = answer([0, 0, width, 1000]) if width else "invalid"
    kwargs = {"reward_mode": reward_mode, "negative_boxes": [[0, 0, 0.2, 1]]}
    original = score_completion(completion, FULL, **kwargs)
    explicit = score_completion(completion, FULL, iou_reward_mode="capped", **kwargs)
    assert original == explicit
    if width:
        expected = float(width >= 500) + 0.2 * min(width / 1000, 0.5)
        if reward_mode == "contrast":
            expected += 0.1 * original["contrast"]
        assert original["reward"] == pytest.approx(expected)
    else:
        assert original["reward"] == -0.2


def test_continuous_iou_distinguishes_hits_and_preserves_hit_priority() -> None:
    completions = [answer([0, 0, width, 1000]) for width in (499, 500, 750, 1000)]
    records = score_completions(completions, FULL, reward_mode="acc05",
                                iou_reward_mode="continuous")
    assert [item["hit_reward"] for item in records] == [0, 1, 1, 1]
    assert [item["iou_reward"] for item in records] == pytest.approx([0.0998, 0.1, 0.15, 0.2])
    assert [item["reward"] for item in records] == pytest.approx([0.0998, 1.1, 1.15, 1.2])
    assert grounding_rewards(completions, FULL, reward_mode="acc05",
                             iou_reward_mode="continuous") == [item["reward"] for item in records]
    capped = grounding_rewards(completions, FULL, reward_mode="acc05")
    assert capped == pytest.approx([0.0998, 1.1, 1.1, 1.1])


@pytest.mark.parametrize("completion,reason", [("invalid", "stop"),
                                               (answer([0, 0, 1000, 1000]), "length")])
def test_continuous_iou_keeps_invalid_and_truncated_penalties(completion, reason) -> None:
    record = score_completion(completion, FULL, reward_mode="acc05", iou_reward_mode="continuous",
                              finish_reason=reason, require_termination=True)
    assert record["valid"] is False
    assert record["reward"] == -0.2
    assert record["hit_reward"] == record["iou_reward"] == record["contrast_reward"] == 0


def test_unknown_iou_reward_mode_fails_fast() -> None:
    with pytest.raises(ValueError, match="unknown IoU reward mode"):
        score_completion(answer([0, 0, 1000, 1000]), FULL, iou_reward_mode="typo")


def test_half_iou_is_a_hit_and_matches_independent_metric_geometry() -> None:
    completion = answer([0, 0, 500, 1000])
    scored = score_completion(completion, FULL)
    expected_iou = iou_xyxy([0.0, 0.0, 0.5, 1.0], FULL)
    assert expected_iou == 0.5
    assert scored["iou"] == expected_iou
    assert scored["hit"] is True
    assert scored["reward"] == pytest.approx(1.1)


def test_success_reward_range_with_reviewed_negatives() -> None:
    half = answer([0, 0, 500, 1000])
    contrast_minimum = score_completion(half, FULL, [[0, 0, 0.5, 1.0]])
    contrast_maximum = score_completion(
        answer([0, 0, 1000, 1000]), FULL, [[0, 0, 0.2, 1.0]]
    )
    assert contrast_minimum["contrast"] == -1.0
    assert contrast_minimum["reward"] == pytest.approx(1.0)
    assert contrast_maximum["contrast"] == 1.0
    assert contrast_maximum["reward"] == pytest.approx(1.2)
    assert contrast_minimum["reward"] == pytest.approx(1.0)
    assert contrast_maximum["reward"] == pytest.approx(1.2)


def test_ground_truth_prediction_scores_above_reviewed_wrong_instance() -> None:
    target = [0.0, 0.0, 0.5, 1.0]
    negative = [0.5, 0.0, 1.0, 1.0]
    correct = score_completion(answer([0, 0, 500, 1000]), target, [negative])
    wrong = score_completion(answer([500, 0, 1000, 1000]), target, [negative])
    assert correct["reward"] > wrong["reward"]
    assert correct["reward"] == pytest.approx(1.2)
    assert wrong["reward"] == pytest.approx(-0.1)


def test_any_hit_beats_any_miss_even_when_contrast_favors_the_miss() -> None:
    hit = score_completion(answer([0, 0, 500, 1000]), FULL, [[0.8, 0.0, 1.0, 1.0]])
    miss = score_completion(answer([0, 0, 499, 1000]), FULL, [[0.8, 0.0, 1.0, 1.0]])
    assert hit["hit"] is True and miss["hit"] is False
    assert hit["reward"] > miss["reward"]
    assert miss["reward"] <= 0.2


def test_no_negative_boxes_gives_zero_contrast() -> None:
    scored = score_completion(answer([0, 0, 500, 1000]), FULL)
    assert scored["contrast"] == 0.0
    assert scored["max_negative_iou"] is None
    assert scored["reward"] == pytest.approx(1.1)


def test_removing_constant_contrast_shift_preserves_group_advantages() -> None:
    completions = [answer([0, 0, 500, 1000]), answer([0, 0, 400, 1000])]
    negatives = [
        [[0.0, 0.0, 0.5, 1.0]],
        [[0.0, 0.0, 0.4, 1.0]],
    ]
    contrast = score_completions(completions, [FULL, FULL], negatives, reward_mode="contrast")
    acc05 = score_completions(completions, [FULL, FULL], negatives, reward_mode="acc05")
    contrast_rewards = [item["reward"] for item in contrast]
    acc05_rewards = [item["reward"] for item in acc05]
    assert [item["contrast"] for item in contrast] == [-1.0, -1.0]
    assert [a - b for a, b in zip(contrast_rewards, acc05_rewards)] == pytest.approx([-0.1, -0.1])
    assert [value - sum(contrast_rewards) / 2 for value in contrast_rewards] == pytest.approx(
        [value - sum(acc05_rewards) / 2 for value in acc05_rewards]
    )


@pytest.mark.parametrize(
    "completion",
    [
        "not json",
        '{"bbox_2d":[0,0,1000,1000]} trailing',
        '{"bbox_2d":[0,0,1000,1000]}\n{"bbox_2d":[1,1,999,999]}',
        '{"bbox_2d":[0,0,1000,1000],"bbox_2d":[0,0,500,500]}',
        '{"bbox_2d":[true,0,1000,1000]}',
        '{"bbox_2d":[NaN,0,1000,1000]}',
        '{"bbox_2d":[Infinity,0,1000,1000]}',
        '{"bbox_2d":[0,0,1e999,1000]}',
        '{"bbox_2d":[-1,0,1000,1000]}',
        '{"bbox_2d":[0,0,1001,1000]}',
        '{"bbox_2d":[10,0,10,1000]}',
        '{"bbox_2d":[[0,0,500,1000],[500,0,1000,1000]]}',
        '{"bbox_2d":[0,0,1000,1000],"extra":1}',
    ],
)
def test_invalid_or_ambiguous_json_gets_invalid_reward(completion: str) -> None:
    assert parse_short_bbox(completion) is None
    scored = score_completion(completion, FULL)
    assert scored["valid"] is False
    assert scored["reward"] == -0.2


def test_short_mode_rejects_qwen_thinking_wrapper() -> None:
    text = '<think>inspect</think>\n\n{"bbox_2d":[0,0,1000,1000]}'
    assert parse_short_bbox(text) is None
    assert score_completion(text, FULL)["reward"] == -0.2


def test_thinking_mode_reads_only_one_final_json_and_accepts_prompt_open_prefix() -> None:
    final = answer([0, 0, 500, 1000])
    prompted_prefix_completion = f"Inspect the image.\n</think>\n\n{final}<|im_end|>"
    explicit_prefix_completion = f"<think>Inspect the image.</think>\n\n{final}"
    assert parse_thinking_bbox(prompted_prefix_completion) == [0.0, 0.0, 0.5, 1.0]
    assert parse_thinking_bbox(explicit_prefix_completion) == [0.0, 0.0, 0.5, 1.0]
    scored = score_completion(
        prompted_prefix_completion, FULL, answer_mode="thinking", finish_reason="stop"
    )
    assert scored["valid"] is True
    assert scored["reward"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    "completion",
    [
        "Inspect the image. {\"bbox_2d\":[0,0,1000,1000]}",  # no think end marker
        "Inspect the image.</think>{\"bbox_2d\":[0,0,500,1000]",  # truncated JSON
        "Inspect.</think>{\"bbox_2d\":[0,0,1000,1000]} {\"bbox_2d\":[0,0,500,500]}",
        "Inspect.</think>{\"bbox_2d\":[0,0,1000,1000]} trailing",
        "<think>Inspect</think> answer <think>again</think> {\"bbox_2d\":[0,0,1000,1000]}",
        "Inspect</think><|im_end|>{\"bbox_2d\":[0,0,1000,1000]}",
    ],
)
def test_thinking_requires_one_closed_reasoning_segment_and_one_final_json(completion: str) -> None:
    assert parse_thinking_bbox(completion) is None
    assert score_completion(completion, FULL, answer_mode="thinking")["reward"] == -0.2


def test_swift_length_finish_reason_invalidates_even_closed_json() -> None:
    scored = score_completion(
        f"Inspect.</think>{answer([0, 0, 1000, 1000])}",
        FULL,
        answer_mode="thinking",
        finish_reason="length",
    )
    assert scored["format_valid"] is True
    assert scored["valid"] is False
    assert scored["truncated"] is True
    assert scored["invalid_reason"] == "truncated"
    assert scored["reward"] == -0.2


def test_swift_missing_finish_reason_requires_explicit_message_end() -> None:
    final = answer([0, 0, 1000, 1000])
    body = f"Inspect.</think>{final}"
    missing_end = score_completion(
        body,
        FULL,
        answer_mode="thinking",
        finish_reason=None,
        require_termination=True,
    )
    explicit_end = score_completion(
        body + "<|im_end|>",
        FULL,
        answer_mode="thinking",
        finish_reason=None,
        require_termination=True,
    )
    natural_stop = score_completion(
        body,
        FULL,
        answer_mode="thinking",
        finish_reason="stop",
        require_termination=True,
    )
    assert missing_end["valid"] is False
    assert missing_end["invalid_reason"] == "truncated"
    assert explicit_end["valid"] is True
    assert natural_stop["valid"] is True


@pytest.mark.parametrize(
    "solution,negative_boxes",
    [
        ([0, 0, 0, 1], None),
        ([0, 0, math.inf, 1], None),
        ([0, 0, 1, 1], [[0, 0, math.nan, 1]]),
        ([0, 0, 1, 1], [[0, 0, 1.1, 1]]),
        ([0, 0, 1, 1], [0, 0, 0.5, 1]),
    ],
)
def test_invalid_ground_truth_or_negative_metadata_fails_fast(solution, negative_boxes) -> None:
    with pytest.raises(ValueError):
        score_completion(answer([0, 0, 500, 1000]), solution, negative_boxes)
