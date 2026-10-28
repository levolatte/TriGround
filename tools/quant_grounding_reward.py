"""严格量化版 grounding 奖励，复用项目独立 ACC@0.5 核验几何。"""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

from tools.verify_acc05_dev import box_is_valid, iou_xyxy

INVALID_REWARD = -0.2
THINK_END = "</think>"
QWEN_MESSAGE_END = "<|im_end|>"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _valid_answer_box(values: Any) -> list[float] | None:
    if not isinstance(values, list) or len(values) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        return None
    if any(not math.isfinite(float(value)) or not 0 <= value <= 1000 for value in values):
        return None
    normalized = [float(value) / 1000.0 for value in values]
    return normalized if box_is_valid(normalized) else None


def _load_answer_json(text: str) -> list[float] | None:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"bbox_2d"}:
        return None
    return _valid_answer_box(payload["bbox_2d"])


def _strip_qwen_message_end(text: str) -> str | None:
    """Accept a decoded Qwen turn terminator only as one terminal wrapper."""
    text = text.strip()
    count = text.count(QWEN_MESSAGE_END)
    if count == 0:
        return text
    if count != 1 or not text.endswith(QWEN_MESSAGE_END):
        return None
    return text[: -len(QWEN_MESSAGE_END)].rstrip()


def _has_qwen_message_end(text: Any) -> bool:
    return (
        isinstance(text, str)
        and text.count(QWEN_MESSAGE_END) == 1
        and text.rstrip().endswith(QWEN_MESSAGE_END)
    )


def parse_short_bbox(completion: str) -> list[float] | None:
    """Parse one JSON answer; short mode rejects all thinking prefixes and repairs."""
    if not isinstance(completion, str):
        return None
    body = _strip_qwen_message_end(completion)
    if body is None or "<think>" in body or THINK_END in body:
        return None
    return _load_answer_json(body)


def parse_thinking_bbox(completion: str) -> list[float] | None:
    """Parse Qwen native-thinking output after its sole ``</think>`` boundary.

    The prompt may already end at ``<think>\n``, so the raw completion is allowed
    to begin with reasoning text and omit the opening tag. If present in the
    completion, ``<think>`` must occur once at its beginning. The final segment
    must itself be exactly one JSON object; reasoning text never earns reward.
    """
    if not isinstance(completion, str):
        return None
    body = _strip_qwen_message_end(completion)
    if body is None or body.count(THINK_END) != 1:
        return None
    reasoning, answer = body.split(THINK_END)
    reasoning = reasoning.lstrip()
    opening_count = reasoning.count("<think>")
    if opening_count:
        if opening_count != 1 or not reasoning.startswith("<think>"):
            return None
        reasoning = reasoning[len("<think>") :]
    if "<think>" in answer or THINK_END in answer:
        return None
    return _load_answer_json(answer.strip())


def parse_bbox(completion: str, answer_mode: str = "short") -> list[float] | None:
    if answer_mode == "short":
        return parse_short_bbox(completion)
    if answer_mode == "thinking":
        return parse_thinking_bbox(completion)
    raise ValueError(f"unknown answer mode: {answer_mode}")


def _looks_like_box(values: Any) -> bool:
    return (
        isinstance(values, (list, tuple))
        and len(values) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
    )


def validate_normalized_box(values: Any, name: str = "box") -> list[float]:
    """Validate GT/negative boxes at the reward system boundary."""
    if not _looks_like_box(values):
        raise ValueError(f"{name} must contain four numeric xyxy coordinates: {values!r}")
    box = [float(value) for value in values]
    if not all(math.isfinite(value) for value in box) or not box_is_valid(box):
        raise ValueError(f"{name} must be a finite valid normalized xyxy box: {values!r}")
    return box


def _targets_for_completions(solutions: Any, count: int) -> list[list[float]]:
    if _looks_like_box(solutions):
        return [validate_normalized_box(solutions, "solution")] * count
    if not isinstance(solutions, (list, tuple)):
        raise ValueError("solution must be one normalized xyxy box or one box per completion")
    if len(solutions) == 1 and _looks_like_box(solutions[0]):
        target = validate_normalized_box(solutions[0], "solution")
        return [target] * count
    if len(solutions) != count:
        raise ValueError(f"solution count {len(solutions)} does not match completions count {count}")
    return [validate_normalized_box(box, f"solution[{index}]") for index, box in enumerate(solutions)]


def _one_negative_batch(values: Any, name: str) -> list[list[float]]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of normalized xyxy boxes")
    return [validate_normalized_box(box, f"{name}[{index}]") for index, box in enumerate(values)]


def _negative_batches_for_completions(negative_boxes: Any, count: int) -> list[list[list[float]]]:
    if negative_boxes is None:
        return [[] for _ in range(count)]
    if not isinstance(negative_boxes, (list, tuple)):
        raise ValueError("negative_boxes must be a list of boxes or per-completion box lists")
    if len(negative_boxes) == 0:
        return [[] for _ in range(count)]

    # A list of boxes describes one sample's reviewed negatives and is shared by
    # its generated completions. A list of box-lists is already per-completion.
    if all(_looks_like_box(box) for box in negative_boxes):
        batch = _one_negative_batch(negative_boxes, "negative_boxes")
        return [batch] * count
    if len(negative_boxes) == 1 and (
        negative_boxes[0] is None or isinstance(negative_boxes[0], (list, tuple))
    ):
        batch = _one_negative_batch(negative_boxes[0], "negative_boxes[0]")
        return [batch] * count
    if len(negative_boxes) != count:
        raise ValueError(
            f"negative_boxes batch count {len(negative_boxes)} does not match completions count {count}"
        )
    return [
        _one_negative_batch(boxes, f"negative_boxes[{index}]")
        for index, boxes in enumerate(negative_boxes)
    ]


def _finish_reasons_for_completions(finish_reasons: Any, count: int) -> list[Any]:
    if finish_reasons is None or isinstance(finish_reasons, str):
        return [finish_reasons] * count
    if not isinstance(finish_reasons, (list, tuple)):
        return [finish_reasons] * count
    if len(finish_reasons) == 1:
        return [finish_reasons[0]] * count
    if len(finish_reasons) != count:
        raise ValueError(
            f"finish_reason count {len(finish_reasons)} does not match completions count {count}"
        )
    return list(finish_reasons)


def score_completion(
    completion: str,
    solution: Sequence[float],
    negative_boxes: Sequence[Sequence[float]] | None = None,
    *,
    reward_mode: str = "contrast",
    iou_reward_mode: str = "capped",
    answer_mode: str = "short",
    finish_reason: Any = None,
    require_termination: bool = False,
) -> dict[str, Any]:
    """Return reward and geometry/formats diagnostics for one response.

    ``contrast`` is the A/D formula; ``acc05`` is B's same formula without the
    reviewed-negative contrast term. Missing negatives contribute ``d = 0``.
    ``continuous`` removes the auxiliary IoU cap for independent candidates;
    the default ``capped`` preserves the original A/B/D rewards.
    """
    if reward_mode not in {"contrast", "acc05"}:
        raise ValueError(f"unknown reward mode: {reward_mode}")
    if iou_reward_mode not in {"capped", "continuous"}:
        raise ValueError(f"unknown IoU reward mode: {iou_reward_mode}")
    target = validate_normalized_box(solution, "solution")
    negatives = _one_negative_batch(negative_boxes, "negative_boxes")
    prediction = parse_bbox(completion, answer_mode)
    reason = str(finish_reason).lower() if finish_reason is not None else None
    finished = reason in {"stop", "eos"} or _has_qwen_message_end(completion)
    truncated = reason == "length" or (require_termination and not finished)
    format_valid = prediction is not None
    valid = format_valid and not truncated

    iou: float | None = None
    max_negative_iou: float | None = None
    contrast = 0.0
    geometric_hit = False
    if prediction is not None:
        iou = iou_xyxy(prediction, target)
        geometric_hit = iou >= 0.5
        if negatives:
            max_negative_iou = max(iou_xyxy(prediction, negative) for negative in negatives)
            contrast = max(-1.0, min(1.0, (iou - max_negative_iou) / 0.2))

    if valid:
        hit_reward = float(iou >= 0.5)
        iou_reward = 0.2 * (min(iou, 0.5) if iou_reward_mode == "capped" else iou)
        contrast_reward = 0.1 * contrast if reward_mode == "contrast" else 0.0
        reward = hit_reward + iou_reward + contrast_reward
        invalid_reason = None
    else:
        hit_reward = iou_reward = contrast_reward = 0.0
        reward = INVALID_REWARD
        invalid_reason = "truncated" if truncated else "format"

    return {
        "valid": valid,
        "format_valid": format_valid,
        "truncated": truncated,
        "invalid_reason": invalid_reason,
        "iou": iou,
        "hit": bool(valid and geometric_hit),
        "geometric_hit": geometric_hit,
        "max_negative_iou": max_negative_iou,
        "contrast": contrast,
        "hit_reward": hit_reward,
        "iou_reward": iou_reward,
        "contrast_reward": contrast_reward,
        "reward": reward,
    }


def score_completions(
    completions: Sequence[str],
    solutions: Any,
    negative_boxes: Any = None,
    *,
    reward_mode: str = "contrast",
    iou_reward_mode: str = "capped",
    answer_mode: str = "short",
    finish_reasons: Any = None,
    require_termination: bool = False,
) -> list[dict[str, Any]]:
    """Score a Swift batch while retaining a per-completion diagnostic record."""
    count = len(completions)
    targets = _targets_for_completions(solutions, count)
    negatives = _negative_batches_for_completions(negative_boxes, count)
    reasons = _finish_reasons_for_completions(finish_reasons, count)
    return [
        score_completion(
            completion,
            target,
            negative_batch,
            reward_mode=reward_mode,
            iou_reward_mode=iou_reward_mode,
            answer_mode=answer_mode,
            finish_reason=reason,
            require_termination=require_termination,
        )
        for completion, target, negative_batch, reason in zip(completions, targets, negatives, reasons)
    ]


def grounding_rewards(
    completions: Sequence[str],
    solutions: Any,
    negative_boxes: Any = None,
    *,
    reward_mode: str = "contrast",
    iou_reward_mode: str = "capped",
    answer_mode: str = "short",
    finish_reasons: Any = None,
    require_termination: bool = False,
) -> list[float]:
    """Swift-compatible float-list wrapper around :func:`score_completions`."""
    return [
        item["reward"]
        for item in score_completions(
            completions,
            solutions,
            negative_boxes,
            reward_mode=reward_mode,
            iou_reward_mode=iou_reward_mode,
            answer_mode=answer_mode,
            finish_reasons=finish_reasons,
            require_termination=require_termination,
        )
    ]
