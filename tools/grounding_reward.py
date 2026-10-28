"""短 JSON 定位奖励；仅依赖标准库，与独立 ACC@0.5 评分共用几何定义。"""

from __future__ import annotations

import json

from tools.verify_acc05_dev import box_is_valid, iou_xyxy


def _unique_object(pairs: list[tuple]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("JSON contains duplicate keys")
    return result


def parse_short_bbox(completion: str) -> list[float] | None:
    """只接受一个 bbox_2d JSON 对象；不抽取末框、不裁剪或修复坐标。"""
    try:
        payload = json.loads(completion, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"bbox_2d"}:
        return None
    box = payload["bbox_2d"]
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in box):
        return None
    normalized = [value / 1000.0 for value in box]
    return normalized if box_is_valid(normalized) else None


def validate_solution(solution: list[float]) -> None:
    """GT 是输入边界：坏标签直接报错，不能伪装成模型低分。"""
    if not isinstance(solution, (list, tuple)) or not box_is_valid(solution):
        raise ValueError(f"solution must be a valid normalized xyxy box: {solution!r}")


def grounding_reward(completion: str, solution: list[float], mode: str = "acc05") -> float:
    """ACC 阈值 + 未过线框的弱 IoU 奖励；成功后饱和，非法输出 -0.1。

    mode='iou' 和 'binary' 用于固定采样预算下的奖励对照。
    主臂使用 scale_rewards=none，避免组内标准差把弱几何奖励重新放大。
    """
    if mode not in {"acc05", "iou", "binary"}:
        raise ValueError(f"unknown reward mode: {mode}")
    validate_solution(solution)
    prediction = parse_short_bbox(completion)
    if prediction is None:
        return -0.1
    iou = iou_xyxy(prediction, solution)
    if mode == "iou":
        return iou
    hit = float(iou >= 0.5)
    return hit if mode == "binary" else hit + 0.2 * min(iou, 0.5)


def grounding_rewards(
    completions: list[str], solutions: list[list[float]], mode: str = "acc05"
) -> list[float]:
    if len(completions) != len(solutions):
        raise ValueError("completions and solutions must have the same length")
    return [grounding_reward(text, target, mode) for text, target in zip(completions, solutions)]
