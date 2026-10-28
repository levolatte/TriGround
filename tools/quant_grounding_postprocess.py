"""限定范围的 bbox 答案格式规范化。"""

from __future__ import annotations

import json
import re

from tools.quant_grounding_reward import parse_bbox

_WS = r"[ \t\r\n]*"
_NUMBER = r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_END = "<|im_end|>"
_THINK_END = "</think>"
_JSON_FENCE = re.compile(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object_pattern(value_pattern: str) -> re.Pattern[str]:
    return re.compile(
        rf"\{{{_WS}\"bbox_2d\"{_WS}:{_WS}{value_pattern}{_WS}\}}"
    )


def _coordinate_pair_patterns() -> tuple[re.Pattern[str], ...]:
    """Known coordinate-pair spellings and the observed crossed delimiter typo."""
    ws = _WS
    number = _NUMBER
    patterns = (
        # Two parenthesized coordinate pairs: [(x,y),(x,y)]
        rf"\[{ws}\({ws}(?P<x1>{number}){ws},{ws}(?P<y1>{number}){ws}\){ws},{ws}"
        rf"\({ws}(?P<x2>{number}){ws},{ws}(?P<y2>{number}){ws}\){ws}\]",
        # Two square-bracketed coordinate pairs: [[x,y],[x,y]]
        rf"\[{ws}\[{ws}(?P<x1>{number}){ws},{ws}(?P<y1>{number}){ws}\]{ws},{ws}"
        rf"\[{ws}(?P<x2>{number}){ws},{ws}(?P<y2>{number}){ws}\]{ws}\]",
        # Observed typo: [x1,y1),(x2,y2]
        rf"\[{ws}(?P<x1>{number}){ws},{ws}(?P<y1>{number}){ws}\){ws},{ws}"
        rf"\({ws}(?P<x2>{number}){ws},{ws}(?P<y2>{number}){ws}\]",
    )
    return tuple(_object_pattern(pattern) for pattern in patterns)


_COORDINATE_PAIR_PATTERNS = _coordinate_pair_patterns()


def _completion_answer_span(text: str, answer_mode: str) -> tuple[int, int] | None:
    """Return the final JSON answer's trimmed span without touching its wrapper."""
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    trimmed = text[start:end]
    if trimmed.count(_END):
        if trimmed.count(_END) != 1 or not trimmed.endswith(_END):
            return None
        end -= len(_END)
        while end > start and text[end - 1].isspace():
            end -= 1

    body = text[start:end]
    if answer_mode == "short":
        if "<think>" in body or _THINK_END in body:
            return None
        left = len(body) - len(body.lstrip())
        right = len(body.rstrip())
        return start + left, start + right

    if body.count(_THINK_END) != 1:
        return None
    delimiter = body.index(_THINK_END)
    reasoning = body[:delimiter].lstrip()
    opening_count = reasoning.count("<think>")
    if opening_count:
        if opening_count != 1 or not reasoning.startswith("<think>"):
            return None
    answer_start = delimiter + len(_THINK_END)
    answer = body[answer_start:]
    if "<think>" in answer or _THINK_END in answer:
        return None
    left = len(answer) - len(answer.lstrip())
    right = len(answer.rstrip())
    return start + answer_start + left, start + answer_start + right


def normalize_bbox_completion(
    text: str,
    answer_mode: str = "short",
    format_policy: str = "strict",
) -> str:
    """Repair only recognized coordinate-pair punctuation in a bbox answer.

    ``strict`` is a no-op. ``punctuation`` preserves valid raw completions and
    repairs only a full-string, single-object match with four JSON numbers.
    Thinking text and all completion wrappers remain byte-for-byte unchanged.
    """
    if format_policy not in {"strict", "punctuation", "single_box"}:
        raise ValueError(f"unknown format policy: {format_policy}")
    if answer_mode not in {"short", "thinking"}:
        raise ValueError(f"unknown answer mode: {answer_mode}")
    if format_policy == "strict" or not isinstance(text, str):
        return text
    if format_policy == "single_box":
        return _normalize_single_box_completion(text, answer_mode)
    if parse_bbox(text, answer_mode) is not None:
        return text

    span = _completion_answer_span(text, answer_mode)
    if span is None:
        return text
    start, end = span
    answer = text[start:end]
    for pattern in _COORDINATE_PAIR_PATTERNS:
        match = pattern.fullmatch(answer)
        if match is None:
            continue
        coords = [match.group(name) for name in ("x1", "y1", "x2", "y2")]
        repaired_answer = '{"bbox_2d":[' + ",".join(coords) + "]}"
        repaired = text[:start] + repaired_answer + text[end:]
        if parse_bbox(repaired, answer_mode) is not None:
            return repaired
    return text


def _normalize_single_box_completion(text: str, answer_mode: str) -> str:
    """Canonicalize one complete final JSON answer under the narrow D protocol.

    This accepts one complete ``json`` fence, one top-level singleton list,
    and a string ``label`` alongside ``bbox_2d``. It never examines reasoning
    text or chooses among boxes.
    """
    if parse_bbox(text, answer_mode) is not None:
        return text

    span = _completion_answer_span(text, answer_mode)
    if span is None:
        return text
    start, end = span
    answer = text[start:end]
    fence = _JSON_FENCE.fullmatch(answer)
    if fence is not None:
        answer = fence.group(1)

    try:
        payload = json.loads(
            answer,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return text

    if isinstance(payload, list):
        if len(payload) != 1:
            return text
        payload = payload[0]
    if not isinstance(payload, dict) or "bbox_2d" not in payload:
        return text
    if set(payload) not in ({"bbox_2d"}, {"bbox_2d", "label"}):
        return text
    if "label" in payload and not isinstance(payload["label"], str):
        return text

    canonical_answer = json.dumps(
        {"bbox_2d": payload["bbox_2d"]}, ensure_ascii=False, separators=(",", ":")
    )
    canonical = text[:start] + canonical_answer + text[end:]
    return canonical if parse_bbox(canonical, answer_mode) is not None else text
