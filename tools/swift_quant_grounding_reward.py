"""ms-swift ORM registrations for strict quantized grounding rewards."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from swift.rewards import ORM, orms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_reward import score_completions  # noqa: E402
from tools.quant_grounding_postprocess import normalize_bbox_completion  # noqa: E402

SHORT_RESPONSE_PREFIX = "<think>\n\n</think>\n\n"
THINKING_RESPONSE_PREFIX = "<think>\n"


def _metadata_for_batch(value: Any, count: int) -> list[Any]:
    if value is None or isinstance(value, str):
        return [value] * count
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return [value[0]] * count
        if len(value) == count:
            return list(value)
    return [value] * count


def _response_token_ids_for_batch(value: Any, count: int) -> list[list[int]]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Swift reward kwargs must include response_token_ids")
    if count == 1 and all(isinstance(token, int) and not isinstance(token, bool) for token in value):
        values = [value]
    elif len(value) == count:
        values = list(value)
    else:
        raise ValueError(
            f"response_token_ids count {len(value)} does not match completions count {count}"
        )
    result = []
    for index, token_ids in enumerate(values):
        if not isinstance(token_ids, (list, tuple)) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in token_ids
        ):
            raise ValueError(f"response_token_ids[{index}] must be one flat generated-token list")
        result.append(list(token_ids))
    return result


def _restore_generated_completions(
    completions: list[str], response_token_ids: Any, finish_reasons: Any, chat_template_kwargs: Any,
    template: Any, answer_mode: str,
) -> tuple[list[str], list[int]]:
    if template is None:
        raise RuntimeError("Swift grounding reward needs trainer.template bound before rollout scoring")
    token_id_rows = _response_token_ids_for_batch(response_token_ids, len(completions))
    reasons = _metadata_for_batch(finish_reasons, len(completions))
    template_kwargs = _metadata_for_batch(chat_template_kwargs, len(completions))
    expected_prefix = SHORT_RESPONSE_PREFIX if answer_mode == "short" else THINKING_RESPONSE_PREFIX
    restored = []
    token_counts = []
    for index, (completion, token_ids, reason, row_kwargs) in enumerate(
        zip(completions, token_id_rows, reasons, template_kwargs)
    ):
        if row_kwargs is None:
            row_kwargs = {}
        if not isinstance(row_kwargs, dict):
            raise ValueError(f"chat_template_kwargs[{index}] must be a mapping")
        template_inputs = SimpleNamespace(chat_template_kwargs=row_kwargs)
        response_prefix = template._get_response_prefix(template_inputs)
        if response_prefix != expected_prefix:
            raise ValueError(
                f"{answer_mode} reward expected Swift response_prefix {expected_prefix!r}, "
                f"got {response_prefix!r}"
            )
        generated_text = template.decode_generate_ids(
            token_ids, first_token=False, template_inputs=template_inputs
        )
        if completion != response_prefix + generated_text:
            raise ValueError(
                "Swift completion does not match response_prefix plus decoded response_token_ids"
            )
        restored.append(generated_text)
        token_counts.append(len(token_ids))
    return restored, token_counts


def _append_reward_metrics(
    records: list[dict[str, Any]], generated_tokens: list[int] | None = None,
    generated_text: list[str] | None = None, canonical_text: list[str] | None = None,
    format_policy: str | None = None, **kwargs: Any
) -> None:
    output_path = os.environ.get("QUANT_REWARD_METRICS_JSONL")
    if not output_path:
        return
    ids = _metadata_for_batch(kwargs.get("id"), len(records))
    occurrence_ids = _metadata_for_batch(kwargs.get("occurrence_id"), len(records))
    group_ids = _metadata_for_batch(kwargs.get("group_id"), len(records))
    sources = _metadata_for_batch(kwargs.get("source"), len(records))
    negative_kinds = _metadata_for_batch(kwargs.get("grounding_negative_kind"), len(records))
    sampling_pools = _metadata_for_batch(kwargs.get("sampling_pool"), len(records))
    finish_reasons = _metadata_for_batch(
        kwargs.get("finish_reason", kwargs.get("finish_reasons")), len(records)
    )
    generated_tokens = generated_tokens or [None] * len(records)
    generated_text = generated_text or [None] * len(records)
    items = [
        {
            "id": sample_id,
            "occurrence_id": occurrence_id,
            "group_id": group_id,
            "source": source,
            "negative_kind": negative_kind,
            "sampling_pool": sampling_pool,
            "finish_reason": finish_reason,
            "generated_tokens": generated_count,
            "generated_text": response_text,
            "valid": record["valid"],
            "format_valid": record["format_valid"],
            "truncated": record["truncated"],
            "iou": record["iou"],
            "max_negative_iou": record.get("max_negative_iou"),
            "hit": record["hit"],
            "contrast": record["contrast"],
            "reward": record["reward"],
        }
        for sample_id, occurrence_id, group_id, source, negative_kind, sampling_pool,
        finish_reason, generated_count, response_text, record in zip(
            ids,
            occurrence_ids,
            group_ids,
            sources,
            negative_kinds,
            sampling_pools,
            finish_reasons,
            generated_tokens,
            generated_text,
            records,
        )
    ]
    if canonical_text is not None:
        for item, canonical in zip(items, canonical_text):
            item["canonical_text"] = canonical
            item["format_policy"] = format_policy
            item["format_repaired"] = canonical != item["generated_text"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        group_field = "occurrence_id" if item["occurrence_id"] is not None else "id"
        group_value = item[group_field]
        if group_value is not None:
            groups.setdefault((group_field, str(group_value)), []).append(item)
    payload = {
        "count": len(records),
        "valid_count": sum(record["valid"] for record in records),
        "invalid_count": sum(not record["valid"] for record in records),
        "hit_count": sum(record["hit"] for record in records),
        "all_batch_rewards_equal": len({record["reward"] for record in records}) <= 1,
        "groups": [
            {
                "key_field": group_field,
                "key": group_value,
                "count": len(group),
                "all_rewards_equal": len({item["reward"] for item in group}) <= 1,
            }
            for (group_field, group_value), group in groups.items()
        ],
        "items": items,
    }
    with Path(output_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


class _QuantBBoxReward(ORM):
    reward_mode = "contrast"
    iou_reward_mode = "capped"
    answer_mode = "short"
    format_policy = "strict"

    def bind_template(self, template) -> None:
        self.template = template

    def __call__(self, completions, solution, **kwargs) -> list[float]:
        finish_reasons = kwargs.get("finish_reason", kwargs.get("finish_reasons"))
        generated_completions, generated_tokens = _restore_generated_completions(
            completions,
            kwargs.get("response_token_ids"),
            finish_reasons,
            kwargs.get("chat_template_kwargs"),
            getattr(self, "template", None),
            self.answer_mode,
        )
        canonical_completions = [
            normalize_bbox_completion(completion, self.answer_mode, self.format_policy)
            for completion in generated_completions
        ]
        records = score_completions(
            canonical_completions,
            solution,
            negative_boxes=kwargs.get("grounding_negative_boxes"),
            reward_mode=self.reward_mode,
            iou_reward_mode=self.iou_reward_mode,
            answer_mode=self.answer_mode,
            finish_reasons=finish_reasons,
            require_termination=True,
        )
        _append_reward_metrics(
            records,
            generated_tokens=generated_tokens,
            generated_text=generated_completions,
            canonical_text=(canonical_completions if self.format_policy != "strict" else None),
            format_policy=(self.format_policy if self.format_policy != "strict" else None),
            **kwargs,
        )
        return [record["reward"] for record in records]


class QuantBBoxContrastReward(_QuantBBoxReward):
    """A arm: ACC threshold, weak IoU, and reviewed-negative contrast."""


class QuantBBoxAcc05Reward(_QuantBBoxReward):
    """B arm: same ACC/IoU reward without the contrast term."""

    reward_mode = "acc05"


class QuantBBoxContrastThinkingReward(_QuantBBoxReward):
    """D arm: contrast reward on the final JSON after Qwen ``</think>``."""

    answer_mode = "thinking"


class QuantBBoxAcc05ContinuousIoUReward(QuantBBoxAcc05Reward):
    """Independent B candidate: retain the hit term and uncapped weak IoU."""

    iou_reward_mode = "continuous"


class QuantBBoxContrastThinkingRepairedReward(QuantBBoxContrastThinkingReward):
    """D repair arm: single-box normalization only for scoring the final answer."""

    format_policy = "single_box"


orms["quant_bbox_contrast"] = QuantBBoxContrastReward
orms["quant_bbox_acc05"] = QuantBBoxAcc05Reward
orms["quant_bbox_acc05_continuous_iou"] = QuantBBoxAcc05ContinuousIoUReward
orms["quant_bbox_contrast_thinking"] = QuantBBoxContrastThinkingReward
orms["quant_bbox_contrast_thinking_repaired"] = QuantBBoxContrastThinkingRepairedReward
