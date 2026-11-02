"""ACC@0.5 纠错训练的分流、题目采样与逐题损失。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000
from tools.quant_grounding_postprocess import normalize_bbox_completion
from tools.quant_grounding_reward import box_is_valid, iou_xyxy, parse_bbox, score_completion


@dataclass(frozen=True)
class Completion:
    text: str
    token_ids: list[int]
    finish_reason: str


@dataclass(frozen=True)
class RouteDecision:
    route: str
    selected_index: int | None = None


def classify(row: dict, completions: list[Completion]) -> dict:
    """只规范化判分文本，训练仍使用 Completion 中的原始词元。"""
    if len(completions) != 2:
        raise ValueError("每题必须恰有两个独立回答")
    normalized_texts = [
        normalize_bbox_completion(
            completion.text, answer_mode="short", format_policy="punctuation"
        )
        for completion in completions
    ]
    scores = [
        score_completion(
            text,
            row["solution"],
            reward_mode="acc05",
            answer_mode="short",
            finish_reason=completion.finish_reason,
            require_termination=True,
        )
        for text, completion in zip(normalized_texts, completions)
    ]
    hits = [score["hit"] for score in scores]
    route = "mixed" if sum(hits) == 1 else "all_hit" if all(hits) else "all_wrong"
    # 失败样本的框要留下来：方向自适应抖动需要知道模型实际偏大还是偏小。
    predicted = [parse_bbox(text, answer_mode="short") for text in normalized_texts]
    return {"hits": hits, "normalized_texts": normalized_texts, "route": route,
            "predicted_boxes": predicted}


def decide_route(
    classification: dict,
    retention_rng: random.Random,
    keep_probability: float = 0.2,
) -> RouteDecision:
    """双错用 GT，混合用 RLOO，双中独立抽签后均匀保留一答。"""
    if not 0 <= keep_probability <= 1:
        raise ValueError("keep_probability 必须在 [0, 1]")
    route = classification["route"]
    if route == "mixed":
        return RouteDecision("mixed")
    if route == "all_wrong":
        return RouteDecision("gt")
    if route != "all_hit":
        raise ValueError(f"未知分类: {route}")
    if retention_rng.random() < keep_probability:
        return RouteDecision("retain", retention_rng.randrange(2))
    return RouteDecision("skip")


def observed_size_ratio(boxes: list[list[float] | None], solution: list[float]) -> float | None:
    """模型失败框相对真值的线性尺寸比中位数（√面积比）。

    这是"抖动该往哪个方向纠"的唯一依据。实测依据（City378 分区解剖）：
    上一版抖动对**所有**"两答全错"的样本无条件放大 1.133 倍，结果近失带
    （IoU 0.3~0.5）的尺寸比中位从 0.7583 推到 0.9646、救回 3 道，
    但总体命中从 311 掉到 305——因为它把本来对的框也放大了，而 ACC@0.5 是
    阈值指标，尺度扰动对已贴近 0.5 的样本是负和的。

    所以现在只按模型**实际偏的方向**去纠：偏小才放大，偏大才缩小，已经对准就不动。
    """
    gt_area = (solution[2] - solution[0]) * (solution[3] - solution[1])
    if gt_area <= 0:
        return None
    ratios: list[float] = []
    for box in boxes:
        if box is None or not box_is_valid(box):
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area <= 0:
            continue
        ratios.append((area / gt_area) ** 0.5)
    if not ratios:
        return None
    ratios.sort()
    middle = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[middle]
    return (ratios[middle - 1] + ratios[middle]) / 2


def jitter_solution(solution: list[float], rng: random.Random, min_iou: float = 0.65,
                    scale_low: float = 0.85, scale_high: float = 1.20,
                    direction: float | None = None,
                    adaptive: bool = False,
                    small_area: float | None = None) -> list[float]:
    """抽一个仍与真值 IoU >= min_iou 的框，作为 ACC@0.5 的容忍带监督目标。

    为什么不用真值框本身：ACC@0.5 只判定 IoU>=0.5。中心对齐时尺寸的容忍带是线性
    0.707~1.414 倍，也就是一个放大 20% 的框与真值同分。而逐 token 的交叉熵把
    "476" 与 "462" 当成完全不同的答案，等于在惩罚一个本该满分的框，把容量花在
    拿不到分的精度上。

    两个可选的方向：

    * **对称抖动**（scale_low<1<scale_high，默认）：主要起正则作用，让模型不要把
      容量花在逐字复现坐标上。实测净上拉力只有约 +2%。
    * **偏大抖动**（scale_low=1.0, scale_high=1.30, min_iou=0.60）：实测近失题的
      预测框尺寸比中位只有 0.79，而偏小到 0.707 以下必挂、偏大到 1.414 才挂，
      所以往上拉是安全方向。这一档能把 0.79 明显拉向 1.0。

    注意 `min_iou` 会硬性夹住尺度上界（同心框 IoU=1/k²），所以想加大上拉力必须
    同时放宽 min_iou，否则改 scale_high 没有效果。

    抽不到合格框就退回真值，绝不产生一个离真值很远的监督目标。

    `adaptive=True` 时启用**方向自适应**（这是修掉上一版负结果的关键）：
    由 `direction`（模型失败框的尺寸比中位）决定往哪边纠，而不是恒定放大。
    幅度也比上一版温和得多（上一版净上拉 1.133 倍，对阈值指标过猛）。

    `small_area` 非空时只在真值面积小于该阈值的样本上抖动——偏小的问题集中在
    小目标上（近失带框面积中位 0.0017），大目标本来就框得准，不需要推。
    """
    if small_area is not None:
        gt_area = (solution[2] - solution[0]) * (solution[3] - solution[1])
        if gt_area >= small_area:
            return list(solution)
    if adaptive:
        if direction is None:
            return list(solution)
        if direction < 0.95:
            # 模型框偏小 → 往上纠，但只到 +15%，避免把框推过头。
            scale_low, scale_high = 1.02, 1.15
        elif direction > 1.05:
            # 模型框偏大 → 往下纠。
            scale_low, scale_high = 0.87, 0.98
        else:
            # 尺寸已经对准，不要再动它。
            return list(solution)

    center_x = (solution[0] + solution[2]) / 2
    center_y = (solution[1] + solution[3]) / 2
    half_w = (solution[2] - solution[0]) / 2
    half_h = (solution[3] - solution[1]) / 2
    for _ in range(20):
        scale = rng.uniform(scale_low, scale_high)
        shift_x = rng.uniform(-0.02, 0.02)
        shift_y = rng.uniform(-0.02, 0.02)
        candidate = [center_x - half_w * scale + shift_x, center_y - half_h * scale + shift_y,
                     center_x + half_w * scale + shift_x, center_y + half_h * scale + shift_y]
        if box_is_valid(candidate) and iou_xyxy(candidate, solution) >= min_iou:
            return candidate
    return list(solution)


def serialize_gt(row: dict, rng: random.Random | None = None, min_iou: float = 0.65,
                 scale_low: float = 0.85, scale_high: float = 1.20,
                 direction: float | None = None, adaptive: bool = False,
                 small_area: float | None = None) -> str:
    """GT 只作为监督目标返回，原始 messages 不作修改。

    `rng` 为 None 时严格等于旧行为（用真值框）；传入后才启用容忍带抖动。
    命中判定始终读 `row["solution"]`，与本函数无关。
    """
    solution = (jitter_solution(row["solution"], rng, min_iou, scale_low, scale_high,
                                direction=direction, adaptive=adaptive, small_area=small_area)
                if rng is not None else row["solution"])
    return json.dumps(
        {"bbox_2d": bbox_to_qwen1000(solution)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def random_state_to_json(state: tuple) -> list:
    """把 random.Random 状态转成可直接存入 JSON 的列表。"""
    return [random_state_to_json(value) if isinstance(value, tuple) else value for value in state]


def random_state_from_json(state: list) -> tuple:
    return tuple(
        random_state_from_json(value) if isinstance(value, list) else value for value in state
    )


class QuestionSampler:
    """均匀 fresh 周期与最新 mixed 池；每批四题且同批不重复。"""

    def __init__(self, rows: list[dict], seed: int = 2026):
        self.rows = {row["id"]: row for row in rows}
        if len(self.rows) != len(rows) or len(rows) < 4:
            raise ValueError("train_index 必须包含至少四个唯一查询 id")
        if any(message["role"] == "assistant" for row in rows for message in row["messages"]):
            raise ValueError("训练输入 messages 不得包含 assistant 回答")
        self.rng = random.Random(seed)
        self.fresh_order = list(self.rows)
        self.rng.shuffle(self.fresh_order)
        self.cursor = 0
        self.mixed_pool: dict[str, None] = {}
        self.batch_count = 0
        self.fresh_count = 0
        self.revisit_count = 0
        self.observation_count = 0
        self.cycle_count = 0

    def _fresh(self, count: int, excluded: set[str]) -> list[str]:
        selected: list[str] = []
        while len(selected) < count:
            if self.cursor == len(self.fresh_order):
                self.rng.shuffle(self.fresh_order)
                self.cursor = 0
                self.cycle_count += 1
            sample_id = self.fresh_order[self.cursor]
            self.cursor += 1
            if sample_id not in excluded:
                selected.append(sample_id)
                excluded.add(sample_id)
        self.fresh_count += len(selected)
        return selected

    def next_batch(self) -> list[dict]:
        selected: set[str] = set()
        ids = self._fresh(2, selected)
        candidates = [sample_id for sample_id in self.mixed_pool if sample_id not in selected]
        revisits = self.rng.sample(candidates, min(2, len(candidates)))
        ids.extend(revisits)
        selected.update(revisits)
        ids.extend(self._fresh(4 - len(ids), selected))
        self.batch_count += 1
        self.revisit_count += len(revisits)
        return [self.rows[sample_id] for sample_id in ids]

    def observe(self, sample_id: str, route: str) -> None:
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        if route not in {"mixed", "all_wrong", "all_hit"}:
            raise ValueError(f"未知分类: {route}")
        if route == "mixed":
            self.mixed_pool[sample_id] = None
        else:
            self.mixed_pool.pop(sample_id, None)
        self.observation_count += 1

    def state_dict(self) -> dict:
        return {
            "fresh_order": self.fresh_order.copy(),
            "cursor": self.cursor,
            "rng_state": random_state_to_json(self.rng.getstate()),
            "mixed_pool": list(self.mixed_pool),
            "batch_count": self.batch_count,
            "fresh_count": self.fresh_count,
            "revisit_count": self.revisit_count,
            "observation_count": self.observation_count,
            "cycle_count": self.cycle_count,
        }

    def load_state_dict(self, state: dict) -> None:
        order = list(state["fresh_order"])
        pool = list(state["mixed_pool"])
        if len(order) != len(self.rows) or set(order) != set(self.rows):
            raise ValueError("采样状态的 fresh_order 与 train_index 不一致")
        if not 0 <= state["cursor"] <= len(order) or not set(pool) <= set(self.rows):
            raise ValueError("采样状态含无效 cursor 或 mixed_pool")
        self.fresh_order = order
        self.cursor = state["cursor"]
        self.rng.setstate(random_state_from_json(state["rng_state"]))
        self.mixed_pool = dict.fromkeys(pool)
        self.batch_count = state["batch_count"]
        self.fresh_count = state["fresh_count"]
        self.revisit_count = state["revisit_count"]
        self.observation_count = state["observation_count"]
        self.cycle_count = state["cycle_count"]


def mixed_question_loss(
    logps: list[Tensor],
    ref_logps: list[Tensor],
    hits: list[bool],
    beta: float = 0.01,
) -> Tensor:
    """二元 RLOO 的逐题损失；两答各自按有效答案词元取均值。"""
    if len(logps) != 2 or len(ref_logps) != 2 or len(hits) != 2 or sum(hits) != 1:
        raise ValueError("mixed 损失需要一中一错的两个回答")
    return torch.stack([
        mixed_response_loss(logp, ref_logp, hit, beta)
        for logp, ref_logp, hit in zip(logps, ref_logps, hits)
    ]).mean()


def mixed_response_loss(
    logp: Tensor, ref_logp: Tensor, hit: bool, beta: float = 0.01
) -> Tensor:
    """一条混合分支回答的损失；由调用方按半题权重即时反传。"""
    if logp.ndim != 1 or not logp.numel() or ref_logp.shape != logp.shape:
        raise ValueError("logps 必须是非空的一维有效答案词元，且与 reference 对齐")
    advantage = 1.0 if hit else -1.0
    delta = ref_logp.detach() - logp
    kl = torch.exp(delta) - delta - 1
    return (-advantage * logp + beta * kl).mean()


def ce_question_loss(logps: Tensor) -> Tensor:
    """一条 GT 或成功回答的逐题 CE（交叉熵）损失。"""
    if logps.ndim != 1 or not logps.numel():
        raise ValueError("logps 必须是非空的一维有效答案词元")
    return -logps.mean()
