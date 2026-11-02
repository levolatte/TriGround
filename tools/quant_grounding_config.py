"""专用 ACC 纠错循环的配置与正式训练索引。"""
from __future__ import annotations
import json
from pathlib import Path

DEFAULTS = {
    "model": "Qwen/Qwen3.6-27B", "model_type": "qwen3_5",
    "model_revision": None, "seed": 2026, "learning_rate": 2e-6,
    "beta": 0.01, "keep_probability": 0.2,
    "min_pixels": 200704, "max_pixels": 1204224, "max_length": 8192,
    "gradient_checkpointing": True, "stop_presentations": 4400,
    "checkpoint_every": 400, "generation_prefill_reuse": False,
    # ACC@0.5 容忍带抖动：None 表示关闭（监督目标用真值框，与旧版逐字一致）。
    "jitter_min_iou": None, "jitter_scale_low": 0.85, "jitter_scale_high": 1.20,
    # 方向自适应：由模型失败框相对真值的尺寸比决定纠错方向，而不是恒定放大。
    # 上一版无条件放大使 City378 从 311 掉到 305（近失带救回 3 道、全局误伤 10 道）。
    # `jitter_small_area` 非空时只对真值面积小于该值的样本抖动（小目标才有偏小问题）。
    "jitter_adaptive": False, "jitter_small_area": None,
}
FIELDS = set(DEFAULTS) | {"init_adapter", "train_index", "media_root", "sequence_path"}

def load_config(path: Path) -> dict:
    given = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(given) - FIELDS
    if unknown:
        raise ValueError(f"未知配置字段: {sorted(unknown)}")
    config = DEFAULTS | given
    for key in ("init_adapter", "train_index", "media_root"):
        if not config.get(key):
            raise ValueError(f"缺少配置: {key}")
    for key in ("stop_presentations", "checkpoint_every"):
        if not isinstance(config[key], int) or config[key] <= 0 or config[key] % 4:
            raise ValueError(f"{key} 必须是正的四题整批数")
    if config["stop_presentations"] > 4400:
        raise ValueError("首轮每组总呈现预算为4400")
    if not isinstance(config["generation_prefill_reuse"], bool):
        raise ValueError("generation_prefill_reuse 必须是布尔值")
    # 抖动参数的合法性：抽出的监督框必须是「仍能命中」的框，否则等于教模型做错。
    if config["jitter_min_iou"] is not None:
        if not 0.5 <= config["jitter_min_iou"] <= 0.75:
            raise ValueError("jitter_min_iou 须在 0.5~0.75：低于 0.5 会教出必挂的框，"
                             "高于 0.75 则几乎无法放大（同心框 IoU=1/k²，上界约 1.15 倍）")
        if not 0.6 <= config["jitter_scale_low"] < config["jitter_scale_high"] <= 1.45:
            raise ValueError("jitter 尺度范围须落在 0.6~1.45 且 low<high")
        if not isinstance(config["jitter_adaptive"], bool):
            raise ValueError("jitter_adaptive 必须是布尔值")
        if config["jitter_small_area"] is not None and not 0 < config["jitter_small_area"] <= 1:
            raise ValueError("jitter_small_area 须在 (0, 1] 或为 None")
    return config

def read_train_index(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 4 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("正式训练索引须含至少四个不同查询")
    for row in rows:
        if any(message["role"] == "assistant" for message in row["messages"]):
            raise ValueError(f"{row['id']}: 标准答案不得进入生成提示")
        if len(row["images"]) not in (2, 3):
            raise ValueError(f"{row['id']}: 只接受正式两图/三图输入")
    return rows

def configure_unrestricted_hf_sampling(engine, request_config) -> dict:
    """评价的采样诊断同样显式覆盖 HF 默认 top-k。"""
    engine.generation_config.top_k = 0
    effective = engine._prepare_generation_config(request_config)
    if effective.do_sample and (effective.temperature != 1 or effective.top_p != 1 or effective.top_k != 0):
        raise RuntimeError("采样必须为 temperature=1/top_p=1/top_k=0")
    return {"temperature": effective.temperature, "top_p": effective.top_p,
            "top_k": effective.top_k, "do_sample": effective.do_sample}
