"""按权重声明的架构动态解析模型类，使同一份推理/评估代码兼容多代基座。

背景：项目原有代码在多处写死 `Qwen3VLForConditionalGeneration`。换成新一代基座后
（Qwen3.5/3.6 的 `architectures` 是 `Qwen3_5ForConditionalGeneration`，`model_type`
是 `qwen3_5`），写死的类会直接加载失败；而旧版 transformers 甚至连 config 都读不了
（实测 4.57.3 报 `KeyError: 'qwen3_5'`）。

这里只做一件事：读 config 的 `architectures[0]`，在 transformers 命名空间里取出对应
类；取不到再回落到统一的 `AutoModelForImageTextToText`。调用方负责保证 transformers
版本足够新，本模块不做版本探测以外的兜底。
"""

from __future__ import annotations

import transformers
from transformers import AutoConfig


def resolve_model_class(model_name: str, *, local_files_only: bool = False):
    """返回 (model_class, class_name)。"""
    config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    architectures = list(config.architectures or [])
    for name in architectures:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class, name
    fallback = getattr(transformers, "AutoModelForImageTextToText", None)
    if fallback is None:
        raise RuntimeError(
            f"transformers {transformers.__version__} 无法加载 {architectures or config.model_type}；"
            "请升级 transformers（Qwen3.5/3.6 需要 >= 5.2.0）"
        )
    return fallback, "AutoModelForImageTextToText"
