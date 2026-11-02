"""同题两次原生 generate 之间转移独立的完整预填充输出。"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import torch


PROMPT_TENSORS = (
    "attention_mask", "position_ids", "mm_token_type_ids", "pixel_values", "image_grid_thw",
)
CACHE_TENSORS = ("keys", "values", "conv_states", "recurrent_states")
CACHE_FLAGS = ("has_previous_state", "is_conv_states_initialized", "is_recurrent_states_initialized")


def cache_tensor_items(cache):
    """覆盖 Qwen3.5 全注意力 KV、卷积状态与线性注意力递归状态。"""
    for index, layer in enumerate(cache.layers):
        for name in CACHE_TENSORS:
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                yield f"{index}.{name}", value
            elif isinstance(value, dict):
                for key, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        yield f"{index}.{name}.{key}", tensor


def generation_cache_snapshot(cache):
    """仅供诊断：在本答 decode 继续前把最终缓存复制到 CPU。"""
    return {
        "seq_length": cache.get_seq_length(),
        "tensors": {key: tensor.detach().cpu().clone() for key, tensor in cache_tensor_items(cache)},
        "layers": [{key: getattr(layer, key, None) for key in CACHE_FLAGS} for layer in cache.layers],
    }


@contextmanager
def native_prefill_reuse(native, audit=None):
    """只包住同一题、同一权重下的两次 generate；第二答接管一次性副本。

    第一答仍执行 HF 原生 _prefill，立即 deepcopy 完整原生输出。第二答只
    转移该副本，保留原生 generate 的采样、停止条件及随机数消费顺序。
    缓存不逃逸此作用域，不得在两次调用之间更新权重或改变输入。
    """
    original = native._prefill
    if getattr(original, "_quant_prefill_reuse", False):
        raise RuntimeError("prefill reuse 作用域不可嵌套")
    had_override = "_prefill" in native.__dict__
    audit = {} if audit is None else audit
    for key in ("native_prefills", "prefill_replays", "prefill_calls"):
        audit.setdefault(key, 0)
    calls = 0
    snapshot = None
    prompt_ids = None
    prompt_tensors = None
    config_state = None

    def replay(input_ids, generation_config, model_kwargs, is_first_iteration=True):
        nonlocal calls, snapshot, prompt_ids, prompt_tensors, config_state
        if calls >= 2:
            raise RuntimeError("prefill reuse 每个作用域必须恰好调用两次，不可跨题复用")
        cache = model_kwargs.get("past_key_values")
        if (not is_first_iteration or not generation_config.use_cache
                or generation_config.num_beams != 1 or generation_config.num_return_sequences != 1
                or generation_config.prefill_chunk_size is not None
                or (cache is not None and cache.get_seq_length() != 0)):
            raise ValueError("prefill reuse 只接受从空缓存开始的单序列原生生成")
        current_tensors = {key: model_kwargs[key] for key in PROMPT_TENSORS if key in model_kwargs}
        # HF 会为每次 generate 添加私有特殊词元张量，它们不属于用户生成配置。
        current_config = {key: value for key, value in vars(generation_config).items() if not key.startswith("_")}
        if calls == 0:
            prompt_ids = input_ids.detach().clone()
            # 大图不额外复制；保留其对象与版本，阻止原地改图后复用语言缓存。
            prompt_tensors = {
                key: (value, value._version) if key == "pixel_values" else value.detach().clone()
                for key, value in current_tensors.items()
            }
            config_state = current_config
            outputs = original(input_ids, generation_config, model_kwargs, is_first_iteration)
            snapshot = deepcopy(outputs)
            audit["native_prefills"] += 1
            audit["prompt_length"] = input_ids.shape[1]
            audit["cache_length"] = snapshot.past_key_values.get_seq_length()
            tensors = list(cache_tensor_items(snapshot.past_key_values))
            audit["cache_tensor_count"] = len(tensors)
            audit["cache_bytes"] = sum(t.numel() * t.element_size() for _, t in tensors)
        else:
            if (not torch.equal(input_ids, prompt_ids) or current_tensors.keys() != prompt_tensors.keys()
                    or current_config != config_state):
                raise ValueError("prefill reuse 两次调用必须使用同一 prompt 和生成配置")
            for key, value in current_tensors.items():
                saved = prompt_tensors[key]
                unchanged = (value is saved[0] and value._version == saved[1]) if key == "pixel_values" else torch.equal(value, saved)
                if not unchanged:
                    raise ValueError(f"prefill reuse 的 prompt 输入发生变化：{key}")
            outputs, snapshot = snapshot, None
            audit["prefill_replays"] += 1
        calls += 1
        audit["prefill_calls"] += 1
        return outputs

    replay._quant_prefill_reuse = True
    native._prefill = replay
    try:
        yield audit
        if calls != 2:
            raise RuntimeError("prefill reuse 每个作用域必须恰好调用两次")
    finally:
        snapshot = None
        if had_override:
            native._prefill = original
        else:
            del native._prefill
