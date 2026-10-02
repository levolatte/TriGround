"""探测 vLLM 前缀缓存相关参数的合法取值与默认值。

存在意义：混合架构（Qwen3.6 有 48/64 层是线性注意力）的前缀缓存需要 `mamba_cache_mode`
配合，而不同 vLLM 版本取值不同。靠猜参数名会浪费一轮完整的加载（每次约 5 分钟），
所以先用一个只读探针把取值列清楚。

用法（云端，vLLM 环境）：
    /root/vllm_env/bin/python scripts/probe_vllm_cache_options.py
"""

from __future__ import annotations

import dataclasses
import inspect


def show_enum(name: str) -> None:
    try:
        from vllm.config import MambaCacheMode
    except ImportError as error:
        print(f"{name}: 导入失败 {error}")
        return
    if name == "MambaCacheMode":
        values = [getattr(MambaCacheMode, item) for item in dir(MambaCacheMode)
                  if not item.startswith("_") and not callable(getattr(MambaCacheMode, item))]
        print(f"MambaCacheMode 取值: {values}")


def main() -> None:
    from vllm.config import CacheConfig, MultiModalConfig

    print("=== CacheConfig 中与缓存相关的字段 ===")
    for field in dataclasses.fields(CacheConfig):
        if any(key in field.name for key in ("mamba", "prefix", "block_size")):
            print(f"  {field.name} = {field.default!r}")

    print("\n=== MultiModalConfig 中与缓存相关的字段 ===")
    for field in dataclasses.fields(MultiModalConfig):
        if "cache" in field.name or "processor" in field.name:
            print(f"  {field.name} = {field.default!r}")

    print("\n=== MambaCacheMode 取值 ===")
    show_enum("MambaCacheMode")

    print("\n=== 离线 LLM.__init__ 是否接受这些参数 ===")
    from vllm import LLM
    signature = inspect.signature(LLM.__init__)
    for name in ("enable_prefix_caching", "mamba_cache_mode", "mm_processor_cache_type",
                 "mamba_block_size", "max_num_seqs", "gpu_memory_utilization"):
        accepted = name in signature.parameters
        print(f"  {name}: {'接受' if accepted else '不接受'}")

    print("\n=== LLM.__init__ 的 **kwargs 说明 ===")
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            print(f"  {name}: {parameter}")


if __name__ == "__main__":
    main()
