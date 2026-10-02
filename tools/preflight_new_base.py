"""新版基座本地部署预检（在云电脑上运行，不下载权重、不改环境）。

它只回答四个会让后面白干的问题，全部 fail fast：
  1) 当前 transformers 是否认识 Qwen3.5/3.6 的架构类；
  2) 目标权重的 config/preprocessor 是否与我们的调用方式兼容（pixel 上限、processor 类）；
  3) 磁盘剩余是否放得下（含 27B BF16 的 54GB 风险）；
  4) 目标 GPU 的显存与预计视觉 token 预算是否匹配。

用法：
    python tools/preflight_new_base.py --model Qwen/Qwen3.5-9B --pixels 602112 1204224 2073600
    python tools/preflight_new_base.py --model /root/rematch_models/Qwen3.5-9B --local-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import transformers
from transformers import AutoConfig, AutoProcessor

# 项目已验证基线的口径：每图 min_pixels=200704，每图 max_pixels=602112。
# 合并视觉 token ≈ pixels / 1024（patch16 × merge2 的 32×32）。
PROJECT_MIN_PIXELS = 200704
BASELINE_MAX_PIXELS = 602112
TOKENS_PER_RECORD_IMAGES = 3


def report_transformers() -> dict:
    names = ["Qwen3VLForConditionalGeneration", "Qwen3_5ForConditionalGeneration",
             "Qwen3_5MoeForConditionalGeneration", "AutoModelForImageTextToText"]
    return {
        "transformers_version": transformers.__version__,
        "available_classes": {name: hasattr(transformers, name) for name in names},
    }


def report_model(model_name: str, local_only: bool) -> dict:
    """读 config 并解析架构类。这里必须捕获：旧版 transformers 会直接拒绝解析新 model_type，
    而预检的价值就在于把这类问题连同其余检查一起报出来，而不是死在第一行。"""
    try:
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_only)
    except Exception as error:  # 预检工具：任何环境/网络失败都要报告，而不是中断其余检查
        return {"model_name": model_name, "config_readable": False,
                "error": f"{type(error).__name__}: {str(error).splitlines()[0]}"}
    result = {
        "model_name": model_name,
        "config_readable": True,
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "class_resolvable": all(
            hasattr(transformers, name) for name in (config.architectures or [])
        ),
    }
    text = getattr(config, "text_config", None)
    if text is not None:
        result["num_hidden_layers"] = getattr(text, "num_hidden_layers", None)
        result["hidden_size"] = getattr(text, "hidden_size", None)
        result["max_position_embeddings"] = getattr(text, "max_position_embeddings", None)
    vision = getattr(config, "vision_config", None)
    if vision is not None:
        result["vision_patch_size"] = getattr(vision, "patch_size", None)
        result["vision_merge_size"] = getattr(vision, "spatial_merge_size", None)
        result["vision_deepstack_indexes"] = list(
            getattr(vision, "deepstack_visual_indexes", []) or []
        )
    return result


def report_processor(model_name: str, local_only: bool) -> dict:
    """确认 pixel 上限能通过公开参数入口生效，而不是被静默忽略（项目踩过这个坑）。"""
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, min_pixels=PROJECT_MIN_PIXELS, max_pixels=BASELINE_MAX_PIXELS,
            local_files_only=local_only,
        )
    except Exception as error:  # 同上：processor 依赖 torchvision 等可选后端，缺失要报告
        return {"processor_loadable": False,
                "error": f"{type(error).__name__}: {str(error).splitlines()[0]}"}
    size = processor.image_processor.size
    applied = getattr(size, "longest_edge", None)
    if applied is None and isinstance(size, dict):
        applied = size.get("longest_edge")
    return {
        "processor_loadable": True,
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "longest_edge_after_override": applied,
        "override_took_effect": applied == BASELINE_MAX_PIXELS,
    }


def report_disk(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        out.append({
            "path": str(probe),
            "total_GiB": round(usage.total / 2**30, 1),
            "free_GiB": round(usage.free / 2**30, 1),
        })
    return out


def report_gpu() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    capability = f"sm_{props.major}{props.minor}"
    # 27B BF16 权重约 54GB；这里是"能不能放得下"的硬门槛，不是性能评估。
    weight_budget = {
        "Qwen3.5-9B BF16": 18,
        "Qwen3.5-27B BF16": 54,
        "Qwen3.5-27B GPTQ-Int4": 15,
        "Qwen3-VL-8B BF16": 16,
        "Qwen3-VL-32B BF16": 64,
    }
    total = props.total_memory / 2**30
    return {
        "cuda_available": True,
        "name": props.name,
        "capability": capability,
        "total_GiB": round(total, 1),
        "count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "fits_weights_only": {k: total >= v for k, v in weight_budget.items()},
        "note": "只比较权重体积；训练还需为激活/优化器留出空间，高分辨率下建议再留 15GiB 以上",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pixels", type=int, nargs="+",
                        default=[BASELINE_MAX_PIXELS, 1204224, 2073600])
    parser.add_argument("--local-only", action="store_true",
                        help="只读本地已有权重，不访问网络")
    parser.add_argument("--disk-paths", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    report = {
        "transformers": report_transformers(),
        "model": report_model(args.model, args.local_only),
        "processor": report_processor(args.model, args.local_only),
        "gpu": report_gpu(),
        "disk": report_disk(args.disk_paths or [Path.cwd()]),
        "pixel_plan": [
            {
                "max_pixels": value,
                "visual_tokens_per_image": value // 1024,
                "visual_tokens_per_record_3_images": TOKENS_PER_RECORD_IMAGES * (value // 1024),
                "ratio_vs_project_baseline": round(value / BASELINE_MAX_PIXELS, 2),
            }
            for value in args.pixels
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    problems = []
    if not report["model"].get("config_readable", False):
        problems.append(
            f"transformers {transformers.__version__} 无法解析该权重的 model_type；"
            f"原报错：{report['model']['error']}。"
            "Qwen3.5/3.6 需要更新版 transformers（架构类名 Qwen3_5ForConditionalGeneration）"
        )
    elif not report["model"].get("class_resolvable", False):
        problems.append(
            f"transformers {transformers.__version__} 不认识 {report['model']['architectures']}；"
            "需要升级 transformers"
        )
    if not report["processor"].get("processor_loadable", False):
        problems.append(f"processor 无法加载：{report['processor']['error']}")
    elif not report["processor"].get("override_took_effect", False):
        problems.append(
            "max_pixels 覆盖没有生效；检查 AutoProcessor 的公开参数入口，不要依赖处理器私有属性"
        )
    for entry in report["pixel_plan"]:
        if entry["visual_tokens_per_record_3_images"] > 8192:
            problems.append(
                f"max_pixels={entry['max_pixels']} 时三图约 {entry['visual_tokens_per_record_3_images']} "
                "个视觉 token，必须同时提高 model_max_length（基线是 4096）"
            )
    if problems:
        print("\n需要先解决的问题：")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\n预检通过：环境可以加载该权重并按计划探测这些像素预算。")


if __name__ == "__main__":
    main()
