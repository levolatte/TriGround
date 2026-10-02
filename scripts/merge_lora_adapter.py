"""把 LoRA 适配器合并进基座并另存，作为 vLLM LoRA 覆盖不全时的兜底路径。

**什么时候需要它**：`scripts/evaluate_trained_adapter.sh` 会用**同一个适配器**分别走 vLLM 与
HF 两条路径跑 City412。若两者 ACC 差超过 ±2 题，说明 vLLM 的 LoRA 实现**没有覆盖全部目标模块**
（我们训练用的是 `--target_modules all-linear`，含 48 层线性注意力），此时 vLLM 会静默产出
"只适配了一部分"的结果。合并后 vLLM 只需加载普通权重，不存在覆盖问题。

**代价（所以要优先用 LoRA 而不是合并）**：
  * 需要额外约 54 GB 磁盘与十几分钟写盘；
  * `merge_and_unload` 在 BF16 下算 `W + BA·scale`，与运行时施加适配器**数值不同**，
    属于提交前引入的一次变化，必须重新过一遍等价性检查。

用法（需要 GPU，用主环境的 python）：
    python scripts/merge_lora_adapter.py \
        --base /root/rematch_models/Qwen3.6-27B \
        --adapter /root/runs/q36_600_2073600/v0-xxx/checkpoint-600 \
        --out /root/rematch_models/Qwen3.6-27B-merged-600
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.native_model_loading import resolve_model_class  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"适配器不存在: {args.adapter}")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"输出目录非空，避免覆盖已有结果: {args.out}")

    import torch
    from peft import PeftModel

    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    print(f"适配器 r={adapter_config.get('r')} alpha={adapter_config.get('lora_alpha')} "
          f"target_modules={adapter_config.get('target_modules')}")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model_class, class_name = resolve_model_class(args.base, local_files_only=True)
    print(f"基座类: {class_name}")

    base = model_class.from_pretrained(args.base, dtype=dtype, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False,
                                      autocast_adapter_dtype=False, local_files_only=True)

    # 合并前后各取一个模块的权重范数，用来证明合并不是空操作——若 LoRA 张量全为零，
    # 合并会"成功"但结果与基座完全相同，这种错误必须能看出来。
    sample_key = None
    for name, _ in model.named_parameters():
        if ".layers.0." in name and name.endswith(".weight") and "lora" not in name:
            sample_key = name
            break
    before = None
    if sample_key:
        before = float(model.get_parameter(sample_key).detach().float().norm())

    merged = model.merge_and_unload()
    merged.eval()

    if sample_key and before is not None:
        after = float(merged.get_parameter(sample_key).detach().float().norm())
        delta = abs(after - before)
        print(f"抽样模块 {sample_key}")
        print(f"  合并前范数 {before:.6f}  合并后 {after:.6f}  变化 {delta:.6f} "
              f"({delta / before * 100 if before else 0:.4f}%)")
        if delta == 0.0:
            print("  ⚠️ 变化为 0 —— 适配器可能是全零，或没有被真正合并")

    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="4GB")
    # 把 processor/tokenizer 相关文件一并复制，否则 vLLM 无法独立加载
    for name in ("preprocessor_config.json", "chat_template.jinja", "chat_template.json",
                 "generation_config.json", "configuration.json", "vocab.json", "merges.txt",
                 "tokenizer.json", "tokenizer_config.json", "config.json"):
        source = Path(args.base) / name
        if source.is_file():
            shutil.copy2(source, args.out / name)
    print(f"已保存合并模型 -> {args.out}")
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"体积 {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
