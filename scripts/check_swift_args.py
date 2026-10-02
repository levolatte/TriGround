"""在云端用权威字段集校验训练脚本用到的每个 ms-swift 参数。

为什么必须做这件事：
  * `swift sft --help` 是懒加载的，只列 6 行基础参数，不能用来验证；
  * ms-swift 支持 `--ignore_args_error`，**参数名写错会被静默忽略**；
  * 4.5.3 里 LoRA 的选择参数是 `--tuner_type`，不是早期版本的 `--train_type`。
    若沿用 `--train_type lora` 而被静默丢弃，27B 会走**全量微调**并在 96GB 上 OOM。

用法（云端）：python scripts/check_swift_args.py
"""

from __future__ import annotations

import dataclasses
import sys

from swift.arguments import SftArguments

FIELDS = {f.name for f in dataclasses.fields(SftArguments)}

# 训练脚本 scripts/train_qwen36_27b_swift.sh 实际传入的参数（去掉 --）
USED = [
    "model", "dataset", "output_dir", "tuner_type", "torch_dtype", "attn_impl",
    "num_train_epochs", "max_steps", "per_device_train_batch_size",
    "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "warmup_ratio",
    "max_grad_norm", "weight_decay", "lora_rank", "lora_alpha", "lora_dropout",
    "target_modules", "freeze_vit", "freeze_aligner", "freeze_parameters_ratio",
    "max_length", "max_pixels", "gradient_checkpointing", "save_strategy", "save_steps",
    "save_total_limit", "logging_steps", "dataloader_num_workers", "dataset_num_proc",
    "split_dataset_ratio", "seed", "report_to", "ignore_args_error",
]

missing = [name for name in USED if name not in FIELDS]
print(f"SftArguments 字段数 {len(FIELDS)}；脚本使用 {len(USED)} 个")
print("缺失参数:", missing if missing else "无（全部存在）")

if "tuner_type" in FIELDS:
    field = next(f for f in dataclasses.fields(SftArguments) if f.name == "tuner_type")
    print(f"tuner_type 默认值: {field.default!r}")
for name in ("freeze_vit", "freeze_aligner", "ignore_args_error"):
    if name in FIELDS:
        field = next(f for f in dataclasses.fields(SftArguments) if f.name == name)
        print(f"{name} 默认值: {field.default!r}")

sys.exit(1 if missing else 0)
