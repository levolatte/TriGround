"""列出 ms-swift 4.5.3 `SftArguments` 的全部字段，确定 LoRA 的正确选择参数。

背景：`swift sft` 支持 `--ignore_args_error True`，参数名写错可能被静默忽略，
导致本意是 LoRA 的训练实际退化成全量微调（27B 全量在 96GB 上必然 OOM）。
因此必须在开卡前把参数名钉死，而不是靠记忆。
"""

import dataclasses
import re

from swift.arguments import SftArguments

fields = [f.name for f in dataclasses.fields(SftArguments)]
print("字段总数:", len(fields))

print("\n--- 含 type 的字段 ---")
print([f for f in fields if "type" in f])

print("\n--- 含 tuner / adapter / lora 的字段 ---")
print([f for f in fields if re.search(r"tuner|adapter|lora", f)])

print("\n--- 含 freeze 的字段 ---")
print([f for f in fields if "freeze" in f])

print("\n--- 含 pixel 的字段 ---")
print([f for f in fields if "pixel" in f])

for name in ("train_type", "tuner_type", "tuner_backend", "peft_type", "model_type"):
    print(f"{name}: {'存在' if name in fields else '不存在'}")
