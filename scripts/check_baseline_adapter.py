"""检查基线 A 的 LoRA 适配器与新环境（transformers 5.16.1 / peft 0.20.0）是否兼容。

不加载基座，只读 adapter_config.json 与 safetensors 键名，确认：
  * 目标模块名与 Qwen3-VL-8B 的语言层命名一致；
  * 张量数量与配置里的目标模块数相符；
  * 记下 base_model_name_or_path，避免换基座时误挂。
"""

import json
import re
import sys

from safetensors import safe_open

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/autodl-tmp/rematch_20260922/results/triground_abv_execution_20260928"
    "/runs/seed2026_600_deterministic/A/main"
)

config = json.load(open(f"{ADAPTER}/adapter_config.json", encoding="utf-8"))
targets = config.get("target_modules") or []
# target_modules 可以写成短名（q_proj/k_proj/...），实际按层展开；因此层号从张量键里数。
explicit_layers = sorted({int(m.group(1)) for name in targets
                          if (m := re.search(r"layers\.(\d+)\.", name))})

with safe_open(f"{ADAPTER}/adapter_model.safetensors", framework="pt") as handle:
    keys = list(handle.keys())
key_layers = sorted({int(m.group(1)) for key in keys if (m := re.search(r"layers\.(\d+)\.", key))})
key_modules = sorted({key.split(".self_attn.")[1].split(".")[0] for key in keys if ".self_attn." in key})

print("peft_type       :", config.get("peft_type"))
print("r / alpha / drop:", config.get("r"), config.get("lora_alpha"), config.get("lora_dropout"))
print("base_model      :", config.get("base_model_name_or_path"))
print("target_modules  :", targets, f"（{len(targets)} 个短名）" if explicit_layers == [] else "")
print("张量数          :", len(keys))
print("张量覆盖层号    :", key_layers[:3], "...", key_layers[-3:], f"共 {len(key_layers)} 层")
print("张量覆盖模块    :", key_modules)
expected = len(key_layers) * len(key_modules) * 2
print(f"期望张量数      : {len(key_modules)} 模块 x {len(key_layers)} 层 x 2(A/B) = {expected}")
print("一致性          :", "OK" if expected == len(keys) else "不一致，需人工确认")
