#!/usr/bin/env bash
# 把 flash-linear-attention (fla) 装进**主 conda 环境**，让 Gated DeltaNet 走优化 kernel。
#
# 背景（这是训练慢的真正原因）：
#   transformers 的 qwen3_5 建模代码对 Gated DeltaNet 用了
#       @use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
#   其优先级为 ① HF Hub kernels → ② 原包(fla) → ③ 纯 torch 兜底。
#   本机 fla 与 causal_conv1d 都没装，于是 48 层线性注意力一直跑第 ③ 档纯 PyTorch 实现，
#   实测只有约 43 TFLOPS（该卡 bf16 稠密算力应在 250+ TFLOPS）。
#
# 为什么装在主环境而不是 venv：训练由 ms-swift 在主 conda 环境里跑，kernel 必须对它可见。
# 因此装完必须校验 **torch / transformers / peft / ms-swift 版本一个都没变**。
#
# fla 的 chunk_gated_delta_rule 是 Triton JIT（triton 已装），不需要编译 CUDA 扩展；
# causal-conv1d 需要编译，Blackwell + CUDA 13 有失败风险，因此本脚本默认不装它。
set -uo pipefail

PY=/root/miniconda3/bin/python
MIRROR="${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"
INSTALL_CAUSAL_CONV1D="${INSTALL_CAUSAL_CONV1D:-0}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

snapshot() {
  "${PY}" - <<'PYEOF'
import importlib.metadata as md
for name in ("torch", "transformers", "peft", "accelerate", "ms-swift", "tokenizers", "triton", "einops"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=<missing>")
PYEOF
}

say "0/5 安装前主环境快照"
BEFORE="$(snapshot)"
echo "${BEFORE}"

say "1/5 安装 flash-linear-attention"
"${PY}" -m pip install -U flash-linear-attention -i "${MIRROR}" 2>&1 | tail -4
fla_status=${PIPESTATUS[0]}

if [ "${INSTALL_CAUSAL_CONV1D}" = "1" ]; then
  say "1b/5 安装 causal-conv1d（需编译 CUDA 扩展，有失败风险）"
  "${PY}" -m pip install -U causal-conv1d -i "${MIRROR}" 2>&1 | tail -4 || true
fi

say "2/5 校验 fla 可导入且提供所需 kernel"
"${PY}" - <<'PYEOF'
try:
    import fla
    print("fla", getattr(fla, "__version__", "<unknown>"))
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    print("chunk_gated_delta_rule 可用: True")
except Exception as error:
    print("fla 校验失败:", type(error).__name__, error)
PYEOF
fla_import=$?

say "3/5 校验 transformers 是否真的会用到它（装饰器在 import 时解析）"
"${PY}" - <<'PYEOF'
try:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    func = m.torch_chunk_gated_delta_rule
    module = getattr(func, "__module__", "?")
    wrapped = getattr(func, "__wrapped__", None)
    print("装饰后函数模块:", module)
    print("包装层:", getattr(wrapped, "__module__", None))
    print("是否来自 fla:", "fla" in str(module) or "fla" in str(getattr(wrapped, "__module__", "")))
except Exception as error:
    print("transformers 侧校验失败:", type(error).__name__, error)
PYEOF
tr_status=$?

say "4/5 安装后主环境快照（除新装包外必须逐行一致）"
AFTER="$(snapshot)"
echo "${AFTER}"
# 逐行比较：只允许出现新增行，不允许已有行发生变化
changed="$(diff <(echo "${BEFORE}") <(echo "${AFTER}") | grep -E '^[<>]' | grep -vE '^(> .*<missing>$)' || true)"
if [ -z "${changed}" ]; then
  echo "MAIN_ENV_UNCHANGED"
else
  echo "MAIN_ENV_CHANGED —— 已有依赖被改动，需排查"
  echo "${changed}"
fi

say "5/5 结论"
echo "fla_pip=${fla_status} fla_import=${fla_import} transformers=${tr_status}"
echo "FLA_SETUP_DONE"
