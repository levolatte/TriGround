#!/usr/bin/env bash
# 在**独立 venv** 里安装 vLLM，绝不触碰已验证的训练环境。
#
# 为什么必须隔离：vLLM 会拉取自己 pin 的 torch 版本，直接装进 conda 主环境会覆盖
# torch 2.12.1+cu130 / transformers 5.16.1 / ms-swift 4.5.3 —— 这三者是刚刚用
# 7/7 云端预检和 12 项单测验证过的训练栈，一旦被动过，训练结果就不再可信。
#
# 安装完成后会同时校验：
#   1) 独立环境里 vLLM 可导入、版本可用；
#   2) vLLM 的模型注册表里存在 Qwen3_5ForConditionalGeneration（我们的稠密 27B）；
#   3) **主环境版本号逐项未变**（这是隔离是否成功的判据）。
set -uo pipefail

VENV="${VENV:-/root/vllm_env}"
MAIN_PY=/root/miniconda3/bin/python
MIRROR="${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

snapshot_main() {
  "${MAIN_PY}" - <<'PYEOF'
import importlib.metadata as md
for name in ("torch", "transformers", "peft", "accelerate", "ms-swift", "tokenizers"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=<missing>")
PYEOF
}

say "0/5 记录主环境版本快照"
BEFORE="$(snapshot_main)"
echo "${BEFORE}"

say "1/5 创建独立 venv ${VENV}"
if [ ! -x "${VENV}/bin/python" ]; then
  /root/miniconda3/bin/python -m venv "${VENV}"
fi
"${VENV}/bin/python" -m pip install -q --upgrade pip -i "${MIRROR}" || true

say "2/5 安装 vLLM（这一步会拉取自己的 torch，属预期行为）"
"${VENV}/bin/python" -m pip install -U vllm -i "${MIRROR}" 2>&1 | tail -5
install_status=${PIPESTATUS[0]}
say "pip 退出码 ${install_status}"

say "3/5 校验 vLLM 可导入"
"${VENV}/bin/python" - <<'PYEOF'
import vllm
print("vllm", vllm.__version__)
import torch
print("venv torch", torch.__version__, "cuda", torch.version.cuda)
print("sm_120 可用:", torch.cuda.get_device_capability() if torch.cuda.is_available() else "无 CUDA")
PYEOF
vllm_status=$?

say "4/5 校验模型注册表是否含稠密 Qwen3.5/3.6"
"${VENV}/bin/python" - <<'PYEOF'
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
hits = sorted(a for a in archs if "Qwen3_5" in a or "Qwen3.5" in a)
print("匹配到的架构:", hits if hits else "无")
target = "Qwen3_5ForConditionalGeneration"
print("目标架构", target, "->", "支持" if target in archs else "不支持")
PYEOF
registry_status=$?

say "5/5 校验主环境是否被改动"
AFTER="$(snapshot_main)"
echo "${AFTER}"
if [ "${BEFORE}" = "${AFTER}" ]; then
  echo "MAIN_ENV_UNCHANGED"
else
  echo "MAIN_ENV_CHANGED —— 隔离失败，需要排查"
  diff <(echo "${BEFORE}") <(echo "${AFTER}") || true
fi

echo "---"
echo "vllm_import=${vllm_status} registry=${registry_status} install=${install_status}"
if [ "${vllm_status}" -eq 0 ] && [ "${registry_status}" -eq 0 ]; then
  echo "VLLM_ENV_OK"
else
  echo "VLLM_ENV_PROBLEM"
fi
