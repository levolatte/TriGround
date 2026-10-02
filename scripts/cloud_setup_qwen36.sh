#!/usr/bin/env bash
# 新实例（RTX PRO 6000 / Blackwell sm_120）环境安装。
#
# 事实约束（2026-10-02 实查）：
#   - 镜像自带 python 3.12.3 + torch 2.12.1+cu130 + torchvision 0.27.1+cu130（支持 Blackwell）
#   - 没有 transformers / peft / accelerate / ms-swift
#   - 本地 transformers 4.57.3 连 qwen3_5 的 config 都读不了，必须 >= 5.2.0
#   - 不装 flash-attn：我们用 sdpa，规避 Blackwell 轮子缺失
set -euo pipefail

PY=/root/miniconda3/bin/python
PIP=/root/miniconda3/bin/pip
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

TORCH_BEFORE="$(${PY} -c 'import torch;print(torch.__version__)')"
echo "=== torch before: ${TORCH_BEFORE}"

echo "=== 1) 基础依赖"
${PIP} install -U pip wheel
${PIP} install -U "transformers>=5.2.0" "qwen_vl_utils>=0.0.14" decord \
    peft accelerate datasets pillow numpy tensorboard "huggingface_hub[cli]"

echo "=== 2) ms-swift（官方支持 Qwen3.6）"
${PIP} install -U ms-swift

TORCH_AFTER="$(${PY} -c 'import torch;print(torch.__version__)')"
echo "=== torch after: ${TORCH_AFTER}"
if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
  echo "!!! ms-swift 改动了 torch（${TORCH_BEFORE} -> ${TORCH_AFTER}），回滚到镜像自带版本"
  ${PIP} install "torch==${TORCH_BEFORE%%+*}" torchvision --index-url https://download.pytorch.org/whl/cu130
fi

echo "=== 3) 关键能力自检"
${PY} - <<'EOF'
import importlib, sys
import transformers
print("python", sys.version.split()[0])
print("transformers", transformers.__version__)
from transformers import Qwen3_5ForConditionalGeneration  # noqa: F401
print("Qwen3_5ForConditionalGeneration: OK")
import torch
print("torch", torch.__version__, "cuda_build", torch.version.cuda, "available", torch.cuda.is_available())
for name in ("peft", "accelerate", "deepspeed", "qwen_vl_utils", "decord", "swift"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'n/a')}")
    except Exception as error:
        print(f"{name}: MISSING ({type(error).__name__})")
EOF

echo "=== 4) hf CLI 与镜像连通"
which hf huggingface-cli || true
export HF_ENDPOINT=https://hf-mirror.com
timeout 30 hf env 2>/dev/null | head -8 || timeout 30 huggingface-cli env 2>/dev/null | head -8 || true
echo "SETUP_DONE"
