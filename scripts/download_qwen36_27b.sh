#!/usr/bin/env bash
# 在新实例下载 Qwen3.6-27B（约 54GB BF16）。
#
# 依赖：先跑完 cloud_setup_qwen36.sh（需要 huggingface_hub CLI）。
# 事实：Qwen3.6-27B 无官方 4-bit，只有 FP8；本机数据盘 200G，54G 权重放得下。
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
TARGET="${TARGET:-/root/rematch_models/Qwen3.6-27B}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$(dirname "${TARGET}")"

CLI=""
for candidate in /root/miniconda3/bin/hf /root/miniconda3/bin/huggingface-cli; do
  [ -x "${candidate}" ] && CLI="${candidate}" && break
done
if [ -z "${CLI}" ]; then
  echo "找不到 hf / huggingface-cli，先执行 cloud_setup_qwen36.sh" >&2
  exit 2
fi
echo "using CLI: ${CLI}  endpoint: ${HF_ENDPOINT}"

# 先单文件测速，避免 54GB 下到一半才发现链路不可用
echo "=== 单文件测速"
timeout 120 curl -sL -o /dev/null -w 'config.json speed=%{speed_download} B/s http=%{http_code}\n' \
  "${HF_ENDPOINT}/${MODEL_ID}/resolve/main/config.json" || true

echo "=== 全量下载（支持断点续传，可重复执行）"
"${CLI}" download "${MODEL_ID}" --local-dir "${TARGET}"

echo "=== 校验"
/root/miniconda3/bin/python - <<EOF
import json, pathlib
target = pathlib.Path("${TARGET}")
config = json.loads((target / "config.json").read_text())
print("architectures:", config.get("architectures"))
print("model_type:", config.get("model_type"))
print("vision deepstack:", (config.get("vision_config") or {}).get("deepstack_visual_indexes"))
total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
print(f"files: {sum(1 for p in target.rglob('*') if p.is_file())}  size: {total/2**30:.1f} GiB")
assert config.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "架构不符"
print("MODEL_OK")
EOF
df -h /root/autodl-tmp | tail -1
echo "DOWNLOAD_DONE"
