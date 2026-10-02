#!/usr/bin/env bash
# 在新实例下载 Qwen3.6-27B（约 54GB BF16）。
#
# 源选择（2026-10-02 实测）：HF 镜像 hf-mirror 在本机下载 **卡死**（20 秒零增长，
# 停在 0.27GB），而 ModelScope 上有同一个官方仓库 `Qwen/Qwen3.6-27B`，国内链路快。
# 因此默认走 ModelScope，HF 只作为备选。
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
TARGET="${TARGET:-/root/rematch_models/Qwen3.6-27B}"
SOURCE="${SOURCE:-modelscope}"          # modelscope | hf
PY=/root/miniconda3/bin/python

mkdir -p "$(dirname "${TARGET}")"

case "${SOURCE}" in
  modelscope)
    CLI=/root/miniconda3/bin/modelscope
    [ -x "${CLI}" ] || { echo "缺少 modelscope CLI" >&2; exit 2; }
    echo "=== ModelScope 下载 ${MODEL_ID}"
    "${CLI}" download --model "${MODEL_ID}" --local_dir "${TARGET}"
    ;;
  hf)
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    CLI=""
    for candidate in /root/miniconda3/bin/hf /root/miniconda3/bin/huggingface-cli; do
      [ -x "${candidate}" ] && CLI="${candidate}" && break
    done
    [ -n "${CLI}" ] || { echo "缺少 hf CLI" >&2; exit 2; }
    echo "=== HF 下载 ${MODEL_ID} via ${HF_ENDPOINT}"
    "${CLI}" download "${MODEL_ID}" --local-dir "${TARGET}"
    ;;
  *)
    echo "SOURCE 只能是 modelscope 或 hf" >&2; exit 2;;
esac

echo "=== 校验权重完整性"
"${PY}" - <<EOF
import json, pathlib
target = pathlib.Path("${TARGET}")
config = json.loads((target / "config.json").read_text())
print("architectures:", config.get("architectures"))
print("model_type:", config.get("model_type"))
print("deepstack:", (config.get("vision_config") or {}).get("deepstack_visual_indexes"))
shards = sorted(target.glob("*.safetensors"))
index = target / "model.safetensors.index.json"
if index.exists():
    wanted = set(json.loads(index.read_text())["weight_map"].values())
    have = {p.name for p in shards}
    missing = sorted(wanted - have)
    print(f"shards: {len(have)}/{len(wanted)}  missing={missing[:5]}")
    assert not missing, "分片缺失"
for name in ("preprocessor_config.json", "tokenizer_config.json", "chat_template.json"):
    path = target / name
    print(f"{name}: {'OK' if path.exists() else 'MISSING'}")
total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
print(f"files={sum(1 for p in target.rglob('*') if p.is_file())}  size={total/2**30:.1f} GiB")
assert config.get("architectures") == ["Qwen3_5ForConditionalGeneration"], "架构不符"
print("MODEL_OK")
EOF
df -h /root/autodl-tmp | tail -1
echo "DOWNLOAD_DONE"
