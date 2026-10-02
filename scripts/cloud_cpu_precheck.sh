#!/usr/bin/env bash
# 在新实例上做「无卡阶段」的 CPU 侧验收：确认代码、数据、processor、模板编码都对。
# 全部只读或写临时目录，不碰训练产物。
set -uo pipefail

CODE=/root/AIC_code
PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/rematch_20260922
MANIFEST="${ROOT}/results/triground_abv_execution_20260928/deployment_600_seed2026/manifests/A.json"
VAL="${ROOT}/results/triground_abv_20260927/inputs/city_val.json"
GT="${ROOT}/results/triground_abv_20260927/inputs/city_gt.json"
DATA_ROOT="${ROOT}/data/city/train"
OUT=/root/sprint_qwen36
mkdir -p "${OUT}"

echo "=== 1) 数据完整性 ==="
for d in visible infrared target_v2/qwen3vl_native_sft/depth_rgb; do
  printf '%-42s %s files\n' "$d" "$(find "${DATA_ROOT}/$d" -type f 2>/dev/null | wc -l)"
done
du -sh "${DATA_ROOT}" /root/rematch_models/* 2>/dev/null
ls -la "${ROOT}/results/triground_abv_execution_20260928/runs/seed2026_600_deterministic/A/main/adapter_model.safetensors"

echo "=== 2) 清单转换（A.json -> ms-swift 格式）==="
cd "${CODE}"
"${PY}" tools/convert_manifest_to_swift.py --input "${MANIFEST}" --output "${OUT}/a_city_swift.jsonl" || exit 1
wc -l "${OUT}/a_city_swift.jsonl"

echo "=== 3) 单元测试（含真实清单无损与防泄漏断言）==="
"${PY}" -m pytest tests/test_convert_manifest_to_swift.py -q 2>&1 | tail -5

echo "=== 4) 部署预检（无卡会如实报 cuda_available=false）==="
"${PY}" tools/preflight_new_base.py --model /root/rematch_models/Qwen3.6-27B \
    --pixels 602112 1204224 2073600 --disk-paths /root/autodl-tmp 2>&1 | tail -45

echo "=== 5) 探针 dry-run（token 预算）==="
"${PY}" tools/probe_native_resolution.py --dry-run \
    --model /root/rematch_models/Qwen3.6-27B \
    --manifest "${VAL}" --target-manifest "${GT}" --data-root "${DATA_ROOT}" \
    --output-dir "${OUT}/dryrun" --pixels 602112 1204224 2073600 --model-max-length 16384 2>&1 | tail -30

echo "=== 6) 真实模板编码（多条样本：验证 <image> 替换、监督完整、不超长被截断）==="
"${PY}" - <<'EOF'
import json, os
os.environ.setdefault("MAX_PIXELS", "1204224")
from swift import get_processor, get_template

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "4096"))
rows = [json.loads(line) for line in open("/root/sprint_qwen36/a_city_swift.jsonl", encoding="utf-8")]
# 取等距样本，兼顾不同图组与低分辨率图
sample = [rows[i] for i in range(0, len(rows), max(1, len(rows)//8))][:8]

processor = get_processor("/root/rematch_models/Qwen3.6-27B")
template = get_template(processor)
template.set_mode("train")

print(f"{'id':<44} {'input_ids':>9} {'supervised':>10} {'images':>6} {'over_max':>8}")
bad = []
for row in sample:
    encoded = template.encode(row, return_template_inputs=True)
    inputs = encoded["template_inputs"]
    n_ids = len(encoded["input_ids"])
    supervised = sum(1 for x in encoded["labels"] if x != -100)
    n_images = len(getattr(inputs, "images", []) or [])
    over = n_ids > MAX_LENGTH
    if over or supervised == 0 or n_images != 3:
        bad.append(row["id"])
    print(f"{row['id'][:44]:<44} {n_ids:9d} {supervised:10d} {n_images:6d} {str(over):>8}")

print("[LABELS 首条]", template.safe_decode(encoded["labels"])[:160].replace("\n", "\\n"))
grid = getattr(inputs, "image_grid_thw", None)
print("[GRID]", grid)
print("[MAX_LENGTH]", MAX_LENGTH)
assert not bad, f"以下样本有问题（超长/无监督/图像数不对）：{bad}"
print("SUPERVISION_OK")
EOF

echo "=== 7) ms-swift 训练入口可用性（只打印帮助，不训练）==="
/root/miniconda3/bin/swift sft --help 2>&1 | head -5
echo "CPU_PRECHECK_DONE"
