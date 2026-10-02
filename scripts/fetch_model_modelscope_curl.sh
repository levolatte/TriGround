#!/usr/bin/env bash
# 从 ModelScope 按文件清单下载 Qwen3.6-27B，带断点续传、重试与大小校验。
#
# 为什么不用 modelscope CLI：实测两次在 ~13GB 处**静默卡死**（进程仍在，速率恒为 0），
# 且它把 15 个分片并发拉到一半就把其余分片建成 0 字节文件。这里改为：
#   1) 用官方 API 取文件清单与字节大小；
#   2) 把遗留的 *.incomplete 改回正式名，交给 curl -C - 续传；
#   3) 逐文件 curl --retry，多轮扫描直到全部大小吻合；
#   4) 最后按 model.safetensors.index.json 校验分片集合完整。
set -uo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
REVISION="${REVISION:-master}"
TARGET="${TARGET:-/root/autodl-tmp/rematch_models/Qwen3.6-27B}"
PARALLEL="${PARALLEL:-3}"
PASSES="${PASSES:-6}"
PY=/root/miniconda3/bin/python
BASE_URL="https://www.modelscope.cn/models/${MODEL_ID}/resolve/${REVISION}"
API_URL="https://www.modelscope.cn/api/v1/models/${MODEL_ID}/repo/files?Revision=${REVISION}"

mkdir -p "${TARGET}"
cd "${TARGET}"

# 磁盘守卫：权重必须落在有大空间的数据盘上。
# 本项目已踩过这个坑：把 27B(54G)+8B(17G) 放到 /root/rematch_models（30GB 系统盘 overlay），
# 结果系统盘 100% 写满，curl 报 "No space left on device" 而下载静默卡死。
AVAIL_GIB=$(df -BG --output=avail "${TARGET}" | tail -1 | tr -dc '0-9')
NEED_GIB=$(( ${TOTAL_BYTES:-55586109578} / 1024 / 1024 / 1024 + 5 ))
if [ "${AVAIL_GIB}" -lt "${NEED_GIB}" ]; then
  echo "目标文件系统仅剩 ${AVAIL_GIB} GiB，需要约 ${NEED_GIB} GiB：${TARGET}" >&2
  echo "请把权重放到数据盘（例如 /root/autodl-tmp/rematch_models）并建立同名符号链接。" >&2
  df -h "${TARGET}" >&2
  exit 3
fi
echo "[$(date +%H:%M:%S)] 目标 ${TARGET} 可用 ${AVAIL_GIB} GiB（需要 ${NEED_GIB} GiB）"

echo "[$(date +%H:%M:%S)] 取文件清单"
curl -sL --retry 5 --retry-delay 3 --max-time 120 "${API_URL}" -o /tmp/ms_files.json || exit 1
"${PY}" - <<'PYEOF' > /tmp/ms_pairs.txt
import json
data = json.load(open("/tmp/ms_files.json", encoding="utf-8"))
files = data["Data"]["Files"]
rows = [(f["Path"], int(f["Size"])) for f in files if f.get("Type") == "blob"]
rows.sort(key=lambda r: (not r[0].endswith(".safetensors"), -r[1]))
for path, size in rows:
    # 文件名不含空格，用空格分隔两列，便于 xargs -n 2 直接拆成两个参数
    print(f"{path} {size}")
PYEOF
TOTAL=$(grep -c . /tmp/ms_pairs.txt)
TOTAL_BYTES=$(awk '{s+=$2} END {print s}' /tmp/ms_pairs.txt)
echo "[$(date +%H:%M:%S)] 清单 ${TOTAL} 个文件，共 ${TOTAL_BYTES} 字节"

# 遗留的 .incomplete 改回正式名，让 curl 能续传
find . -name '*.incomplete' -size +0c -print0 | while IFS= read -r -d '' f; do mv -f "$f" "${f%.incomplete}"; done
find . -name '*.incomplete' -size 0c -delete

download_one() {
  local path="$1" size="$2"
  local current=0
  [ -f "$path" ] && current=$(stat -c%s "$path")
  if [ "$current" -eq "$size" ]; then return 0; fi
  curl -L -C - --retry 8 --retry-delay 5 --retry-all-errors \
       --connect-timeout 20 --max-time 7200 \
       -o "$path" "${BASE_URL}/${path}" >/dev/null 2>&1
  local after=0
  [ -f "$path" ] && after=$(stat -c%s "$path")
  [ "$after" -eq "$size" ] && echo "  ok  $path ($(numfmt --to=iec "$after" 2>/dev/null || echo "$after"))"
}
export -f download_one
export BASE_URL

for pass in $(seq 1 "${PASSES}"); do
  echo "[$(date +%H:%M:%S)] === 第 ${pass}/${PASSES} 轮 ==="
  xargs -P "${PARALLEL}" -n 2 -a /tmp/ms_pairs.txt bash -c 'download_one "$0" "$1"' || true
  done_bytes=$(du -sb . | cut -f1)
  echo "[$(date +%H:%M:%S)] 已下载 ${done_bytes} / ${TOTAL_BYTES} 字节 ($(( 100 * done_bytes / TOTAL_BYTES ))%)"
  missing=$(while read -r p s; do
      [ -f "$p" ] && [ "$(stat -c%s "$p")" -eq "$s" ] || echo "$p"
    done < /tmp/ms_pairs.txt | wc -l)
  echo "[$(date +%H:%M:%S)] 未完成 ${missing} 个"
  [ "${missing}" -eq 0 ] && break
  sleep 3
done

echo "=== 校验 ==="
"${PY}" - <<PYEOF
import json, pathlib, sys
target = pathlib.Path("${TARGET}")
expected = {}
for line in open("/tmp/ms_pairs.txt", encoding="utf-8"):
    path, size = line.split()
    expected[path] = int(size)
bad = []
for path, size in expected.items():
    actual = (target / path).stat().st_size if (target / path).exists() else -1
    if actual != size:
        bad.append((path, size, actual))
print(f"文件 {len(expected)} 个，大小不符 {len(bad)} 个")
for path, size, actual in bad[:8]:
    print(f"  MISMATCH {path}: expect {size} got {actual}")
index = target / "model.safetensors.index.json"
if index.exists():
    wanted = set(json.loads(index.read_text())["weight_map"].values())
    have = {p.name for p in target.glob("*.safetensors")}
    missing = sorted(wanted - have)
    print(f"分片 {len(have)}/{len(wanted)}，缺失 {missing[:5]}")
    bad += [(m, 0, 0) for m in missing]
config = json.loads((target / "config.json").read_text())
print("architectures:", config.get("architectures"))
total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
print(f"总大小 {total/2**30:.1f} GiB")
sys.exit(1 if bad else 0)
PYEOF
status=$?
df -h /root/autodl-tmp | tail -1
if [ "${status}" -eq 0 ]; then echo "MODEL_DOWNLOAD_OK"; else echo "MODEL_DOWNLOAD_INCOMPLETE"; fi
