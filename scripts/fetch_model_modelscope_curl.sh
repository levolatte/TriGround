#!/usr/bin/env bash
# 从 ModelScope 按文件清单下载模型，带 sha256 校验、断点续传与重试。
#
# 为什么需要这么麻烦（本项目实测三次踩坑）：
#   1. hf-mirror 跑到 0.27GB 后静默卡死；
#   2. modelscope CLI 在 ~13GB 处卡死（根因是 30GB 系统盘被写满）；
#   3. 直接把遗留 *.incomplete 改名后交给 `curl -C -` 续传，在**并发写同一文件**后
#      出现文件**超过目标大小**（如 model-00013 多出 999MB），而续传只会继续追加，
#      永远追不上 —— 静默产出损坏权重。
#
# 因此这里：
#   * 用官方 API 的 Size + **Sha256** 作为唯一判据（下载完整性属于系统边界，值得校验）；
#   * 统一下到 `<name>.part`，只有大小与 sha256 都通过才改名成正式文件；
#   * 大小超出预期直接删除重下，绝不续传。
set -uo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
REVISION="${REVISION:-master}"
TARGET="${TARGET:-/root/autodl-tmp/rematch_models/${MODEL_ID##*/}}"
PARALLEL="${PARALLEL:-6}"
PASSES="${PASSES:-8}"
PY=/root/miniconda3/bin/python
BASE_URL="https://www.modelscope.cn/models/${MODEL_ID}/resolve/${REVISION}"
API_URL="https://www.modelscope.cn/api/v1/models/${MODEL_ID}/repo/files?Revision=${REVISION}"
MANIFEST=/tmp/ms_manifest.tsv

# 单实例锁：本项目实测过"上一轮的 curl 没被杀掉 + 新一轮同时写同一个文件"，
# 结果分片被追加到超过目标大小。锁 + 唯一 .part 名双重保证不会再有并发写同名文件。
exec 9>/var/lock/fetch_model.lock
if ! flock -n 9; then
  echo "已有另一个下载实例在运行，退出以免并发写同一文件" >&2
  exit 4
fi

mkdir -p "${TARGET}"
cd "${TARGET}"

echo "[$(date +%H:%M:%S)] 取文件清单"
curl -sL --retry 5 --retry-delay 3 --max-time 120 "${API_URL}" -o /tmp/ms_files.json || exit 1
"${PY}" - <<'PYEOF' > "${MANIFEST}"
import json
data = json.load(open("/tmp/ms_files.json", encoding="utf-8"))
rows = [(f["Path"], int(f["Size"]), f.get("Sha256") or "-")
        for f in data["Data"]["Files"] if f.get("Type") == "blob"]
# 大文件优先，便于并行铺开
rows.sort(key=lambda r: (not r[0].endswith(".safetensors"), -r[1]))
for path, size, sha in rows:
    print(f"{path}\t{size}\t{sha}")
PYEOF
TOTAL_FILES=$(wc -l < "${MANIFEST}")
# 注意两个坑：awk 的 print 会输出 5.55861e+10（bash 算术报错），而 printf "%d" 在 awk 里
# 会对超过 2^31 的值溢出成 2147483647。用 %.0f 才能拿到完整整数。
TOTAL_BYTES=$(awk -F'\t' '{s+=$2} END {printf "%.0f\n", s}' "${MANIFEST}")
echo "[$(date +%H:%M:%S)] ${TOTAL_FILES} 个文件，共 ${TOTAL_BYTES} 字节"

AVAIL_GIB=$(df -BG --output=avail "${TARGET}" | tail -1 | tr -dc '0-9')
NEED_GIB=$(( TOTAL_BYTES / 1024 / 1024 / 1024 + 5 ))
if [ "${AVAIL_GIB}" -lt "${NEED_GIB}" ]; then
  echo "目标文件系统仅剩 ${AVAIL_GIB} GiB，需要约 ${NEED_GIB} GiB：${TARGET}" >&2
  df -h "${TARGET}" >&2
  exit 3
fi
echo "[$(date +%H:%M:%S)] 可用 ${AVAIL_GIB} GiB（需要 ${NEED_GIB} GiB）"

verify() {  # $1=path $2=sha  -> 0 通过（无 sha 时只做存在性）
  local path="$1" sha="$2"
  [ "${sha}" = "-" ] && return 0
  echo "${sha}  ${path}" | sha256sum -c --status -
}

fetch() {  # $1=path $2=size $3=sha256
  local path="$1" size="$2" sha="$3"
  local part="${path}.part"

  if [ -f "${path}" ] && [ "$(stat -c%s "${path}")" -eq "${size}" ] && verify "${path}" "${sha}"; then
    return 0
  fi
  # 正式名存在但大小或校验不符 -> 不可信，删除重来
  [ -f "${path}" ] && rm -f "${path}"
  # 超出目标大小的残片绝不可续传
  if [ -f "${part}" ] && [ "$(stat -c%s "${part}")" -gt "${size}" ]; then
    rm -f "${part}"
  fi

  curl -L -C - --retry 6 --retry-delay 4 --retry-all-errors \
       --connect-timeout 20 --max-time 7200 \
       -o "${part}" "${BASE_URL}/${path}" >/dev/null 2>&1

  local got=0
  [ -f "${part}" ] && got=$(stat -c%s "${part}")
  if [ "${got}" -ne "${size}" ]; then
    echo "  size-mismatch ${path}: ${got}/${size}"
    return 1
  fi
  if ! verify "${part}" "${sha}"; then
    echo "  sha-mismatch  ${path}"
    rm -f "${part}"
    return 1
  fi
  mv -f "${part}" "${path}"
  echo "  ok  ${path}"
}
export -f fetch verify
export BASE_URL

for pass in $(seq 1 "${PASSES}"); do
  echo "[$(date +%H:%M:%S)] === 第 ${pass}/${PASSES} 轮 ==="
  xargs -P "${PARALLEL}" -a "${MANIFEST}" -d '\n' bash -c \
    'IFS=$'"'"'\t'"'"' read -r p s h <<< "$0"; fetch "$p" "$s" "$h"' || true
  done_bytes=$(du -sb . | cut -f1)
  remaining=$(awk -F'\t' '{print $1"\t"$2"\t"$3}' "${MANIFEST}" | while IFS=$'\t' read -r p s h; do
      if [ -f "$p" ] && [ "$(stat -c%s "$p")" -eq "$s" ]; then :; else echo "$p"; fi
    done | wc -l)
  echo "[$(date +%H:%M:%S)] 已占 ${done_bytes} / ${TOTAL_BYTES} 字节（$(awk -v a="$done_bytes" -v b="$TOTAL_BYTES" 'BEGIN{printf "%.1f", 100*a/b}')%），未完成 ${remaining} 个"
  [ "${remaining}" -eq 0 ] && break
  sleep 3
done
