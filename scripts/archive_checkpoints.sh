#!/usr/bin/env bash
# 检查点归档看护：把每个检查点里**推理真正需要的那部分**硬链接到安全目录。
#
# ## 为什么需要它
#
# 本轮训练用 `--save_total_limit 2`，只保留最后两个检查点（500/600）。若最佳步数在 300，
# 它会被训练器在保存 500 时删掉，**永远看不到**。而训练已经跑了数小时，重启去改这个参数
# 代价太高。这个看护在**不打断训练**的前提下补上可观测性。
#
# ## 为什么是硬链接
#
# 检查点 2.7 GB 中，`optimizer.pt` 占 1.78 GB，但**它只用于断点续训，评估完全用不到**。
# 评估需要的是 `adapter_model.safetensors`（891 MB）+ 若干小配置。
# 用硬链接（`ln`）共享 inode：**零额外磁盘**、瞬间完成，且训练器之后 rmtree 掉原目录时，
# 我们这边的链接仍然有效（删的是目录项，不是 inode）。
#
# ## 为什么要等文件稳定
#
# 检查点目录是在保存开始时出现的，此时 safetensors 可能还没写完。这里在链接前
# 连续两次采样文件大小，只有稳定后才归档，避免锁住一个半成品。
set -uo pipefail

RUN_DIR="${RUN_DIR:-/root/runs/q36_600_2073600}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${RUN_DIR}/archived}"
LOG="${LOG:-/root/checkpoint_archive.log}"
POLL_SECONDS="${POLL_SECONDS:-60}"
# 推理需要的文件；optimizer.pt / scheduler.pt / rng_state.pth 不归档
KEEP=(adapter_model.safetensors adapter_config.json additional_config.json args.json)

exec > >(tee -a "${LOG}") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "${ARCHIVE_DIR}"
say "看护启动：监视 ${RUN_DIR}，归档到 ${ARCHIVE_DIR}"

sha_of_sizes() {
  find "$1" -maxdepth 1 -type f -printf '%f:%s\n' 2>/dev/null | sort
}

while :; do
  for checkpoint in "${RUN_DIR}"/v*/checkpoint-*; do
    [ -d "${checkpoint}" ] || continue
    name="$(basename "${checkpoint}")"
    target="${ARCHIVE_DIR}/${name}"

    # 已经归档且含 safetensors 就跳过
    if [ -f "${target}/adapter_model.safetensors" ]; then
      continue
    fi

    first="$(sha_of_sizes "${checkpoint}")"
    sleep 5
    second="$(sha_of_sizes "${checkpoint}")"
    if [ "${first}" != "${second}" ] || [ -z "${second}" ]; then
      continue   # 还在写，下一轮再看
    fi
    if [ ! -f "${checkpoint}/adapter_model.safetensors" ]; then
      continue   # 关键文件还没出现
    fi

    mkdir -p "${target}"
    linked=0
    for file in "${KEEP[@]}"; do
      source="${checkpoint}/${file}"
      [ -f "${source}" ] || continue
      # -f 强制覆盖，避免半成品链接残留；同一文件系统才能硬链接
      if ln -f "${source}" "${target}/${file}" 2>/dev/null; then
        linked=$((linked + 1))
      else
        cp -f "${source}" "${target}/${file}"   # 跨文件系统时退化为复制
      fi
    done
    if [ "${linked}" -gt 0 ]; then
      say "已归档 ${name}（${linked} 个文件，硬链接）"
    fi
  done
  sleep "${POLL_SECONDS}"
done
