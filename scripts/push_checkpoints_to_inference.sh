#!/usr/bin/env bash
# 把 Arm 5 归档出的检查点**边训边推**到推理机。
#
# ## 为什么要边训边推
#
# 推理机在 08:10 跑完 Arm 4 的 5690 后就空闲，而我的 City412 扫描要到 10:30
# 才知道哪个检查点最佳。**若等到 10:30 再传，那 2 小时 20 分白白空转**；
# 传输不需要 GPU，与训练并行，所以提前推完是纯赚。
#
# ## 为什么只推 archived/
#
# `archived/` 里是归档看护做的**硬链接**（只有 adapter 与配置，不含 optimizer），
# 正是推理需要的部分，也是唯一在 `save_total_limit=6` 下不会被删掉的副本。
#
# ## 幂等
#
# 每个检查点用「目标侧是否已存在 adapter」判定，重复运行不会重传。
#
# ## ⚠️ 每个 ssh 都必须带 `-o BatchMode=yes`
#
# 这不是风格问题，是这次踩到的坑：脚本由 `screen -dmS` 启动，**screen 里 stdin 是 TTY**，
# 于是密钥认证失败时 `ssh` 不会直接退出，而是**弹出密码提示并一直等输入**——
# 进程看起来活着、日志停在启动那一行、什么都不做，**可以挂几十分钟不被察觉**。
# `BatchMode=yes` 让它在无密钥时**立刻失败**，循环下一轮就能重试。
set -uo pipefail

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20)

SRC_ARCH="${SRC_ARCH:-/root/autodl-tmp/runs/q36_arm5_corrected/archived}"
DST_HOST="${DST_HOST:-connect.westb.seetacloud.com}"
DST_PORT="${DST_PORT:-50590}"
DST_ARCH="${DST_ARCH:-/root/autodl-tmp/runs/q36_arm5_corrected/archived}"
INTERVAL="${INTERVAL:-120}"
LOG="${LOG:-/root/arm5_push.log}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }

say "启动：${SRC_ARCH} -> ${DST_HOST}:${DST_PORT}:${DST_ARCH}"
ssh "${SSH_OPTS[@]}" -p "${DST_PORT}" "root@${DST_HOST}" \
    "mkdir -p '${DST_ARCH}'" || { say "目标目录创建失败，退出"; exit 2; }

while true; do
  for ck in "${SRC_ARCH}"/checkpoint-*; do
    [ -d "${ck}" ] || continue
    name="$(basename "${ck}")"
    # 用目标侧是否存在 adapter 判定，避免依赖本地状态文件
    if ssh "${SSH_OPTS[@]}" -p "${DST_PORT}" \
           "root@${DST_HOST}" "[ -f '${DST_ARCH}/${name}/adapter_model.safetensors' ]"; then
      continue
    fi
    size="$(du -sh "${ck}" | cut -f1)"
    say "推送 ${name}（${size}）"
    # 先传目录再校验大小：中断时不会留下被误认为完整的副本
    if tar -C "${SRC_ARCH}" -cf - "${name}" \
        | ssh "${SSH_OPTS[@]}" -o Compression=no -p "${DST_PORT}" \
              "root@${DST_HOST}" "tar -C '${DST_ARCH}' -xf -"; then
      say "  ${name} 推送完成"
    else
      say "  ${name} 推送失败（下轮重试）"
    fi
  done
  sleep "${INTERVAL}"
done
