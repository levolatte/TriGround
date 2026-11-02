"""把 new-rgbdt500 被选中的子集从本地传到 B 机。

只传清单里的 4,146 个文件（约 9.4 GB），不传全量 23.3 GB。
用 SFTP 逐个传，因为清单是"散落在三个子目录里"的，打 tar 再传反而多一次落盘。

断点续传：目标文件已存在且大小一致就跳过，所以中断后重跑即可。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time

import paramiko

LOCAL_ROOT = Path(r"F:\AIC\data\new-rgbdt500")
REMOTE_ROOT = "/root/autodl-tmp/rematch_20260922/data/external/newrgbdt500"
LIST = Path(r"F:\AIC\.work\newrgbdt_upload_list.txt")

HOST, PORT, USER, PASSWORD = "connect.westd.seetacloud.com", 24014, "root", "CII6IHYgC8/J"


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, look_for_keys=False,
                   allow_agent=False, timeout=60, banner_timeout=60, auth_timeout=60)
    return client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rels = [line.strip() for line in LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"清单 {len(rels)} 个文件", flush=True)

    admin = connect()
    # 建好三个子目录，避免每个 worker 都去 mkdir。
    _, out, _ = admin.exec_command(f"mkdir -p {REMOTE_ROOT}/visible {REMOTE_ROOT}/infrared "
                                   f"{REMOTE_ROOT}/depth && echo ok", timeout=120)
    print("远端目录:", out.read().decode().strip(), flush=True)
    existing = {}
    _, out, _ = admin.exec_command(f"find {REMOTE_ROOT} -type f -printf '%P\\t%s\\n'", timeout=300)
    for line in out.read().decode("utf-8", "replace").splitlines():
        if "\t" in line:
            name, size = line.rsplit("\t", 1)
            existing[name] = int(size)
    print(f"远端已有 {len(existing)} 个文件（断点续传基线）", flush=True)
    admin.close()

    todo = []
    total_bytes = 0
    for rel in rels:
        local = LOCAL_ROOT / rel
        size = local.stat().st_size
        if existing.get(rel.replace("\\", "/")) == size:
            continue
        todo.append(rel)
        total_bytes += size
    print(f"待传 {len(todo)} 个，共 {total_bytes / 1e9:.2f} GB", flush=True)
    if not todo:
        print("已全部传完")
        return 0

    done = {"files": 0, "bytes": 0}
    started = time.time()

    def transfer(rel: str) -> None:
        client = connect()
        try:
            sftp = client.open_sftp()
            local = LOCAL_ROOT / rel
            remote = f"{REMOTE_ROOT}/{rel}"
            sftp.put(str(local), remote)
            sftp.close()
        finally:
            client.close()
        done["files"] += 1
        done["bytes"] += local.stat().st_size

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transfer, rel): rel for rel in todo}
        for index, future in enumerate(as_completed(futures), 1):
            rel = futures[future]
            try:
                future.result()
            except Exception as error:  # 传输失败必须暴露，不能静默跳过
                print(f"失败 {rel}: {type(error).__name__}: {error}", flush=True)
                raise
            if index % 200 == 0 or index == len(todo):
                elapsed = time.time() - started
                rate = done["bytes"] / elapsed / 1e6 if elapsed else 0
                remain = (total_bytes - done["bytes"]) / max(1, rate) / 1e6
                print(f"  {index}/{len(todo)}  {done['bytes'] / 1e9:.2f}/{total_bytes / 1e9:.2f} GB  "
                      f"{rate:.1f} MB/s  剩余约 {remain:.0f} 秒", flush=True)

    print(f"完成：{done['files']} 个文件，{done['bytes'] / 1e9:.2f} GB，"
          f"用时 {(time.time() - started) / 60:.1f} 分钟", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
