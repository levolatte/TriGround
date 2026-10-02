"""从 ModelScope 并行下载模型权重，带 Range 续传、sha256 校验与逐文件重试。

为什么不用 shell：本项目的 bash 版下载器连续踩了四个坑（自匹配的 pkill 杀掉清理脚本、
陈旧 curl 并发写同一文件导致分片超过目标大小、awk 的 print/`%d` 分别给出科学计数法与
32 位溢出、xargs 里导出函数的引号问题）。这些都不是网络问题，用 Python 一次写清更可靠。

判据只有两个：官方 API 的 Size 与 Sha256。只有两者都通过，`.part` 才会改名成正式文件。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.6-27B"
REVISION = "master"
API = "https://www.modelscope.cn/api/v1/models/{model}/repo/files?Revision={rev}"
BASE = "https://www.modelscope.cn/models/{model}/resolve/{rev}"
CHUNK = 1 << 20
LOCK = threading.Lock()


def fetch_json(url: str, retries: int = 5) -> dict:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - 预检式重试，失败要报告不能静默
            print(f"  清单请求第 {attempt} 次失败: {type(error).__name__}: {error}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"无法获取清单: {url}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def download_file(base_url: str, target: Path, name: str, size: int, sha: str,
                  retries: int) -> tuple[str, str]:
    final = target / name
    part = target / f"{name}.part"

    if final.exists() and final.stat().st_size == size and (not sha or sha256_of(final) == sha):
        return name, "already-ok"
    if final.exists():
        final.unlink()

    for attempt in range(1, retries + 1):
        start = part.stat().st_size if part.exists() else 0
        if start > size:
            part.unlink()
            start = 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        request = urllib.request.Request(f"{base_url}/{name}", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, part.open("ab") as handle:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    handle.write(block)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            log(f"  {name} 第 {attempt} 次中断于 {part.stat().st_size if part.exists() else 0} 字节: "
                f"{type(error).__name__}")
            time.sleep(2 * attempt)
            continue

        got = part.stat().st_size
        if got != size:
            log(f"  {name} 大小不符 {got}/{size}（第 {attempt} 次）")
            continue
        if sha and sha256_of(part) != sha:
            log(f"  {name} sha256 不符（第 {attempt} 次），删除重下")
            part.unlink()
            continue
        shutil.move(str(part), str(final))
        return name, f"ok(attempt {attempt})"

    return name, "FAILED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    api = API.format(model=args.model, rev=args.revision)
    base = BASE.format(model=args.model, rev=args.revision)
    payload = fetch_json(api)
    files = [(f["Path"], int(f["Size"]), f.get("Sha256") or "")
             for f in payload["Data"]["Files"] if f.get("Type") == "blob"]
    files.sort(key=lambda item: (not item[0].endswith(".safetensors"), -item[1]))
    total = sum(size for _, size, _ in files)

    usage = shutil.disk_usage(args.target)
    need = total + 5 * 2**30
    if usage.free < need:
        raise SystemExit(
            f"目标文件系统剩余 {usage.free/2**30:.1f} GiB，需要约 {need/2**30:.1f} GiB：{args.target}"
        )
    log(f"{len(files)} 个文件，共 {total/2**30:.2f} GiB；目标 {args.target} 剩余 {usage.free/2**30:.1f} GiB")

    done_bytes = 0
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_file, base, args.target, name, size, sha, args.retries):
                   (name, size) for name, size, sha in files}
        for future in concurrent.futures.as_completed(futures):
            name, size = futures[future]
            try:
                result_name, status = future.result()
            except Exception as error:  # noqa: BLE001 - 单文件失败不应终止整体，但要记录
                result_name, status = name, f"EXC {type(error).__name__}: {error}"
            results.append((result_name, status))
            if status.startswith("ok") or status == "already-ok":
                done_bytes += size
            log(f"  {result_name}: {status}   累计 {done_bytes/2**30:.2f}/{total/2**30:.2f} GiB")

    failed = [item for item in results if "FAILED" in item[1] or item[1].startswith("EXC")]
    print(f"\n完成 {len(results) - len(failed)}/{len(results)}")
    for name, status in failed:
        print(f"  失败 {name}: {status}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
