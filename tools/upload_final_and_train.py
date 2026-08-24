from __future__ import annotations

import concurrent.futures
import json
import os
import posixpath
import stat
import threading
import time
from pathlib import Path

import paramiko


HOST = "connect.bjb1.seetacloud.com"
PORT = 24482
USER = "root"
PASSWORD = os.environ["AIC_REMOTE_PASS"]
LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(r"D:\AIC\city_detection_prepared\train")
REMOTE_REPO = "/root/autodl-tmp/mm-grounding-adapters"
REMOTE_DATA = "/root/autodl-tmp/city_detection_prepared/train"


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=20)
    return client


def run(client, command, timeout=None):
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode(errors="replace")
    error = stderr.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"remote command failed ({status}): {command}\n{output}\n{error}")
    return output


def manifest_files():
    relative = set()
    for name in ("grounding_final_train.json", "grounding_final_val.json"):
        records = json.loads((LOCAL_DATA / name).read_text(encoding="utf-8"))
        for record in records.values():
            relative.update(record[key] for key in ("visible", "infrared", "depth"))
    relative.update(("grounding_final_train.json", "grounding_final_val.json", "grounding_final_report.json"))
    return sorted(relative)


def project_files():
    files = []
    for name in ("train.py", "evaluate.py", "pyproject.toml", "requirements-experiment.txt", "README.md"):
        files.append(LOCAL_REPO / name)
    for directory in ("configs", "src", "tools", "scripts"):
        files.extend(
            path for path in (LOCAL_REPO / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        )
    return files


def ensure_dir(sftp, path, cache):
    if path in cache:
        return
    pieces = path.strip("/").split("/")
    current = ""
    for piece in pieces:
        current += "/" + piece
        if current in cache:
            continue
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)
        cache.add(current)


progress_lock = threading.Lock()
uploaded_bytes = 0
skipped_bytes = 0
completed_files = 0
started = time.time()


def upload_batch(batch):
    global uploaded_bytes, skipped_bytes, completed_files
    client = connect()
    cache = set()
    with client.open_sftp() as sftp:
        for local, remote in batch:
            size = local.stat().st_size
            ensure_dir(sftp, posixpath.dirname(remote), cache)
            try:
                remote_size = sftp.stat(remote).st_size
            except OSError:
                remote_size = -1
            if remote_size == size:
                with progress_lock:
                    skipped_bytes += size
                    completed_files += 1
                continue
            sftp.put(str(local), remote)
            with progress_lock:
                uploaded_bytes += size
                completed_files += 1
    client.close()


def main():
    data_rel = manifest_files()
    data_pairs = [(LOCAL_DATA / rel, posixpath.join(REMOTE_DATA, rel.replace("\\", "/"))) for rel in data_rel]
    code_pairs = []
    for local in project_files():
        relative = local.relative_to(LOCAL_REPO).as_posix()
        code_pairs.append((local, posixpath.join(REMOTE_REPO, relative)))
    pairs = code_pairs + data_pairs
    required = sum(local.stat().st_size for local, _ in pairs)
    print(json.dumps({"files": len(pairs), "size_gib": round(required / 2**30, 2)}), flush=True)

    batches = [[] for _ in range(4)]
    for index, pair in enumerate(sorted(pairs, key=lambda item: item[0].stat().st_size, reverse=True)):
        batches[index % len(batches)].append(pair)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(upload_batch, batch) for batch in batches]
        while not all(future.done() for future in futures):
            time.sleep(20)
            with progress_lock:
                elapsed = max(time.time() - started, 1)
                print(json.dumps({"files_done": completed_files, "files_total": len(pairs), "uploaded_gib": round(uploaded_bytes/2**30, 2), "skipped_gib": round(skipped_bytes/2**30, 2), "upload_mib_s": round(uploaded_bytes/2**20/elapsed, 2)}), flush=True)
        for future in futures:
            future.result()

    client = connect()
    base = f"cd {REMOTE_REPO} && export HF_HOME=/root/autodl-tmp/aic/huggingface && export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1"
    print(run(client, base + " && /root/miniconda3/bin/python tools/preflight.py --config configs/multimodal.yaml --offline", timeout=300), flush=True)
    print(run(client, base + " && /root/miniconda3/bin/python tools/preflight.py --config configs/multimodal.yaml --device cuda --backward", timeout=900), flush=True)
    launch = base + " && mkdir -p logs runs/multimodal && nohup /root/miniconda3/bin/python -u train.py --config configs/multimodal.yaml > logs/multimodal_final.log 2>&1 < /dev/null & echo $!"
    pid = run(client, launch).strip()
    time.sleep(8)
    status = run(client, f"ps -p {pid} -o pid=,stat=,etime=,cmd= || true; tail -n 30 {REMOTE_REPO}/logs/multimodal_final.log 2>/dev/null || true")
    print(json.dumps({"pid": pid, "log": f"{REMOTE_REPO}/logs/multimodal_final.log", "run_dir": f"{REMOTE_REPO}/runs/multimodal"}), flush=True)
    print(status, flush=True)
    client.close()


if __name__ == "__main__":
    main()
