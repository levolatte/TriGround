import os
from pathlib import Path

import paramiko


root = Path(__file__).resolve().parents[1]
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.bjb1.seetacloud.com", port=24482, username="root", password=os.environ["AIC_REMOTE_PASS"], timeout=15)
with client.open_sftp() as sftp:
    sftp.put(str(root / "tools" / "eval_qwen_native_grounding.py"), "/root/autodl-tmp/mm-grounding-adapters/tools/eval_qwen_native_grounding.py")
command = r'''set -euo pipefail
cd /root/autodl-tmp/mm-grounding-adapters
snapshot=$(find /root/autodl-tmp/aic/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots -mindepth 1 -maxdepth 1 -type d | head -n1)
test -n "$snapshot"
HF_HOME=/root/autodl-tmp/aic/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /root/miniconda3/bin/python tools/eval_qwen_native_grounding.py --model "$snapshot" --manifest data/refcoco/val.jsonl --samples 96 --output runs/qwen_native_grounding/results.jsonl'''
_, stdout, stderr = client.exec_command(command, timeout=1800)
for line in iter(stdout.readline, ""):
    print(line, end="")
error = stderr.read().decode()
if error:
    print(error)
status = stdout.channel.recv_exit_status()
client.close()
raise SystemExit(status)
