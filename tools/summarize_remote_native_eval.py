import os

import paramiko


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("connect.bjb1.seetacloud.com", port=24482, username="root", password=os.environ["AIC_REMOTE_PASS"], timeout=15)
command = r'''cd /root/autodl-tmp/mm-grounding-adapters
/root/miniconda3/bin/python -c "import json,collections; rows=[json.loads(x) for x in open('runs/qwen_native_grounding/results.jsonl')]; groups=collections.defaultdict(list); [groups[r['source']].append(r) for r in rows]; print(json.dumps({k:{'n':len(v),'mean_iou':sum(x['iou'] for x in v)/len(v),'acc_0.5':sum(x['iou']>=.5 for x in v)/len(v),'acc_0.7':sum(x['iou']>=.7 for x in v)/len(v)} for k,v in groups.items()},indent=2)); print('WORST'); [print(json.dumps({k:r[k] for k in ('id','query','gt','prediction','iou','answer')},ensure_ascii=False)) for r in sorted(rows,key=lambda x:x['iou'])[:8]]"'''
_, stdout, stderr = client.exec_command(command, timeout=60)
print(stdout.read().decode())
print(stderr.read().decode())
status = stdout.channel.recv_exit_status()
client.close()
raise SystemExit(status)
