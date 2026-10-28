import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

root = Path('/root/autodl-tmp/quant_rl_20261026')
run = root / 'runs/d_repair_20261027'
sys.path.insert(0, str(root / 'code'))
from tools.eval_quant_grounding import score_rows, read_jsonl
from tools.run_quant_d_repair_gate import gate_summary

state = json.loads((run / 'state.json').read_text())
report = {'at': datetime.now(timezone.utc).isoformat(),
          'status': state['status'], 'stage': state['stage'],
          'completed_stages': state['completed_stages']}
for name, dataset in [('gate4', 'train4'), ('expanded12', 'train12')]:
    path = run / name / 'predictions_raw.jsonl'
    if not path.exists():
        continue
    raw = read_jsonl(path)
    by_id = {row['id']: row for row in read_jsonl(run / (dataset + '.jsonl'))}
    if raw:
        scored = score_rows([by_id[row['id']] for row in raw], raw, 'thinking', 'contrast', 'single_box')
        summary = gate_summary(scored, expected_queries=len(raw))
        traces = [sample['thinking_trace'] for row in raw for sample in row['samples']]
        report[name] = {key: summary[key] for key in ['queries', 'answers', 'valid', 'truncated', 'geometry_reward_groups']}
        report[name]['forced_boundaries'] = sum(trace['thought_boundary']=='forced' for trace in traces)
        report[name]['natural_boundaries'] = sum(trace['thought_boundary']=='natural' for trace in traces)
    else:
        report[name] = {'queries': 0}
for name in ['gate4_gate.json', 'training/gate_result.json', 'expanded12_gate.json', 'controller.json']:
    path = run / name
    if path.exists():
        report[name] = json.loads(path.read_text())
report['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(report, ensure_ascii=True))
