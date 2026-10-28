"""Run the bounded real-rollout qualification for one numerically qualified variant.

This controller never starts B/C/D training. Each measurement is a separate
optimization_job invocation, so loading, failures, and idle gaps use the same
two-hour optimization ledger as the fixed-answer tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path('/root/autodl-tmp/quant_rl_20261026')
DATA = Path('/root/autodl-tmp/rematch_20260922/data')
MODEL = Path('/root/autodl-tmp/rematch_models/Qwen3.6-27B')
DEADLINE = '2026-10-26T10:16:00+00:00'
CASE_DEFINITIONS = (
    ('original_A', 'A', 2, False, False, False, 0.32),
    ('candidate_A', 'A', 2, True, True, True, 0.45),
    ('candidate_C', 'C', 1, True, True, False, 0.18),
    ('candidate_D', 'D', 1, True, True, False, 0.50),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path: Path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def strict_fixed_gate(path: Path, variant: str) -> dict:
    comparison = read_json(path)
    repeat = comparison['candidates']['baseline_repeat']['comparison']
    candidate = comparison['candidates'][variant]['comparison']
    for name, record in (('baseline_repeat', repeat), (variant, candidate)):
        if (record['numeric_pass'] is not True
                or record['gradient']['cosine'] < 0.999
                or record['gradient']['relative_l2'] > 0.01
                or record['update']['cosine'] < 0.999
                or record['update']['relative_l2'] > 0.01):
            raise RuntimeError(f'{name} 未通过原定固定回答数值门槛')
    return {'comparison': str(path), 'reference_repeat_passed': True,
            'variant': variant, 'variant_numeric_passed': True}


def remaining_optimization_hours(ledger: Path) -> float:
    rows = read_json(ledger)
    if any(row['status'] == 'running' for row in rows):
        raise RuntimeError('统一账本仍有运行中的作业')
    used = sum(row['gpu_hours'] for row in rows)
    stage_used = sum(row['gpu_hours'] for row in rows if row['stage'] == 'optimization')
    latest = max(datetime.fromisoformat(row.get('ended_at', row.get('recorded_at'))) for row in rows)
    # optimization_job first charges this idle interval to the same stage.
    pending_gap = max(0.0, (datetime.now(timezone.utc) - latest).total_seconds() / 3600)
    return max(0.0, min(72 - used, 2 - stage_used) - pending_gap - 2 / 3600)


def actual_tokens(arm: str, output: Path, updates: int) -> dict:
    resources = jsonl(output / 'resource_steps.jsonl')
    if [row['step'] for row in resources] != list(range(1, updates + 1)):
        raise RuntimeError(f'{arm} 资源记录未覆盖全部实际更新: {output}')
    prompt_maxima = [row.get('prompt_tokens_step_max') for row in resources]
    if arm == 'C':
        audit = read_json(output / 'runtime_audit.json')
        sft = audit['sft_input_observations']
        if sft['count'] != 8 or audit['training_batch']['answers_per_optimizer_step'] != 8:
            raise RuntimeError(f'C臂没有完整8题监督更新: {sft}')
        return {'prompt_tokens_step_max': prompt_maxima,
                'sft_forward_count': sft['count'],
                'sft_selected_logits_first': sft['first']['selected_logits_tokens'],
                'sft_selected_logits_max': sft['maximum']['selected_logits_tokens'],
                'supervised_target_tokens_total': None,
                'token_scope_note': '训练审计未记录每题全部监督词元；只报告实际前向的选取logits长度'}

    if any(row['rollout_groups'] != 4 or row['rollout_answers'] != 32 for row in resources):
        raise RuntimeError(f'{arm} 每步必须是4题、32回答: {output}')
    reward_batches = jsonl(output / 'reward_metrics.jsonl')
    if len(reward_batches) != 4 * updates:
        raise RuntimeError(f'{arm} 奖励日志未覆盖每步四个G8题组: {output}')
    per_step = []
    for start in range(0, len(reward_batches), 4):
        batches = reward_batches[start:start + 4]
        counts = [item['generated_tokens'] for batch in batches for item in batch['items']]
        if len(counts) != 32 or any(not isinstance(count, int) or count <= 0 for count in counts):
            raise RuntimeError(f'{arm} 生成词元日志不完整: {output}')
        per_step.append(sum(counts))
    return {'prompt_tokens_step_max': prompt_maxima,
            'generated_completion_tokens_per_step': per_step,
            'generated_completion_tokens_total': sum(per_step),
            'token_scope_note': '回答token来自实际奖励日志；prompt只记录每步最大长度，非总token'}


def make_config(code: Path, case: tuple, adapter: Path, variant: str, root: Path) -> Path:
    name, arm, _steps, optimized, deterministic, _reload, _hours = case
    config = read_json(code / 'configs/quant_grounding' / f'arm_{arm.lower()}.json')
    config.update(model=str(MODEL), dataset=str(DATA / 'quant_grounding_20261026_formal/schedule.jsonl'),
                  data_root=str(DATA), adapters=str(adapter), seed=2026,
                  output_dir=str(root / name), hf_max_batch_size=2,
                  train_micro_batch_size=1, gradient_checkpointing_every_n_layers=1,
                  runtime_optimization=optimized, deterministic_training=deterministic,
                  max_gpu_hours=1.5, save_hours=[])
    path = root / 'configs' / f'{name}_{variant}.json'
    if path.exists():
        raise RuntimeError(f'配置已存在，拒绝复用: {path}')
    write_json(path, config)
    return path


def run_case(case: tuple, root: Path, code: Path, base_python: Path,
             candidate_python: Path, adapter: Path, variant: str,
             ledger: Path, deadline: datetime) -> dict:
    name, arm, steps, optimized, _deterministic, reload, cap_hours = case
    output = root / name
    if output.exists():
        raise RuntimeError(f'真实检查输出已存在，拒绝覆盖: {output}')
    config = make_config(code, case, adapter, variant, root)
    python = candidate_python if optimized else base_python
    label = f'live_{variant}_{name}_20261026'
    wall_hours = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600 - 1 / 60
    budget_hours = remaining_optimization_hours(ledger)
    limit = min(cap_hours, wall_hours, budget_hours)
    if limit <= 0:
        raise RuntimeError(f'{name} 开始前优化预算或10:16 UTC硬截止已耗尽')
    command = [str(base_python), str(ROOT / 'optimization_job.py'),
               '--label', label, '--max-hours', str(limit), '--',
               str(python), str(ROOT / 'live_optimization_check.py'),
               '--config', str(config), '--output', str(output), '--steps', str(steps)]
    if reload:
        command.append('--reload')
    exit_code = subprocess.run(command, cwd=code).returncode
    measurement_path = ROOT / 'runs/optimization_measurements' / f'{label}.json'
    partial = {'name': name, 'arm': arm, 'config': str(config), 'python': str(python),
               'label': label, 'max_hours': limit, 'exit_code': exit_code,
               'output': str(output), 'measurement': str(measurement_path)}
    if exit_code:
        raise RuntimeError(f'{name} 真实检查退出码{exit_code}; measurement={measurement_path}')
    measurement = read_json(measurement_path)
    result = read_json(output / 'live_result.json')
    audit = read_json(output / 'runtime_audit.json')
    if optimized and 'causal_conv1d' not in audit['kernel_implementations']['causal_conv1d_fn']['module']:
        raise RuntimeError(f'{name} 没有实际调用优化因果卷积核')
    if audit['runtime_optimization'] is not optimized or audit['deterministic_training'] is not _deterministic:
        raise RuntimeError(f'{name} 运行审计与候选配置不一致')
    expected_updates = steps + int(reload)
    if len(result['updates']) != expected_updates or result['arm'] != arm:
        raise RuntimeError(f'{name} 更新数或臂错误: {result["updates"]}')
    if result['saved_and_reloaded'] is not reload:
        raise RuntimeError(f'{name} 保存重载标记错误')
    grad_norms = [float(value) for value in result['update_grad_norms'].values()]
    if (len(grad_norms) != expected_updates or not all(math.isfinite(value) for value in grad_norms)
            or (arm != 'D' and not all(value > 0 for value in grad_norms))):
        raise RuntimeError(f'{name} 梯度范数不完整或无效: {grad_norms}')
    headroom = int(measurement['minimum_device_headroom_mib'])
    if headroom < 6144:
        raise RuntimeError(f'{name} 整卡最小余量{headroom}MiB，低于6144MiB')
    tokens = actual_tokens(arm, output, expected_updates)
    return {**partial, 'status': 'passed', 'steps': expected_updates,
            'step_seconds': [row['complete_update_seconds'] for row in result['updates']],
            'first_segment_median_complete_update_seconds': result['first_segment_median_complete_update_seconds'],
            'saved_and_reloaded': reload, 'minimum_device_headroom_mib': headroom,
            'peak_device_used_mib': measurement['peak_device_used_mib'],
            'all_update_grad_norms': grad_norms, 'actual_tokens': tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='audit_conv', choices=('audit_conv',))
    parser.add_argument('--python', type=Path, default=ROOT / 'train_env_opt/bin/python',
                        help='candidate interpreter; baseline always uses train_env')
    parser.add_argument('--deadline-utc', default=DEADLINE)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    code = ROOT / 'code'
    base_python = ROOT / 'train_env/bin/python'
    candidate_python = args.python.absolute()
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('deadline must include UTC offset')
    output_root = ROOT / 'runs/optimization_live'
    report_path = output_root / 'sequence_complete.json'
    if report_path.exists():
        raise RuntimeError(f'序列已有结果，拒绝重复启动: {report_path}')
    comparison_path = ROOT / 'runs/optimization_deterministic/runs/comparison.json'
    ledger = ROOT / 'runs/budget.json'
    adapter = ROOT / 'runs/shared_format_warmup/training/checkpoint-50'
    try:
        gate = strict_fixed_gate(comparison_path, args.variant)
    except Exception as error:
        if not args.check_only:
            if output_root.exists():
                raise RuntimeError(f'真实检查目录已存在，拒绝覆盖: {output_root}') from error
            write_json(report_path, {'status': 'fixed_numeric_gate_failed', 'all_passed': False,
                                     'fallback_required': True, 'variant': args.variant,
                                     'failure': f'{type(error).__name__}: {error}', 'results': {}})
        raise
    if not base_python.is_file() or not candidate_python.is_file() or not adapter.is_dir():
        raise FileNotFoundError('训练解释器或共享warmup50缺失')
    if remaining_optimization_hours(ledger) <= 0:
        raise RuntimeError('优化阶段2 GPU小时余额已用尽')
    if args.check_only:
        print(json.dumps({'gate': gate, 'base_python': str(base_python),
                          'candidate_python': str(candidate_python),
                          'remaining_optimization_hours': remaining_optimization_hours(ledger),
                          'deadline_utc': deadline.isoformat(), 'cases': [case[0] for case in CASE_DEFINITIONS]},
                         ensure_ascii=False, indent=2))
        return
    output_root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'all_passed': False, 'fallback_required': True,
              'fixed_numeric_gate': gate, 'deadline_utc': deadline.isoformat(),
              'candidate_python': str(candidate_python), 'baseline_python': str(base_python),
              'results': {}, 'speed': None}
    try:
        for case in CASE_DEFINITIONS:
            report['active_case'] = case[0]
            write_json(report_path, report)
            row = run_case(case, output_root, code, base_python, candidate_python,
                           adapter, args.variant, ledger, deadline)
            report['results'][case[0]] = row
            report['active_case'] = None
            write_json(report_path, report)
            if case[0] == 'candidate_A':
                baseline = report['results']['original_A']
                baseline_seconds = baseline['first_segment_median_complete_update_seconds']
                candidate_seconds = row['first_segment_median_complete_update_seconds']
                ratio = candidate_seconds / baseline_seconds
                base_tokens = baseline['actual_tokens']['generated_completion_tokens_per_step'][:2]
                candidate_tokens = row['actual_tokens']['generated_completion_tokens_per_step'][:2]
                token_ratio = sum(candidate_tokens) / sum(base_tokens)
                prompt_base = baseline['actual_tokens']['prompt_tokens_step_max'][:2]
                prompt_candidate = row['actual_tokens']['prompt_tokens_step_max'][:2]
                token_confounded = (abs(token_ratio - 1) > 0.10
                                    or any(abs(c - b) / b > 0.10 for b, c in zip(prompt_base, prompt_candidate)))
                report['speed'] = {
                    'baseline_two_step_median_seconds': baseline_seconds,
                    'candidate_two_step_median_seconds': candidate_seconds,
                    'candidate_over_baseline': ratio, 'passed_10_percent': ratio <= 0.90,
                    'baseline_completion_tokens_first_two': base_tokens,
                    'candidate_completion_tokens_first_two': candidate_tokens,
                    'candidate_over_baseline_completion_tokens': token_ratio,
                    'prompt_tokens_step_max_baseline': prompt_base,
                    'prompt_tokens_step_max_candidate': prompt_candidate,
                    'token_difference_material': token_confounded,
                    'attribution': ('token lengths differ materially; speed cannot be attributed solely to engineering'
                                    if token_confounded else 'token lengths similar; fixed-answer phase remains the paired control')}
                write_json(report_path, report)
                if ratio > 0.90:
                    raise RuntimeError(f'候选真实完整更新仅达到{ratio:.3f}倍原始耗时，未达至少10%提速门槛')
        report.update(status='passed', all_passed=True, fallback_required=False)
    except Exception as error:
        report.update(status='failed', failure=f'{type(error).__name__}: {error}',
                      all_passed=False, fallback_required=True)
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
