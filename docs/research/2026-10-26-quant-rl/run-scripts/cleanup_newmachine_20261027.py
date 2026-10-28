"""Remove superseded experiments on the user-authorized replacement instance."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

BASE = Path('/root/autodl-tmp').resolve()
ROOT = BASE / 'quant_rl_20261026'
REPORT = ROOT / 'runs/newmachine_cleanup_20261027.json'


def checked(path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(BASE) or resolved == BASE:
        raise ValueError(f'Deletion or move outside the authorized data disk: {resolved}')
    return resolved


def same_file_contents(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            chunk = a.read(8 * 1024 * 1024)
            if chunk != b.read(len(chunk)):
                return False
            if not chunk:
                return True


def main():
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError(f'Unexpected active GPU job before cleanup: {gpu}')
    required = [BASE / 'rematch_models/Qwen3.6-27B/config.json',
                BASE / 'rematch_20260922/data/quant_grounding_20261026_formal/schedule.jsonl',
                ROOT / 'train_env/bin/python',
                ROOT / 'runs/shared_format_warmup/training/checkpoint-50/adapter_model.safetensors',
                ROOT / 'runs/initial_seed2026/A_seed2026/train/checkpoint-59/adapter_model.safetensors',
                ROOT / 'runs/optimized_B_seed2026/train/checkpoint-59/adapter_model.safetensors',
                ROOT / 'runs/optimized_C_seed2026/train/checkpoint-550/adapter_model.safetensors',
                ROOT / 'runs/optimized_C_seed2026/train/matched-exposure-checkpoint-29/adapter_model.safetensors']
    if not all(path.is_file() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.is_file()])
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'disk_before': shutil.disk_usage(BASE)._asdict(), 'removed': [], 'retained_legacy': []}
    duplicates = [('optimized_B_seed2026', 'checkpoint-59', 'matched-exposure-checkpoint-59'),
                  ('optimized_C_seed2026', 'matched-exposure-checkpoint-29', 'checkpoint-29')]
    for arm, keep, remove in duplicates:
        directory = ROOT / 'runs' / arm / 'train'
        if not same_file_contents(directory / keep / 'adapter_model.safetensors',
                                  directory / remove / 'adapter_model.safetensors'):
            raise RuntimeError(f'Duplicate adapter differs: {arm}/{remove}')
    legacy = [BASE / 'runs' / name for name in ['q36_arm2_capability', 'q36_arm4_matched', 'q36_arm5_corrected']]
    with zipfile.ZipFile(ROOT / 'runs/legacy_training_metadata.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for directory in legacy:
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix in {'.json', '.jsonl', '.log', '.md'}:
                    archive.write(path, path.relative_to(BASE / 'runs'))
    retained = BASE / 'runs/retained_baselines'
    retained.mkdir(exist_ok=True)
    for arm, step in [('q36_arm4_matched', 600), ('q36_arm5_corrected', 700)]:
        source = checked(BASE / 'runs' / arm / 'archived' / f'checkpoint-{step}')
        destination = retained / f'{arm}_checkpoint-{step}'
        if destination.exists() or not destination.resolve().is_relative_to(BASE):
            raise ValueError(f'Unexpected retention destination: {destination}')
        shutil.move(str(source), str(destination))
        report['retained_legacy'].append(str(destination))
    targets = [*legacy, BASE / 'rematch_models/Qwen3-VL-8B-Instruct', ROOT / 'train_env_opt',
               ROOT / 'code_before_optimization']
    for arm, checkpoints in {
        'optimized_B_seed2026': ['checkpoint-18', 'checkpoint-36', 'matched-exposure-checkpoint-59'],
        'optimized_C_seed2026': ['checkpoint-29', 'checkpoint-162', 'checkpoint-325'],
        'optimized_D_seed2026': ['checkpoint-6'],
    }.items():
        targets.extend(ROOT / 'runs' / arm / 'train' / checkpoint for checkpoint in checkpoints)
    for target in targets:
        target = checked(target)
        if any(path.resolve().is_relative_to(target) for path in required):
            raise ValueError(f'Protected experiment asset under deletion target: {target}')
        size = int(subprocess.check_output(['du', '-s', '-B1', str(target)], text=True).split()[0])
        shutil.rmtree(target)
        report['removed'].append({'path': str(target), 'allocated_bytes_before_removal': size})
        REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    report['disk_after'] = shutil.disk_usage(BASE)._asdict()
    report['freed_bytes'] = report['disk_after']['free'] - report['disk_before']['free']
    report['ended_at'] = datetime.now(timezone.utc).isoformat()
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'removed_paths': len(report['removed']), 'freed_gib': report['freed_bytes']/2**30,
                      'free_gib': report['disk_after']['free']/2**30, 'report': str(REPORT)}))


if __name__ == '__main__':
    main()
