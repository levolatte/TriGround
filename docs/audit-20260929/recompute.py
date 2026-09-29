"""Recompute all scores and consumption order from the unpacked audit bundle."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code'))
from tools.report_triground_abv import build_report, write_outputs
from tools.report_rematch_experiment import load_manifest

BASE = ROOT / 'results/triground_abv_execution_20260928'
RUN = BASE / 'cloud_final/runs/seed2026_600_deterministic'
OUT = BASE / 'verification/package_recomputed'
OUT.mkdir(exist_ok=True, parents=True)
arms = ['C0', 'A', 'B', 'V']
diag = BASE / 'deployment_600_seed2026/manifests/diagnostics'
for label, manifest, condition in [
    ('city412', BASE / 'verification/AB_analysis/city_gt.json', 'city412'),
    ('diagnostics', diag / 'gt_manifest.json', 'normal'),
]:
    paths = {a: RUN / 'evaluation' / a / condition / 'predictions.jsonl' for a in arms}
    kw = {}
    if label == 'diagnostics':
        kw = dict(class_map=json.loads((diag / 'class_map.json').read_text(encoding="utf-8")),
                  scene_map=json.loads((diag / 'scene_map.json').read_text(encoding="utf-8")),
                  modal_paths={(a, c): RUN / 'evaluation' / a / c / 'predictions.jsonl'
                               for a in arms for c in ['ir_missing', 'depth_missing', 'both_missing']})
    report = build_report(load_manifest(manifest), paths, require_run_config=True, **kw)
    write_outputs(report, OUT / label)
    print(label, report['ranking'])

checks = {}
for a in ['A', 'B', 'V']:
    actual = [json.loads(line)['id'] for line in (RUN / a / 'main/consumed_samples.jsonl').read_text(encoding="utf-8").splitlines()]
    expected = [r['id'] for r in json.loads((BASE / 'deployment_600_seed2026/manifests' / f'{a}.json').read_text(encoding="utf-8"))]
    resume = json.loads((RUN / a / 'resume_comparison.json').read_text(encoding="utf-8"))
    assert len(actual) == 4800 and actual == expected
    assert resume['status'] == 'pass'
    checks[a] = dict(samples=len(actual), matches_manifest=True, resume_report_status=resume['status'])
(OUT / 'consumption_check.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(checks)
