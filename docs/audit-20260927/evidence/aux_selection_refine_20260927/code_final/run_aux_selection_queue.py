"""Bounded, serial, frozen-inference queue. No training or automatic retries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.prepare_aux_selection import read_rows, dump, dump_rows


def check_complete(manifest, predictions, require_parse=False):
    ids = [r['id'] for r in read_rows(manifest)]
    rows = read_rows(predictions)
    actual = [r['id'] for r in rows]
    assert len(actual) == len(set(actual)) == len(ids) and set(actual) == set(ids), (predictions, len(actual), len(ids))
    if require_parse:
        assert all(r['parsed'] for r in rows), f'preflight invalid output: {predictions}'
        assert not any(r.get('generation_cap_hit') or r.get('selection_generation_cap_hit') or r.get('refine_generation_cap_hit') for r in rows), 'preflight generation cap reached'
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'full'], required=True)
    p.add_argument('--smoke-attempt', type=int, default=1)
    a = p.parse_args()
    root = a.root.resolve()
    out = root/'results/aux_selection_refine_20260927'
    m = out/'manifests'
    window = json.loads((out/'execution_window.json').read_text())
    deadline = window['deadline_epoch']
    assert 0 < deadline-window['start_epoch'] <= 8*3600, 'invalid eight-hour execution window'
    assert shutil.disk_usage(root).free >= 10*1024**3, 'need 10GiB at queue start'
    assert time.time() < deadline, '8-hour window expired'
    (out/'logs').mkdir(exist_ok=True)
    (out/'completed_stages').mkdir(exist_ok=True)
    (out/f'{a.phase}_runner.pid').write_text(str(os.getpid())+'\n')
    py = '/root/miniconda3/bin/python'
    auxpy = str(root/'.venvs/aux_selection/bin/python')
    qwen = ['--model', '/root/rematch_models/Qwen3-VL-8B-Instruct', '--adapter',
            str(root/'results/next_stage_20260925/c_phase2/checkpoint-500')]
    dl = ['--deadline-epoch', str(deadline)]
    model_root = root/'models'
    status = out/'stage_status.tsv'

    def stage(name, command, verify=None):
        done = out/'completed_stages'/f'{name}.json'
        if done.exists():
            if verify: verify()
            print('ALREADY_COMPLETE', name, flush=True)
            return
        assert shutil.disk_usage(root).free >= 3*1024**3, 'less than 3GiB remaining'
        left = deadline-time.time()
        assert left > 0, '8-hour window expired'
        started = time.time()
        with status.open('a') as h:
            h.write(f'{time.strftime("%F %T")}\t{name}\tstart\n')
        print('STAGE_START', name, time.strftime('%F %T'), flush=True)
        log = out/'logs'/f'{name}.log'
        with log.open('a') as h, (out/'logs'/f'{name}.gpu.csv').open('a') as gpu_log:
            sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits', '-l', '1'], stdout=gpu_log, stderr=subprocess.STDOUT)
            try:
                result = subprocess.run([str(x) for x in command], cwd=root/'code', stdout=h, stderr=subprocess.STDOUT, timeout=left)
            except subprocess.TimeoutExpired:
                with status.open('a') as s: s.write(f'{time.strftime("%F %T")}\t{name}\tbudget_expired\n')
                raise
            finally:
                sampler.terminate()
                sampler.wait(timeout=10)
        if result.returncode:
            with status.open('a') as h: h.write(f'{time.strftime("%F %T")}\t{name}\tfailed\t{result.returncode}\n')
            raise RuntimeError(f'{name} failed: {result.returncode}; see {log}')
        if verify: verify()
        record = {'stage': name, 'started_epoch': started, 'finished_epoch': time.time(), 'elapsed_seconds': time.time()-started,
                  'command': [str(x) for x in command], 'log': str(log)}
        dump(done, record)
        with status.open('a') as h: h.write(f'{time.strftime("%F %T")}\t{name}\tcomplete\t{record["elapsed_seconds"]:.3f}\n')
        print('STAGE_COMPLETE', name, record['elapsed_seconds'], flush=True)

    def qstage(name, args, config_dir, verify):
        if (config_dir/'run_config.json').exists(): args += ['--resume']
        stage(name, [py, '-m', 'tools.predict_aux_selection', *args, *qwen, *dl], verify)

    def pipeline(cohort, baseline, *, output_name=None):
        label = output_name or cohort
        folder = out/label
        folder.mkdir(exist_ok=True)
        manifest = m/f'{cohort}.jsonl'
        querydir = folder/'query'
        queryfile = querydir/'query_info.jsonl'
        qstage(label+'_parse', ['parse-query', '--manifest', manifest, '--output-dir', querydir], querydir,
               lambda: check_complete(manifest, queryfile, require_parse=True))
        candidates, evidence = folder/'candidates.jsonl', folder/'evidence.jsonl'
        stage(label+'_candidates', [auxpy, '-m', 'tools.aux_selection_evidence', 'candidates', '--manifest', manifest,
              '--baseline', baseline, '--query-info', queryfile, '--data-root', root, '--output', candidates,
              '--model', model_root/'grounding-dino-tiny', *dl], lambda: check_complete(manifest, candidates))
        stage(label+'_depth', [auxpy, '-m', 'tools.aux_selection_evidence', 'depth', '--candidates', candidates,
              '--data-root', root, '--output', evidence, '--model', model_root/'sam2.1-hiera-tiny', *dl],
              lambda: check_complete(manifest, evidence))
        pred = folder/'trimodal'
        qstage(label+'_select_refine', ['select-refine', '--manifest', manifest, '--evidence', evidence,
               '--query-info', queryfile, '--condition', 'trimodal', '--output-dir', pred], pred,
               lambda: check_complete(manifest, pred/'final_predictions.jsonl', require_parse=cohort=='train16'))

    if a.phase == 'smoke':
        envdir = out/'env8'
        qstage('env8', ['baseline', '--manifest', m/'env8.jsonl', '--prompts', m/'env8_baseline_prompts.json',
               '--output', envdir/'predictions.jsonl'], envdir,
               lambda: check_complete(m/'env8.jsonl', envdir/'predictions.jsonl', require_parse=True))
        stage('verify_env8', [py, '-m', 'tools.prepare_aux_selection', 'verify-env', '--expected', m/'env8_expected.json',
               '--actual', envdir/'predictions.jsonl'])
        trainout = out/'train16'/'baseline'
        qstage('train16_baseline', ['baseline', '--manifest', m/'train16.jsonl', '--prompts', m/'train16_baseline_prompts.json',
               '--output', trainout/'predictions.jsonl'], trainout,
               lambda: check_complete(m/'train16.jsonl', trainout/'predictions.jsonl', require_parse=True))
        baseline = out/'train16'/'c_baseline.jsonl'
        if not baseline.exists():
            dump_rows(baseline, [{'id': r['id'], 'bbox': r['prediction']} for r in read_rows(trainout/'predictions.jsonl')])
        smoke_name = 'train16' if a.smoke_attempt == 1 else f'train16_attempt{a.smoke_attempt}'
        pipeline('train16', baseline, output_name=smoke_name)
        dump(out/'smoke_completed.json', {'finished': time.strftime('%F %T'), 'environment8_exact': True,
             'training_samples': 16, 'smoke_output': smoke_name, 'gt_used_for_inference': False, 'awaits_interface_inspection_before_full': True})
        return

    assert (out/'smoke_completed.json').exists() and (out/'configuration_frozen.json').exists(), 'smoke and interface inspection must finish first'
    pipeline('city412', m/'c_baseline.jsonl')
    full = out/'city412'
    stage('report412', [py, '-m', 'tools.report_aux_selection', 'main', '--gt', m/'scoring/city412_gt.json',
          '--c', m/'c_report_predictions.jsonl', '--selected', full/'trimodal/selected_predictions.jsonl',
          '--final', full/'trimodal/final_predictions.jsonl', '--evidence', full/'evidence.jsonl', '--output-dir', out/'report412'])
    gate = json.loads((out/'report412/gate.json').read_text())
    if gate['positive_gain']:
        subset = out/'city96'
        subset.mkdir(exist_ok=True)
        ids = {r['id'] for r in read_rows(m/'city96.jsonl')}
        for src, target in [(full/'evidence.jsonl', subset/'evidence.jsonl'), (full/'query/query_info.jsonl', subset/'query_info.jsonl')]:
            if not target.exists(): dump_rows(target, [r for r in read_rows(src) if r['id'] in ids])
        for condition in ['rgb_evidence', 'rgb_ir']:
            pred = subset/condition
            qstage('city96_'+condition, ['select-refine', '--manifest', m/'city96.jsonl', '--evidence', subset/'evidence.jsonl',
                   '--query-info', subset/'query_info.jsonl', '--condition', condition, '--output-dir', pred], pred,
                   lambda p=pred: check_complete(m/'city96.jsonl', p/'final_predictions.jsonl'))
        stage('report96', [py, '-m', 'tools.report_aux_selection', 'diagnostics', '--gt', m/'scoring/city96_gt.json',
              '--rgb', subset/'rgb_evidence/final_predictions.jsonl', '--rgb-ir', subset/'rgb_ir/final_predictions.jsonl',
              '--trimodal', full/'trimodal/final_predictions.jsonl', '--output-dir', out/'report96'])
    dump(out/'gpu_queue_complete.json', {'finished': time.strftime('%F %T'), 'gate': gate,
         'diagnostic96_run': gate['positive_gain'], 'training_run': False, 'submission_generated': False})


if __name__ == '__main__':
    main()
