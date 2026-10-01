"""Create real initial predictions and detector candidates for mixed-modality tasks.

The public manifest is the only task input. Ground truth is never read here.
Existing City caches can be reused; missing modalities are omitted, not fabricated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

from PIL import Image

from .bootstrap import cpu_dino
from tools.aux_selection_evidence import build_candidate_row, detect_image_proposals
from tools.predict_aux_selection import (
    _generate, _load_qwen, build_query_parse_prompt, load_manifest, parse_query_json,
)
from tools.prepare_qwen3vl_native_sft import native_prompt


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def public_rows(path):
    validated = load_manifest(Path(path), require_images=True)
    originals = {str(row['id']): row for row in read_rows(path)}
    for row in validated:
        source = originals[str(row['id'])]
        for key in ('source', 'image_group', 'scene_id', 'ir_rgb_registration',
                    'registration_source', 'depth_visual_encoding'):
            if key in source:
                row[key] = source[key]
        for key, value in row['images'].items():
            if value:
                p = Path(value)
                row['images'][key] = str(p if p.is_absolute() else (Path(path).parent / p).resolve())
    return validated


def run(args):
    rows = public_rows(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    done = {str(row['id']) for row in read_rows(args.output)} if args.output.exists() else set()
    cached = {}
    for path in args.reuse:
        cached.update({str(row['id']): row for row in read_rows(path)})
    started = time.monotonic()
    failures, reused, generated = [], 0, 0
    model_state = None
    if args.stage == 'candidates':
        baseline = {str(row['id']): row for row in read_rows(args.baseline)}
        query_info = {str(row['id']): row for row in read_rows(args.query_info)}
    with args.output.open('a', encoding='utf-8') as handle:
        for row in rows:
            sample_id = str(row['id'])
            if sample_id in done:
                continue
            if time.monotonic() - started >= args.max_run_seconds:
                break
            previous = cached.get(sample_id)
            if previous and previous.get('query') == row['query']:
                result = {**previous, 'reused_from_existing_cache': True}
                if args.stage == 'candidates':
                    result.update(images=row['images'], depth_encoding=row.get('depth_encoding'))
                reused += 1
            elif args.stage == 'candidates':
                prediction = baseline.get(sample_id, {})
                initial_box = prediction.get('bbox', prediction.get('prediction'))
                info = query_info.get(sample_id, {})
                if initial_box is None or not info.get('parsed'):
                    failures.append({'id': sample_id, 'reason': 'missing_or_invalid_initial_prediction_or_query_parse'})
                    continue
                detections = []
                if info['scope'] == 'single':
                    if model_state is None:
                        model_state = cpu_dino(args.model, args.threads)
                        if args.detector_device != 'cpu':
                            model_state[1].to(args.detector_device)
                    processor, model = model_state
                    modalities = ['rgb']
                    if row['images'].get('ir') and (row.get('depth_encoding') == 'city_mm' or
                            row.get('ir_rgb_registration') == 'normalized_shared_frame'):
                        modalities.append('ir')
                    for modality in modalities:
                        with Image.open(row['images'][modality]) as image:
                            detections.extend(detect_image_proposals(image.convert('RGB'), modality,
                                              info, processor, model, args.detector_device))
                result = build_candidate_row(row, {'bbox': initial_box}, info, detections,
                                             random.Random(args.seed + len(done)))
                result['detector_device'] = args.detector_device
                generated += 1
            else:
                if model_state is None:
                    model_state = _load_qwen(args.model, args.adapter)
                torch, processor, model = model_state
                if args.stage == 'query':
                    prompt, images = build_query_parse_prompt(row['query']), []
                else:
                    available = [(name, key) for name, key in
                        [('rgb', 'rgb'), ('infrared', 'ir'), ('depth', 'depth_visual')]
                        if row['images'].get(key)]
                    policy = 'millimeter' if row.get('depth_encoding') == 'city_mm' else 'visual'
                    prompt = native_prompt(row['query'], tuple(name for name, _ in available), policy)
                    images = []
                    for _, key in available:
                        with Image.open(row['images'][key]) as image:
                            images.append(image.convert('RGB').copy())
                with torch.inference_mode():
                    output = _generate(torch, processor, model, prompt, images,
                                       prompt_has_image_placeholders=args.stage == 'baseline')
                result = {'id': sample_id, 'query': row['query'], **output}
                if args.stage == 'query':
                    parsed = parse_query_json(output['raw_text'])
                    result.update(parsed=parsed is not None, **(parsed or {}))
                else:
                    from tools.evaluate_pretrained_grounder import parse_generated_bbox
                    prediction = parse_generated_bbox(output['raw_text'])
                    result.update(bbox=prediction, parsed=prediction is not None)
                if not result['parsed']:
                    failures.append({'id': sample_id, 'reason': 'invalid_model_output'})
                generated += 1
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            done.add(sample_id)
            print(json.dumps({'id': sample_id, 'stage': args.stage, 'done': len(done),
                              'total': len(rows), 'seconds': time.monotonic() - started}), flush=True)
    if args.stage == 'baseline':
        prompt_policy = 'native_prompt; one query plus source full-frame modalities; model receives each listed image with native prompt image placeholder'
        modality_rows = [[name for name, key in [('rgb', 'rgb'), ('infrared', 'ir'), ('depth', 'depth_visual')]
                          if row['images'].get(key)] for row in rows]
    elif args.stage == 'query':
        prompt_policy = 'text-only build_query_parse_prompt; no image pixels'
        modality_rows = [[] for _ in rows]
    else:
        prompt_policy = 'no Qwen prompt; use the referenced cached C baseline and parsed query info, then run detector proposals'
        modality_rows = []
        for row in rows:
            info = query_info.get(str(row['id']), {})
            used = ['rgb'] if info.get('scope') == 'single' else []
            if used and row['images'].get('ir') and (row.get('depth_encoding') == 'city_mm' or
                    row.get('ir_rgb_registration') == 'normalized_shared_frame'):
                used.append('infrared')
            modality_rows.append(used)
    modality_counts = {}
    for modalities in modality_rows:
        for modality in modalities:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
    summary = {'stage': args.stage, 'manifest': str(args.manifest), 'output': str(args.output),
               'model': str(args.model), 'adapter': str(args.adapter) if args.adapter else None,
               'prompt_policy': prompt_policy,
               'actual_modalities': modality_counts,
               'fresh_inference': bool(args.stage in {'baseline', 'query'} and generated == len(rows)
                                      and reused == 0 and not failures),
               'fresh_inference_rows': generated,
               'baseline_path': str(args.baseline) if args.stage == 'candidates' else None,
               'query_info_path': str(args.query_info) if args.stage == 'candidates' else None,
               'expected': len(rows), 'completed': len(done),
               'reused': reused, 'generated': generated, 'failures': failures,
               'elapsed_seconds': time.monotonic() - started}
    if args.stage == 'candidates':
        summary['detector_device'] = args.detector_device
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['baseline', 'query', 'candidates'], required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--query-info', type=Path)
    parser.add_argument('--reuse', type=Path, action='append', default=[])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--detector-device', choices=['cpu', 'cuda:0'], default='cpu',
                        help='Use CUDA only in a separately scheduled detector phase without Qwen resident')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-run-seconds', type=float, default=3600)
    args = parser.parse_args()
    if args.stage == 'candidates' and (args.baseline is None or args.query_info is None):
        parser.error('candidates needs --baseline and --query-info')
    if args.stage != 'candidates' and args.adapter is None:
        parser.error('prediction preparation needs --adapter')
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
