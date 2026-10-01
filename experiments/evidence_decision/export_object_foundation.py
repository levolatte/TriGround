"""Offline candidate-choice lessons from actual detector pools and private labels.

These are supervised labels, not successful teacher or student trajectories.
No ground-truth crop, additional candidate, or invented evidence is created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from PIL import Image

from .evaluate import iou
from .object_controller import DECISION_INSTRUCTION, TOOL_SCHEMAS, build_initial_messages, build_messages
from .object_tools import ObjectTools
from .prepare_object_assets import public_rows, read_rows
from .vision_tools import CandidatePool


def export(args):
    rows = public_rows(args.manifest)
    candidates = {str(row['id']): row for row in read_rows(args.candidate_cache)}
    if args.labels.suffix == '.jsonl':
        labels = {str(row['id']): row for row in read_rows(args.labels)}
    else:
        labels = json.loads(args.labels.read_text(encoding='utf-8-sig'))
    eligible, skipped = [], []
    for row in rows:
        sample_id = str(row['id'])
        if sample_id not in candidates:
            skipped.append({'id': sample_id, 'reason': 'no_candidate_cache'})
            continue
        label = labels[sample_id]
        target = label.get('bbox', label.get('bbox_xyxy_normalized'))
        pool = candidates[sample_id]
        options = [c for c in pool['candidates'] if c['role'] == 'target']
        covered = [c for c in options if iou(c['bbox'], target) >= .5]
        if not covered:
            skipped.append({'id': sample_id, 'reason': 'uncovered'})
            continue
        best = max(covered, key=lambda c: iou(c['bbox'], target))
        eligible.append((row, best['bbox'], iou(pool['c_bbox'], target) >= .5))
    random.Random(args.seed).shuffle(eligible)
    # Alternate initial-correct and initial-wrong tasks when both are available.
    groups = [[entry for entry in eligible if entry[2] == value] for value in (False, True)]
    selected = []
    while any(groups) and len(selected) < args.limit:
        for group in groups:
            if group and len(selected) < args.limit:
                selected.append(group.pop())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'decisions.jsonl'
    if output.exists():
        raise FileExistsError(output)
    with output.open('w', encoding='utf-8') as handle:
        for index, (row, target_box, initial_correct) in enumerate(selected):
            sample_id = str(row['id'])
            with Image.open(row['images']['rgb']) as image:
                pool = CandidatePool(candidates[sample_id], image.size)
            tool = ObjectTools(row, pool, args.output_dir / 'observations' / sample_id,
                               seed=args.seed + index)
            public = tool.public_candidates()
            chosen = next(item['id'] for item in public if item['bbox'] == target_box and item['role'] == 'target')
            modalities = [mode for mode, key in [('rgb','rgb'),('ir','ir'),('depth','depth_visual')]
                          if row['images'].get(key)]
            atlas = tool.atlas(modalities=modalities)
            initial = build_initial_messages(row, public, atlas)
            messages = build_messages(initial, [], public, [], {},
                DECISION_INSTRUCTION)
            call = {'name': 'finish', 'arguments': {'id': chosen}}
            decision = {'id': sample_id, 'decision_index': 0, 'messages': messages,
                        'tools': TOOL_SCHEMAS, 'target_action': '<tool_call>' + json.dumps(call) + '</tool_call>',
                        'label_source': 'offline_gt_candidate_choice',
                        'trajectory_source': 'foundation_candidate_choice_not_online_success',
                        'initial_correct': initial_correct, 'image_group': row['images']['rgb']}
            handle.write(json.dumps(decision, ensure_ascii=False) + '\n')
    summary = {'available_questions': len(rows), 'eligible': len(eligible), 'exported': len(selected),
               'initial_correct': sum(entry[2] for entry in selected),
               'initial_wrong': sum(not entry[2] for entry in selected), 'skipped': skipped,
               'label_source': 'offline_gt_candidate_choice', 'autonomous_success_claimed': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
