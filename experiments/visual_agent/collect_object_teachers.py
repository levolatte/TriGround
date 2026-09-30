"""Export reviewed real steps; offline GT filters final answers only.

Review JSONL has one row per step, using the zero-based step index from
episode.json::steps:
{"id":"sample-id","step":0,"use_action":true,"note_supported":true,"reason":"..."}

No review row means no supervision. A supported tool action can be retained
without its note by setting note_supported=false. A wrong final answer never
removes supported earlier evidence actions, and never becomes a positive label.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
from pathlib import Path

from .evaluate import iou
from .object_controller import DECISION_INSTRUCTION
from .presentation import compact_json


def _read_labels(path: Path | None):
    if path is None:
        return None
    if path.suffix == '.jsonl':
        return {str(row['id']): row for row in map(json.loads, path.read_text(encoding='utf-8-sig').splitlines())
                if row}
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _read_reviews(path: Path):
    reviews = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row['id'])
        step_index = row['step']
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError(f'{path}:{line_number}: step must be a zero-based nonnegative integer')
        for field in ('use_action', 'note_supported'):
            if not isinstance(row.get(field), bool):
                raise ValueError(f'{path}:{line_number}: {field} must be an explicit boolean')
        reason = row.get('reason')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f'{path}:{line_number}: reason must explain the step review')
        key = (sample_id, step_index)
        if key in reviews:
            raise ValueError(f'{path}:{line_number}: duplicate review for {sample_id} step {step_index}')
        reviews[key] = {**row, 'id': sample_id, 'reason': reason.strip()}
    return reviews


def _training_row(state, step, review, final_iou):
    messages = copy.deepcopy(step['messages'])
    last_content = messages[-1].get('content')
    if isinstance(last_content, list):
        for part in last_content:
            if (part.get('type') == 'text' and
                    part.get('text', '').startswith('Teacher: review the visible evidence')):
                part['text'] = DECISION_INSTRUCTION

    target_action = step['target_action']
    if not review['note_supported']:
        target_action = f"<tool_call>{compact_json(step['call'])}</tool_call>"
    return {
        'schema_version': 'visual-agent-object-teacher-v1',
        'id': state['sample_id'],
        'query': state['row']['query'],
        'image_group': state['row']['images']['rgb'],
        'step': step['index'],
        'origin': 'teacher',
        'label_source': 'explicit_step_review',
        'supervision_status': 'teacher_reviewed_action',
        'messages': messages,
        'target_action': target_action,
        'tools': copy.deepcopy(step['tools']),
        'is_final': step['call']['name'] == 'finish',
        'observation_status': step['observation']['status'],
        'protocol_valid': bool(step.get('protocol_valid')),
        'step_review': {
            'use_action': review['use_action'],
            'note_supported': review['note_supported'],
            'reason': review['reason'],
        },
        'final_iou_diagnostic': final_iou,
        'final_iou_used_for_supervision': step['call']['name'] == 'finish',
    }


def collect(episodes: Path, reviews_path: Path, output: Path, labels_path: Path | None = None):
    gold = _read_labels(labels_path)
    reviews = _read_reviews(reviews_path)
    records, decisions, action_counts = [], [], Counter()
    review_keys_seen = set()
    episode_paths = sorted(episodes.glob('*/episode.json'))
    if not episode_paths:
        raise ValueError(f'no immediate child episode.json files under {episodes}')
    sample_ids = set()

    for path in episode_paths:
        state = json.loads(path.read_text(encoding='utf-8'))
        sample_id = str(state['sample_id'])
        if sample_id in sample_ids:
            raise ValueError(f'duplicate sample ID under {episodes}: {sample_id}')
        sample_ids.add(sample_id)
        final = next((step for step in reversed(state['steps'])
                      if (step.get('call') or {}).get('name') == 'finish' and
                      step['observation']['status'] == 'OK' and
                      step['observation'].get('data', {}).get('bbox') is not None), None)
        bbox = final['observation']['data']['bbox'] if final else None
        score = None
        if gold is not None:
            label = gold[sample_id]
            target = label.get('bbox', label.get('bbox_xyxy_normalized'))
            score = iou(bbox, target)

        raw_steps = []
        for step in state['steps']:
            key = (sample_id, step['index'])
            review = reviews.get(key)
            if review is not None:
                review_keys_seen.add(key)
            raw_steps.append({
                'index': step['index'],
                'note': step['note'],
                'call': copy.deepcopy(step['call']),
                'target_action': step['target_action'],
                'protocol_valid': bool(step.get('protocol_valid')),
                'observation_status': step['observation']['status'],
                'review': copy.deepcopy(review),
            })
            if review is None or not review['use_action'] or not step.get('protocol_valid'):
                continue
            if step['call']['name'] == 'finish' and (score is None or score < 0.5):
                raw_steps[-1]['supervision_excluded'] = ('final_not_scored' if score is None
                                                        else 'final_iou_below_0_5')
                continue
            decisions.append(_training_row(state, step, review, score))
            action_counts[step['call']['name']] += 1

        records.append({
            'id': sample_id,
            'complete': bool(state['complete']),
            'terminal_bbox': bbox,
            'final_iou': score,
            'final_iou_ge_0_5': None if score is None else score >= 0.5,
            'steps': raw_steps,
            'episode': str(path),
            'label_source': 'real_teacher_episode; optional_offline_gt_score',
        })

    unknown_reviews = set(reviews) - review_keys_seen
    if unknown_reviews:
        sample_id, step_index = sorted(unknown_reviews)[0]
        raise ValueError(f'review references missing episode step: {sample_id} step {step_index}')

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('records.jsonl', records), ('decisions.jsonl', decisions)]:
        (output / name).write_text(
            ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows),
            encoding='utf-8')
    scored = [row['final_iou'] for row in records if row['final_iou'] is not None]
    summary = {
        'episodes': len(records),
        'complete_episodes': sum(row['complete'] for row in records),
        'scored_episodes': len(scored),
        'episodes_final_iou_ge_0_5': sum(row['final_iou_ge_0_5'] is True for row in records),
        'raw_steps': sum(len(row['steps']) for row in records),
        'review_rows': len(reviews),
        'supervised_decisions': len(decisions),
        'supervised_actions': dict(action_counts),
        'mean_final_iou_diagnostic': sum(scored) / len(scored) if scored else None,
        'note': ('Every decision requires an explicit use_action=true step review and a protocol-valid original action. '
                 'Unsupported notes are removed from target_action. Final actions additionally require offline '
                 'IoU>=0.5; final correctness never filters earlier evidence actions or changes recorded actions.'),
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, required=True,
                        help='Directory whose immediate child directories contain episode.json')
    parser.add_argument('--reviews', type=Path, required=True, help='Explicit per-step human review JSONL')
    parser.add_argument('--labels', type=Path,
                        help='Offline GT labels: required to supervise finish, not to retain reviewed evidence actions')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.episodes, args.reviews, args.output_dir, args.labels), ensure_ascii=False))


if __name__ == '__main__':
    main()
