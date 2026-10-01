"""Offline labels for GT-free, actually executed scripted demonstrations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .export_trajectories import (
    _gt_boxes, _rows, load_frozen_manifest, revised_final_label, verify_frozen_splits,
)


def evidence_types(trace):
    events = trace['events']
    actions = [event['action']['action'] for event in events]
    recovery = any(events[i]['observation']['status'] in ('UNKNOWN', 'EMPTY') and
                   events[i]['action'] != events[i + 1]['action'] for i in range(len(events) - 1))
    return dict(recovery=recovery, search='search_candidates' in actions,
                depth='measure_depth' in actions,
                supported_depth=any(event['action']['action'] == 'measure_depth' and
                    event['observation'].get('data', {}).get('pair', {}).get('status') == 'supported'
                    for event in events),
                ir=any('ir' in event['action'].get('modalities', []) or
                       event['action'].get('modality') == 'ir' for event in events))


def export(args):
    train = load_frozen_manifest(args.train_manifest)
    holdout = load_frozen_manifest(args.holdout_manifest)
    debug = load_frozen_manifest(args.debug_manifest)
    verify_frozen_splits(train, holdout, debug)
    boxes = _gt_boxes(args.gt)
    prepared, inventory = [], Counter()
    seen = set()
    for trace in _rows(args.traces):
        sample_id = str(trace['id'])
        if sample_id not in train or sample_id in seen:
            raise ValueError(f'{sample_id}: not a unique frozen training start')
        seen.add(sample_id)
        if trace['manifest_row']['images']['rgb'] != train[sample_id]['images']['rgb']:
            raise ValueError(f'{sample_id}: training image group changed')
        if trace['manifest_row']['query'] != train[sample_id]['query'] or trace['query'] != train[sample_id]['query']:
            raise ValueError(f'{sample_id}: training query changed')
        if trace.get('trajectory_origin') != 'scripted':
            raise ValueError('This exporter accepts explicitly scripted observations only')
        inventory['collected_trajectories'] += 1
        final, source = revised_final_label(trace, trace['final_candidates'], boxes[sample_id])
        if final is None:
            inventory['uncovered_trajectories'] += 1
            continue
        rows = []
        for index, event in enumerate(trace['events']):
            action = event.get('action') or {}
            observation = event.get('observation') or {}
            if action.get('action') not in ('inspect_regions', 'measure_depth', 'search_candidates'):
                continue
            if observation.get('status') not in ('OK', 'UNKNOWN', 'EMPTY'):
                inventory['excluded_error_decisions'] += 1
                continue
            rows.append(dict(id=sample_id, decision_index=index,
                image_group=train[sample_id]['images']['rgb'], origin='scripted',
                messages=event['messages'], target_action=json.dumps(action, separators=(',', ':')),
                original_action=action, original_raw_output=None,
                label_source='scripted_query_and_actual_observation_policy',
                online_final_status=trace['final_status'], is_final=False))
        rows.append(dict(id=sample_id, decision_index=len(trace['events']),
            image_group=train[sample_id]['images']['rgb'], origin='scripted',
            messages=trace['terminal_messages'], target_action=json.dumps(final, separators=(',', ':')),
            original_action=None, original_raw_output=None,
            label_source='offline_' + source, online_final_status=trace['final_status'], is_final=True))
        prepared.append((final['candidate_id'] == 'KEEP', rows, trace))
    # Preserve scarce real recovery/search examples when limiting KEEP labels.
    prepared.sort(key=lambda item: (item[0], *(-int(evidence_types(item[2])[key])
        for key in ('supported_depth', 'recovery', 'search', 'depth', 'ir')), item[1][0]['id']))
    selected, selected_traces = [], []
    composition = Counter()
    keeps = 0
    for keep, rows, trace in prepared:
        if keep and (keeps + 1) * 2 > len(selected_traces) + 1:
            inventory['excluded_keep_cap_trajectories'] += 1
            continue
        keeps += int(keep)
        selected.extend(rows)
        selected_traces.append(trace['id'])
        composition.update(key for key, present in evidence_types(trace).items() if present)
    if not selected:
        raise ValueError('No demonstrations survive terminal coverage and KEEP cap')
    inventory.update(training_trajectories=len(selected_traces), training_decisions=len(selected),
        keep_trajectories=keeps, missing_training_starts=len(set(train) - seen))
    result = dict(inventory)
    result.update(training_ids=selected_traces, tool_actions=dict(Counter(
        json.loads(row['target_action'])['action'] for row in selected)),
        training_evidence_trajectories=dict(composition),
        label_policy='Scripted actions use query/current candidates/real observations only. GT labels only the final available ID. No online model success is claimed.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'train.jsonl').open('w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (args.output_dir / 'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('traces', 'train-manifest', 'holdout-manifest', 'debug-manifest', 'gt', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(export(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
