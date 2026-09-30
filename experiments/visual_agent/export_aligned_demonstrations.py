"""Offline branch selection: supervise a real observation of the final target.

Collection has no GT input. This exporter uses training GT to choose among
already executed branches; these are offline expert labels, not online success.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .export_demonstrations import export


def align_rows(rows, branches, exclusions):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['id'], []).append(row)
    selected, revised = [], []
    for sample_id, decisions in grouped.items():
        final = next(row for row in decisions if row['is_final'])
        candidate_id = str(json.loads(final['target_action'])['candidate_id'])
        if candidate_id == 'KEEP':
            selected.extend(row for row in decisions
                            if (sample_id, row['decision_index']) not in exclusions)
            continue
        branch = branches[(sample_id, candidate_id)]
        event = branch['event']
        observation = event['observation']
        action = event['action']
        if (action['action'] != 'inspect_regions' or
                candidate_id not in map(str, action['candidate_ids']) or
                observation['status'] != 'OK'):
            raise ValueError(f'{sample_id}: branch does not inspect its final candidate')
        images = [image for image in observation['images']
                  if candidate_id in map(str, image.get('candidate_ids', []))]
        paths = {image['path'] for image in images}
        visible_paths = {block['image'] for message in branch['terminal_messages']
                         if isinstance(message['content'], list)
                         for block in message['content'] if block['type'] == 'image'}
        if not paths or not paths.issubset(visible_paths):
            raise ValueError(f'{sample_id}: final-target images missing from terminal input')
        prefix = branch['prefix_events']
        metadata = dict(id=sample_id, image_group=final['image_group'],
                        origin='scripted_offline_branch_selection',
                        original_raw_output=None, online_final_status='OBSERVATION_COMPLETE',
                        branch_candidate_id=candidate_id)
        selected.append(dict(**metadata, decision_index=len(prefix),
            messages=event['messages'], target_action=json.dumps(action, separators=(',', ':')),
            original_action=action, label_source='offline_GT_selects_executed_target_observation',
            is_final=False))
        selected.append(dict(**metadata, decision_index=len(prefix)+1,
            messages=branch['terminal_messages'], target_action=final['target_action'],
            original_action=None, label_source=final['label_source'], is_final=True))
        revised.append(dict(id=sample_id, candidate_id=candidate_id,
                            prior_decisions=len(decisions), new_decisions=2,
                            target_observation_images=sorted(paths)))
    return selected, revised


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('traces', 'train-manifest', 'holdout-manifest', 'debug-manifest',
                 'gt', 'output-dir', 'branches', 'action-review'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    inventory = export(args)
    path = args.output_dir/'train.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    branches = {}
    for source in sorted(args.branches.glob('*/branches.jsonl')):
        for line in source.read_text().splitlines():
            branch = json.loads(line)
            key = (str(branch['id']), str(branch['candidate_id']))
            if key in branches:
                raise ValueError(f'Duplicate evidence branch: {key}')
            branches[key] = branch
    review = json.loads(args.action_review.read_text())
    exclusions = {(row['id'], row['decision_index']) for row in review['excluded']}
    aligned, revised = align_rows(rows, branches, exclusions)
    path.rename(args.output_dir/'train_before_alignment.jsonl')
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in aligned))
    (args.output_dir/'inventory_before_alignment.json').write_text(json.dumps(inventory, indent=2))
    inventory.update(training_decisions=len(aligned),
        tool_actions=dict(Counter(json.loads(row['target_action'])['action'] for row in aligned)),
        aligned_corrective_trajectories=len(revised),
        excluded_semantic_review_decisions=len(exclusions),
        label_policy='GT-free real collection; offline training GT selects an already executed target observation branch. KEEP uses reviewed original demonstrations. Not autonomous success.',
        training_evidence_trajectories_scope='Original selected collection, before corrective branch replacement; positive action labels counted separately.')
    (args.output_dir/'inventory.json').write_text(json.dumps(inventory, indent=2))
    (args.output_dir/'alignment_review.json').write_text(json.dumps(revised, indent=2))
    (args.output_dir/'semantic_review.json').write_text(json.dumps(review, indent=2))
    print(json.dumps(inventory, indent=2))


if __name__ == '__main__':
    main()
