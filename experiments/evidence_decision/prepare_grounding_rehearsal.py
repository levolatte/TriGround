"""Export original boxes as native open-tool finish decisions.

These are source-annotation bbox rehearsals, not agent trajectories. Candidate
snapshots are included when an existing cache is available; missing caches are
left empty and never synthesized from the box label.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import tempfile

from .object_controller import DECISION_INSTRUCTION, OBJECT_PROFILE, TOOL_SCHEMAS, build_initial_messages, build_messages
from .object_tools import ObjectTools
from .presentation import compact_json
from .vision_tools import CandidatePool


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def _selected_tasks(tasks, count, seed, selection_rows=None):
    if selection_rows is not None:
        ids = [str(row.get('sample_id', row.get('id'))) for row in selection_rows]
        by_id = {str(row['id']): row for row in tasks}
        return [by_id[sample_id] for sample_id in ids[:count]]

    groups = defaultdict(list)
    for row in tasks:
        groups[row.get('source', 'city')].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected, selected_ids, selected_groups = [], set(), set()
    while any(groups.values()) and len(selected) < count:
        for source in sorted(groups):
            if not groups[source] or len(selected) >= count:
                continue
            index = next((i for i, row in enumerate(groups[source])
                          if str(row['id']) not in selected_ids and
                          str(row.get('image_group', row['id'])) not in selected_groups), None)
            if index is None:
                index = next((i for i, row in enumerate(groups[source])
                              if str(row['id']) not in selected_ids), None)
            if index is None:
                groups[source].clear()
                continue
            row = groups[source].pop(index)
            selected.append(row)
            selected_ids.add(str(row['id']))
            selected_groups.add(str(row.get('image_group', row['id'])))
    return selected


def _public_candidates(task, candidate_row, temporary_dir, seed):
    if candidate_row is None:
        return []
    pool = CandidatePool(candidate_row, (1, 1))
    tools = ObjectTools(task, pool, temporary_dir, seed=seed)
    return tools.public_candidates()


def _build_messages(task, candidates):
    initial = build_initial_messages(task, candidates, {'status': 'OK', 'images': []}, OBJECT_PROFILE)
    # No atlas pixels are manufactured here: the export runs offline and uses
    # only the frozen candidate cache. Remove the stock sentence implying an
    # atlas was attached when this scene has no cached overview image.
    initial[1]['content'] = [part for part in initial[1]['content']
                             if not (part.get('type') == 'text' and
                                     part.get('text', '').startswith(
                                         'The initial candidate overview contains'))]
    return build_messages(initial, [], candidates, [], {}, DECISION_INSTRUCTION, OBJECT_PROFILE)


def export(manifest, labels, output, count=750, seed=2031, *, candidates=None,
           selection_manifest=None):
    tasks = read_rows(manifest)
    gold = {str(row['id']): row for row in read_rows(labels)}
    candidate_by_id = ({str(row['id']): row for row in read_rows(candidates)}
                       if candidates else {})
    selection_rows = read_rows(selection_manifest) if selection_manifest else None
    selected = _selected_tasks(tasks, count, seed, selection_rows)
    rows = []
    with tempfile.TemporaryDirectory(prefix='grounding_rehearsal_') as temporary_dir:
        for task in selected:
            sample_id = str(task['id'])
            label = gold[sample_id]
            box = label.get('bbox', label.get('bbox_xyxy_normalized'))
            candidate_row = candidate_by_id.get(sample_id)
            candidates_public = _public_candidates(task, candidate_row, temporary_dir, seed)
            messages = _build_messages(task, candidates_public)
            target = '<tool_call>' + compact_json({
                'name': 'finish', 'arguments': {'bbox': [float(value) for value in box]},
            }) + '</tool_call>'
            rows.append({
                'id': sample_id + ':grounding_rehearsal',
                'sample_id': sample_id,
                'decision_index': 0,
                'messages': messages,
                'tools': TOOL_SCHEMAS,
                'target_action': target,
                'source': task.get('source'),
                'image_group': task.get('image_group', task.get('scene_id', sample_id)),
                'label_source': 'source_annotation_original_grounding',
                'trajectory_source': 'grounding_rehearsal_not_agent_trajectory',
                'candidate_snapshot_available': candidate_row is not None,
                'bbox_frame': 'normalized_xyxy_global_rgb',
                'bbox_target_kind': 'native_finish_tool_call',
            })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    return {
        'requested': count,
        'written': len(rows),
        'unique_questions': len({r['sample_id'] for r in rows}),
        'unique_groups': len({r['image_group'] for r in rows}),
        'with_candidate_snapshot': sum(row['candidate_snapshot_available'] for row in rows),
        'without_candidate_snapshot': sum(not row['candidate_snapshot_available'] for row in rows),
        'target_action': 'native finish tool call with normalized xyxy bbox in global RGB',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Training-only manifest after group exclusions')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--candidates', type=Path,
                        help='Existing candidate cache; rows with missing cache retain an empty snapshot')
    parser.add_argument('--selection-manifest', type=Path,
                        help='Optional JSONL listing the exact sample IDs to re-export')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=750)
    parser.add_argument('--seed', type=int, default=2031)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.labels, args.output, args.count, args.seed,
                            candidates=args.candidates,
                            selection_manifest=args.selection_manifest), ensure_ascii=False))


if __name__ == '__main__':
    main()
