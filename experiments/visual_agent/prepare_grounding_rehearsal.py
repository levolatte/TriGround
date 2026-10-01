"""Export original grounding supervision for an already isolated training split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000, native_prompt


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def export(manifest, labels, output, count=750, seed=2031):
    tasks = read_rows(manifest)
    gold = {str(row['id']): row for row in read_rows(labels)}
    groups = defaultdict(list)
    for row in tasks:
        groups[row.get('source', 'city')].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while any(groups.values()) and len(selected) < count:
        for source in sorted(groups):
            if groups[source] and len(selected) < count:
                selected.append(groups[source].pop())
    rows = []
    for task in selected:
        sample_id = str(task['id'])
        label = gold[sample_id]
        box = label.get('bbox', label.get('bbox_xyxy_normalized'))
        available = [(name, task['images'][key]) for name, key in
                     [('rgb', 'rgb'), ('infrared', 'ir'), ('depth', 'depth_visual')]
                     if task['images'].get(key)]
        policy = 'millimeter' if task.get('depth_encoding') == 'city_mm' else 'visual'
        prompt = native_prompt(task['query'], tuple(name for name, _ in available), policy).replace('<image>\n', '')
        content = [{'type': 'image', 'image': path, 'max_pixels': 602112,
                    'modality': name, 'view': 'global'} for name, path in available]
        content.append({'type': 'text', 'text': prompt})
        rows.append({'id': sample_id + ':grounding_rehearsal', 'sample_id': sample_id,
                     'decision_index': 0, 'messages': [{'role': 'user', 'content': content}],
                     'target_action': json.dumps({'bbox_2d': bbox_to_qwen1000(box)}, separators=(',', ':')),
                     'source': task.get('source'), 'image_group': task.get('image_group', task['images']['rgb']),
                     'label_source': 'source_annotation_original_grounding',
                     'trajectory_source': 'grounding_rehearsal_not_agent_trajectory'})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    return {'requested': count, 'written': len(rows), 'unique_questions': len({r['sample_id'] for r in rows})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Training-only manifest after group exclusions')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=750)
    parser.add_argument('--seed', type=int, default=2031)
    args = parser.parse_args()
    print(json.dumps(export(args.manifest, args.labels, args.output, args.count, args.seed)))


if __name__ == '__main__':
    main()
