"""Reuse mainline's new natural queries with their original full-frame images.

Only task inputs are copied into the public manifest. Existing target labels and
manually chosen evidence regions remain offline and never generate candidates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from .evaluate import SHARED_RE
from .prepare_object_data import read_jsonl, normalized_query, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cloud-output-root', required=True)
    args = parser.parse_args()
    excluded = {r['image_group'] for r in read_jsonl(args.data_dir / 'group_splits.jsonl')
                if r['split'] == 'preserved_holdout'}
    base = read_jsonl(args.data_dir / 'manifest.jsonl')
    existing = {(r['image_group'], normalized_query(r['query'])): r['id'] for r in base}
    output, cloud, labels, skipped = [], [], [], []
    copied = {}
    assets = args.output_dir / 'images'
    assets.mkdir(parents=True, exist_ok=True)
    for row in read_jsonl(args.reviews):
        if not row.get('selected_for_training'):
            continue
        source, scene = row['source'], row['scene_id']
        group = scene if source == 'city' else source + ':' + scene
        if group in excluded or (source == 'city' and SHARED_RE.search(scene)):
            skipped.append({'source_task_id': row['task_id'], 'reason': 'preserved_holdout_group'})
            continue
        query = row.get('query', row.get('proposed_query'))
        identity = (group, normalized_query(query))
        if identity in existing:
            skipped.append({'source_task_id': row['task_id'], 'reason': 'same_scene_query_in_base',
                            'existing_id': existing[identity]})
            continue
        sample_id = f'reviewed_{len(output):04d}'
        images, remote_images = {}, {}
        for original, mode in [('rgb','rgb'),('infrared','ir'),('depth','depth_visual'),('depth_raw','depth_raw')]:
            if not row['images'].get(original):
                continue
            path = Path(row['images'][original])
            if str(path) not in copied:
                destination = assets / (sample_id + '_' + mode + path.suffix)
                shutil.copy2(path, destination)
                copied[str(path)] = destination
            destination = copied[str(path)]
            images[mode] = str(destination.resolve())
            remote_images[mode] = args.cloud_output_root.rstrip('/') + '/images/' + destination.name
        record = {'id': sample_id, 'source_id': row['task_id'], 'source': source,
                  'image_group': group, 'scene_id': scene, 'query': query, 'images': images,
                  'available_modalities': list(images),
                  'depth_encoding': 'city_mm' if source == 'city' else 'unknown',
                  'ir_rgb_registration': 'normalized_shared_frame' if source == 'city' or source.startswith('rgbt_') else 'not_available',
                  'origin': 'mainline_new_queries', 'task_category': row['category']}
        output.append(record)
        cloud.append({**record, 'images': remote_images})
        labels.append({'id': sample_id, 'bbox_xyxy_normalized': row['bbox'],
                       'label_source': 'mainline_reviewed_query', 'source_task_id': row['task_id']})
        existing[identity] = sample_id
    write_jsonl(args.output_dir / 'manifest.jsonl', output)
    write_jsonl(args.output_dir / 'cloud_manifest.jsonl', cloud)
    write_jsonl(args.output_dir / 'private_labels_offline.jsonl', labels)
    summary = {'additional_queries': len(output), 'skipped': skipped,
               'integration': 'Prefer these focused questions within source quotas when assembling final training; base collection remains frozen.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'additional_queries': len(output), 'skipped': len(skipped)}))


if __name__ == '__main__':
    main()
