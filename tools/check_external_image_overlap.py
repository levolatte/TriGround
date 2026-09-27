"""List RGB overlap candidates using thumbnail distances, without changing data."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def thumbnail(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((24, 24)), dtype=np.float32).reshape(-1)


def reference(manifest, root, label):
    payload = json.loads(manifest.read_text(encoding='utf-8-sig'))
    rows = payload.items() if isinstance(payload, dict) else ((r['id'], r) for r in payload)
    grouped = {}
    for sample_id, row in rows:
        path = (root / row.get('visible', row.get('rgb', ''))).resolve()
        grouped.setdefault(str(path), []).append(str(sample_id))
    return [{'path': path, 'source': label, 'ids': ids} for path, ids in grouped.items()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', nargs=3, action='append', metavar=('LABEL', 'MANIFEST', 'ROOT'))
    p.add_argument('--reference-cache', type=Path, required=True)
    p.add_argument('--new-manifest', type=Path)
    p.add_argument('--new-root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--max-distance', type=float, default=12.0)
    a = p.parse_args()
    if a.reference:
        rows = [r for label, manifest, root in a.reference for r in reference(Path(manifest), Path(root), label)]
        vectors = np.stack([thumbnail(r['path']) for r in rows])
        a.reference_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.reference_cache, vectors=vectors, records=json.dumps(rows))
        print(json.dumps({'reference_images': len(rows), 'cache': str(a.reference_cache)}))
    if not a.new_manifest:
        return
    if a.new_root is None or a.output is None:
        p.error('--new-manifest requires --new-root and --output')
    cache = np.load(a.reference_cache, allow_pickle=False)
    rows, vectors = json.loads(str(cache['records'])), cache['vectors']
    candidates = []
    new_rows = [json.loads(line) for line in a.new_manifest.read_text(encoding='utf-8').splitlines() if line]
    for row in new_rows:
        path = a.new_root / row['rgb']
        distance = np.abs(vectors - thumbnail(path)).mean(axis=1)
        matches = []
        for index in np.argsort(distance)[:3]:
            if distance[index] > a.max_distance:
                continue
            matches.append({**rows[index], 'thumbnail_mean_absolute_distance': float(distance[index])})
        if matches:
            candidates.append({'id': row['id'], 'rgb': str(path), 'matches': matches})
    report = {'new_image_groups': len(new_rows), 'candidate_groups': len(candidates),
              'method': '24x24 RGB mean absolute pixel distance; candidate threshold 12/255; not proof of duplicates',
              'candidates': candidates}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'new_image_groups': len(new_rows), 'candidate_groups': len(candidates)}))


if __name__ == '__main__':
    main()
