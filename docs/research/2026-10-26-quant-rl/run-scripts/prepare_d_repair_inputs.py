"""Freeze D repair queries by source order, without reading model predictions."""

import argparse
import json
from pathlib import Path


HARD4_IDS = (
    'city_shuming_78_00000046_010',
    'city_002041_001',
    'rgbdt500_323_00000118::q01',
    'roborefit_train_000919',
)
EVIDENCE_DIR = Path(__file__).resolve().parents[1] / 'evidence'


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def select(rows, quotas, excluded=(), excluded_scenes=()):
    excluded = set(excluded)
    excluded_scenes = set(excluded_scenes)
    counts = {source: 0 for source in quotas}
    selected = []
    seen = set()
    for row in rows:
        source, identity = row['source'], row['id']
        if (source not in quotas or counts[source] >= quotas[source]
                or identity in excluded or row['group_id'] in excluded_scenes
                or identity in seen):
            continue
        selected.append(row)
        seen.add(identity)
        counts[source] += 1
    if counts != quotas:
        raise ValueError(f'Cannot fill source quotas: {counts} != {quotas}')
    return selected


def hard4_from(rows):
    selected = [row for row in rows if row['id'] in HARD4_IDS]
    if {row['id'] for row in selected} != set(HARD4_IDS) or len(selected) != 4:
        raise ValueError('Formal diagnostic must contain the four fixed hard queries')
    if {source: sum(row['source'] == source for row in selected)
            for source in ('city', 'rgbdt', 'robo')} != {'city': 2, 'rgbdt': 1, 'robo': 1}:
        raise ValueError('Fixed hard queries must contain two city, one rgbdt, and one robo query')
    scored = [row for row in read(EVIDENCE_DIR / 'warmup50-diagnostic32-g8-scored.jsonl')
              if row['id'] in HARD4_IDS]
    if {row['id'] for row in scored} != set(HARD4_IDS) or len(scored) != 4:
        raise ValueError('Warmup50 scored diagnostic must contain the four fixed hard queries')
    for row in scored:
        if len(row['samples']) != 8:
            raise ValueError(f"{row['id']}: warmup diagnostic must contain G8")
        scores = [sample.get('score', sample) for sample in row['samples']]
        valid = sum(score['valid'] for score in scores)
        truncated = sum(score['truncated'] for score in scores)
        hits = sum(score['hit'] for score in scores)
        if valid != 8 or truncated != 0 or not 2 <= hits <= 6:
            raise ValueError(f"{row['id']}: warmup evidence misses 8 valid, 0 truncated, 2-6 hits")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = read(args.diagnostic)
    four = hard4_from(rows)
    hard_scenes = {row['group_id'] for row in four}
    twelve = select(rows, {'city': 6, 'rgbdt': 3, 'robo': 3}, HARD4_IDS, hard_scenes)
    if ({row['id'] for row in twelve} & set(HARD4_IDS)
            or {row['group_id'] for row in twelve} & hard_scenes):
        raise ValueError('The 12 training queries overlap a hard4 query or scene')
    selected = {'train4': four, 'train12': twelve}
    if args.dataset_dir:
        city = read(args.dataset_dir / 'city400_dev.jsonl')
        external = read(args.dataset_dir / 'external94_dev.jsonl')
        if len(city) != 378 or len(external) != 88:
            raise ValueError('Expected frozen formal development sets with 378/88 queries')
        # The frozen external development split contains RGBDT only; Robo is training-only.
        heldout = select(city + external, {'city': 6, 'rgbdt': 6})
        train_groups = {row['group_id'] for row in four + twelve}
        if train_groups & {row['group_id'] for row in heldout}:
            raise ValueError('Training and heldout scenes overlap')
        selected['heldout12'] = heldout
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in selected.items():
        (args.output_dir / (name + '.jsonl')).write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in subset), encoding='utf-8')
    manifest = {name: [{'id': row['id'], 'source': row['source'], 'group_id': row['group_id']}
                      for row in subset] for name, subset in selected.items()}
    (args.output_dir / 'input_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: len(subset) for name, subset in selected.items()}))


if __name__ == '__main__':
    main()
