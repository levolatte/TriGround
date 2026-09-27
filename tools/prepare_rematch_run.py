"""Prepare the eight-image training pressure set; never change formal manifests."""
import argparse
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root / 'target_v2/qwen3vl_native_sft'
    tri = json.loads((base / 'trimodal_train.json').read_text())
    rgb = json.loads((base / 'rgb_train.json').read_text())
    assert len(tri) == len(rgb) == 3707
    assert [r['id'] for r in tri] == [r['id'] for r in rgb]
    sizes = {}
    for row in tri:
        for name in row['image']:
            if name not in sizes:
                with Image.open(args.data_root / name) as im:
                    sizes[name] = list(im.size)
    def area(row):
        return sum(min(602112, sizes[p][0] * sizes[p][1]) for p in row['image'])
    def length(row):
        return len(row['conversations'][0]['value'])
    selected, seen = [], set()
    for order, limit in [(sorted(tri, key=lambda r: (area(r), length(r)), reverse=True), 4),
                         (sorted(tri, key=lambda r: (length(r), area(r)), reverse=True), 8)]:
        for row in order:
            key = tuple(row['image'])
            if key not in seen:
                selected.append(row)
                seen.add(key)
            if len(selected) >= limit:
                break
    assert len(selected) == 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        'trimodal_pressure_8.json': selected,
        'trimodal_pressure_16.json': selected + selected,
        'pressure_selection.json': {
            'policy': 'four largest pixel-budget examples then four longest prompts; unique image groups',
            'samples': [{'id': r['id'], 'image_sizes': [sizes[p] for p in r['image']],
                         'prompt_characters': length(r)} for r in selected],
        },
    }
    for name, payload in payloads.items():
        (args.output_dir / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(payloads['pressure_selection.json'], ensure_ascii=False))


if __name__ == '__main__':
    main()
