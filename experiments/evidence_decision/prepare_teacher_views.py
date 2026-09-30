"""Save the exact processor-sized teacher views on the machine holding images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .remote_object_teacher import _prepare_visible_images


def prepare(episode_dir: Path, step_index: int | None = None):
    state = json.loads((episode_dir / 'episode.json').read_text(encoding='utf-8'))
    if step_index is None:
        index = len(state['steps'])
        output = episode_dir / 'visible' / f'step_{index:04d}'
        messages = json.loads((episode_dir / 'current_messages.json').read_text(encoding='utf-8'))
        tools = json.loads((episode_dir / 'current_tools.json').read_text(encoding='utf-8'))
    else:
        index = step_index
        output = episode_dir / 'visible' / f'before_step_{index:04d}'
        step = state['steps'][index]
        messages, tools = step['messages'], step['tools']
    prepared, metadata = _prepare_visible_images(messages, output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'current_messages.json').write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'image_preparation.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'current_tools.json').write_text(json.dumps(tools, ensure_ascii=False), encoding='utf-8')
    return {'directory': str(output), 'images': len(metadata), 'step': index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-dir', type=Path, required=True)
    parser.add_argument('--step-index', type=int, help='Prepare the recorded input before a historical action')
    args = parser.parse_args()
    print(json.dumps(prepare(args.episode_dir, args.step_index), ensure_ascii=False))


if __name__ == '__main__':
    main()
