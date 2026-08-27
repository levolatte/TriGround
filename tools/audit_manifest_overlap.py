from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_target_v2_data import sequence_key, visible_path


def load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not an object-keyed manifest")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    args = parser.parse_args()
    train = load(args.train)
    test = load(args.test)
    exact_ids = sorted(set(train) & set(test))
    train_sequences: dict[str, list[str]] = {}
    for sample_id in train:
        train_sequences.setdefault(sequence_key(sample_id), []).append(sample_id)
    test_sequences: dict[str, list[str]] = {}
    for sample_id in test:
        test_sequences.setdefault(sequence_key(sample_id), []).append(sample_id)
    shared_sequences = sorted(set(train_sequences) & set(test_sequences))
    train_visible = {
        str(visible_path(record, args.train)): sample_id
        for sample_id, record in train.items()
    }
    test_visible = {
        str(visible_path(record, args.test)): sample_id
        for sample_id, record in test.items()
    }
    exact_images = sorted(set(train_visible) & set(test_visible))
    affected_test = sorted(
        sample_id
        for sequence in shared_sequences
        for sample_id in test_sequences[sequence]
    )
    print(json.dumps({
        "train_samples": len(train),
        "test_samples": len(test),
        "exact_id_overlap_count": len(exact_ids),
        "exact_id_overlap_examples": exact_ids[:10],
        "exact_image_overlap_count": len(exact_images),
        "exact_image_overlap_examples": exact_images[:10],
        "shared_sequence_count": len(shared_sequences),
        "affected_test_samples": len(affected_test),
        "shared_sequence_examples": shared_sequences[:10],
        "affected_test_examples": affected_test[:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
