from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_target_v2_data import sequence_key, visible_path


def load(path: Path) -> dict[str, dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} is not an object-keyed manifest")
        return payload
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {str(record.get("id", index)): record for index, record in enumerate(rows)}


def scene_key(sample_id: str, record: dict) -> str:
    for field in ("scene_id", "sequence_id", "video_id", "original_image_id"):
        if record.get(field) is not None:
            return f"{field}:{record[field]}"
    return f"sample_id:{sequence_key(sample_id)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument(
        "--additional-train",
        type=Path,
        action="append",
        default=[],
        help="Additional upstream training manifest; repeat for the initialization chain.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()
    train_manifests = [args.train, *args.additional_train]
    train_sets = [(path, load(path)) for path in train_manifests]
    test = load(args.test)
    exact_ids = sorted(
        {
            sample_id
            for _, train in train_sets
            for sample_id in set(train) & set(test)
        }
    )
    train_sequences: dict[str, list[str]] = {}
    for path, train in train_sets:
        for sample_id, record in train.items():
            train_sequences.setdefault(scene_key(sample_id, record), []).append(
                f"{path}:{sample_id}"
            )
    test_sequences: dict[str, list[str]] = {}
    for sample_id, record in test.items():
        test_sequences.setdefault(scene_key(sample_id, record), []).append(sample_id)
    shared_sequences = sorted(set(train_sequences) & set(test_sequences))
    train_visible = {
        str(visible_path(record, path)): f"{path}:{sample_id}"
        for path, train in train_sets
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
    report = {
        "train_manifests": [str(path) for path in train_manifests],
        "train_samples": sum(len(train) for _, train in train_sets),
        "test_samples": len(test),
        "exact_id_overlap_count": len(exact_ids),
        "exact_id_overlap_examples": exact_ids[:10],
        "exact_image_overlap_count": len(exact_images),
        "exact_image_overlap_examples": exact_images[:10],
        "shared_sequence_count": len(shared_sequences),
        "affected_test_samples": len(affected_test),
        "shared_sequence_examples": shared_sequences[:10],
        "affected_test_examples": affected_test[:10],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.fail_on_overlap and (exact_ids or exact_images or shared_sequences):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
