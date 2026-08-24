from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> tuple[list[tuple[str, dict]], str]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("JSON manifest must map sample IDs to records")
        return [(str(sample_id), record) for sample_id, record in payload.items()], "json"
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [(str(record.get("id", index)), record) for index, record in enumerate(rows)], "jsonl"


def record_group(sample_id: str, record: dict, group_key: str | None) -> str:
    if group_key:
        if record.get(group_key) is None:
            raise KeyError(f"sample {sample_id!r} has no {group_key!r} group field")
        return str(record[group_key])
    for key in ("scene_id", "sequence_id", "video_id", "original_image_id"):
        if record.get(key) is not None:
            return str(record[key])
    return sample_id


def grouped_split(
    rows: list[tuple[str, dict]],
    fractions: tuple[float, ...],
    validation_fraction: float,
    seed: int,
    group_key: str | None,
) -> tuple[dict[float, list[tuple[str, dict]]], list[tuple[str, dict]]]:
    if not rows:
        raise ValueError("manifest is empty")
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("training fractions must be in (0, 1]")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id, record in rows:
        groups[record_group(sample_id, record, group_key)].append(sample_id)
    order = sorted(groups)
    random.Random(seed).shuffle(order)

    validation_groups: set[str] = set()
    validation_target = math.ceil(len(rows) * validation_fraction)
    validation_count = 0
    while order and validation_count < validation_target:
        group = order.pop(0)
        validation_groups.add(group)
        validation_count += len(groups[group])

    training_groups = order
    training_total = sum(len(groups[group]) for group in training_groups)
    if not training_groups:
        raise ValueError("validation split consumed every group")
    selected_groups: set[str] = set()
    cursor = 0
    subsets: dict[float, list[tuple[str, dict]]] = {}
    for fraction in sorted(set(fractions)):
        target = math.ceil(training_total * fraction)
        selected_count = sum(len(groups[group]) for group in selected_groups)
        while cursor < len(training_groups) and selected_count < target:
            group = training_groups[cursor]
            cursor += 1
            selected_groups.add(group)
            selected_count += len(groups[group])
        subsets[fraction] = [
            row
            for row in rows
            if record_group(row[0], row[1], group_key) in selected_groups
        ]
    validation = [
        row
        for row in rows
        if record_group(row[0], row[1], group_key) in validation_groups
    ]
    return subsets, validation


def write_manifest(
    path: Path, rows: list[tuple[str, dict]], kind: str, source_root: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rebased = []
    for sample_id, source in rows:
        record = dict(source)
        for key in ("rgb", "visible", "ir", "infrared", "thermal", "depth", "aux"):
            value = record.get(key)
            if value and not Path(value).is_absolute():
                record[key] = Path(
                    os.path.relpath(source_root / value, path.parent)
                ).as_posix()
        rebased.append((sample_id, record))
    if kind == "json":
        payload = {sample_id: record for sample_id, record in rebased}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for _, record in rebased
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic, group-safe nested training subsets and optional validation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-key")
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.25, 0.5, 1.0))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows, kind = load_manifest(args.manifest)
    subsets, validation = grouped_split(
        rows, tuple(args.fractions), args.validation_fraction, args.seed, args.group_key
    )
    suffix = ".json" if kind == "json" else ".jsonl"
    report = {"input": len(rows), "validation": len(validation), "train": {}}
    for fraction, subset in subsets.items():
        label = f"{fraction * 100:g}"
        write_manifest(
            args.output_dir / f"train_{label}{suffix}",
            subset,
            kind,
            args.manifest.resolve().parent,
        )
        report["train"][label] = len(subset)
    if args.validation_fraction:
        write_manifest(
            args.output_dir / f"val{suffix}",
            validation,
            kind,
            args.manifest.resolve().parent,
        )
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
