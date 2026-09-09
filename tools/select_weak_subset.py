from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tools.build_grouped_subsets import load_manifest, record_group, write_manifest


def _proportional_quotas(counts: Counter, size: int) -> dict[tuple[str, str], int]:
    total = sum(counts.values())
    exact = {key: value * size / total for key, value in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    priority = sorted(
        counts,
        key=lambda key: (exact[key] - quotas[key], counts[key], key),
        reverse=True,
    )
    for key in priority[:remaining]:
        quotas[key] += 1
    return quotas


def select_scene_diverse_stratified(
    rows: list[tuple[str, dict]], size: int, seed: int
) -> list[tuple[str, dict]]:
    if not 0 < size <= len(rows):
        raise ValueError("size must be within the manifest length")
    scene_count = len({record_group(sample_id, record, None) for sample_id, record in rows})
    if size > scene_count:
        raise ValueError(
            f"requested {size} records but only {scene_count} distinct scenes are available"
        )
    strata = Counter(
        (str(record.get("class_name", "")), str(record.get("scale_bin", "")))
        for _, record in rows
    )
    quotas = _proportional_quotas(strata, size)
    order = list(rows)
    random.Random(seed).shuffle(order)
    selected: list[tuple[str, dict]] = []
    selected_scenes: set[str] = set()
    selected_strata: Counter = Counter()
    for sample_id, record in order:
        scene = record_group(sample_id, record, None)
        stratum = (str(record.get("class_name", "")), str(record.get("scale_bin", "")))
        if scene in selected_scenes or selected_strata[stratum] >= quotas[stratum]:
            continue
        selected.append((sample_id, record))
        selected_scenes.add(scene)
        selected_strata[stratum] += 1
        if len(selected) == size:
            return selected
    # Some small strata may not have enough distinct scenes. Fill the remainder
    # without relaxing the one-record-per-scene diversity constraint.
    for sample_id, record in order:
        scene = record_group(sample_id, record, None)
        if scene in selected_scenes:
            continue
        selected.append((sample_id, record))
        selected_scenes.add(scene)
        if len(selected) == size:
            return selected
    raise RuntimeError("unable to construct the requested subset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select an unchanged, deterministic, scene-diverse weak subset"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rows, kind = load_manifest(args.manifest)
    selected = select_scene_diverse_stratified(rows, args.size, args.seed)
    write_manifest(args.output, selected, kind, args.manifest.resolve().parent)
    records = [record for _, record in selected]
    report = {
        "source": str(args.manifest),
        "output": str(args.output),
        "seed": args.seed,
        "input_samples": len(rows),
        "selected_samples": len(selected),
        "unique_scenes": len(
            {record_group(sample_id, record, None) for sample_id, record in selected}
        ),
        "labels_modified": False,
        "classes": dict(sorted(Counter(str(r.get("class_name", "")) for r in records).items())),
        "scales": dict(sorted(Counter(str(r.get("scale_bin", "")) for r in records).items())),
    }
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
