from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tools.review_grounding import query_type
except ModuleNotFoundError:  # Support direct execution as tools/script.py.
    from review_grounding import query_type


def load_manifest(path: Path) -> list[tuple[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object keyed by sample id")
    return list(data.items())


def resolved_image(manifest: Path, relative: str) -> Path:
    return (manifest.resolve().parent / relative).resolve()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def image_metadata(visible: Path, infrared: Path) -> dict[str, object]:
    with Image.open(visible) as image:
        rgb = np.asarray(image.convert("L").resize((96, 64)), dtype=np.float32)
    with Image.open(infrared) as image:
        ir = np.asarray(image.convert("L").resize((96, 64)), dtype=np.float32)
    brightness = float(rgb.mean())
    contrast = float(rgb.std())
    ir_contrast = float(ir.std())
    return {
        "brightness": round(brightness, 2),
        "rgb_contrast": round(contrast, 2),
        "ir_contrast": round(ir_contrast, 2),
        "light_bin": "dark" if brightness < 70 else "normal" if brightness < 150 else "bright",
        "rgb_contrast_bin": "low" if contrast < 35 else "normal" if contrast < 65 else "high",
        "ir_contrast_bin": "low" if ir_contrast < 35 else "normal" if ir_contrast < 65 else "high",
    }


def spatial_bin(box: list[float]) -> tuple[str, str]:
    x = (float(box[0]) + float(box[2])) / 2
    y = (float(box[1]) + float(box[3])) / 2
    horizontal = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
    vertical = "upper" if y < 1 / 3 else "lower" if y > 2 / 3 else "middle"
    return horizontal, vertical


def source_family(visible: Path) -> str:
    stem = visible.stem
    return "shuming" if stem.startswith("shuming_") else "numeric"


def feature_values(record: dict) -> list[tuple[str, str]]:
    metadata = record["candidate_metadata"]
    return [
        ("class", str(record.get("class_name", "unknown"))),
        ("scale", str(record.get("scale_bin", "unknown"))),
        ("query", str(metadata["query_type"])),
        ("light", str(metadata["light_bin"])),
        ("rgb_contrast", str(metadata["rgb_contrast_bin"])),
        ("ir_contrast", str(metadata["ir_contrast_bin"])),
        ("horizontal", str(metadata["horizontal_bin"])),
        ("vertical", str(metadata["vertical_bin"])),
        ("source", str(metadata["source_family"])),
    ]


FEATURE_WEIGHTS = {
    "class": 2.5,
    "scale": 2.0,
    "query": 2.0,
    "light": 2.0,
    "rgb_contrast": 1.0,
    "ir_contrast": 1.0,
    "horizontal": 1.0,
    "vertical": 1.0,
    "source": 1.5,
}


def greedy_coverage_select(
    candidates: list[tuple[str, dict]], size: int, seed: int
) -> list[tuple[str, dict]]:
    if size > len(candidates):
        raise ValueError(f"requested {size} candidates but only {len(candidates)} are eligible")
    rng = random.Random(seed)
    tie_break = {sample_id: rng.random() for sample_id, _ in candidates}
    remaining = dict(candidates)
    counts: Counter[tuple[str, str]] = Counter()
    class_counts: Counter[str] = Counter()
    max_per_class = math.ceil(size * 0.25)
    selected_scenes: set[str] = set()
    selected: list[tuple[str, dict]] = []
    while len(selected) < size:
        def score(item: tuple[str, dict]) -> tuple[float, float]:
            sample_id, record = item
            coverage = sum(
                FEATURE_WEIGHTS[name] / (1 + counts[(name, value)])
                for name, value in feature_values(record)
            )
            return coverage, tie_break[sample_id]

        allowed = [
            item for item in remaining.items()
            if str(item[1]["candidate_metadata"]["scene_key"]) not in selected_scenes
            if class_counts[str(item[1].get("class_name", "unknown"))] < max_per_class
        ]
        if not allowed:
            allowed = [
                item for item in remaining.items()
                if str(item[1]["candidate_metadata"]["scene_key"]) not in selected_scenes
            ]
        if not allowed:
            raise RuntimeError("not enough distinct eligible scenes")
        best_id, best_record = max(allowed, key=score)
        selected.append((best_id, best_record))
        counts.update(feature_values(best_record))
        class_counts[str(best_record.get("class_name", "unknown"))] += 1
        selected_scene = str(best_record["candidate_metadata"]["scene_key"])
        selected_scenes.add(selected_scene)
        for sample_id in [
            sample_id for sample_id, record in remaining.items()
            if str(record["candidate_metadata"]["scene_key"]) == selected_scene
        ]:
            del remaining[sample_id]
    return selected


def build_candidates(
    source_manifest: Path,
    exclusion_manifests: list[Path],
    size: int,
    seed: int,
    hash_distance: int,
) -> tuple[dict[str, dict], dict]:
    exclusions: set[Path] = set()
    exclusion_hashes: set[int] = set()
    for manifest in exclusion_manifests:
        for _, record in load_manifest(manifest):
            visible = resolved_image(manifest, record["visible"])
            exclusions.add(visible)
    for visible in sorted(exclusions):
        if visible.is_file():
            exclusion_hashes.add(dhash(visible))

    grouped: dict[Path, list[tuple[str, dict]]] = defaultdict(list)
    for sample_id, record in load_manifest(source_manifest):
        visible = resolved_image(source_manifest, record["visible"])
        grouped[visible].append((sample_id, record))

    eligible: list[tuple[str, dict]] = []
    exact_excluded = near_excluded = 0
    for visible, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        if visible in exclusions:
            exact_excluded += 1
            continue
        value = dhash(visible)
        if any((value ^ other).bit_count() <= hash_distance for other in exclusion_hashes):
            near_excluded += 1
            continue
        image_info = image_metadata(
            visible, resolved_image(source_manifest, rows[0][1]["infrared"])
        )
        for sample_id, original in rows:
            record = dict(original)
            horizontal, vertical = spatial_bin(record["bbox"])
            record["candidate_metadata"] = {
                **image_info,
                "query_type": query_type(str(record.get("query", ""))),
                "horizontal_bin": horizontal,
                "vertical_bin": vertical,
                "source_family": source_family(visible),
                "scene_key": str(visible),
                "source_manifest": str(source_manifest.resolve()),
                "selection_seed": seed,
            }
            eligible.append((sample_id, record))

    # Relative bins are more useful than fixed camera thresholds for a dataset
    # whose exposure range is narrow. Numeric values remain in the metadata.
    for numeric_key, bin_key in (
        ("brightness", "light_bin"),
        ("rgb_contrast", "rgb_contrast_bin"),
        ("ir_contrast", "ir_contrast_bin"),
    ):
        values = np.asarray(
            [float(record["candidate_metadata"][numeric_key]) for _, record in eligible]
        )
        lower, upper = np.quantile(values, [1 / 3, 2 / 3])
        for _, record in eligible:
            value = float(record["candidate_metadata"][numeric_key])
            record["candidate_metadata"][bin_key] = (
                "low" if value <= lower else "high" if value >= upper else "mid"
            )

    selected = greedy_coverage_select(eligible, size, seed)
    output = dict(selected)
    report = {
        "source_manifest": str(source_manifest.resolve()),
        "exclusion_manifests": [str(path.resolve()) for path in exclusion_manifests],
        "seed": seed,
        "hash_distance": hash_distance,
        "source_scenes": len(grouped),
        "exact_excluded_scenes": exact_excluded,
        "near_duplicate_excluded_scenes": near_excluded,
        "eligible_scenes": len({r["candidate_metadata"]["scene_key"] for _, r in eligible}),
        "selected_samples": len(output),
        "unique_visible_images": len({record["visible"] for record in output.values()}),
        "classes": dict(sorted(Counter(str(r.get("class_name")) for r in output.values()).items())),
        "scales": dict(sorted(Counter(str(r.get("scale_bin")) for r in output.values()).items())),
        "light_bins": dict(sorted(Counter(str(r["candidate_metadata"]["light_bin"]) for r in output.values()).items())),
        "query_types": dict(sorted(Counter(str(r["candidate_metadata"]["query_type"]) for r in output.values()).items())),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an unseen scene-coverage review queue")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hash-distance", type=int, default=3)
    args = parser.parse_args()
    output, report = build_candidates(
        args.source.resolve(), [path.resolve() for path in args.exclude],
        args.size, args.seed, args.hash_distance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
