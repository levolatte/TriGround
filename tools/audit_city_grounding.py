from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"D:\AIC\city_detection_prepared\train")
CLASS_NAMES = ("person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball", "light", "garbage_can", "uav", "tricycle")


def group_key(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2])
    if stem.isdigit():
        return f"plain_{len(stem):02d}_{int(stem) // 50:06d}"
    return stem


def percentile(values, p):
    return float(np.percentile(np.asarray(values), p))


def main():
    manifests = {
        split: json.loads((ROOT / f"stage4_{split}_v2.json").read_text(encoding="utf-8"))
        for split in ("train", "val")
    }
    report = {"splits": {}}
    all_records = {}
    for split, records in manifests.items():
        all_records.update(records)
        scenes = defaultdict(list)
        query_words, areas, widths, heights = [], [], [], []
        for sample_id, record in records.items():
            scenes[record["visible"]].append((sample_id, record))
            query_words.append(len(record["query"].split()))
            x1, y1, x2, y2 = record["bbox"]
            widths.append(x2 - x1); heights.append(y2 - y1); areas.append((x2 - x1) * (y2 - y1))
        duplicate_query_scenes = 0
        duplicate_bbox_scenes = 0
        for values in scenes.values():
            queries = [record["query"].casefold() for _, record in values]
            boxes = [tuple(record["bbox"]) for _, record in values]
            duplicate_query_scenes += len(queries) != len(set(queries))
            duplicate_bbox_scenes += len(boxes) != len(set(boxes))
        stems = {Path(path).stem for path in scenes}
        groups = {group_key(stem) for stem in stems}
        report["splits"][split] = {
            "records": len(records), "scenes": len(scenes), "groups": len(groups),
            "queries_per_scene": {"mean": len(records) / len(scenes), "max": max(map(len, scenes.values()))},
            "query_words": {"mean": statistics.mean(query_words), "p50": percentile(query_words, 50), "p95": percentile(query_words, 95)},
            "bbox_area": {"p01": percentile(areas, 1), "p10": percentile(areas, 10), "p50": percentile(areas, 50), "p90": percentile(areas, 90)},
            "boxes_under_0.1pct": sum(area < .001 for area in areas),
            "boxes_under_1pct": sum(area < .01 for area in areas),
            "duplicate_query_scenes": duplicate_query_scenes,
            "duplicate_bbox_scenes": duplicate_bbox_scenes,
            "comma_queries": sum("," in record["query"] for record in records.values()),
        }

    train_stems = {Path(record["visible"]).stem for record in manifests["train"].values()}
    val_stems = {Path(record["visible"]).stem for record in manifests["val"].values()}
    report["leakage"] = {
        "stem_overlap": len(train_stems & val_stems),
        "group_overlap": len({group_key(x) for x in train_stems} & {group_key(x) for x in val_stems}),
    }
    split_of = {stem: "train" for stem in train_stems} | {stem: "val" for stem in val_stems}
    numeric = sorted((int(stem), stem) for stem in split_of if stem.isdigit())
    cross_boundary_neighbors = []
    for (left_number, left), (right_number, right) in zip(numeric, numeric[1:]):
        if right_number - left_number == 1 and split_of[left] != split_of[right]:
            cross_boundary_neighbors.append((left, right))
    report["leakage"]["adjacent_numeric_cross_split"] = len(cross_boundary_neighbors)
    report["leakage"]["adjacent_examples"] = cross_boundary_neighbors[:10]

    source_counts, included_counts, excluded_counts = Counter(), Counter(), Counter()
    included_stems = train_stems | val_stems
    excluded_stems = set()
    path_errors = []
    dimension_mismatch = []
    for label in (ROOT / "labels_clean").glob("*.txt"):
        stem = label.stem
        target = included_counts if stem in included_stems else excluded_counts
        if stem not in included_stems:
            excluded_stems.add(stem)
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                class_id = int(line.split()[0]); source_counts[class_id] += 1; target[class_id] += 1
    for record in all_records.values():
        paths = [ROOT / record[key] for key in ("visible", "infrared", "depth")]
        if not all(path.is_file() for path in paths):
            path_errors.append([str(path) for path in paths])
            continue
        sizes = []
        for path in paths:
            with Image.open(path) as image:
                sizes.append(image.size)
        if len(set(sizes)) != 1:
            dimension_mismatch.append((record["visible"], sizes))
    report["classes"] = {
        CLASS_NAMES[i]: {"source": source_counts[i], "included": included_counts[i], "excluded": excluded_counts[i], "retained_fraction": included_counts[i] / source_counts[i] if source_counts[i] else 0}
        for i in range(len(CLASS_NAMES))
    }
    report["paths"] = {"missing": len(path_errors), "dimension_mismatch_records": len(dimension_mismatch)}
    report["excluded_scenes"] = len(excluded_stems)

    rng = random.Random(2026)
    scene_records = {}
    for record in all_records.values():
        scene_records.setdefault(record["visible"], record)
    sample = rng.sample(list(scene_records.values()), min(300, len(scene_records)))
    depth_stats = []
    correlations = {"rgb_ir": [], "rgb_depth": []}
    for record in sample:
        arrays = []
        for key in ("visible", "infrared", "depth"):
            with Image.open(ROOT / record[key]) as image:
                array = np.asarray(image.convert("L").resize((160, 90))).astype(np.float32)
                arrays.append(array)
        rgb, ir, depth_gray = arrays
        def gradient(a):
            return np.hypot(np.diff(a, axis=1, append=a[:, -1:]), np.diff(a, axis=0, append=a[-1:, :])).ravel()
        for name, other in (("rgb_ir", ir), ("rgb_depth", depth_gray)):
            corr = float(np.corrcoef(gradient(rgb), gradient(other))[0, 1])
            if math.isfinite(corr): correlations[name].append(corr)
        with Image.open(ROOT / record["depth"]) as depth_image:
            raw = np.asarray(depth_image)
            depth_stats.append({"mode": depth_image.mode, "dtype": str(raw.dtype), "min": int(raw.min()), "max": int(raw.max()), "zero_fraction": float((raw == 0).mean()), "unique": int(len(np.unique(raw)))})
    report["alignment_proxy"] = {
        key: {"mean": statistics.mean(values), "p10": percentile(values, 10), "p50": percentile(values, 50)}
        for key, values in correlations.items()
    }
    report["depth_sample_300"] = {
        "modes": Counter(item["mode"] for item in depth_stats),
        "dtypes": Counter(item["dtype"] for item in depth_stats),
        "max_p50": percentile([item["max"] for item in depth_stats], 50),
        "max_p95": percentile([item["max"] for item in depth_stats], 95),
        "zero_fraction_mean": statistics.mean(item["zero_fraction"] for item in depth_stats),
        "unique_p50": percentile([item["unique"] for item in depth_stats], 50),
    }
    # dHash is only a screening proxy; very small Hamming distance indicates
    # exact or near-duplicate visual frames across the nominal split.
    hashes = {"train": [], "val": []}
    for split, records in manifests.items():
        seen = set()
        for record in records.values():
            path = record["visible"]
            if path in seen:
                continue
            seen.add(path)
            with Image.open(ROOT / path) as image:
                gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            bits = (gray[:, 1:] > gray[:, :-1]).ravel()
            value = sum(int(bit) << index for index, bit in enumerate(bits))
            hashes[split].append((path, value))
    near = []
    for train_path, train_hash in hashes["train"]:
        for val_path, val_hash in hashes["val"]:
            distance = (train_hash ^ val_hash).bit_count()
            if distance <= 3:
                near.append((distance, train_path, val_path))
    near.sort()
    report["leakage"]["dhash_distance_le_3_pairs"] = len(near)
    report["leakage"]["dhash_examples"] = near[:20]
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
