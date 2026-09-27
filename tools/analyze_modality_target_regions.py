"""GT-box pixel diagnostics for changed and distance-word validation queries.

Uses ground truth only for diagnosis. Pixel contrast is not evidence that a
modality was used by the model or that it can distinguish the target.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_modality_inputs import DISTANCE_WORDS, read_predictions
from report_rematch_experiment import box_iou


def masks(shape: tuple[int, int], bbox: list[float]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"GT bbox must be normalized nonempty xyxy: {bbox}")
    yy, xx = np.ogrid[:height, :width]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_width, half_height = (x2 - x1) * 0.75, (y2 - y1) * 0.75
    target = (xx >= x1 * width) & (xx < x2 * width) & (yy >= y1 * height) & (yy < y2 * height)
    outer = (xx >= (cx - half_width) * width) & (xx < (cx + half_width) * width) & (yy >= (cy - half_height) * height) & (yy < (cy + half_height) * height)
    return target, outer & ~target


def median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def spread_5_95(values: np.ndarray) -> float | None:
    return float(np.percentile(values, 95) - np.percentile(values, 5)) if values.size else None


def difference(first: float | None, second: float | None) -> float | None:
    return first - second if first is not None and second is not None else None


def score(prediction: list[float] | None, target: list[float]) -> float:
    return box_iou(prediction, target) if prediction is not None else 0.0


def region_stats(raw: np.ndarray, mapped: np.ndarray, infrared: np.ndarray, target: np.ndarray, ring: np.ndarray) -> dict:
    if raw.shape != target.shape or mapped.shape[:2] != target.shape or infrared.shape[:2] != target.shape:
        raise ValueError("RGB/IR/depth pixel grids differ; this diagnostic requires equal shapes")
    gray = mapped[:, :, 0] if mapped.ndim == 3 else mapped
    if mapped.ndim == 3 and not np.all(mapped == mapped[:, :, :1]):
        raise ValueError("mapped depth is not grayscale RGB")
    ir_gray = infrared.astype(np.float64).mean(axis=2) if infrared.ndim == 3 else infrared.astype(np.float64)
    depth_valid_target = raw[target] > 0
    depth_valid_ring = raw[ring] > 0
    raw_target = raw[target][depth_valid_target]
    raw_ring = raw[ring][depth_valid_ring]
    mapped_target = gray[target][depth_valid_target]
    mapped_ring = gray[ring][depth_valid_ring]
    raw_target_median, raw_ring_median = median(raw_target), median(raw_ring)
    mapped_target_median, mapped_ring_median = median(mapped_target), median(mapped_ring)
    ir_target_median, ir_ring_median = median(ir_gray[target]), median(ir_gray[ring])
    return {
        "target_pixels": int(target.sum()), "ring_pixels": int(ring.sum()),
        "target_depth_valid_ratio": float(depth_valid_target.mean()),
        "ring_depth_valid_ratio": float(depth_valid_ring.mean()) if ring.any() else None,
        "target_raw_depth_median_valid": raw_target_median,
        "ring_raw_depth_median_valid": raw_ring_median,
        "target_minus_ring_raw_depth_median": difference(raw_target_median, raw_ring_median),
        "target_mapped_gray_median_valid": mapped_target_median,
        "ring_mapped_gray_median_valid": mapped_ring_median,
        "target_mapped_gray_p95_minus_p05_valid": spread_5_95(mapped_target),
        "target_mapped_gray_distinct_valid": int(np.unique(mapped_target).size),
        "target_minus_ring_mapped_gray_median": difference(mapped_target_median, mapped_ring_median),
        "target_ir_brightness_median": ir_target_median,
        "ring_ir_brightness_median": ir_ring_median,
        "target_minus_ring_ir_brightness_median": difference(ir_target_median, ir_ring_median),
    }


def run(args: argparse.Namespace) -> dict:
    root = args.data_root.resolve()
    records = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    rgb = read_predictions(args.rgb_predictions)
    m2 = read_predictions(args.m2_predictions)
    if set(records) != set(rgb) or set(records) != set(m2):
        raise ValueError("manifest and prediction ID sets differ")
    rows = []
    for sample_id, record in records.items():
        target_box = record["bbox"]
        rgb_iou, m2_iou = score(rgb[sample_id], target_box), score(m2[sample_id], target_box)
        rgb_hit, m2_hit = rgb_iou >= 0.5, m2_iou >= 0.5
        outcome = "corrected" if m2_hit and not rgb_hit else "regressed" if rgb_hit and not m2_hit else "unchanged"
        keywords = sorted(set(word.lower() for word in DISTANCE_WORDS.findall(record["query"])))
        if outcome == "unchanged" and not keywords:
            continue
        paths = {key: (root / "target_v2" / record[key]).resolve() for key in ("visible", "infrared", "depth")}
        mapped_path = root / "target_v2" / "qwen3vl_native_sft" / "depth_rgb" / paths["depth"].relative_to(root / "depth")
        with Image.open(paths["depth"]) as image:
            raw = np.asarray(image)
        if raw.ndim != 2 or raw.dtype != np.uint16:
            raise ValueError(f"expected uint16 depth: {paths['depth']}")
        with Image.open(mapped_path) as image:
            mapped = np.asarray(image)
        with Image.open(paths["infrared"]) as image:
            infrared = np.asarray(image)
        target, ring = masks(raw.shape, target_box)
        if not target.any():
            raise ValueError(f"GT bbox contains no depth pixel: {sample_id}")
        rows.append({
            "id": sample_id, "outcome": outcome, "distance_keywords": ",".join(keywords),
            "rgb_iou": rgb_iou, "m2_iou": m2_iou, "query": record["query"],
            "visible_path": str(paths["visible"]), "infrared_path": str(paths["infrared"]),
            "raw_depth_path": str(paths["depth"]), "mapped_depth_path": str(mapped_path),
            **region_stats(raw, mapped, infrared, target, ring),
        })
    recommendations = {}
    for outcome in ("corrected", "regressed"):
        candidates = [row for row in rows if row["outcome"] == outcome]
        # Prefer results with both IoUs comfortably away from the ACC@0.5 boundary.
        candidates.sort(key=lambda row: min(abs(row["rgb_iou"] - 0.5), abs(row["m2_iou"] - 0.5)), reverse=True)
        selected, seen = [], set()
        for row in candidates:
            if row["raw_depth_path"] not in seen:
                selected.append({key: row[key] for key in ("id", "outcome", "rgb_iou", "m2_iou", "visible_path", "infrared_path", "raw_depth_path", "mapped_depth_path")})
                seen.add(row["raw_depth_path"])
            if len(selected) == 2:
                break
        for row in candidates:
            if len(selected) == 2:
                break
            if row["id"] not in {item["id"] for item in selected}:
                selected.append({key: row[key] for key in ("id", "outcome", "rgb_iou", "m2_iou", "visible_path", "infrared_path", "raw_depth_path", "mapped_depth_path")})
        recommendations[outcome] = selected
    result = {
        "assumptions": [
            "Target and 1.5x outer ring are defined by original GT bbox on the shared pixel grid.",
            "Depth medians use positive raw pixels only; zero is reported via valid ratio.",
            "IR brightness is the mean of stored channels, without radiometric calibration.",
            "GT-conditioned pixel differences diagnose input quality only; they do not establish model modality use, target separability, physical distance units, or geometric alignment.",
        ],
        "analyzed_queries": len(rows),
        "outcome_ids": {name: [row["id"] for row in rows if row["outcome"] == name] for name in ("corrected", "regressed")},
        "distance_keyword_ids": [row["id"] for row in rows if row["distance_keywords"]],
        "visual_inspection_recommendations": recommendations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "target_region_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "target_region_stats.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgb-predictions", type=Path, required=True)
    parser.add_argument("--m2-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("analyzed_queries", "outcome_ids", "visual_inspection_recommendations")}, indent=2))


if __name__ == "__main__":
    main()
