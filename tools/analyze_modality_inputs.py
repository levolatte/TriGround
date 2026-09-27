"""CPU-only input-quality and paired-outcome audit for native RGB vs three-image runs.

Pixel summaries describe information presented to the model; they cannot establish
whether the model used a modality or whether RGB/IR/depth are geometrically aligned.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_qwen3vl_native_sft import depth_mm_to_grayscale_rgb
from report_rematch_experiment import box_iou


DISTANCE_WORDS = re.compile(
    r"\b(?:near|nearer|nearest|close|closer|closest|far|farther|farthest|"
    r"furthest|distance|front|behind|back|foreground|background)\b", re.I
)
QUANTILES = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not values.size:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def read_predictions(path: Path) -> dict[str, list[float] | None]:
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record["id"])
        if sample_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate ID {sample_id}")
        prediction = record["prediction"]
        if prediction is not None and len(prediction) != 4:
            raise ValueError(f"{path}:{line_number}: prediction must be xyxy")
        rows[sample_id] = prediction
    return rows


def image_stats(data_root: Path, manifest_base: Path, record: dict) -> dict:
    paths = {key: (manifest_base / record[key]).resolve() for key in ("visible", "infrared", "depth")}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
    with Image.open(paths["depth"]) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"expected 16-bit depth: {paths['depth']} is {raw.dtype} {raw.shape}")
    depth_relative = paths["depth"].relative_to(data_root / "depth")
    mapped_path = data_root / "target_v2" / "qwen3vl_native_sft" / "depth_rgb" / depth_relative
    with Image.open(mapped_path) as image:
        mapped = np.asarray(image)
    expected = depth_mm_to_grayscale_rgb(raw)
    if mapped.shape != expected.shape:
        raise ValueError(f"mapped depth shape mismatch: {mapped_path}")
    gray = mapped[:, :, 0]
    valid = raw > 0
    with Image.open(paths["infrared"]) as image:
        ir = np.asarray(image)
    ir_gray = ir.astype(np.float64).mean(axis=2) if ir.ndim == 3 else ir.astype(np.float64)
    return {
        "visible_path": str(paths["visible"]),
        "infrared_path": str(paths["infrared"]),
        "depth_path": str(paths["depth"]),
        "mapped_depth_path": str(mapped_path),
        "height": int(raw.shape[0]),
        "width": int(raw.shape[1]),
        "depth_valid_ratio": float(valid.mean()),
        "depth_over_19999_ratio": float((raw > 19_999).mean()),
        "depth_valid_quantiles": quantiles(raw[valid]),
        "depth_distinct_valid": int(np.unique(raw[valid]).size),
        "mapped_nonzero_ratio": float((gray > 0).mean()),
        "mapped_distinct_gray": int(np.unique(gray).size),
        "mapped_gray_quantiles": quantiles(gray),
        "mapped_valid_gray_quantiles": quantiles(gray[valid]),
        "mapped_rgb_channels_equal": bool(np.all(mapped == mapped[:, :, :1])),
        "mapped_expected_equal": bool(np.array_equal(mapped, expected)),
        "ir_shape": list(ir.shape),
        "ir_dtype": str(ir.dtype),
        "ir_distinct_gray": int(np.unique(ir_gray).size),
        "ir_gray_quantiles": quantiles(ir_gray),
    }


def group_summary(rows: list[dict]) -> dict:
    return {
        "queries": len(rows),
        "images": len({row["depth_path"] for row in rows}),
        "corrected": sum(row["outcome"] == "corrected" for row in rows),
        "regressed": sum(row["outcome"] == "regressed" for row in rows),
        "rgb_hits": sum(row["rgb_hit"] for row in rows),
        "m2_hits": sum(row["m2_hit"] for row in rows),
        "mean_depth_valid_ratio_by_image": float(np.mean([r["depth_valid_ratio"] for r in {x["depth_path"]: x for x in rows}.values()])),
    }


def run(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    rgb = read_predictions(args.rgb_predictions)
    m2 = read_predictions(args.m2_predictions)
    ids = set(manifest)
    if set(rgb) != ids or set(m2) != ids:
        raise ValueError(f"prediction ID mismatch: manifest={len(ids)} rgb={len(rgb)} m2={len(m2)}; missing/extras must be resolved")
    manifest_base = data_root / "target_v2"
    groups = defaultdict(list)
    for sample_id, record in manifest.items():
        groups[tuple(record[key] for key in ("visible", "infrared", "depth"))].append(sample_id)
    image_rows = []
    by_group = {}
    for group_key, sample_ids in groups.items():
        stats = image_stats(data_root, manifest_base, manifest[sample_ids[0]])
        stats["sample_ids"] = sample_ids
        image_rows.append(stats)
        by_group[group_key] = stats
    rows = []
    for sample_id, record in manifest.items():
        group = tuple(record[key] for key in ("visible", "infrared", "depth"))
        stats = by_group[group]
        target = record["bbox"]
        rgb_iou = box_iou(rgb[sample_id], target) if rgb[sample_id] is not None else 0.0
        m2_iou = box_iou(m2[sample_id], target) if m2[sample_id] is not None else 0.0
        rgb_hit, m2_hit = rgb_iou >= 0.5, m2_iou >= 0.5
        outcome = "corrected" if m2_hit and not rgb_hit else "regressed" if rgb_hit and not m2_hit else "unchanged"
        words = sorted(set(word.lower() for word in DISTANCE_WORDS.findall(record["query"])))
        rows.append({
            "id": sample_id, "query": record["query"], "distance_keywords": ",".join(words),
            "distance_group": "distance" if words else "other", "outcome": outcome,
            "rgb_iou": rgb_iou, "m2_iou": m2_iou, "rgb_hit": rgb_hit, "m2_hit": m2_hit,
            "depth_path": stats["depth_path"], "visible_path": stats["visible_path"],
            "infrared_path": stats["infrared_path"], "mapped_depth_path": stats["mapped_depth_path"],
            "depth_valid_ratio": stats["depth_valid_ratio"],
            "mapped_distinct_gray": stats["mapped_distinct_gray"],
            "mapped_gray_p50": stats["mapped_gray_quantiles"]["p50"],
            "ir_gray_p50": stats["ir_gray_quantiles"]["p50"],
        })
    outcomes = {name: [row for row in rows if row["outcome"] == name] for name in ("corrected", "regressed", "unchanged")}
    distance = {name: group_summary([row for row in rows if row["distance_group"] == name]) for name in ("distance", "other") if any(row["distance_group"] == name for row in rows)}
    chosen = []
    for name in ("corrected", "regressed", "unchanged"):
        seen_images = set()
        for row in outcomes[name]:
            if row["depth_path"] not in seen_images:
                chosen.append({key: row[key] for key in ("id", "query", "outcome", "visible_path", "infrared_path", "depth_path", "mapped_depth_path")})
                seen_images.add(row["depth_path"])
            if sum(item["outcome"] == name for item in chosen) >= 3:
                break
    report = {
        "assumptions": ["Each unique visible/infrared/depth triplet is counted once for pixel statistics.",
                        "The supplied manifest is original bbox ground truth; predictions are rescored with IoU >= 0.5.",
                        "Distance groups are English query keyword matches, not verified semantic categories.",
                        "Pixel statistics and mapping equality cannot prove modality use or visual alignment."],
        "manifest": str(args.manifest.resolve()), "data_root": str(data_root),
        "queries": len(rows), "image_groups": len(image_rows),
        "outcomes": {name: group_summary(group) for name, group in outcomes.items() if group},
        "outcome_ids": {name: [row["id"] for row in group] for name, group in outcomes.items()},
        "distance_groups": distance,
        "distance_keyword_counts": {word: sum(word in row["distance_keywords"].split(",") for row in rows)
                                    for word in sorted({word for row in rows for word in row["distance_keywords"].split(",") if word})},
        "mapped_expected_equal_count": sum(row["mapped_expected_equal"] for row in image_rows),
        "representative_raw_paths": chosen,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "modality_diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "image_stats.json").write_text(json.dumps(image_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "query_outcomes.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 模态输入与成对结果诊断", "", f"样本 {len(rows)} 条，独立图像组 {len(image_rows)} 组。", "",
             "| 分组 | Query 数 | RGB 命中 | M2 命中 | M2 修正 | M2 退化 |", "|---|---:|---:|---:|---:|---:|"]
    for name, group in (("全部", rows), ("距离关键词", [r for r in rows if r["distance_group"] == "distance"]), ("其他", [r for r in rows if r["distance_group"] == "other"])):
        if group:
            s = group_summary(group)
            lines.append(f"| {name} | {s['queries']} | {s['rgb_hits']} | {s['m2_hits']} | {s['corrected']} | {s['regressed']} |")
    lines += ["", f"深度映射与预处理函数完全一致：{report['mapped_expected_equal_count']}/{len(image_rows)} 组。", "",
              "## M2 修正与退化样本", "",
              "| ID | 变化 | RGB IoU | M2 IoU | 距离关键词 |", "|---|---|---:|---:|---|"]
    for row in rows:
        if row["outcome"] != "unchanged":
            lines.append(f"| `{row['id']}` | {row['outcome']} | {row['rgb_iou']:.3f} | {row['m2_iou']:.3f} | {row['distance_keywords']} |")
    lines += ["",
              "## 判读边界", "", *[f"- {item}" for item in report["assumptions"]], "", "## 建议人工检查的原始路径", ""]
    for item in chosen:
        lines.append(f"- {item['outcome']} `{item['id']}`：RGB `{item['visible_path']}`；IR `{item['infrared_path']}`；原始深度 `{item['depth_path']}`；映射深度 `{item['mapped_depth_path']}`")
    (args.output_dir / "modality_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="Original validation bbox ground truth JSON")
    parser.add_argument("--rgb-predictions", type=Path, required=True)
    parser.add_argument("--m2-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report[key] for key in ("queries", "image_groups", "outcomes", "distance_groups", "mapped_expected_equal_count")}, indent=2))


if __name__ == "__main__":
    main()
