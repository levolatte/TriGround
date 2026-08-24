from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def normalize_bbox(raw_bbox: Any, width: int, height: int) -> list[float] | None:
    """Convert a Visual Genome absolute xywh box to normalized xyxy."""
    if width <= 0 or height <= 0:
        return None
    if isinstance(raw_bbox, dict):
        try:
            x = float(raw_bbox["x"])
            y = float(raw_bbox["y"])
            box_width = float(raw_bbox.get("w", raw_bbox.get("width")))
            box_height = float(raw_bbox.get("h", raw_bbox.get("height")))
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            x, y, box_width, box_height = map(float, raw_bbox)
        except (TypeError, ValueError):
            return None
    else:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + box_width))
    y2 = max(0.0, min(float(height), y + box_height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)]


def split_for_image(image_id: str, val_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < val_fraction else "train"


def _image_id(row: dict[str, Any]) -> str:
    value = row.get("image_id", row.get("id"))
    if value is None:
        image = row.get("image")
        if isinstance(image, dict) and image.get("path"):
            value = Path(image["path"]).stem
    if value is None:
        raise ValueError("Visual Genome row 缺少 image_id/id/image.path")
    text = str(value)
    return f"{int(text):06d}" if text.isdigit() else text


def _image_size(row: dict[str, Any], image_path: Path) -> tuple[int, int]:
    width = row.get("width", row.get("image_width"))
    height = row.get("height", row.get("image_height"))
    if width and height:
        return int(width), int(height)
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def _ensure_image(row: dict[str, Any], image_id: str, image_dir: Path) -> Path | None:
    candidates = list(image_dir.glob(f"{image_id}.*"))
    if candidates:
        return candidates[0]
    image = row.get("image")
    if not isinstance(image, dict) or not image.get("bytes"):
        return None
    suffix = Path(str(image.get("path", ".jpg"))).suffix or ".jpg"
    output = image_dir / f"{image_id}{suffix.lower()}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image["bytes"])
    return output


def records_from_row(
    row: dict[str, Any],
    image_dir: Path,
    manifest_root: Path,
    aux_mode: str,
) -> tuple[str, list[tuple[str, dict[str, Any]]], str | None]:
    image_id = _image_id(row)
    image_path = _ensure_image(row, image_id, image_dir)
    if image_path is None:
        return image_id, [], "missing_image"
    width, height = _image_size(row, image_path)
    regions = row.get("region_phrases", row.get("regions", [])) or []
    output: list[tuple[str, dict[str, Any]]] = []
    relative_image = Path(os.path.relpath(image_path, manifest_root)).as_posix()
    for region_index, region in enumerate(regions, start=1):
        if not isinstance(region, dict):
            continue
        query = str(region.get("phrase", region.get("query", ""))).strip()
        bbox = normalize_bbox(region.get("bbox"), width, height)
        if not query or bbox is None:
            continue
        sample_id = f"{image_id}_{region_index:03d}"
        record: dict[str, Any] = {
            "visible": relative_image,
            "query": query,
            "bbox": bbox,
        }
        if aux_mode == "rgb-copy":
            record["infrared"] = relative_image
            record["depth"] = relative_image
        output.append((sample_id, record))
    return image_id, output, None


def _iter_rows(parquet_paths: Iterable[Path]):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit("缺少 pyarrow；请先执行 `python -m pip install pyarrow`。") from error
    for parquet_path in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=16):
            yield from batch.to_pylist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Visual Genome parquet to grounding JSON")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--aux-mode",
        choices=("omit", "rgb-copy"),
        default="omit",
        help="rgb-copy 仅用于验证 joint 数据管线，不代表真实 IR/Depth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction 必须在 (0,1) 内")
    parquet_paths = sorted((args.source / "parquet").glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"没有找到 parquet: {args.source / 'parquet'}")
    image_dir = args.source / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "val": {}}
    missing_images = 0
    processed_rows = 0
    duplicate_ids = 0
    for row in _iter_rows(parquet_paths):
        processed_rows += 1
        image_id, records, error = records_from_row(
            row, image_dir, args.output_dir, args.aux_mode
        )
        if error:
            missing_images += 1
            continue
        split = split_for_image(image_id, args.val_fraction, args.seed)
        for sample_id, record in records:
            if sample_id in datasets[split]:
                duplicate_ids += 1
                continue
            datasets[split][sample_id] = record
            if args.max_queries and sum(map(len, datasets.values())) >= args.max_queries:
                break
        if args.max_queries and sum(map(len, datasets.values())) >= args.max_queries:
            break

    for split, records in datasets.items():
        output = args.output_dir / f"{split}.json"
        output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source": str(args.source),
        "parquet_files": [path.name for path in parquet_paths],
        "processed_image_rows": processed_rows,
        "train_queries": len(datasets["train"]),
        "val_queries": len(datasets["val"]),
        "missing_images": missing_images,
        "duplicate_sample_ids": duplicate_ids,
        "auxiliary_mode": args.aux_mode,
        "scientific_warning": (
            "infrared/depth are RGB copies; do not use for multimodal-gain claims"
            if args.aux_mode == "rgb-copy"
            else "IR/depth omitted; output is suitable for RGB grounding only"
        ),
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
