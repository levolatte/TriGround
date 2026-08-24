from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image


SOURCES = ("flir", "mfad", "m3fd")


def _relative(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def convert_split(
    dataset_root: Path,
    split: str,
    output: Path,
    verify_images: bool = False,
) -> dict[str, int]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    source_counts: dict[str, int] = {}
    verified: set[tuple[Path, Path]] = set()
    for source in SOURCES:
        annotation = (
            dataset_root
            / f"rgbtvg_{source}"
            / f"rgbtvg_{source}_{split}.pth"
        )
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        rows = torch.load(annotation, map_location="cpu", weights_only=False)
        if not isinstance(rows, list):
            raise ValueError(f"{annotation} must contain a list")
        source_counts[source] = len(rows)
        for index, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) < 4:
                raise ValueError(f"invalid annotation row {source}:{index}")
            filename, size, xywh, query = row[:4]
            if not isinstance(size, dict) or not {"width", "height"} <= set(size):
                raise ValueError(f"invalid image size metadata {source}:{index}")
            width, height = int(size["width"]), int(size["height"])
            x, y, box_width, box_height = (float(value) for value in xywh)
            if (
                width <= 0
                or height <= 0
                or box_width <= 0
                or box_height <= 0
                or x < 0
                or y < 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError(f"invalid xywh bbox {source}:{index}: {xywh}")
            rgb = dataset_root / "image_data" / source / "rgb" / str(filename)
            infrared = dataset_root / "image_data" / source / "ir" / str(filename)
            if not rgb.is_file() or not infrared.is_file():
                raise FileNotFoundError(f"missing RGB/IR pair for {source}:{filename}")
            pair = (rgb, infrared)
            if verify_images and pair not in verified:
                with Image.open(rgb) as rgb_image, Image.open(infrared) as ir_image:
                    if rgb_image.size != (width, height):
                        raise ValueError(f"RGB metadata mismatch: {rgb}")
                    if ir_image.size != rgb_image.size:
                        raise ValueError(f"unaligned RGB/IR sizes: {rgb} vs {infrared}")
                verified.add(pair)
            stem = Path(str(filename)).stem
            records.append(
                {
                    "id": f"rgbt_{source}_{split}_{index:06d}",
                    "source": f"rgbt_groundbench_{source}",
                    "stage": "ir",
                    "rgb": _relative(rgb, output),
                    "aux": _relative(infrared, output),
                    "aux_type": "ir",
                    "query": str(query).strip(),
                    "bbox": [
                        round(x / width, 8),
                        round(y / height, 8),
                        round((x + box_width) / width, 8),
                        round((y + box_height) / height, 8),
                    ],
                    "scene_id": f"{source}:{stem}",
                    "original_image_id": f"{source}:{stem}",
                }
            )
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return {"total": len(records), **source_counts, "verified_pairs": len(verified)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official RGBT-GroundBench .pth annotations to staged JSONL"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    report = {}
    for split in ("train", "val"):
        report[split] = convert_split(
            args.dataset_root.resolve(),
            split,
            (args.output_dir / f"{split}.jsonl").resolve(),
            verify_images=args.verify_images,
        )
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
