from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ANNOTATION_NAMES = {
    "train": ("roborefit_train.json", "refindoor_train_with_scene.json"),
    "testA": ("roborefit_testA.json", "refindoor_testA_with_scene.json"),
    "testB": ("roborefit_testB.json", "refindoor_testB_with_scene.json"),
}


def _relative(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path, manifest.parent)).as_posix()


def _find_annotation(dataset_root: Path, split: str) -> Path:
    for name in ANNOTATION_NAMES[split]:
        matches = sorted(dataset_root.rglob(name))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple annotations named {name}: {matches}")
    raise FileNotFoundError(
        f"could not find a RoboRefIt {split} annotation below {dataset_root}"
    )


def _resolve_media(dataset_root: Path, annotation: Path, raw_path: str) -> Path:
    normalized = str(raw_path).replace("\\", "/").lstrip("./")
    relative = Path(normalized)
    candidates = [dataset_root / relative]
    candidates.extend(parent / relative for parent in annotation.parents)
    candidates.append(annotation.parent / relative.name)
    # Some releases include RoboRefIt/data in JSON paths while the caller already
    # points at that directory (or at final_dataset).
    parts = relative.parts
    for marker in ("data", "final_dataset"):
        if marker in parts:
            suffix = Path(*parts[parts.index(marker) :])
            candidates.extend(parent / suffix for parent in annotation.parents)
            if marker == "final_dataset":
                tail = Path(*parts[parts.index(marker) + 1 :])
                candidates.extend(parent / tail for parent in annotation.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen and candidate.is_file():
            return candidate
        seen.add(candidate)
    raise FileNotFoundError(f"cannot resolve {raw_path!r} from {annotation}")


def convert_split(
    dataset_root: Path,
    split: str,
    output: Path,
    audit_depth_images: int = 100,
    clip_bboxes: bool = False,
) -> dict:
    annotation = _find_annotation(dataset_root, split)
    payload = json.loads(annotation.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{annotation} must contain a non-empty JSON list")

    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    image_sizes: dict[Path, tuple[int, int]] = {}
    depth_audit: dict[Path, tuple[str, str, int, int]] = {}
    clipped_boxes = 0
    max_bbox_clip_pixels = 0.0
    bbox_clip_examples = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"invalid annotation row {split}:{index}")
        rgb = _resolve_media(dataset_root, annotation, row["rgb_path"])
        depth = _resolve_media(dataset_root, annotation, row["depth_path"])
        if rgb not in image_sizes:
            with Image.open(rgb) as image:
                image_sizes[rgb] = image.size
        width, height = image_sizes[rgb]
        original_box = tuple(float(value) for value in row["bbox"])
        if len(original_box) != 4 or not all(math.isfinite(value) for value in original_box):
            raise ValueError(f"invalid bbox values {split}:{index}: {row['bbox']}")
        x1, y1, x2, y2 = original_box
        if not (
            width > 0
            and height > 0
            and 0 <= x1 < x2 <= width
            and 0 <= y1 < y2 <= height
        ):
            if not clip_bboxes:
                raise ValueError(
                    f"invalid pixel xyxy bbox {split}:{index}: {row['bbox']} for {width}x{height}"
                )
            clipped_box = (
                min(max(x1, 0.0), float(width)),
                min(max(y1, 0.0), float(height)),
                min(max(x2, 0.0), float(width)),
                min(max(y2, 0.0), float(height)),
            )
            x1, y1, x2, y2 = clipped_box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(
                    f"bbox remains invalid after clipping {split}:{index}: "
                    f"{row['bbox']} -> {list(clipped_box)} for {width}x{height}"
                )
            correction = max(
                abs(before - after)
                for before, after in zip(original_box, clipped_box, strict=True)
            )
            clipped_boxes += 1
            max_bbox_clip_pixels = max(max_bbox_clip_pixels, correction)
            if len(bbox_clip_examples) < 20:
                bbox_clip_examples.append(
                    {
                        "index": index,
                        "original": list(original_box),
                        "clipped": list(clipped_box),
                        "image_size": [width, height],
                    }
                )
        query = str(row.get("text", "")).strip()
        if not query:
            raise ValueError(f"empty referring expression {split}:{index}")
        image_key = rgb.relative_to(dataset_root).as_posix()
        record = {
            "id": f"roborefit_{split}_{index:06d}",
            "source": "roborefit",
            "stage": "depth",
            "rgb": _relative(rgb, output),
            "depth": _relative(depth, output),
            "query": query,
            "bbox": [
                round(x1 / width, 8),
                round(y1 / height, 8),
                round(x2 / width, 8),
                round(y2 / height, 8),
            ],
            "scene_id": f"roborefit:{split}:{image_key}",
            "original_image_id": image_key,
        }
        if row.get("scene") is not None:
            record["scene"] = str(row["scene"])
        if row.get("class") is not None:
            record["class_name"] = str(row["class"])
        records.append(record)

        if len(depth_audit) < audit_depth_images and depth not in depth_audit:
            with Image.open(depth) as image:
                array = np.asarray(image)
                depth_audit[depth] = (
                    image.mode,
                    str(array.dtype),
                    int(array.min()),
                    int(array.max()),
                )

    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "annotation": str(annotation),
        "samples": len(records),
        "unique_rgb": len(image_sizes),
        "audited_depth_images": len(depth_audit),
        "depth_modes": dict(Counter(value[0] for value in depth_audit.values())),
        "depth_dtypes": dict(Counter(value[1] for value in depth_audit.values())),
        "depth_min": min((value[2] for value in depth_audit.values()), default=None),
        "depth_max": max((value[3] for value in depth_audit.values()), default=None),
        "bbox_clipped": clipped_boxes,
        "max_bbox_clip_pixels": max_bbox_clip_pixels,
        "bbox_clip_examples": bbox_clip_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official RoboRefIt RGB-D annotations to staged JSONL"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-depth-images", type=int, default=100)
    parser.add_argument(
        "--clip-bboxes",
        action="store_true",
        help="Clip edge-overflowing boxes to image bounds and report every repair",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(ANNOTATION_NAMES),
        default=tuple(ANNOTATION_NAMES),
        help="Dataset splits to convert (default: train testA testB)",
    )
    args = parser.parse_args()
    if args.audit_depth_images < 0:
        parser.error("--audit-depth-images must be non-negative")
    report = {}
    for split in args.splits:
        report[split] = convert_split(
            args.dataset_root.resolve(),
            split,
            (args.output_dir / f"{split}.jsonl").resolve(),
            audit_depth_images=args.audit_depth_images,
            clip_bboxes=args.clip_bboxes,
        )
    report_path = args.output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
