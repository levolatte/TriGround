from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


REQUIRED_FIELDS = ("visible", "infrared", "depth", "query", "bbox")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare city weak grounding data for Stage 4")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--include-visualized-depth",
        action="store_true",
        help="允许三通道/8-bit 深度可视化图；正式训练不推荐",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    source = (args.source or root / "queries" / "weak_grounding.json").resolve()
    output_dir = (args.output_dir or root).resolve()
    records = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(records, dict):
        raise SystemExit("weak grounding 顶层必须是 sample ID -> record 映射")

    split_stems = {
        split: set((root / "splits" / f"{split}.txt").read_text(encoding="utf-8-sig").split())
        for split in ("train", "val")
    }
    if split_stems["train"] & split_stems["val"]:
        raise SystemExit("train/val split 存在场景重叠")

    outputs: dict[str, dict[str, dict]] = {"train": {}, "val": {}}
    errors: list[str] = []
    excluded_visualized_depth: set[str] = set()
    for sample_id, raw in records.items():
        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            errors.append(f"{sample_id}: missing {missing}")
            continue
        bbox = raw["bbox"]
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and 0 <= bbox[0] < bbox[2] <= 1
            and 0 <= bbox[1] < bbox[3] <= 1
        ):
            errors.append(f"{sample_id}: invalid bbox {bbox}")
            continue
        if not str(raw["query"]).strip():
            errors.append(f"{sample_id}: empty query")
            continue
        for modality in ("visible", "infrared", "depth"):
            if not (root / raw[modality]).is_file():
                errors.append(f"{sample_id}: missing path {raw[modality]}")
                break
        else:
            stem = Path(raw["visible"]).stem
            depth_path = root / raw["depth"]
            with Image.open(depth_path) as depth_image:
                metric_depth = depth_image.mode in {"I;16", "I;16L", "I;16B", "I"}
            if not metric_depth and not args.include_visualized_depth:
                excluded_visualized_depth.add(stem)
                continue
            split = "train" if stem in split_stems["train"] else "val" if stem in split_stems["val"] else None
            if split is None:
                errors.append(f"{sample_id}: scene {stem} is outside splits")
                continue
            outputs[split][sample_id] = {
                "visible": raw["visible"],
                "infrared": raw["infrared"],
                "depth": raw["depth"],
                "query": str(raw["query"]).strip(),
                "bbox": [round(float(value), 6) for value in bbox],
            }

    if errors:
        preview = "\n".join(errors[:20])
        raise SystemExit(f"转换前校验失败，共 {len(errors)} 条:\n{preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_records in outputs.items():
        (output_dir / f"stage4_{split}.json").write_text(
            json.dumps(split_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report = {
        "source": str(source),
        "weak_labels": True,
        "source_train_scenes": len(split_stems["train"]),
        "source_val_scenes": len(split_stems["val"]),
        "converted_train_scenes": len({Path(item["visible"]).stem for item in outputs["train"].values()}),
        "converted_val_scenes": len({Path(item["visible"]).stem for item in outputs["val"].values()}),
        "train_queries": len(outputs["train"]),
        "val_queries": len(outputs["val"]),
        "total_queries": sum(map(len, outputs.values())),
        "excluded_visualized_depth_scenes": len(excluded_visualized_depth),
        "depth_policy": "single-channel integer depth only" if not args.include_visualized_depth else "includes visualized depth",
        "format": "sample_id -> {visible, infrared, depth, query, bbox(normalized xyxy)}",
    }
    (output_dir / "stage4_conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
