from __future__ import annotations

import argparse
import json
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRAIN_MEMBER = "qwen_generation_train_100.json"
VAL_MEMBER = "qwen_generation_val.json"
MAX_VALID_DEPTH_MM = 19_999
INVALID_DEPTH_LEVEL = 0
FARTHEST_DEPTH_LEVEL = 1
NEAREST_DEPTH_LEVEL = 255


def bbox_to_qwen1000(bbox: list[float]) -> list[int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have four coordinates, got {bbox}")
    values = [Decimal(str(value)) for value in bbox]
    if not all(Decimal(0) <= value <= Decimal(1) for value in values):
        raise ValueError(f"bbox coordinates must be in [0, 1], got {bbox}")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError(f"bbox must be non-empty xyxy, got {bbox}")

    result = [int((value * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) for value in values]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"bbox collapsed after 0-1000 quantization: {bbox} -> {result}")
    return result


def depth_mm_to_grayscale_rgb(depth_mm: np.ndarray) -> np.ndarray:
    if depth_mm.ndim != 2 or not np.issubdtype(depth_mm.dtype, np.integer):
        raise ValueError(f"expected a 2D integer depth image, got {depth_mm.shape} {depth_mm.dtype}")

    depth = depth_mm.astype(np.int64, copy=False)
    valid = depth > 0
    clipped = np.minimum(depth, MAX_VALID_DEPTH_MM)
    gray = np.full(depth.shape, INVALID_DEPTH_LEVEL, dtype=np.uint8)
    scale = (NEAREST_DEPTH_LEVEL - FARTHEST_DEPTH_LEVEL) / (MAX_VALID_DEPTH_MM - 1)
    gray[valid] = np.rint(
        NEAREST_DEPTH_LEVEL - (clipped[valid] - 1) * scale
    ).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def render_depth(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        depth = np.asarray(image)
    rgb = depth_mm_to_grayscale_rgb(depth)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(destination)


def _source_path(record: dict[str, Any], key: str, source_manifest_dir: Path) -> Path:
    source = (source_manifest_dir / record[key]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{key} file not found: {source}")
    return source


def _relative_to_data_root(path: Path, data_root: Path) -> str:
    return path.relative_to(data_root.resolve()).as_posix()


def _answer(bbox_2d: list[int]) -> str:
    return json.dumps({"bbox_2d": bbox_2d}, separators=(",", ":"))


def _rgb_prompt(query: str) -> str:
    return (
        "<image>\nThe image is the RGB view. Locate the object described by this query: "
        f"{query}\nReturn only JSON in this exact form with coordinates normalized to 0-1000: "
        '{"bbox_2d":[x1,y1,x2,y2]}'
    )


def _trimodal_prompt(query: str) -> str:
    return (
        "<image>\n<image>\n<image>\n"
        "These are aligned views of the same scene in this order: RGB, infrared, depth. "
        "In the depth image, brighter valid pixels are nearer, darker valid pixels are "
        "farther, and black pixels are invalid. Locate the object described by this query: "
        f"{query}\nReturn only JSON in this exact form with coordinates normalized to 0-1000: "
        '{"bbox_2d":[x1,y1,x2,y2]}'
    )


def native_prompt(query: str, modalities: tuple[str, ...], depth_policy: str = "millimeter") -> str:
    """Prompt a real combination of views; keep the original City prompts unchanged."""
    if modalities == ("rgb",):
        return _rgb_prompt(query)
    if modalities == ("rgb", "infrared", "depth") and depth_policy == "millimeter":
        return _trimodal_prompt(query)
    descriptions = {"rgb": "RGB", "infrared": "infrared", "depth": "depth"}
    if not modalities or modalities[0] != "rgb" or len(set(modalities)) != len(modalities):
        raise ValueError(f"unsupported view order: {modalities}")
    if any(modality not in descriptions for modality in modalities):
        raise ValueError(f"unsupported modality in {modalities}")
    images = "\n".join("<image>" for _ in modalities)
    names = ", ".join(descriptions[modality] for modality in modalities)
    guidance = ""
    if "depth" in modalities and depth_policy == "millimeter":
        guidance = " Brighter valid depth pixels are nearer; black means invalid."
    elif "depth" in modalities and depth_policy == "sensor_linear_20000":
        guidance = (
            " This depth visualization uses a fixed inverse mapping of the raw sensor values: "
            "smaller positive stored values are brighter and zero is black. "
            "Its physical distance unit is not established."
        )
    elif "depth" in modalities and depth_policy != "visual":
        raise ValueError(f"unsupported depth policy: {depth_policy}")
    return (
        f"{images}\nThese are views of the same scene in this order: {names}."
        f"{guidance} Locate the object described by this query: {query}\n"
        "Return only JSON in this exact form with coordinates normalized to 0-1000: "
        '{"bbox_2d":[x1,y1,x2,y2]}'
    )


def _sample(
    sample_id: str,
    split: str,
    record: dict[str, Any],
    image: str | list[str],
    prompt: str,
    bbox_2d: list[int],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "split": split,
        "class_name": record["class_name"],
        "image": image,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": _answer(bbox_2d)},
        ],
    }


def convert_split(
    source: dict[str, dict[str, Any]],
    split: str,
    data_root: Path,
    source_manifest_dir: Path,
    output_dir: Path,
    converted_depths: dict[Path, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rgb_samples: list[dict[str, Any]] = []
    trimodal_samples: list[dict[str, Any]] = []
    depth_root = (data_root / "depth").resolve()

    for sample_id, record in source.items():
        visible = _source_path(record, "visible", source_manifest_dir)
        infrared = _source_path(record, "infrared", source_manifest_dir)
        depth = _source_path(record, "depth", source_manifest_dir)
        visible_rel = _relative_to_data_root(visible, data_root)
        infrared_rel = _relative_to_data_root(infrared, data_root)

        if depth not in converted_depths:
            depth_rel = depth.relative_to(depth_root)
            rendered = output_dir / "depth_rgb" / depth_rel
            render_depth(depth, rendered)
            converted_depths[depth] = _relative_to_data_root(rendered, data_root)
        rendered_depth_rel = converted_depths[depth]

        bbox_2d = bbox_to_qwen1000(record["bbox"])
        rgb = _sample(
            sample_id,
            split,
            record,
            visible_rel,
            _rgb_prompt(record["query"]),
            bbox_2d,
        )
        trimodal = _sample(
            sample_id,
            split,
            record,
            [visible_rel, infrared_rel, rendered_depth_rel],
            _trimodal_prompt(record["query"]),
            bbox_2d,
        )
        if rgb["conversations"][1] != trimodal["conversations"][1]:
            raise AssertionError(f"RGB/trimodal supervision differs for {sample_id}")
        rgb_samples.append(rgb)
        trimodal_samples.append(trimodal)

    return rgb_samples, trimodal_samples


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert_archive(manifest_zip: Path, data_root: Path, output_dir: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    source_manifest_dir = data_root / "target_v2"
    with zipfile.ZipFile(manifest_zip) as archive:
        train_source = json.loads(archive.read(TRAIN_MEMBER))
        val_source = json.loads(archive.read(VAL_MEMBER))

    overlap = set(train_source) & set(val_source)
    if overlap:
        raise ValueError(f"train/val ID overlap: {sorted(overlap)[:5]}")

    converted_depths: dict[Path, str] = {}
    rgb_train, trimodal_train = convert_split(
        train_source, "train", data_root, source_manifest_dir, output_dir, converted_depths
    )
    rgb_val, trimodal_val = convert_split(
        val_source, "val", data_root, source_manifest_dir, output_dir, converted_depths
    )

    outputs = {
        "rgb_train": output_dir / "rgb_train.json",
        "rgb_val": output_dir / "rgb_val.json",
        "trimodal_train": output_dir / "trimodal_train.json",
        "trimodal_val": output_dir / "trimodal_val.json",
    }
    for name, samples in (
        ("rgb_train", rgb_train),
        ("rgb_val", rgb_val),
        ("trimodal_train", trimodal_train),
        ("trimodal_val", trimodal_val),
    ):
        _write_json(outputs[name], samples)

    report = {
        "source_zip": str(manifest_zip.resolve()),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "train_samples": len(train_source),
        "val_samples": len(val_source),
        "rendered_depth_images": len(converted_depths),
        "depth_policy": {
            "source_unit": "millimeter",
            "linear_range_mm": [1, MAX_VALID_DEPTH_MM],
            "far_clip_condition": "value > 19999 maps to RGB [1, 1, 1]",
            "invalid_condition": "value == 0",
            "invalid_rgb": [0, 0, 0],
            "valid_rgb_range": [FARTHEST_DEPTH_LEVEL, NEAREST_DEPTH_LEVEL],
            "direction": "near is bright; far is dark",
            "normalization": "fixed global linear mapping; never per-image",
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    _write_json(output_dir / "conversion_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the reviewed City split into official Qwen3-VL SFT manifests."
    )
    parser.add_argument("--manifest-zip", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="City train directory containing target_v2, visible, infrared, and depth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: DATA_ROOT/target_v2/qwen3vl_native_sft",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.data_root / "target_v2" / "qwen3vl_native_sft"
    report = convert_archive(args.manifest_zip, args.data_root, output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
