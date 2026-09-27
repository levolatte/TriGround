from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.prepare_qwen3vl_native_sft import (
    bbox_to_qwen1000,
    convert_archive,
    depth_mm_to_grayscale_rgb,
)


def test_depth_mapping_is_fixed_monotonic_and_marks_invalid_black():
    depth = np.array([[0, 1, 10_000, 19_999, 20_000, 65_535]], dtype=np.uint16)
    rgb = depth_mm_to_grayscale_rgb(depth)

    assert rgb.shape == (1, 6, 3)
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 1] == rgb[:, :, 2])
    assert rgb[0, 0].tolist() == [0, 0, 0]
    assert rgb[0, 1].tolist() == [255, 255, 255]
    assert 1 < int(rgb[0, 2, 0]) < 255
    assert rgb[0, 3].tolist() == [1, 1, 1]
    assert rgb[0, 4].tolist() == [1, 1, 1]
    assert rgb[0, 5].tolist() == [1, 1, 1]


def test_bbox_uses_half_up_rounding_and_rejects_collapsed_boxes():
    assert bbox_to_qwen1000([0, 0.0005, 0.9995, 1]) == [0, 1, 1000, 1000]
    with pytest.raises(ValueError, match="collapsed"):
        bbox_to_qwen1000([0.0001, 0.1, 0.0002, 0.2])


def _save_rgb(path: Path, value: int) -> None:
    Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(path)


def test_archive_conversion_preserves_split_id_and_identical_supervision(tmp_path: Path):
    data_root = tmp_path / "train"
    for directory in ("target_v2", "visible", "infrared", "depth"):
        (data_root / directory).mkdir(parents=True)

    for stem in ("train_frame", "val_frame"):
        _save_rgb(data_root / "visible" / f"{stem}.png", 20)
        _save_rgb(data_root / "infrared" / f"{stem}.png", 40)
        Image.fromarray(np.array([[0, 1, 19_999]], dtype=np.uint16)).save(
            data_root / "depth" / f"{stem}.png"
        )

    def record(stem: str, bbox: list[float]) -> dict[str, object]:
        return {
            "visible": f"../visible/{stem}.png",
            "infrared": f"../infrared/{stem}.png",
            "depth": f"../depth/{stem}.png",
            "query": f"the object in {stem}",
            "bbox": bbox,
            "class_name": "person",
        }

    archive_path = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "qwen_generation_train_100.json",
            json.dumps({"train-id": record("train_frame", [0, 0.5, 0.9995, 1])}),
        )
        archive.writestr(
            "qwen_generation_val.json",
            json.dumps({"val-id": record("val_frame", [0.1, 0.2, 0.3, 0.4])}),
        )

    output_dir = data_root / "target_v2" / "qwen3vl_native_sft"
    report = convert_archive(archive_path, data_root, output_dir)
    rgb_train = json.loads((output_dir / "rgb_train.json").read_text(encoding="utf-8"))
    rgb_val = json.loads((output_dir / "rgb_val.json").read_text(encoding="utf-8"))
    trimodal_train = json.loads(
        (output_dir / "trimodal_train.json").read_text(encoding="utf-8")
    )

    assert report["train_samples"] == 1
    assert report["val_samples"] == 1
    assert report["rendered_depth_images"] == 2
    assert rgb_train[0]["id"] == "train-id"
    assert rgb_train[0]["split"] == "train"
    assert rgb_val[0]["id"] == "val-id"
    assert rgb_val[0]["split"] == "val"
    assert isinstance(rgb_train[0]["image"], str)
    assert trimodal_train[0]["image"] == [
        "visible/train_frame.png",
        "infrared/train_frame.png",
        "target_v2/qwen3vl_native_sft/depth_rgb/train_frame.png",
    ]
    assert trimodal_train[0]["conversations"][0]["value"].count("<image>") == 3
    assert rgb_train[0]["conversations"][1] == trimodal_train[0]["conversations"][1]
    assert json.loads(rgb_train[0]["conversations"][1]["value"])["bbox_2d"] == [
        0,
        500,
        1000,
        1000,
    ]
    rendered = np.asarray(
        Image.open(output_dir / "depth_rgb" / "train_frame.png")
    )
    assert rendered[0].tolist() == [[0, 0, 0], [255, 255, 255], [1, 1, 1]]
