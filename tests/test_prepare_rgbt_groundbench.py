import json

import torch
from PIL import Image

from tools.prepare_rgbt_groundbench import SOURCES, convert_split


def test_converts_official_rgbt_rows_to_ir_manifest(tmp_path):
    root = tmp_path / "extracted"
    for source in SOURCES:
        annotation = root / f"rgbtvg_{source}"
        annotation.mkdir(parents=True)
        image_name = f"{source}.png"
        for modality in ("rgb", "ir"):
            image_dir = root / "image_data" / source / modality
            image_dir.mkdir(parents=True)
            Image.new("RGB", (20, 10)).save(image_dir / image_name)
        row = [image_name, {"width": 20, "height": 10}, [2, 1, 8, 5], "target"]
        torch.save([row], annotation / f"rgbtvg_{source}_train.pth")

    output = root / "manifests" / "train.jsonl"
    report = convert_split(root, "train", output, verify_images=True)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report == {"total": 3, "flir": 1, "mfad": 1, "m3fd": 1, "verified_pairs": 3}
    assert records[0]["stage"] == records[0]["aux_type"] == "ir"
    assert records[0]["bbox"] == [0.1, 0.1, 0.5, 0.6]
    assert (output.parent / records[0]["rgb"]).resolve().is_file()
    assert (output.parent / records[0]["aux"]).resolve().is_file()
