import json

import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from mm_grounding.data import GroundingDataset, NativeGroundingCollator, encode_depth_image


def test_depth_uses_fixed_scale_and_validity_mask():
    depth = np.array([[0, 1000], [2000, 300000]], dtype=np.uint32)
    encoded = np.asarray(encode_depth_image(depth, unit_scale=1000.0, clip_distance=20.0))
    assert encoded.shape == (2, 2, 3)
    assert encoded[..., 1].tolist() == [[0, 255], [255, 0]]
    assert 0 < encoded[0, 1, 0] < encoded[1, 0, 0] < 255


def test_depth_value_does_not_depend_on_image_minmax():
    first = np.asarray(encode_depth_image(np.array([[1000]], dtype=np.uint16), 1000.0, 20.0))
    second = np.asarray(
        encode_depth_image(np.array([[1000, 9000]], dtype=np.uint16), 1000.0, 20.0)
    )
    assert first[0, 0, 0] == second[0, 0, 0]


def test_coordinate_mask_excludes_bbox_key_and_marks_only_values():
    pieces = {
        1: '{"bbox_2d":',
        2: "[",
        3: "123",
        4: ",",
        5: "456",
        6: "]}",
    }
    tokenizer = SimpleNamespace(decode=lambda ids, **_: pieces[ids[0]])
    collator = object.__new__(NativeGroundingCollator)
    collator.processor = SimpleNamespace(tokenizer=tokenizer)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = input_ids.clone()
    mask = collator._coordinate_token_mask(input_ids, labels)
    assert mask.tolist() == [[False, False, True, False, True, False]]


def test_ir_stage_accepts_canonical_aux_field_without_fake_depth(tmp_path):
    Image.new("RGB", (8, 6), "white").save(tmp_path / "rgb.png")
    Image.new("L", (8, 6), 127).save(tmp_path / "ir.png")
    manifest = tmp_path / "ir.json"
    manifest.write_text(
        json.dumps(
            {
                "sample": {
                    "stage": "ir",
                    "rgb": "rgb.png",
                    "aux": "ir.png",
                    "aux_type": "ir",
                    "query": "the warm object",
                    "bbox": [0.1, 0.2, 0.7, 0.8],
                }
            }
        ),
        encoding="utf-8",
    )
    sample = GroundingDataset(manifest, stage="ir")[0]
    assert set(sample) == {"rgb", "ir", "query", "bbox", "sample_id"}
    assert sample["ir"].mode == "RGB"


def test_depth_stage_loads_depth_without_fake_ir(tmp_path):
    Image.new("RGB", (8, 6), "white").save(tmp_path / "rgb.png")
    Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(tmp_path / "depth.png")
    manifest = tmp_path / "depth.json"
    manifest.write_text(
        json.dumps(
            {
                "sample": {
                    "stage": "depth",
                    "rgb": "rgb.png",
                    "depth": "depth.png",
                    "query": "the closest object",
                    "bbox": [0.1, 0.2, 0.7, 0.8],
                }
            }
        ),
        encoding="utf-8",
    )
    sample = GroundingDataset(manifest, stage="depth")[0]
    assert set(sample) == {"rgb", "depth", "query", "bbox", "sample_id"}
    assert sample["depth"].mode == "RGB"
