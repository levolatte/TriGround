import json

from PIL import Image

from tools.prepare_roborefit import convert_split
from tools.build_grouped_subsets import record_group


def test_convert_roborefit_normalizes_bbox_and_resolves_release_paths(tmp_path):
    data = tmp_path / "RoboRefIt" / "data"
    split = data / "final_dataset" / "train"
    (split / "image").mkdir(parents=True)
    (split / "depth").mkdir()
    Image.new("RGB", (640, 480)).save(split / "image" / "0000000.png")
    Image.new("I;16", (640, 480), 1200).save(split / "depth" / "0000000.png")
    annotation = [
        {
            "num": 0,
            "text": "the cup on the left",
            "bbox": [64, 48, 320, 240],
            "rgb_path": "final_dataset\\train\\image\\0000000.png",
            "depth_path": "final_dataset\\train\\depth\\0000000.png",
            "mask_path": "final_dataset\\train\\mask\\0000000\\0.png",
            "scene": "table",
            "class": "cup",
        }
    ]
    (split / "roborefit_train.json").write_text(json.dumps(annotation), encoding="utf-8")
    output = data / "manifests" / "train.jsonl"

    report = convert_split(data, "train", output, audit_depth_images=1)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["stage"] == "depth"
    assert row["bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert row["class_name"] == "cup"
    assert (output.parent / row["rgb"]).resolve() == split / "image" / "0000000.png"
    assert (output.parent / row["depth"]).resolve() == split / "depth" / "0000000.png"
    assert report["samples"] == 1
    assert report["depth_max"] == 1200


def test_grouped_subsets_default_to_visible_scene_path():
    record = {"visible": "visible/scene_001.png"}
    assert record_group("different-query-id", record, None) == record["visible"]


def test_convert_roborefit_clips_and_reports_edge_overflow(tmp_path):
    split = tmp_path / "final_dataset" / "train"
    (split / "image").mkdir(parents=True)
    (split / "depth").mkdir()
    Image.new("RGB", (640, 480)).save(split / "image" / "0000000.png")
    Image.new("L", (640, 480), 100).save(split / "depth" / "0000000.png")
    annotation = [
        {
            "text": "the clipped object",
            "bbox": [-1, 20, 649, 481],
            "rgb_path": "final_dataset/train/image/0000000.png",
            "depth_path": "final_dataset/train/depth/0000000.png",
        }
    ]
    (split / "roborefit_train.json").write_text(json.dumps(annotation), encoding="utf-8")
    output = tmp_path / "manifests" / "train.jsonl"

    report = convert_split(tmp_path, "train", output, clip_bboxes=True)
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["bbox"] == [0.0, 0.04166667, 1.0, 1.0]
    assert report["bbox_clipped"] == 1
    assert report["max_bbox_clip_pixels"] == 9
    assert report["bbox_clip_examples"][0]["original"] == [-1.0, 20.0, 649.0, 481.0]
