import json

from PIL import Image

from tools.prepare_roborefit import convert_split


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
