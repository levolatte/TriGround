import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.prepare_semifinal_sft import build, read_records


def manifest(tmp_path, name, scenes=4, queries=2, source=None):
    root = tmp_path / name
    root.mkdir()
    rows = []
    for scene in range(scenes):
        Image.new("RGB", (40, 30)).save(root / f"{scene}.png")
        for query in range(queries):
            rows.append({
                "id": f"{scene}_{query}", "source": source or name,
                "scene_id": f"scene{scene}", "rgb": f"{scene}.png",
                "query": f"the object beside the red box, item {query}",
                "bbox": [0.1, 0.2, 0.6, 0.8],
            })
    path = root / "data.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def scene_keys(rows):
    return {(row["source"], row["scene_id"]) for row in rows}


def test_scene_split_preserves_full_query_and_supervision(tmp_path):
    source = manifest(tmp_path, "city")
    records = read_records(source)
    records[0]["query"] = "a very specific relation " * 40
    source.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    output = tmp_path / "out"
    report = build({"city": source}, output, val_ratio=0.25)
    train, val = read_records(output / "train.jsonl"), read_records(output / "val.jsonl")
    assert len(train) == 6 and len(val) == 2
    assert not scene_keys(train) & scene_keys(val)
    target = next(row for row in train + val if row["id"] == "0_0")
    assert target["query"] == records[0]["query"]
    assert target["query"] in target["messages"][0]["content"]
    assert json.loads(target["messages"][1]["content"])["bbox_2d"] == [100, 200, 600, 800]
    assert report["train_unique_ids"] == 6
    assert json.loads((output / "val_gt.json").read_text()) == {row["id"]: row["bbox"] for row in val}


def test_weights_are_exact_presentations_and_reproducible(tmp_path):
    sources = {name: manifest(tmp_path, name, 3, 1) for name in ("rgb", "rgbd")}
    report = build(sources, tmp_path / "first", weights={"rgb": 3, "rgbd": 1},
                   train_samples=16, val_ratio=0)
    build(sources, tmp_path / "second", weights={"rgb": 3, "rgbd": 1},
          train_samples=16, val_ratio=0)
    assert report["source_pools"]["rgb"]["train_presentations"] == 12
    assert report["source_pools"]["rgbd"]["train_presentations"] == 4
    assert report["train_unique_ids"] == 6
    assert (tmp_path / "first/train.jsonl").read_bytes() == (tmp_path / "second/train.jsonl").read_bytes()


def test_exclusions_remove_whole_scene_without_colliding_source_names(tmp_path):
    sources = {name: manifest(tmp_path, name, 2, 2) for name in ("a", "b")}
    exclusions = tmp_path / "exclude.jsonl"
    exclusions.write_text(json.dumps({"source": "a", "id": "0_0"}) + "\n", encoding="utf-8")
    report = build(sources, tmp_path / "out", val_ratio=0, exclude=[exclusions])
    rows = read_records(tmp_path / "out/train.jsonl")
    assert ("a", "scene0") not in scene_keys(rows)
    assert ("b", "scene0") in scene_keys(rows)
    assert report["source_pools"]["a"]["excluded_rows"] == 2
    assert report["source_pools"]["b"]["excluded_rows"] == 0


def test_exact_image_exclusion_handles_official_dictionary(tmp_path):
    source = manifest(tmp_path, "a", 2, 2)
    excluded = tmp_path / "official.json"
    excluded.write_text(json.dumps({"official_id": {"visible": "a/0.png", "query": "unused"}}))
    build({"a": source}, tmp_path / "out", val_ratio=0, exclude=[excluded])
    assert len(read_records(tmp_path / "out/train.jsonl")) == 2


def test_raw_depth_requires_explicit_policy_and_reuses_renderer(tmp_path):
    source = manifest(tmp_path, "rgbd", 2, 1)
    depth = source.parent / "depth.png"
    Image.fromarray(np.array([[0, 1], [10000, 19999]], dtype=np.uint16)).save(depth)
    rows = read_records(source)
    for row in rows:
        row["depth"] = "depth.png"
    source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    with pytest.raises(ValueError, match="raw depth needs"):
        build({"rgbd": source}, tmp_path / "bad", val_ratio=0)
    report = build({"rgbd": source}, tmp_path / "good", val_ratio=0,
                   depth_policies={"rgbd": "millimeter"})
    sample = read_records(tmp_path / "good/train.jsonl")[0]
    assert report["rendered_depth_images"] == 1
    assert sample["modalities"] == ["rgb", "depth"]
    assert sample["messages"][0]["content"].count("<image>") == 2
    with Image.open(sample["images"][1]) as image:
        assert np.asarray(image)[0, :, 0].tolist() == [0, 255]


def test_native_reasoning_keeps_answer_and_extracts_query_from_user_only(tmp_path):
    source = manifest(tmp_path, "expert", 1, 1)
    answer = '<think>The other candidate is irrelevant.</think>\n{"bbox_2d":[100,200,600,800]}'
    row = {
        "id": "reason", "image": "0.png", "scene_id": "scene0",
        "conversations": [
            {"from": "human", "value": "<image>\nLocate the object described by this query: the left car\nReturn JSON"},
            {"from": "gpt", "value": answer},
        ],
    }
    source.write_text(json.dumps(row), encoding="utf-8")
    build({"expert": source}, tmp_path / "out", val_ratio=0)
    output = read_records(tmp_path / "out/train.jsonl")[0]
    assert output["query"] == "the left car"
    assert output["messages"][1]["content"] == answer
    assert output["bbox"] == [0.1, 0.2, 0.6, 0.8]


def test_ambiguous_two_image_input_requires_explicit_modalities(tmp_path):
    source = manifest(tmp_path, "two", 1, 1)
    row = {
        "id": "two", "images": ["0.png", "0.png"],
        "messages": [{"role": "user", "content": "<image><image>the car"},
                     {"role": "assistant", "content": '{"bbox_2d":[100,200,600,800]}'}],
    }
    source.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="supply --modalities"):
        build({"two": source}, tmp_path / "bad", val_ratio=0)
    build({"two": source}, tmp_path / "good", val_ratio=0, modalities={"two": ("rgb", "infrared")})
    assert read_records(tmp_path / "good/train.jsonl")[0]["modalities"] == ["rgb", "infrared"]


def test_validation_source_cannot_silently_enter_training(tmp_path):
    source = manifest(tmp_path, "a", 1, 1)
    row = read_records(source)[0]
    row["split"] = "val"
    source.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="only training splits"):
        build({"a": source}, tmp_path / "out", val_ratio=0)
