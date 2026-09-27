from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.prepare_next_stage_data import (
    QUOTAS,
    _native_sample,
    _nearfar_task,
    _select_auxiliary,
    _source_row,
    build_schedule,
    prepare,
)
from tools.prepare_qwen3vl_native_sft import native_prompt


def _rgb(path: Path) -> None:
    Image.fromarray(np.full((20, 20, 3), 50, dtype=np.uint8)).save(path)


def test_true_modality_prompts_do_not_make_up_missing_views() -> None:
    ir = native_prompt("left lamp", ("rgb", "infrared"))
    depth_visual = native_prompt("box", ("rgb", "depth"), "visual")
    depth_metric = native_prompt("box", ("rgb", "depth"), "millimeter")
    assert ir.count("<image>") == 2 and "depth" not in ir
    assert depth_visual.count("<image>") == 2 and "nearer" not in depth_visual
    assert "nearer" in depth_metric
    sensor = native_prompt("box", ("rgb", "infrared", "depth"), "sensor_linear_20000")
    assert "stored values" in sensor and "physical distance unit is not established" in sensor
    assert "nearer" not in sensor


def test_schedule_follows_exact_quotas_and_balances_groups() -> None:
    sources = {
        key: [
            {"id": f"{key}-{index}", "group_id": f"{key}-group-{index}", "source": key}
            for index in range(3)
        ]
        for key in ("city", "rgbdt", "rgbt", "robo")
    }
    first = build_schedule(sources, QUOTAS[("D", 1)], 2026)
    second = build_schedule(sources, QUOTAS[("T", 1)], 2026)
    assert [(row["source"], row["id"]) for row in first] == [
        (row["source"], row["id"]) for row in second
    ]
    assert Counter(row["source"] for row in first) == QUOTAS[("D", 1)]
    assert len({row["group_id"] for row in first if row["source"] == "rgbdt"}) == 3


def test_city_continuation_has_original_prompt_answer_and_private_metadata(tmp_path: Path) -> None:
    for modality in ("rgb", "infrared", "depth"):
        _rgb(tmp_path / f"{modality}.png")
    city = [{
        "id": "city-original", "image": ["rgb.png", "infrared.png", "depth.png"],
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\n<image>\nFind the lamp"},
            {"from": "gpt", "value": '{"bbox_2d":[100,100,300,300]}'},
        ],
        "class_name": "wrong metadata",
    }]
    city_path = tmp_path / "city.json"
    city_path.write_text(json.dumps(city), encoding="utf-8")
    output = tmp_path / "schedule"
    report = prepare(argparse.Namespace(
        city_native=city_path, city_root=tmp_path, rgbdt=None, rgbt=None, robo=None,
        robo_fallback_rgbdt=False, branch="C", phase=1, seed=2026, output_dir=output,
    ))
    assert report["presentations"] == 8000
    assert report["actual_sources"] == {"city": 8000}
    sample = json.loads((output / "c_phase1.json").read_text(encoding="utf-8"))[0]
    assert sample["conversations"] == city[0]["conversations"]
    assert all(Path(path).is_absolute() for path in sample["image"])
    assert "class_name" not in sample and "review_status" not in sample
    metadata = json.loads((output / "c_phase1_metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metadata["source"] == "city" and metadata["task"] == "bbox"


def test_rgbdt_requires_review_and_metric_depth_for_relation(tmp_path: Path) -> None:
    _rgb(tmp_path / "rgb.png")
    _rgb(tmp_path / "ir.png")
    depth = np.zeros((20, 20), dtype=np.uint16)
    depth[:, :10] = 1000
    depth[:, 10:] = 2500
    Image.fromarray(depth).save(tmp_path / "depth.png")
    row = {
        "id": "item-1", "rgb": "rgb.png", "infrared": "ir.png", "depth": "depth.png",
        "query": "the left item", "bbox": [0.05, 0.05, 0.35, 0.9],
        "review_status": "approved_batch", "depth_policy": "millimeter",
        "scene_id": "group-1", "nearfar_objects": [
            {"object_id": "near", "bbox": [0.05, 0.05, 0.35, 0.9], "confirmed": True},
            {"object_id": "far", "bbox": [0.65, 0.05, 0.95, 0.9], "confirmed": True},
        ],
    }
    manifest = tmp_path / "rows.jsonl"
    record = _source_row(row, "rgbdt", manifest, tmp_path)
    assert _nearfar_task(record)["near_object_id"] == "near"
    sample, metadata = _native_sample(record, 1, tmp_path, {}, "nearfar", random.Random(2026))
    prompt = sample["conversations"][0]["value"]
    assert prompt.count("<image>") == 3
    assert "1000" not in prompt and "2500" not in prompt
    assert sample["conversations"][1]["value"] in {"A", "B"}
    assert "nearfar_objects" not in sample and metadata["task"] == "nearfar"
    rejected = dict(row, review_status="auto_pass")
    with pytest.raises(ValueError, match="human-review gate"):
        _source_row(rejected, "rgbdt", manifest, tmp_path)


def test_unverified_sensor_depth_uses_fixed_visible_mapping_without_metric_aux(tmp_path: Path) -> None:
    _rgb(tmp_path / "rgb.png")
    _rgb(tmp_path / "ir.png")
    raw = np.array([[0, 1000, 10000, 19999]], dtype=np.uint16)
    Image.fromarray(raw).save(tmp_path / "depth.png")
    row = {
        "id": "sensor-1", "source": "rgbdt", "query": "small object",
        "bbox": [0.1, 0.1, 0.3, 0.3], "group_id": "scene",
        "rgb_path": tmp_path / "rgb.png", "infrared_path": tmp_path / "ir.png",
        "depth_path": tmp_path / "depth.png", "depth_policy": "sensor_linear_20000",
        "nearfar_objects": [
            {"object_id": "a", "bbox": [0.1, 0.1, 0.3, 0.3], "confirmed": True},
            {"object_id": "b", "bbox": [0.6, 0.6, 0.8, 0.8], "confirmed": True},
        ],
    }
    sample, meta = _native_sample(row, 0, tmp_path, {}, "bbox", random.Random(1))
    rendered = np.asarray(Image.open(sample["image"][2]))
    assert rendered[0, :, 0].tolist()[0] == 0
    assert rendered[0, 1, 0] > rendered[0, 2, 0] > rendered[0, 3, 0] > 0
    assert _nearfar_task(row) is None
    assert meta["depth_policy"] == "sensor_linear_20000"


def test_confirmed_instance_shuffles_answer_without_leaking_object_ids(tmp_path: Path) -> None:
    _rgb(tmp_path / "rgb.png")
    _rgb(tmp_path / "ir.png")
    _rgb(tmp_path / "depth.png")
    record = {
        "id": "item", "source": "rgbdt", "query": "the left box", "bbox": [0.1, 0.1, 0.3, 0.3],
        "rgb_path": tmp_path / "rgb.png", "infrared_path": tmp_path / "ir.png",
        "depth_path": tmp_path / "depth.png", "depth_policy": "visual", "group_id": "one",
        "target_object_id": "secret-positive",
        "confirmed_objects": [
            {"object_id": "secret-positive", "bbox": [0.1, 0.1, 0.3, 0.3], "confirmed": True},
            {"object_id": "secret-negative", "bbox": [0.6, 0.6, 0.8, 0.8], "confirmed": True},
        ],
    }
    answers = set()
    for seed in range(20):
        sample, _ = _native_sample(record, seed, tmp_path, {}, "instance", random.Random(seed))
        prompt, answer = (part["value"] for part in sample["conversations"])
        assert "secret-positive" not in prompt and "secret-negative" not in prompt
        line = next(line for line in prompt.splitlines() if line.startswith(answer + ":"))
        assert "[100, 100, 300, 300]" in line
        answers.add(answer)
    assert answers == {"A", "B"}


def test_t_replaces_exactly_800_plus_800_underlying_presentations(tmp_path: Path) -> None:
    _rgb(tmp_path / "rgb.png")
    _rgb(tmp_path / "ir.png")
    depth = np.full((20, 20), 2500, dtype=np.uint16)
    depth[:, :10] = 1000
    Image.fromarray(depth).save(tmp_path / "depth.png")
    rows = []
    for index in range(200):
        common = {
            "source": "rgbdt", "rgb_path": tmp_path / "rgb.png",
            "infrared_path": tmp_path / "ir.png", "depth_path": tmp_path / "depth.png",
            "depth_policy": "millimeter", "group_id": f"g{index % 40}",
        }
        rows.append({
            **common, "id": f"instance-{index}", "query": "left item",
            "target_object_id": "one", "confirmed_objects": [
                {"object_id": "one", "bbox": [0.05, 0.05, 0.35, 0.9], "confirmed": True},
                {"object_id": "two", "bbox": [0.65, 0.05, 0.95, 0.9], "confirmed": True},
            ],
        })
        rows.append({
            **common, "id": f"nearfar-{index}", "nearfar_objects": [
                {"object_id": "near", "bbox": [0.05, 0.05, 0.35, 0.9], "confirmed": True},
                {"object_id": "far", "bbox": [0.65, 0.05, 0.95, 0.9], "confirmed": True},
            ],
        })
    schedule = rows * 5
    assignments = _select_auxiliary(schedule, 2026)
    assert Counter(assignments.values()) == {"instance": 800, "nearfar": 800}
    assert all(schedule[index]["id"].startswith(kind) for index, kind in assignments.items())


def test_d_materialization_keeps_real_two_and_three_view_inputs(tmp_path: Path) -> None:
    for name in ("rgb", "ir", "rgbdt_depth", "robo_depth", "city_depth"):
        _rgb(tmp_path / f"{name}.png")
    city = [{
        "id": "city1", "split": "train",
        "image": ["rgb.png", "ir.png", "city_depth.png"],
        "conversations": [
            {"from": "human", "value": "<image>\n<image>\n<image>\nCity prompt"},
            {"from": "gpt", "value": '{"bbox_2d":[100,100,300,300]}'},
        ],
    }]
    city_path = tmp_path / "city.json"
    city_path.write_text(json.dumps(city), encoding="utf-8")
    sources = {
        "rgbdt": [{
            "id": "rgbdt1", "rgb": "rgb.png", "infrared": "ir.png", "depth": "rgbdt_depth.png",
            "depth_policy": "visual", "review_status": "human_accepted", "scene_id": "seq1",
            "query": "RGBDT query", "bbox": [0.1, 0.1, 0.3, 0.3],
        }],
        "rgbt": [{
            "id": "rgbt1", "rgb": "rgb.png", "aux": "ir.png", "aux_type": "ir",
            "scene_id": "frame1", "query": "RGB-T query", "bbox": [0.1, 0.1, 0.3, 0.3],
        }],
        "robo": [{
            "id": "robo1", "rgb": "rgb.png", "depth": "robo_depth.png",
            "scene_id": "frame2", "query": "RGB-D query", "bbox": [0.1, 0.1, 0.3, 0.3],
        }],
    }
    paths = {}
    for source, rows in sources.items():
        path = tmp_path / f"{source}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths[source] = path
    output = tmp_path / "mixed"
    report = prepare(argparse.Namespace(
        city_native=city_path, city_root=tmp_path, rgbdt=paths["rgbdt"],
        rgbt=paths["rgbt"], robo=paths["robo"], robo_fallback_rgbdt=False,
        branch="D", phase=1, seed=2026, output_dir=output,
    ))
    assert report["actual_sources"] == QUOTAS[("D", 1)]
    rows = json.loads((output / "d_phase1.json").read_text(encoding="utf-8"))
    by_source = {}
    for row in rows:
        by_source.setdefault(row["id"].split(":")[0], row)
    assert [len(by_source[source]["image"]) for source in ("city", "rgbdt", "rgbt", "robo")] == [3, 3, 2, 2]
    assert "RGB-T query" in by_source["rgbt"]["conversations"][0]["value"]
    assert "depth" not in by_source["rgbt"]["conversations"][0]["value"]
    assert "nearer" not in by_source["robo"]["conversations"][0]["value"]
