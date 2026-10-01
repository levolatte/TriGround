from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from tools.prepare_triground_abv_data import read_jsonl
from tools.prepare_triground_tm_data import (
    ABV, HOLDOUT_LOCATIONS, OUTPUT, _bbox_task, _category_rows, _depth_pair, _ir_read, prepare,
)


def test_depth_holdout_and_reading_use_manual_rois(tmp_path: Path) -> None:
    accepted = read_jsonl(ABV / "accepted_reviews.jsonl")
    pools, diagnostic, locations = _category_rows(accepted, {}, [])
    assert len(pools["depth"]) == 39
    assert len({row["scene_id"] for row in pools["depth"]}) == 17
    assert len([row for row in pools["depth"] if row["relation_type"] != "middle"]) == 32
    assert len(pools["ir_joint"]) == 60
    assert len(pools["competition"]) == 56
    assert len(pools["reliability"]) == 60
    held = [row for row in diagnostic if row["category"] == "depth_relation"]
    assert len(held) == 17
    assert len({row["scene_id"] for row in held}) == 8
    assert {row["location_group"] for row in held} == HOLDOUT_LOCATIONS
    assert locations.isdisjoint({row.get("location_group") for row in pools["depth"]})

    row = next(row for row in pools["depth"] if row["relation_type"] == "bounded_pair")
    normal = _depth_pair(row, tmp_path, random.Random(1), invalid=False,
                         sample_id="depth-normal", split="train")
    invalid = _depth_pair(row, tmp_path, random.Random(2), invalid=True,
                          sample_id="depth-invalid", split="train")
    assert normal["expected_answer"] in {"A_nearer", "B_nearer"}
    assert invalid["expected_answer"] == "unknown"
    assert normal["modalities"] == ["depth", "depth"]
    assert invalid["invalid_side"] in {0, 1}
    assert all(Path(path).is_file() for path in normal["image"] + invalid["image"])
    with Image.open(normal["image"][0]) as a, Image.open(normal["image"][1]) as b:
        assert a.size == b.size
    assert str(row["bbox"]) not in normal["conversations"][0]["value"]


def test_ir_read_uses_reviewed_ir_box_and_null(tmp_path: Path) -> None:
    row = next(row for row in read_jsonl(ABV / "accepted_reviews.jsonl")
               if row["category"] == "ir_complement")
    reviewed_box = [0.1, 0.2, 0.3, 0.4]
    review = {"task_id": row["task_id"], "ir_query": "A warm object at left",
              "ir_bbox": reviewed_box, "accepted": True, "split": "train"}
    normal = _ir_read(row, review, tmp_path, blank=False,
                      sample_id="ir-normal", split="train")
    blank = _ir_read(row, review, tmp_path, blank=True,
                     sample_id="ir-blank", split="train")
    assert normal["expected_answer"] == {"bbox_2d": [100, 200, 300, 400]}
    assert blank["expected_answer"] == {"bbox_2d": None}
    assert normal["modalities"] == ["ir"]
    assert blank["missing_modalities_actual"] == ["ir"]
    assert row["query"] not in normal["conversations"][0]["value"]
    assert "100,200,300,400" not in normal["conversations"][0]["value"]


def test_reliability_ir_is_degraded_without_missing_hint(tmp_path: Path) -> None:
    row = next(row for row in read_jsonl(ABV / "accepted_reviews.jsonl")
               if row.get("selected_for_training") and row["category"] == "reliability"
               and "infrared" in row["images"])
    sample = _bbox_task(row, "ir-low", tmp_path, low="infrared", condition="reliability_ir_low")
    assert sample["missing_modalities_actual"] == []
    assert "unavailable" not in sample["conversations"][0]["value"]
    assert sample["image"][sample["modalities"].index("ir")] != row["images"]["infrared"]
    with Image.open(sample["image"][sample["modalities"].index("ir")]) as image:
        assert image.getbbox() is not None


@pytest.mark.parametrize("steps,ir_deficit", [(400, 133), (600, 200)])
def test_preview_schedule_and_release_contract(tmp_path: Path, steps: int, ir_deficit: int) -> None:
    release = prepare(steps=steps, output=tmp_path)
    assert release["status"] == "provisional"
    assert release["deficits"]["ir_read_normal"] + release["deficits"]["ir_read_blank"] == ir_deficit
    assert sum(release["quotas"].values()) == 2*steps
    assert release["counts"]["M_extra_city_fallback"] == ir_deficit
    assert release["counts"]["M_mother_max"] <= 8
    assert release["counts"]["depth_required_pair_max_exposure_gap"] == 0
    assert release["counts"]["diagnostic_tasks"] == 87
    assert release["counts"]["depth_subject_pair_tasks"] == 32
    T = json.loads(Path(release["manifests"]["T"]).read_text(encoding="utf-8"))
    M = json.loads(Path(release["manifests"]["M"]).read_text(encoding="utf-8"))
    assert len(T) == len(M) == steps*10
    for index in range(steps):
        start = index*10
        assert T[start:start+8] == M[start:start+8]
        assert all(row["task_category"] == "old_city" for row in T[start:start+10])
    diag_locations = {"000003", "000005", "000016", "001179", "000610", "000013"}
    for rows in (T, M):
        city = Counter(row["source_task_id"] for row in rows if row["task_category"] == "old_city")
        assert max(city.values()) <= 2
        assert all(row["scene_id"].split(":", 1)[1].split("_")[0] not in diag_locations
                   for row in rows if row["source"] == "city")
        assert all(len(row["image"]) == len(row["modalities"])
                   == row["conversations"][0]["value"].count("<image>") for row in rows)
    new_used = Counter(row["source_task_id"] for row in M if row["task_category"] != "old_city")
    assert max(new_used.values()) <= 8
    assert len(json.loads(Path(release["training_probes"]["M"]).read_text(encoding="utf-8"))) == 215
    assert len(json.loads(Path(release["probes"]["depth_read"]).read_text(encoding="utf-8"))) == 46
    assert Path(release["release_file"]).is_file()


@pytest.mark.parametrize("steps", [400, 600])
def test_formal_release(steps: int) -> None:
    release = json.loads((OUTPUT / f"release_{steps}_seed2028/release.json").read_text(encoding="utf-8"))
    assert release["status"] == "ready"
    assert all(value == 0 for value in release["deficits"].values())
    assert release["counts"]["M_extra_actual"] == release["quotas"]
    assert release["counts"]["depth_required_pair_max_exposure_gap"] == 0
    assert release["counts"]["training_probe_mothers"] == 215
    assert release["counts"]["diagnostic_tasks"] == 103
    assert release["counts"]["ir_probe_rows"] == 32
    assert len(read_jsonl(Path(release["sources"]["ir_reviews_retained"]))) == 118
    T = json.loads(Path(release["manifests"]["T"]).read_text(encoding="utf-8"))
    M = json.loads(Path(release["manifests"]["M"]).read_text(encoding="utf-8"))
    assert len(T) == len(M) == 10*steps
    assert all(T[i:i+8] == M[i:i+8] for i in range(0, 10*steps, 10))
    for rows in (T, M):
        old_city = Counter(row["source_task_id"] for row in rows if row["task_category"] == "old_city")
        assert max(old_city.values()) <= 2
    assert all(sum(row["task_category"] == "old_city" for row in M[i:i+10]) == 8
               for i in range(0, 10*steps, 10))
    assert max(Counter(row["source_task_id"] for row in M if row["task_category"] != "old_city").values()) <= 8
    accepted = {row["task_id"]: row for row in read_jsonl(ABV / "accepted_reviews.jsonl")}
    assert all(accepted[row["source_task_id"]]["selected_for_training"]
               for row in M if row["task_category"] not in {"old_city"})
    review = {row["task_id"]: row for row in read_jsonl(Path(release["sources"]["ir_reviews_retained"]))}
    ir_read = [row for row in M if row["task_type"] == "ir_bbox"]
    assert len(ir_read) == release["quotas"]["ir_read_normal"] + release["quotas"]["ir_read_blank"]
    for row in ir_read:
        source = review[row["source_task_id"]]
        assert source["accepted"] and source["split"] == "train"
        if row["task_category"] == "ir_read_blank":
            assert row["ir_gt_bbox"] is None and row["expected_answer"] == {"bbox_2d": None}
        else:
            assert row["ir_gt_bbox"] == source["ir_bbox"]
        assert str(source["ir_bbox"]) not in row["conversations"][0]["value"]
    assert len(json.loads(Path(release["training_probes"]["M"]).read_text(encoding="utf-8"))) == 215
    assert len(json.loads(Path(release["diagnostics"]["normal"]).read_text(encoding="utf-8"))) == 103
    assert len(json.loads(Path(release["probes"]["ir_read"]).read_text(encoding="utf-8"))) == 32
    class_map = json.loads(Path(release["class_map"]).read_text(encoding="utf-8"))
    assert Counter(class_map.values()) == {"depth": 23, "ir": 48, "rgb_sufficient": 32}
    ir_missing = json.loads(Path(release["diagnostics"]["ir_missing"]).read_text(encoding="utf-8"))
    diag_ir = next(row for row in ir_missing if "ir" in row["modalities"])
    assert diag_ir["missing_modalities_actual"] == ["ir"]
    with Image.open(diag_ir["image"][diag_ir["modalities"].index("ir")]) as image:
        assert image.getbbox() is None
    scene_exposure = json.loads(Path(release["city_scene_exposure"]).read_text(encoding="utf-8"))
    for arm, rows in (("T", T), ("M", M)):
        counted = Counter(row["scene_id"] for row in rows if row["source"] == "city")
        assert scene_exposure[arm] == dict(counted)
        assert sum(counted.values()) == release["counts"]["city_scenes"][arm]["total_presentations"]
