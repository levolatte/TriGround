from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pytest

from tools.prepare_gu_diagnosis import _normalized_query, build
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def _repo_sources():
    root = Path(__file__).resolve().parents[2]
    snapshot = root / "results/gu_diagnosis_20260927/source_snapshot"
    old_root = root / "results/gu_pilot_20260926/final_cloud_snapshot"
    release_root = old_root
    candidates = root / "results/same_day_multimodal_pilot_20260926/pending_candidates.jsonl"
    paths = [
        snapshot / "city_train.json", snapshot / "city_val.json", snapshot / "city_gt.json",
        old_root / "manifests/seed2026/g_train.json", candidates,
        release_root / "released_cloud.jsonl", release_root / "released_metadata.jsonl",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("downloaded diagnosis source snapshot is unavailable")
    return root, snapshot, old_root, candidates


def test_real_seed2026_diagnosis_manifests_preserve_pairing_and_raw_gt(tmp_path: Path) -> None:
    root, snapshot, old_root, candidates_path = _repo_sources()
    output = tmp_path / "diagnosis"
    args = argparse.Namespace(
        old_root=old_root,
        released_jsonl=old_root / "released_cloud.jsonl",
        released_metadata=old_root / "released_metadata.jsonl",
        released_candidates=candidates_path,
        city_native=snapshot / "city_train.json",
        city_root="/root/autodl-tmp/rematch_20260922/data/city/train",
        city_val=snapshot / "city_val.json",
        city_gt=snapshot / "city_gt.json",
        output_dir=output,
    )
    report = build(args)

    old_dir = old_root / "manifests/seed2026"
    old_g = _json(old_dir / "g_train.json")
    old_u = _json(old_dir / "u_train.json")
    old_meta = _jsonl(old_dir / "metadata.jsonl")
    train_b, train_g, train_u = (_json(output / f"{name}.json") for name in ("train_b", "train_gstar", "train_ustar"))
    train_b_meta = _jsonl(output / "train_b_metadata.jsonl")
    train_g_meta = _jsonl(output / "train_gstar_metadata.jsonl")
    train_u_meta = _jsonl(output / "train_ustar_metadata.jsonl")
    released = {row["id"]: row for row in _jsonl(args.released_jsonl)}
    released_meta = {row["id"]: row for row in _jsonl(args.released_metadata)}
    candidates = {row["task_id"]: row for row in _jsonl(candidates_path)}

    assert report["train_rows"] == {"train_b": 1600, "train_gstar": 1600, "train_ustar": 1600}
    assert [row["id"] for row in train_b] == [row["id"] for row in old_g]
    assert [row["id"] for row in train_g] == [row["id"] for row in old_g]
    assert [row["id"] for row in train_u] == [row["id"] for row in old_u]
    city_positions = [i for i, row in enumerate(old_meta) if row["source"] == "city"]
    for i in city_positions:
        assert train_b[i] == old_g[i]
        assert train_g[i] == old_g[i]
        assert train_u[i] == old_u[i]
        for row in (train_b_meta[i], train_g_meta[i], train_u_meta[i]):
            assert row["target_modality"] == row["output_modality"] == "rgb"
            assert row["source_task_id"] == old_meta[i]["source_task_id"]

    old_new = [row for row in old_meta if row["source"] != "city"]
    old_frequency = Counter(released_meta[row["source_task_id"]]["source_id"] for row in old_new)
    mapping = _jsonl(output / "b_query_mapping.jsonl")
    assert len(mapping) == 98
    assert sum(row["frequency"] for row in mapping) == 400
    city_train = {row["id"]: row for row in _json(snapshot / "city_train.json")}
    old_city_ids = {row["source_task_id"] for row in old_meta if row["source"] == "city"}
    old_city_queries = {
        _normalized_query(row["query"]) for row in old_meta if row["source"] == "city"
    }
    for item in mapping:
        assert item["frequency"] == old_frequency[item["source_id"]]
        assert item["city_source_id"] not in old_city_ids
        assert item["city_source_id"] in city_train
        assert _normalized_query(item["city_query"]) not in old_city_queries
        assert _normalized_query(item["city_query"]) != _normalized_query(item["query"])
        assert item["target_bbox_qwen1000"] == json.loads(
            city_train[item["city_source_id"]]["conversations"][1]["value"]
        )["bbox_2d"]
    mapping_by_source = {row["source_id"]: row for row in mapping}
    assert set(mapping_by_source) == set(old_frequency)

    differing_auxiliary_boxes = 0
    for position, schedule_row in enumerate(old_meta):
        if schedule_row["source"] == "city":
            continue
        source_task_id = schedule_row["source_task_id"]
        candidate = candidates[source_task_id]
        u_modality = released_meta[source_task_id]["target_modality"]
        g_row, u_row = train_g[position], train_u[position]
        assert g_row["image"] == u_row["image"] == released[source_task_id]["image"]
        g_prompt = g_row["conversations"][0]["value"]
        u_prompt = u_row["conversations"][0]["value"]
        assert "physical distance unit is not established" in g_prompt
        assert candidate["query"] in g_prompt and candidate["query"] in u_prompt
        assert g_prompt.replace("the RGB image coordinates", "the TARGET image coordinates") == u_prompt.replace(
            f"the {'RGB' if u_modality == 'rgb' else u_modality} image coordinates", "the TARGET image coordinates"
        )
        g_box = json.loads(g_row["conversations"][1]["value"])["bbox_2d"]
        u_box = json.loads(u_row["conversations"][1]["value"])["bbox_2d"]
        assert g_box == bbox_to_qwen1000(candidate["rgb_gt_bbox_xyxy_normalized"])
        assert u_box == json.loads(released[source_task_id]["conversations"][1]["value"])["bbox_2d"]
        if u_modality != "rgb":
            differing_auxiliary_boxes += g_box != u_box
        assert train_g_meta[position]["query_key"] == train_u_meta[position]["query_key"]
        assert train_g_meta[position]["target_modality"] == "rgb"
        assert train_u_meta[position]["target_modality"] == u_modality
    assert differing_auxiliary_boxes > 0

    legacy_gt = _json(output / "fit_legacy181_gt.json")
    canonical_gt = _json(output / "fit_canonical_aux83_gt.json")
    assert len(legacy_gt) == 181 and len(canonical_gt) == 83
    legacy_meta = _jsonl(output / "fit_legacy181_metadata.jsonl")
    canonical_meta = _jsonl(output / "fit_canonical_aux83_metadata.jsonl")
    assert len({row["query_key"] for row in legacy_meta}) == 98
    assert {row["id"] for row in canonical_meta} <= {row["id"] for row in legacy_meta}
    non_quantized = 0
    for sample_id, gt in legacy_gt.items():
        candidate = candidates[sample_id]
        assert gt["bbox"] == candidate["bbox_xyxy_normalized"]
        assert all(field in gt for field in ("visible", "infrared", "depth"))
        non_quantized += any(value * 1000 != round(value * 1000) for value in gt["bbox"])
    assert non_quantized > 0

    city_variants = {name: _json(output / f"city96_{name}.json") for name in ("rgb", "rgb_ir", "rgb_depth", "trimodal")}
    selected_ids = [row["id"] for row in city_variants["rgb"]]
    assert all([row["id"] for row in rows] == selected_ids for rows in city_variants.values())
    city96_gt = _json(output / "city96_gt.json")
    source_city_gt = _json(snapshot / "city_gt.json")
    assert len(city96_gt) == 96 and len({row["image"][0] for row in city_variants["trimodal"]}) == 78
    for sample_id in selected_ids:
        assert city96_gt[sample_id]["bbox"] == source_city_gt[sample_id]["bbox"]
        assert all(field in city96_gt[sample_id] for field in ("visible", "infrared", "depth"))
