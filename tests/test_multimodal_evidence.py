import copy
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.prepare_multimodal_evidence import prepare_review, merge_review, export_rows, read_rows, attach_city_tasks
from tools.prepare_next_stage_data import (
    QUOTAS, build_schedule, _native_sample, _select_joint_tasks,
    _select_rgb_augmentations, _augment_rgb, prepare,
)
from tools.render_object_evidence_review import render


def bundle(tmp_path):
    images = {}
    for m in ("rgb", "infrared", "depth"):
        p = tmp_path / f"{m}.png"
        Image.fromarray(np.full((30, 40, 3), 180, dtype=np.uint8)).save(p)
        images[m] = str(p)
    return {"id": "scene1", "source": "rgbdt", "scene_id": "seq1", "split": "train",
            "images": images, "depth_policy": "visual", "review_status": "provisional",
            "objects": [{"object_id": "target", "extent": "whole",
                         "boxes": {"rgb": [.1, .2, .3, .4], "infrared": [.4, .2, .6, .4], "depth": None}}],
            "queries": [{"id": "q1", "query": "the standing person", "target_object_id": "target"}]}


def test_blind_jobs_hide_proposed_boxes_and_keep_coordinate_target(tmp_path):
    b = bundle(tmp_path)
    result = prepare_review([b], tmp_path / "review")
    text = Path(result["safe_task"]).read_text()
    jobs = json.loads(text)["jobs"]
    assert len(jobs) == 2
    assert [j["output_modality"] for j in jobs] == ["rgb", "infrared"]
    assert all(set(j) == {"query_id", "images", "query", "output_modality"} for j in jobs)
    assert "proposed_bbox" not in text and "semantic_label" not in text
    index = read_rows(tmp_path / "review/private_index.jsonl")
    reviews = [{"query_id": c["id"], "predicted_bbox_xyxy_normalized": c["proposed_bbox"],
                "ambiguous": False, "evidence_confirmed": True} for c in index]
    merged, summary = merge_review([b], index, reviews)
    assert summary["blind_passed"] == 2 and summary["human_approved"] == 0
    with pytest.raises(ValueError, match="acceptance"):
        export_rows(merged, {})
    approvals = {"scene1": {"status": "approved_batch", "record": "human_review.csv", "tasks": ["bbox", "cross_bbox"]}}
    rows = export_rows(merged, approvals)
    assert rows[0]["bbox"] == [.1, .2, .3, .4]
    assert rows[0]["task_pool"][0]["bbox"] == [.4, .2, .6, .4]
    assert rows[0]["augmentation_eligible"] is False


def test_missing_modality_not_copied_and_external_never_exported(tmp_path):
    b = bundle(tmp_path)
    b["objects"][0]["boxes"]["infrared"] = None
    result = prepare_review([b], tmp_path / "review")
    assert result["review_cases"] == 1
    b["split"] = "external_review"
    with pytest.raises(ValueError, match="external"):
        export_rows([b], {}, preview=True)


def test_candidate_derivation_and_city_attachment_keep_original_labels(tmp_path):
    b = bundle(tmp_path)
    b["source"] = "city"
    b["objects"][0]["confirmed_modalities"] = ["rgb"]
    b["objects"].append({"object_id": "other", "extent": "whole", "confirmed_modalities": ["rgb"],
                         "boxes": {"rgb": [.6, .2, .9, .4], "infrared": None, "depth": None}})
    b["queries"][0].update(review_status="human_accepted", origin_query_id="original1")
    approval = {"scene1": {"status": "approved_batch", "record": "review.csv", "tasks": ["bbox", "relation"]}}
    rows = export_rows([b], approval)
    assert len(rows[0]["task_pool"]) == 1 and rows[0]["task_pool"][0]["answer_object_id"] == "target"
    native = [{"id": "original1", "image": ["r", "i", "d"], "conversations": [{"value": "unchanged query"}, {"value": "unchanged answer"}]}]
    out = attach_city_tasks(native, rows)
    assert out[0]["conversations"] == native[0]["conversations"] and out[0]["image"] == native[0]["image"]
    assert "task_pool" not in native[0]
    with pytest.raises(ValueError, match="duplicate"):
        attach_city_tasks(native, rows + rows)


def test_duplicate_or_foreign_review_cannot_be_attached_to_wrong_image(tmp_path):
    b = bundle(tmp_path)
    prepare_review([b], tmp_path / "review")
    index = read_rows(tmp_path / "review/private_index.jsonl")
    with pytest.raises(ValueError, match="unknown"):
        merge_review([b], index, [{"query_id": "other_scene::q1::rgb"}])
    with pytest.raises(ValueError, match="duplicate"):
        merge_review([b], index, [{"query_id": index[0]["id"]}] * 2)


def test_human_packet_keeps_originals_and_does_not_overwrite_decisions(tmp_path):
    b = bundle(tmp_path)
    out = tmp_path / "human"
    result = render([b], out)
    assert result["human_accepted"] == 0 and result["queries"] == 1
    assert result["review_items"] == 3  # RGB, independently proposed IR, Query.
    assert len(list((out / "images").glob("*original.jpg"))) == 3
    assert "scene1::target::depth" not in (out / "decisions.csv").read_text(encoding="utf-8-sig")
    (out / "decisions.csv").write_text("human work", encoding="utf-8")
    with pytest.raises(FileExistsError):
        render([b], out)
    assert (out / "decisions.csv").read_text() == "human work"


def test_real_g_u_materialization_shares_scenes_and_augmentation(tmp_path):
    b = bundle(tmp_path)
    city = [{"id": "city1", "image": list(b["images"].values()), "conversations": [
        {"from": "human", "value": "<image>\n<image>\n<image>\nFind the person"},
        {"from": "gpt", "value": '{"bbox_2d":[100,200,300,400]}'}]}]
    city_path = tmp_path / "city.json"
    city_path.write_text(json.dumps(city))
    base = {"id": "sample", "query": "the person", "bbox": [.1, .2, .3, .4],
            "scene_id": "scene", "split": "train", "depth_policy": "visual",
            "review_status": "approved_batch", "augmentation_eligible": True,
            "augmentation_review_status": "approved_batch", **b["images"],
            "task_pool": [{"id": "cross", "type": "cross_bbox", "query": "the person",
                           "target_modality": "infrared", "bbox": [.4, .2, .6, .4],
                           "review_status": "approved_batch"}]}
    paths = {}
    for source in ("rgbdt", "rgbt", "robo"):
        row = copy.deepcopy(base)
        if source == "rgbt":
            row.pop("depth")
        if source == "robo":
            row.pop("infrared")
            row["task_pool"] = []
        paths[source] = tmp_path / f"{source}.jsonl"
        paths[source].write_text(json.dumps(row) + "\n")
    metadata = {}
    for branch in ("G", "U"):
        out = tmp_path / branch
        report = prepare(argparse.Namespace(city_native=city_path, city_root=tmp_path, **paths,
                                             robo_fallback_rgbdt=False, branch=branch, phase=1,
                                             seed=2026, output_dir=out, augmentation_fraction=.2))
        assert report["presentations"] == 8000
        assert report["actual_sources"] == QUOTAS[(branch, 1)]
        assert report["augmentation_fraction_actual"] <= .2
        metadata[branch] = [json.loads(s) for s in (out / f"{branch.lower()}_phase1_metadata.jsonl").read_text().splitlines()]
        if branch == "G":
            assert report["tasks"] == {"bbox": 8000}
        else:
            assert report["tasks"]["cross_bbox"] == 1200
            assert report["auxiliary_shortfall_to_bbox"]["relation"] == 2000
    assert [(r["source"], r["source_id"], r["augmentation"]) for r in metadata["G"]] == [
        (r["source"], r["source_id"], r["augmentation"]) for r in metadata["U"]]
    assert all(r["augmentation"] is None for r in metadata["U"] if r["task"] != "bbox")
    # Preparing a later phase in the same manifest directory cannot change any
    # pixels referenced by an earlier phase (sequential cache names once did).
    earlier_images = {p: p.read_bytes() for p in (tmp_path / "G").rglob("*.png")}
    assert earlier_images
    prepare(argparse.Namespace(city_native=city_path, city_root=tmp_path, **paths,
                               robo_fallback_rgbdt=False, branch="G", phase=2,
                               seed=2026, output_dir=tmp_path / "G", augmentation_fraction=.2))
    assert all(p.read_bytes() == pixels for p, pixels in earlier_images.items())


def native_row(tmp_path):
    b = bundle(tmp_path)
    return {"id": "q1", "source": "rgbdt", "group_id": "seq1", "query": "person",
            "bbox": [.1, .2, .3, .4], "depth_policy": "visual",
            **{f"{m}_path": Path(p) for m, p in b["images"].items()}}


def test_cross_modal_output_uses_ir_not_rgb_box(tmp_path):
    row = native_row(tmp_path)
    task = {"id": "ir1", "type": "cross_bbox", "target_modality": "infrared",
            "query": "person", "bbox": [.4, .2, .6, .4], "review_status": "approved_batch"}
    sample, metadata = _native_sample(row, 0, tmp_path, {}, "cross_bbox", random.Random(1), task)
    assert json.loads(sample["conversations"][1]["value"])["bbox_2d"] == [400, 200, 600, 400]
    assert "infrared image" in sample["conversations"][0]["value"]
    assert metadata["target_modality"] == "infrared"


def test_relation_number_shuffle_tracks_answer_and_hides_object_ids(tmp_path):
    row = native_row(tmp_path)
    task = {"id": "relation1", "type": "relation", "query": "the target beside the post",
            "review_status": "approved_batch", "answer_object_id": "secret-positive",
            "candidates": [{"object_id": "secret-positive", "bbox": [.1, .2, .3, .4]},
                           {"object_id": "secret-negative", "bbox": [.6, .6, .8, .8]}]}
    answers = set()
    for seed in range(10):
        sample, _ = _native_sample(row, seed, tmp_path, {}, "relation", random.Random(seed), task)
        prompt = sample["conversations"][0]["value"]
        answer = sample["conversations"][1]["value"]
        assert f"{answer}: [100, 200, 300, 400]" in prompt
        assert "secret" not in prompt
        answers.add(answer)
    assert answers == {"A", "B"}


def test_g_u_schedule_matches_and_auxiliary_shortage_stays_bbox(tmp_path):
    base = native_row(tmp_path)
    sources = {s: [{**base, "id": s, "source": s}] for s in ("city", "rgbdt", "rgbt", "robo")}
    g = build_schedule(sources, QUOTAS[("G", 1)], 2026)
    u = build_schedule(sources, QUOTAS[("U", 1)], 2026)
    assert [(r["source"], r["id"]) for r in g] == [(r["source"], r["id"]) for r in u]
    assert _select_joint_tasks(u, 2026) == {}
    both = [{**base, "task_pool": [
        {"type": "relation", "review_status": "approved_batch"},
        {"type": "cross_bbox", "review_status": "approved_batch"}]} for _ in range(100)]
    assigned = _select_joint_tasks(both, 2026)
    assert sum(t["type"] == "relation" for t in assigned.values()) == 25
    assert sum(t["type"] == "cross_bbox" for t in assigned.values()) == 15


def test_augmentation_is_shared_capped_and_preserves_original_and_auxiliary(tmp_path):
    row = native_row(tmp_path)
    row.update(augmentation_eligible=True, augmentation_review_status="approved_batch")
    schedule = [copy.copy(row) for _ in range(100)]
    first = _select_rgb_augmentations(schedule, 2026, .2)
    assert first == _select_rgb_augmentations(schedule, 2026, .2) and len(first) == 20
    assert _select_rgb_augmentations([row], 2026, .2) == {}
    before = Path(row["infrared_path"]).read_bytes()
    original = Path(row["rgb_path"]).read_bytes()
    augmented = _augment_rgb(row["rgb_path"], ("brightness", .35), tmp_path / "out", {})
    assert Image.open(augmented).size == Image.open(row["rgb_path"]).size
    assert np.array(Image.open(augmented)).mean() < np.array(Image.open(row["rgb_path"])).mean()
    assert Path(row["rgb_path"]).read_bytes() == original
    assert Path(row["infrared_path"]).read_bytes() == before
    with pytest.raises(ValueError, match="0.2"):
        _select_rgb_augmentations(schedule, 2026, .5)
