from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.prepare_rgbdt_annotation_jobs import (
    merge_reviews,
    prepare_generator,
    prepare_human_pack,
    prepare_reviewer,
)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _raw_fixture(tmp_path: Path, count: int = 10) -> tuple[Path, Path]:
    root = tmp_path / "RGBDT500"
    root.mkdir()
    rows = []
    for index in range(count):
        image_dir = root / f"pilot/{index:03d}"
        image_dir.mkdir(parents=True)
        for modality in ("color", "infrared", "depth"):
            if modality == "depth":
                Image.fromarray(np.full((10, 20), 1000 + index, dtype=np.uint16)).save(image_dir / "depth.png")
            else:
                Image.fromarray(np.full((10, 20, 3), index, dtype=np.uint8)).save(image_dir / f"{modality}.png")
        rows.append({
            "id": f"rgbdt500_{index:03d}_0001", "source": "rgbdt500", "split": "pilot",
            "sequence": f"{index:03d}",
            "rgb": f"pilot/{index:03d}/color.png",
            "infrared": f"pilot/{index:03d}/infrared.png",
            "depth": f"pilot/{index:03d}/depth.png",
            "bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.7],
            "depth_units": "unverified", "depth_mode": "I;16", "query": None,
        })
    manifest = root / "pilot_manifest.jsonl"
    _jsonl(manifest, rows)
    return root, manifest


def test_generator_batches_ten_and_blind_reviewer_has_no_target_information(tmp_path: Path) -> None:
    root, manifest = _raw_fixture(tmp_path)
    generator_dir = tmp_path / "generator"
    report = prepare_generator(manifest, root, generator_dir)
    assert report["groups"] == 10 and report["batches"] == 1
    batch = json.loads((generator_dir / "generator_batch_000.json").read_text(encoding="utf-8"))
    assert len(batch["jobs"]) == 10
    assert "target_bbox_xyxy_normalized" in batch["jobs"][0]
    assert Path(batch["jobs"][0]["images"]["depth_preview"]).is_file()
    generated = tmp_path / "generated.jsonl"
    _jsonl(generated, [{
        "query_id": "q1", "image_group_id": batch["jobs"][0]["image_group_id"],
        "query": "the small item on the left", "focus": "appearance",
        "generation_status": "proposed", "rewrite_count": 0,
        "evidence": {"visible": "small item on the left"},
    }])
    reviewer_dir = tmp_path / "reviewer"
    reviewer_report = prepare_reviewer(manifest, root, generated, reviewer_dir)
    assert reviewer_report["questions"] == 1
    blind = json.loads((reviewer_dir / "blind_batch_000.json").read_text(encoding="utf-8"))
    assert set(blind["jobs"][0]) == {"query_id", "images", "query"}
    serialized = json.dumps(blind)
    assert "target_bbox" not in serialized
    assert "generation_evidence" not in serialized
    assert "0.1, 0.2, 0.4, 0.7" not in serialized


def test_merge_requires_all_three_checks_and_never_human_approves(tmp_path: Path) -> None:
    root, manifest = _raw_fixture(tmp_path, 2)
    groups = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    generated = tmp_path / "generated.jsonl"
    _jsonl(generated, [
        {"query_id": "good", "image_group_id": groups[0]["id"], "query": "left item",
         "focus": "appearance", "generation_status": "proposed", "rewrite_count": 1,
         "evidence": {"visible": "left item"}},
        {"query_id": "ambiguous", "image_group_id": groups[1]["id"], "query": "item",
         "focus": "instance_ordinal", "generation_status": "proposed", "rewrite_count": 0,
         "evidence": {"visible": "two items"}},
    ])
    reviews = tmp_path / "reviews.jsonl"
    _jsonl(reviews, [
        {"query_id": "good", "predicted_bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.7],
         "ambiguous": False, "evidence_confirmed": True},
        {"query_id": "ambiguous", "predicted_bbox_xyxy_normalized": [0.1, 0.2, 0.4, 0.7],
         "ambiguous": True, "evidence_confirmed": True},
    ])
    output_dir = tmp_path / "joined"
    summary = merge_reviews(manifest, root, generated, reviews, output_dir)
    assert summary["provisional_train_questions"] == 1
    assert summary["human_approved_questions"] == 0
    joined = [json.loads(line) for line in (output_dir / "joined_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["review_status"] for row in joined] == ["provisional", "needs_review"]
    training = [json.loads(line) for line in (output_dir / "provisional_train_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert training[0]["review_status"] == "provisional" and training[0]["depth_policy"] == "sensor_linear_20000"
    assert training[0]["bbox"] == groups[0]["bbox_xyxy_normalized"]


def test_generation_allows_only_one_rewrite_and_human_pack_needs_real_rows(tmp_path: Path) -> None:
    root, manifest = _raw_fixture(tmp_path, 1)
    group = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    generated = tmp_path / "generated.jsonl"
    _jsonl(generated, [{
        "query_id": "q1", "image_group_id": group["id"], "query": "item",
        "focus": "appearance", "generation_status": "proposed", "rewrite_count": 2,
        "evidence": {"visible": "item"},
    }])
    with pytest.raises(ValueError, match="at most one rewrite"):
        prepare_reviewer(manifest, root, generated, tmp_path / "blind")
    pilot_joined = tmp_path / "pilot_joined.jsonl"
    external_joined = tmp_path / "external_joined.jsonl"
    _jsonl(pilot_joined, [])
    _jsonl(external_joined, [])
    with pytest.raises(ValueError, match="25 provisional"):
        prepare_human_pack(pilot_joined, external_joined, tmp_path / "human")


def test_actual_generator_aliases_stay_unreviewed_and_blind(tmp_path: Path) -> None:
    root, manifest = _raw_fixture(tmp_path, 2)
    groups = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    generated = tmp_path / "generated.jsonl"
    _jsonl(generated, [
        {"query_id": "g1", "image_group_id": groups[0]["id"], "query": "The left item.",
         "focus": "instance", "generation_status": "generated", "rewrite_count": 0,
         "evidence": "Visible in RGB."},
        {"query_id": "g2", "image_group_id": groups[1]["id"], "query": "",
         "focus": "depth", "generation_status": "skipped", "rewrite_count": 0,
         "evidence": "Depth units unknown."},
    ])
    output = tmp_path / "blind"
    report = prepare_reviewer(manifest, root, generated, output)
    assert report["questions"] == 1
    batch = json.loads((output / "blind_batch_000.json").read_text(encoding="utf-8"))
    assert batch["jobs"][0]["query_id"] == "g1"
    assert "evidence" not in batch["jobs"][0]
    assert "focus" not in batch["jobs"][0]


def test_human_pack_uses_100_pilot_and_two_queries_per_external_group(tmp_path: Path) -> None:
    foci = ("appearance", "instance_ordinal", "depth_relation", "infrared_evidence")
    common = {
        "query": "a visible item", "rgb": str(tmp_path / "rgb.png"),
        "infrared": str(tmp_path / "ir.png"), "depth": str(tmp_path / "depth.png"),
        "bbox": [0.1, 0.2, 0.4, 0.7], "blind_iou": 0.9,
    }
    pilot = [{**common, "query_id": f"p-{focus}-{i}", "image_group_id": f"pilot-{i}",
              "split": "pilot", "focus": focus, "review_status": "provisional"}
             for focus in foci for i in range(25)]
    external = [{**common, "query_id": f"e-{group}-{question}",
                 "image_group_id": f"external-{group}", "split": "external_review",
                 "focus": "appearance", "review_status": "needs_review"}
                for group in range(50) for question in range(2)]
    pilot_path, external_path = tmp_path / "pilot.jsonl", tmp_path / "external.jsonl"
    _jsonl(pilot_path, pilot)
    _jsonl(external_path, external)
    output = tmp_path / "human"
    result = prepare_human_pack(pilot_path, external_path, output)
    assert result["total"] == 200
    csv_lines = (output / "human_review_200.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(csv_lines) == 201
    page = (output / "human_review_200.html").read_text(encoding="utf-8")
    assert page.count("<article>") == 200
    assert "class=\"mark\"" in page
