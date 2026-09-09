from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.apply_candidate_reviews import apply_candidate_reviews


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_candidate_reviews_export_mutually_exclusive_manifests(tmp_path: Path):
    manifest = tmp_path / "candidates.json"
    write_json(
        manifest,
        {
            "test": {"query": "old", "bbox": [0, 0, 1, 1], "candidate_metadata": {"x": 1}},
            "train": {"query": "keep", "bbox": [0, 0, 1, 1], "candidate_metadata": {"x": 2}},
            "drop": {"query": "drop", "bbox": [0, 0, 1, 1]},
        },
    )
    reviews = tmp_path / "reviews.json"
    write_json(
        reviews,
        {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "reviews": {
                "test": {"decision": "query_corrected", "corrected_query": "new", "destination": "test", "modality_need": "ir"},
                "train": {"decision": "valid", "corrected_query": "valid but corrected", "destination": "train_supplement", "modality_need": "depth"},
                "drop": {"decision": "exclude"},
            },
        },
    )
    test_output, train_output = tmp_path / "test.json", tmp_path / "train.json"

    report = apply_candidate_reviews(manifest, reviews, test_output, train_output, True)
    test = json.loads(test_output.read_text(encoding="utf-8"))
    train = json.loads(train_output.read_text(encoding="utf-8"))

    assert set(test) == {"test"}
    assert set(train) == {"train"}
    assert test["test"]["query"] == "new"
    assert test["test"]["modality_need"] == "ir"
    assert train["train"]["query"] == "valid but corrected"
    assert train["train"]["original_query"] == "keep"
    assert "candidate_metadata" not in test["test"]
    assert report["test_samples"] == report["train_supplement_samples"] == 1


def test_candidate_reviews_can_require_complete_assignment(tmp_path: Path):
    manifest = tmp_path / "candidates.json"
    write_json(manifest, {"sample": {"query": "x"}})
    reviews = tmp_path / "reviews.json"
    write_json(
        reviews,
        {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "reviews": {}},
    )

    with pytest.raises(ValueError, match="incomplete"):
        apply_candidate_reviews(
            manifest, reviews, tmp_path / "test.json", tmp_path / "train.json", True
        )
