from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.apply_grounding_reviews import DEFAULT_KEEP, apply_reviews


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_apply_reviews_keeps_valid_and_replaces_corrected_query(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "valid": {"query": "original valid", "bbox": [0, 0, 1, 1]},
            "fixed": {"query": "bad description", "bbox": [0, 0, 1, 1]},
            "excluded": {"query": "exclude me", "bbox": [0, 0, 1, 1]},
        },
    )
    reviews = tmp_path / "reviews.json"
    write_json(
        reviews,
        {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "reviews": {
                "valid": {"decision": "valid"},
                "fixed": {
                    "decision": "query_corrected",
                    "corrected_query": "correct description",
                },
                "excluded": {"decision": "exclude"},
            },
        },
    )
    output = tmp_path / "clean.json"
    report_path = tmp_path / "report.json"

    report = apply_reviews(manifest, reviews, output, report_path, DEFAULT_KEEP)
    cleaned = json.loads(output.read_text(encoding="utf-8"))

    assert set(cleaned) == {"valid", "fixed"}
    assert cleaned["fixed"]["query"] == "correct description"
    assert cleaned["fixed"]["original_query"] == "bad description"
    assert report["output_samples"] == 2
    assert report["corrected_queries"] == 1


def test_apply_reviews_rejects_incomplete_coverage(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    reviews = tmp_path / "reviews.json"
    write_json(manifest, {"sample": {"query": "query"}})
    write_json(reviews, {"reviews": {}})

    with pytest.raises(ValueError, match="coverage mismatch"):
        apply_reviews(manifest, reviews, tmp_path / "out.json", tmp_path / "report.json", DEFAULT_KEEP)
