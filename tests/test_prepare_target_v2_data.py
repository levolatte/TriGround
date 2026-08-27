from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.prepare_target_v2_data import reviewed_candidates


def write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_query_only_reviews_are_kept_and_hold_becomes_train(tmp_path: Path):
    manifest = tmp_path / "candidates.json"
    write(
        manifest,
        {
            "query_only": {"query": "old", "visible": "a.png"},
            "hold": {"query": "good", "visible": "b.png"},
            "drop": {"query": "bad", "visible": "c.png"},
        },
    )
    reviews = tmp_path / "reviews.json"
    write(
        reviews,
        {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "reviews": {
                "query_only": {
                    "corrected_query": "new",
                    "destination": "test",
                    "modality_need": "depth",
                },
                "hold": {"decision": "valid", "destination": "hold"},
                "drop": {"decision": "wrong_bbox", "destination": "test"},
            },
        },
    )
    test, train, report = reviewed_candidates(manifest, reviews)
    assert set(test) == {"query_only"}
    assert set(train) == {"hold"}
    assert test["query_only"]["query"] == "new"
    assert test["query_only"]["original_query"] == "old"
    assert "modality_need" not in test["query_only"]
    assert not test["query_only"]["weak_label"]
    assert report["query_only_corrections"] == 1
    assert report["rejected_samples"] == 1
