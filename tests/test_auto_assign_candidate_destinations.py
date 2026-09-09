from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.auto_assign_candidate_destinations import auto_assign


def candidate(index: int) -> dict:
    return {
        "query": f"candidate {index}",
        "class_name": "person" if index % 2 else "car",
        "scale_bin": "large" if index % 3 else "small",
        "candidate_metadata": {
            "scene_key": f"scene-{index}",
            "query_type": "other",
            "light_bin": ["low", "mid", "high"][index % 3],
            "rgb_contrast_bin": "mid",
            "ir_contrast_bin": "mid",
            "horizontal_bin": ["left", "center", "right"][index % 3],
            "vertical_bin": "middle",
            "source_family": "numeric",
        },
    }


def test_auto_assignment_preserves_quality_reviews_and_is_mutually_exclusive(tmp_path: Path):
    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps({f"sample-{i}": candidate(i) for i in range(12)}), encoding="utf-8"
    )
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "reviews": {"sample-0": {"decision": "valid"}},
            }
        ),
        encoding="utf-8",
    )

    report = auto_assign(manifest, reviews, test_size=7, train_size=3, seed=2026)
    saved = json.loads(reviews.read_text(encoding="utf-8"))["reviews"]

    assert saved["sample-0"]["decision"] == "valid"
    assert sum(row["destination"] == "test" for row in saved.values()) == 7
    assert sum(row["destination"] == "train_supplement" for row in saved.values()) == 3
    assert sum(row["destination"] == "hold" for row in saved.values()) == 2
    assert report["preserved_quality_decisions"] == 1
