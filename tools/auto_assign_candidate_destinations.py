from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from tools.review_grounding import ReviewStore
    from tools.select_scene_coverage_candidates import greedy_coverage_select
except ModuleNotFoundError:  # Support direct execution as tools/script.py.
    from review_grounding import ReviewStore
    from select_scene_coverage_candidates import greedy_coverage_select


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auto_assign(
    manifest_path: Path,
    reviews_path: Path,
    test_size: int,
    train_size: int,
    seed: int,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if test_size < 0 or train_size < 0 or test_size + train_size > len(manifest):
        raise ValueError("invalid test/train destination sizes")
    store = ReviewStore(reviews_path, manifest_sha256(manifest_path))
    reviews = store.data["reviews"]
    valid_destinations = {"test", "train_supplement", "hold"}
    existing = {
        sample_id: review["destination"]
        for sample_id, review in reviews.items()
        if review.get("destination") in valid_destinations
    }
    existing_counts = Counter(existing.values())
    if existing_counts["test"] > test_size or existing_counts["train_supplement"] > train_size:
        raise ValueError("existing manual destinations exceed requested target sizes")

    available = [
        (sample_id, record)
        for sample_id, record in manifest.items()
        if sample_id not in existing
    ]
    test_needed = test_size - existing_counts["test"]
    selected_test = greedy_coverage_select(available, test_needed, seed) if test_needed else []
    test_ids = {sample_id for sample_id, _ in selected_test}
    remaining = [item for item in available if item[0] not in test_ids]
    train_needed = train_size - existing_counts["train_supplement"]
    selected_train = (
        greedy_coverage_select(remaining, train_needed, seed + 1) if train_needed else []
    )
    train_ids = {sample_id for sample_id, _ in selected_train}
    hold_ids = {sample_id for sample_id, _ in remaining if sample_id not in train_ids}

    if test_ids:
        store.update(sorted(test_ids), {"destination": "test"})
    if train_ids:
        store.update(sorted(train_ids), {"destination": "train_supplement"})
    if hold_ids:
        store.update(sorted(hold_ids), {"destination": "hold"})

    final_reviews = store.data["reviews"]
    counts = Counter(review.get("destination", "unset") for review in final_reviews.values())
    report = {
        "manifest": str(manifest_path.resolve()),
        "reviews": str(reviews_path.resolve()),
        "seed": seed,
        "samples": len(manifest),
        "preserved_manual_destinations": len(existing),
        "preserved_quality_decisions": sum(bool(review.get("decision")) for review in final_reviews.values()),
        "destination_counts": dict(sorted(counts.items())),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fill balanced candidate destinations")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--test-size", type=int, default=180)
    parser.add_argument("--train-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = auto_assign(
        args.manifest.resolve(), args.reviews.resolve(),
        args.test_size, args.train_size, args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
