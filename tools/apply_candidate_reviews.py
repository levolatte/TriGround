from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


KEEP = {"valid", "query_corrected"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_candidate_reviews(
    manifest_path: Path,
    reviews_path: Path,
    test_output: Path,
    train_output: Path,
    require_complete: bool = False,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    document = json.loads(reviews_path.read_text(encoding="utf-8-sig"))
    if document.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Review file belongs to a different candidate manifest")
    reviews = document.get("reviews", {})
    outputs: dict[str, dict[str, dict]] = {"test": {}, "train_supplement": {}}
    incomplete = []
    for sample_id, record in manifest.items():
        review = reviews.get(sample_id, {})
        decision = review.get("decision")
        destination = review.get("destination")
        if decision in KEEP and destination not in outputs:
            incomplete.append(sample_id)
            continue
        if decision not in KEEP or destination not in outputs:
            if not decision:
                incomplete.append(sample_id)
            continue
        revised = dict(record)
        revised.pop("candidate_metadata", None)
        corrected = str(review.get("corrected_query", "")).strip()
        if decision == "query_corrected" and not corrected:
            raise ValueError(f"query_corrected sample {sample_id} has no corrected_query")
        if corrected:
            revised["original_query"] = revised["query"]
            revised["query"] = corrected
        revised["review_decision"] = decision
        revised["modality_need"] = review.get("modality_need", "uncertain")
        revised["review_note"] = review.get("note", "")
        outputs[destination][sample_id] = revised
    if require_complete and incomplete:
        raise ValueError(f"{len(incomplete)} candidate reviews are incomplete")
    for path, key in ((test_output, "test"), (train_output, "train_supplement")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(outputs[key], ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "candidate_manifest": str(manifest_path.resolve()),
        "candidate_manifest_sha256": sha256(manifest_path),
        "reviews": str(reviews_path.resolve()),
        "reviewed": len(reviews),
        "incomplete": len(incomplete),
        "test_samples": len(outputs["test"]),
        "train_supplement_samples": len(outputs["train_supplement"]),
        "destinations": dict(sorted(Counter(r.get("destination", "unset") for r in reviews.values()).items())),
        "modality_needs": dict(sorted(Counter(r.get("modality_need", "unset") for r in reviews.values()).items())),
        "test_output": str(test_output.resolve()),
        "train_output": str(train_output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reviewed candidates into test and supplemental train manifests")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = apply_candidate_reviews(
        args.manifest.resolve(), args.reviews.resolve(),
        args.test_output.resolve(), args.train_output.resolve(), args.require_complete,
    )
    report_path = (args.report or args.test_output.with_suffix(".report.json")).resolve()
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
