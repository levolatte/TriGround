from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DEFAULT_KEEP = {"valid", "query_corrected"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_reviews(
    manifest_path: Path,
    reviews_path: Path,
    output_path: Path,
    report_path: Path,
    keep_decisions: set[str],
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    review_document = json.loads(reviews_path.read_text(encoding="utf-8-sig"))
    reviews = review_document.get("reviews", {})
    manifest_hash = sha256(manifest_path)
    expected_hash = review_document.get("manifest_sha256")
    if expected_hash and expected_hash != manifest_hash:
        raise ValueError("Review file was created for a different manifest")
    if set(reviews) != set(manifest):
        missing = sorted(set(manifest) - set(reviews))
        extra = sorted(set(reviews) - set(manifest))
        raise ValueError(f"Review coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    decisions = Counter(review.get("decision", "unclassified") for review in reviews.values())
    output = {}
    corrected = 0
    for sample_id, record in manifest.items():
        review = reviews[sample_id]
        if review.get("decision") not in keep_decisions:
            continue
        revised = dict(record)
        corrected_query = str(review.get("corrected_query", "")).strip()
        if review.get("decision") == "query_corrected":
            if not corrected_query:
                raise ValueError(f"query_corrected sample {sample_id} has no corrected_query")
            revised["original_query"] = revised["query"]
            revised["query"] = corrected_query
            corrected += 1
        revised["review_decision"] = review["decision"]
        output[sample_id] = revised

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_hash,
        "reviews": str(reviews_path.resolve()),
        "reviews_sha256": sha256(reviews_path),
        "review_updated_at": review_document.get("updated_at"),
        "input_samples": len(manifest),
        "output_samples": len(output),
        "excluded_samples": len(manifest) - len(output),
        "corrected_queries": corrected,
        "keep_decisions": sorted(keep_decisions),
        "decision_counts": dict(sorted(decisions.items())),
        "output_manifest": str(output_path.resolve()),
        "output_manifest_sha256": sha256(output_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual review decisions to a grounding manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--keep", nargs="+", default=sorted(DEFAULT_KEEP))
    args = parser.parse_args()
    output = args.output.resolve()
    report = (args.report or output.with_suffix(".report.json")).resolve()
    result = apply_reviews(
        args.manifest.resolve(),
        args.reviews.resolve(),
        output,
        report,
        set(args.keep),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
