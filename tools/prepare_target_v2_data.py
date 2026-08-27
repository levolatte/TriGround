from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


REJECTED = {
    "exclude",
    "ambiguous",
    "wrong_bbox",
    "modality_misaligned",
    "tiny_unusable",
}
PATH_FIELDS = ("rgb", "visible", "ir", "infrared", "thermal", "depth", "aux")


def load_json(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object keyed by sample id")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebase_record(record: dict, source: Path, output: Path) -> dict:
    revised = dict(record)
    for field in PATH_FIELDS:
        value = revised.get(field)
        if not value:
            continue
        path = Path(value)
        resolved = path if path.is_absolute() else (source.parent / path).resolve()
        revised[field] = Path(os.path.relpath(resolved, output.parent)).as_posix()
    return revised


def visible_path(record: dict, manifest: Path) -> Path:
    value = record.get("visible") or record.get("rgb")
    if not value:
        raise KeyError("record has neither visible nor rgb path")
    path = Path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def sequence_key(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) <= 3:
        parts = parts[:-1]
    if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) >= 6:
        parts = parts[:-1]
    return "_".join(parts)


def reviewed_candidates(
    manifest_path: Path, reviews_path: Path
) -> tuple[dict[str, dict], dict[str, dict], dict]:
    manifest = load_json(manifest_path)
    document = json.loads(reviews_path.read_text(encoding="utf-8-sig"))
    if document.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("review file belongs to a different candidate manifest")
    reviews = document.get("reviews", {})
    test: dict[str, dict] = {}
    train: dict[str, dict] = {}
    decisions: Counter[str] = Counter()
    destinations: Counter[str] = Counter()
    assumed_valid = []
    query_only = []
    rejected = []
    for sample_id, source_record in manifest.items():
        review = reviews.get(sample_id, {})
        decision = review.get("decision")
        corrected = str(review.get("corrected_query", "")).strip()
        if not decision:
            if corrected:
                decision = "query_corrected"
                query_only.append(sample_id)
            else:
                decision = "valid"
                assumed_valid.append(sample_id)
        decisions[decision] += 1
        if decision in REJECTED:
            rejected.append(sample_id)
            continue
        if decision not in {"valid", "query_corrected"}:
            raise ValueError(f"unsupported decision {decision!r} for {sample_id}")
        record = dict(source_record)
        record.pop("candidate_metadata", None)
        if corrected:
            record["original_query"] = record["query"]
            record["query"] = corrected
        record["review_decision"] = decision
        record["label_source"] = "manual_review_v2"
        record["weak_label"] = False
        # Modality-need annotations from this review round are deliberately
        # excluded: they are not supervision and must not affect sampling.
        record.pop("modality_need", None)
        destination = review.get("destination")
        if destination == "test":
            test[sample_id] = record
            destinations["test"] += 1
        elif destination in {None, "", "hold", "train_supplement"}:
            train[sample_id] = record
            destinations["train"] += 1
        else:
            raise ValueError(f"unsupported destination {destination!r} for {sample_id}")
    return test, train, {
        "candidate_samples": len(manifest),
        "usable_samples": len(test) + len(train),
        "rejected_samples": len(rejected),
        "decision_counts": dict(sorted(decisions.items())),
        "destination_counts": dict(sorted(destinations.items())),
        "query_only_corrections": len(query_only),
        "assumed_valid_without_correction": assumed_valid,
        "rejected_ids": rejected,
        "modality_need_used": False,
    }


def add_source(
    target: dict[str, dict], source: dict[str, dict], source_path: Path, label: str
) -> None:
    collisions = set(target) & set(source)
    if collisions:
        raise ValueError(f"sample-id collision in {label}: {sorted(collisions)[:5]}")
    for sample_id, record in source.items():
        revised = dict(record)
        revised.setdefault("label_source", label)
        target[sample_id] = (revised, source_path)


def write_rebased(
    output: Path, records: dict[str, tuple[dict, Path]]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        sample_id: rebase_record(record, source_path, output)
        for sample_id, (record, source_path) in records.items()
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build target-v2 test and mixed manual/weak training manifests"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-data-root", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--previous-manual-train", type=Path, required=True)
    parser.add_argument("--previous-manual-test", type=Path, required=True)
    parser.add_argument("--weak-train", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--new-test-output", type=Path, required=True)
    parser.add_argument("--manual-train-output", type=Path, required=True)
    parser.add_argument("--weak-train-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        name: value.resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    test_new, train_new, review_report = reviewed_candidates(
        paths["candidates"], paths["reviews"]
    )
    candidate_source = paths["candidate_data_root"] / "__candidate_source__.json"
    train_records: dict[str, tuple[dict, Path]] = {}
    add_source(
        train_records,
        load_json(paths["previous_manual_train"]),
        paths["previous_manual_train"],
        "manual_review_v1",
    )
    add_source(
        train_records,
        train_new,
        candidate_source,
        "manual_review_v2",
    )
    add_source(
        train_records,
        load_json(paths["weak_train"]),
        paths["weak_train"],
        "weak_supervision",
    )
    new_test_records = {
        sample_id: (record, candidate_source)
        for sample_id, record in test_new.items()
    }
    test_records: dict[str, tuple[dict, Path]] = {}
    add_source(
        test_records,
        load_json(paths["previous_manual_test"]),
        paths["previous_manual_test"],
        "manual_review_v1_test",
    )
    add_source(
        test_records,
        test_new,
        candidate_source,
        "manual_review_v2_test",
    )

    # A different frame from the same sequence is still leakage for a visual
    # grounding test.  Keep the manually reviewed test destination and remove
    # matching training-sequence rows, without modifying their source files.
    test_sequences = {sequence_key(sample_id) for sample_id in test_records}
    sequence_overlap_removed = {
        sample_id: record.get("label_source", "unknown")
        for sample_id, (record, _) in train_records.items()
        if sequence_key(sample_id) in test_sequences
    }
    for sample_id in sequence_overlap_removed:
        del train_records[sample_id]

    train_visible = {
        visible_path(record, source): sample_id
        for sample_id, (record, source) in train_records.items()
    }
    test_visible = {
        visible_path(record, source): sample_id
        for sample_id, (record, source) in test_records.items()
    }
    exact_overlap = sorted(str(path) for path in set(train_visible) & set(test_visible))
    if exact_overlap:
        raise ValueError(f"exact train/test image overlap: {exact_overlap[:5]}")
    train_sequences = {sequence_key(sample_id) for sample_id in train_records}
    sequence_overlap = sorted(train_sequences & test_sequences)
    if sequence_overlap:
        raise RuntimeError("sequence-overlap filtering failed")
    for sample_id, (record, _) in train_records.items():
        record["sequence_id"] = sequence_key(sample_id)
    for sample_id, (record, _) in test_records.items():
        record["sequence_id"] = sequence_key(sample_id)
    for sample_id, (record, _) in new_test_records.items():
        record["sequence_id"] = sequence_key(sample_id)

    write_rebased(paths["train_output"], train_records)
    write_rebased(paths["test_output"], test_records)
    write_rebased(paths["new_test_output"], new_test_records)
    manual_train_records = {
        sample_id: value
        for sample_id, value in train_records.items()
        if value[0].get("label_source") != "weak_supervision"
    }
    weak_train_records = {
        sample_id: value
        for sample_id, value in train_records.items()
        if value[0].get("label_source") == "weak_supervision"
    }
    write_rebased(paths["manual_train_output"], manual_train_records)
    write_rebased(paths["weak_train_output"], weak_train_records)
    sources = Counter(
        record.get("label_source", "unknown") for record, _ in train_records.values()
    )
    report = {
        **review_report,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "new_test_samples": len(new_test_records),
        "previous_test_samples": len(test_records) - len(new_test_records),
        "manual_train_samples": len(manual_train_records),
        "weak_train_samples": len(weak_train_records),
        "train_sources": dict(sorted(sources.items())),
        "exact_train_test_image_overlap": 0,
        "sequence_overlap_removed_count": len(sequence_overlap_removed),
        "sequence_overlap_removed_sources": dict(
            sorted(Counter(sequence_overlap_removed.values()).items())
        ),
        "sequence_overlap_removed_ids": sorted(sequence_overlap_removed),
        "remaining_sequence_overlap_count": 0,
        "train_output": str(paths["train_output"]),
        "test_output": str(paths["test_output"]),
        "new_test_output": str(paths["new_test_output"]),
        "manual_train_output": str(paths["manual_train_output"]),
        "weak_train_output": str(paths["weak_train_output"]),
        "reviews_sha256": sha256(paths["reviews"]),
        "weak_labels_modified": False,
    }
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
