"""Merge approved TriGround review packages into one release input.

The four source packages remain untouched. This command only prepares reviewed
candidate and decision files; the separate ``release`` command makes manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.prepare_triground_abv_data import (
    DECISION_FIELDS,
    ROOT,
    _atomic_groups,
    _requires_pair,
    accepted_candidates,
    ancestor_overlap,
    read_jsonl,
    write_jsonl,
)


DEFAULT_PACKAGES = [
    ROOT / "results/triground_abv_depthv2_20260928",
    ROOT / "results/triground_review_repairs_20260928",
    ROOT / "results/triground_depth_supplement_20260928",
    ROOT / "results/triground_depth_expansion_20260928",
]
M3FD_ROLE_CORRECTION = "rgbt_m3fd_train_002847:reliability"
SOURCE_FIELDS = ("task_id", "category", "split", "bundle_id", "scene_id",
                 "source_id", "original_query", "proposed_query")


def _read_package(package: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    candidates_path = package / "data/candidates.jsonl"
    decisions_path = package / "review/decisions.csv"
    candidates = read_jsonl(candidates_path)
    with decisions_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != DECISION_FIELDS:
            raise ValueError(f"unexpected decision columns in {decisions_path}")
        decisions = list(reader)
    by_id = {row["task_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError(f"duplicate candidate task ID in {candidates_path}")
    decision_ids = [row["task_id"] for row in decisions]
    if len(set(decision_ids)) != len(decision_ids) or set(decision_ids) != set(by_id):
        raise ValueError(f"decision IDs do not match candidates in {package}")
    for decision in decisions:
        candidate = by_id[decision["task_id"]]
        for field in SOURCE_FIELDS:
            if decision[field] != str(candidate.get(field, "")):
                raise ValueError(f"source {field} mismatch: {package.name}: {decision['task_id']}")
        if decision["decision"].strip().casefold() not in {"approve", "reject", "skip"}:
            raise ValueError(f"invalid review decision: {package.name}: {decision['task_id']}")
    return candidates, decisions


def _check_splits(rows: list[dict[str, Any]]) -> None:
    for field in ("scene_id", "sequence_group", "location_group"):
        groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            if row.get(field):
                groups[str(row[field])].add(row["split"])
        mixed = [group for group, splits in groups.items() if len(splits) > 1]
        if mixed:
            raise ValueError(f"train/diagnostic {field} collision: {mixed[:5]}")
    # A rejected member of a mandatory pair must not yield a one-sided task.
    _atomic_groups([row for row in rows if _requires_pair(row)])


def merge_review_packages(packages: list[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"merged review directory already exists: {output}")
    if not packages or len({path.resolve() for path in packages}) != len(packages):
        raise ValueError("provide distinct reviewed packages")

    merged_candidates: list[dict[str, Any]] = []
    merged_decisions: list[dict[str, str]] = []
    source_summaries = []
    role_corrections = []
    for package in packages:
        candidates, decisions = _read_package(package)
        by_id = {row["task_id"]: row for row in candidates}
        # Reuse the release gate so a quick approval and an explicitly verified
        # structured approval are interpreted exactly as in the training path.
        accepted_ids = {row["task_id"] for row in accepted_candidates(candidates, package / "review/decisions.csv")}
        approved_decisions = [row for row in decisions if row["decision"].strip().casefold() == "approve"]
        if accepted_ids != {row["task_id"] for row in approved_decisions}:
            raise ValueError(f"an approved required pair was dropped in {package}")
        source_summaries.append({
            "package": str(package.resolve()),
            "candidates": len(candidates),
            "decisions": dict(Counter(row["decision"].strip().casefold() for row in decisions)),
            "approved": len(approved_decisions),
        })
        for decision in approved_decisions:
            original = by_id[decision["task_id"]]
            candidate = dict(original)
            merged_decision = dict(decision)
            candidate["query"] = decision["approved_query"].strip() or candidate["proposed_query"]
            candidate["approval_origin"] = {
                "package": package.name,
                "candidate_file": str((package / "data/candidates.jsonl").resolve()),
                "decision_file": str((package / "review/decisions.csv").resolve()),
                "source_task_id": original["task_id"],
                "source_category": original["category"],
                "source_bundle_id": original["bundle_id"],
            }
            if original["task_id"] == M3FD_ROLE_CORRECTION:
                if original["category"] != "reliability" or original["split"] != "train":
                    raise ValueError("M3FD role correction source changed")
                corrected_id = "rgbt_m3fd_train_002847:ir_complement"
                candidate["task_id"] = corrected_id
                candidate["bundle_id"] = corrected_id
                candidate["category"] = "ir_complement"
                candidate.pop("reliability_variants", None)
                candidate["predecessor_task_ids"] = [original["task_id"]]
                merged_decision.update(task_id=corrected_id, bundle_id=corrected_id,
                                       category="ir_complement")
                role_corrections.append({"source_task_id": original["task_id"],
                                         "curated_task_id": corrected_id,
                                         "from": "reliability", "to": "ir_complement",
                                         "blank_ir_presentations": 0,
                                         "reason": "reviewer identified this as an IR-useful case"})
            merged_candidates.append(candidate)
            merged_decisions.append(merged_decision)

    task_ids = [row["task_id"] for row in merged_candidates]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate merged task ID")
    _check_splits(merged_candidates)
    if len(role_corrections) != 1 and any(package.name == DEFAULT_PACKAGES[0].name for package in packages):
        raise ValueError("expected exactly one M3FD role correction")
    overlap = ancestor_overlap(merged_candidates)
    summary = {
        "status": "merged_approved_for_release",
        "source_packages": source_summaries,
        "approved_total": len(merged_candidates),
        "by_category": dict(sorted(Counter(row["category"] for row in merged_candidates).items())),
        "by_split": dict(sorted(Counter(row["split"] for row in merged_candidates).items())),
        "role_corrections": role_corrections,
        "release_created": False,
    }

    (output / "data").mkdir(parents=True)
    (output / "review").mkdir()
    source_archive = output / "audit/source_packages"
    for index, package in enumerate(packages):
        archive = source_archive / f"{index:02d}_{package.name}"
        archive.mkdir(parents=True)
        shutil.copyfile(package / "data/candidates.jsonl", archive / "candidates.jsonl")
        shutil.copyfile(package / "review/decisions.csv", archive / "decisions.csv")
    write_jsonl(output / "data/candidates.jsonl", merged_candidates)
    with (output / "review/decisions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        writer.writerows(merged_decisions)
    (output / "data/ancestor_overlap.json").write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "data/role_corrections.json").write_text(
        json.dumps(role_corrections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "data/merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, action="append", dest="packages",
                        help="review package; repeat in preferred output order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_review_packages(args.packages or DEFAULT_PACKAGES, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
