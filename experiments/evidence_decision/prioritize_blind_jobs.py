"""Rank GT-free object-teacher jobs by query and public metadata only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


DEFAULT_DATA = Path(__file__).resolve().parents[3] / "results/visual_agent/evidence_decision_20260930/data"
BUCKETS = ("binding", "initial_open_choice", "finish_sufficiency_review")

CAMERA_DISTANCE = re.compile(
    r"\b(?:closest|nearest|farthest|furthest|farther|further|closer|nearer|near|far|close)"
    r"(?:\s+away)?\s+(?:to|from)\s+(?:the\s+)?camera\b|"
    r"\b(?:middle|mid)\s+distance\s+(?:to|from)\s+(?:the\s+)?camera\b|"
    r"\bcamera[- ](?:nearest|closest|distance)\b",
    re.IGNORECASE,
)
LEFT_RIGHT = re.compile(r"\b(?:left|right|leftmost|rightmost)\b", re.IGNORECASE)
ORDINAL = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|leftmost|rightmost|topmost|bottommost|"
    r"nearest|farthest|furthest|closest|middle one|middle distance)\b",
    re.IGNORECASE,
)
POSE_ATTRIBUTE = re.compile(
    r"\b(?:standing|sitting|seated|walking|running|parked|lying|crouching|crouched|"
    r"facing|turned|wearing|holding|carrying|riding|open|closed|leaning|bending)\b",
    re.IGNORECASE,
)
DEPTH_LANGUAGE = re.compile(
    r"\b(?:foreground|background|in the foreground|in the background|"
    r"closest|nearest|farthest|furthest|farther|further|closer|nearer)\b",
    re.IGNORECASE,
)

FEATURE_WEIGHTS = {
    "camera_distance_city_mm_clean": 10.0,
    "camera_distance_city_mm_mixed": 6.0,
    "depth_language_city_mm": 3.0,
    "reference_proposal_gap": 5.0,
    "group_ordinal_low_target_proposals": 4.0,
    "multi_category_ordinal": 3.0,
    "pose_attribute": 1.0,
    "left_right_ordinal": 1.0,
    "target_reference": 1.0,
    "ir_registered": 0.5,
    "ir_unregistered": 0.5,
    "ir_registration_unknown": 0.5,
    "ir_unavailable": 0.25,
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def _role_counts(candidate: dict) -> Counter:
    return Counter(str(row.get("role", "unknown")) for row in candidate.get("candidates", []))


def classify(job: dict, task: dict, candidate: dict) -> dict:
    """Return conservative evidence-need proxies, never an action label."""
    query = str(job.get("query") or task.get("query") or "")
    query_info = candidate.get("query_info") or {}
    available = set(task.get("available_modalities") or job.get("available_modalities") or [])
    images = task.get("images") or job.get("images") or {}
    city_mm = (task.get("depth_encoding") == "city_mm" and
               bool(images.get("depth_raw")) and "depth_raw" in available)
    has_camera_distance = bool(CAMERA_DISTANCE.search(query))
    has_left_right = bool(LEFT_RIGHT.search(query))
    reference_categories = list(query_info.get("reference_categories") or [])
    relation_type = query_info.get("relation_type")
    has_reference = bool(reference_categories) or relation_type not in (
        None, "", "none", "camera_near", "camera_far"
    )
    excluded_camera_distance = has_left_right or has_reference
    roles = _role_counts(candidate)
    target_count = roles.get("target", 0)
    reference_count = roles.get("reference", 0)
    target_category = query_info.get("target_category")
    all_categories = list(dict.fromkeys(
        str(value) for value in [target_category, *reference_categories] if value
    ))
    source_categories = {
        str(source.get("category")).casefold()
        for item in candidate.get("candidates", [])
        for source in item.get("sources", [])
        if source.get("category")
    }
    ordinal = bool(ORDINAL.search(query))
    scope = query_info.get("scope")
    tags: list[str] = []
    if city_mm:
        tags.append("depth_city_mm_available")
    if has_camera_distance and city_mm:
        tags.append("camera_distance_city_mm")
        if excluded_camera_distance:
            tags.append("camera_distance_mixed_with_left_right_or_reference")
            camera_clean = False
        else:
            tags.append("camera_distance_city_mm_clean")
            camera_clean = True
    else:
        camera_clean = False
    if city_mm and DEPTH_LANGUAGE.search(query):
        tags.append("depth_language_city_mm")
    if POSE_ATTRIBUTE.search(query):
        tags.append("pose_attribute")
    if ordinal or has_left_right:
        tags.append("left_right_ordinal")
    if has_reference:
        tags.append("target_reference")
    registration = task.get("ir_rgb_registration")
    if "ir" not in available or registration == "not_available":
        tags.append("ir_unavailable")
    elif registration == "normalized_shared_frame":
        tags.append("ir_registered")
    elif registration in {"unregistered", "not_registered", "unaligned"}:
        tags.append("ir_unregistered")
    else:
        tags.append("ir_registration_unknown")

    if reference_categories and reference_count < len(reference_categories):
        tags.append("reference_proposal_gap")
    if ordinal and len(all_categories) >= 2:
        tags.append("multi_category_ordinal")
    if (scope == "group" or ordinal) and target_count <= 1:
        tags.append("group_ordinal_low_target_proposals")

    # Candidate source rows in this cache do not record semantic category names.
    # Do not call a category a missing proposal when exact matching is unknowable.
    category_match = None
    if source_categories and target_category:
        wanted = str(target_category).casefold()
        category_match = any(wanted == value or wanted in value or value in wanted
                             for value in source_categories)

    score = sum(FEATURE_WEIGHTS.get(tag, 0.0) for tag in tags)
    return {
        "priority_score": score,
        "priority_reasons": tags,
        "priority_features": {
            "camera_distance_city_mm_any": has_camera_distance and city_mm,
            "camera_distance_city_mm_clean": camera_clean,
            "camera_distance_city_mm_mixed": has_camera_distance and city_mm and excluded_camera_distance,
            "depth_language_city_mm": city_mm and bool(DEPTH_LANGUAGE.search(query)),
            "pose_attribute": bool(POSE_ATTRIBUTE.search(query)),
            "left_right_ordinal": ordinal or has_left_right,
            "target_reference": has_reference,
            "ir_registered": "ir" in available and registration == "normalized_shared_frame",
            "ir_unregistered": "ir" in available and registration in {"unregistered", "not_registered", "unaligned"},
            "ir_registration_unknown": "ir" in available and registration not in (
                "normalized_shared_frame", "unregistered", "not_registered", "unaligned", "not_available"
            ),
            "ir_unavailable": "ir" not in available or registration == "not_available",
            "reference_proposal_gap": "reference_proposal_gap" in tags,
            "multi_category_ordinal": "multi_category_ordinal" in tags,
            "group_ordinal_low_target_proposals": "group_ordinal_low_target_proposals" in tags,
            "semantic_category_match": category_match,
        },
        "candidate_role_counts": dict(roles),
        "query_scope": scope,
        "depth_encoding": task.get("depth_encoding"),
        "ir_rgb_registration": task.get("ir_rgb_registration"),
    }


def _quotas(jobs: list[dict], batch_size: int) -> dict[str, int]:
    counts = Counter(job["selection_bucket"] for job in jobs)
    total = len(jobs)
    raw = {bucket: batch_size * counts[bucket] / total for bucket in BUCKETS}
    result = {bucket: min(counts[bucket], int(raw[bucket])) for bucket in BUCKETS}
    left = batch_size - sum(result.values())
    order = sorted(BUCKETS, key=lambda bucket: (-(raw[bucket] - int(raw[bucket])), BUCKETS.index(bucket)))
    for bucket in order:
        if left == 0:
            break
        if result[bucket] < counts[bucket]:
            result[bucket] += 1
            left -= 1
    return result


def _first_batch(classified: list[dict], batch_size: int) -> list[dict]:
    quotas = _quotas(classified, batch_size)
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in classified:
        pools[row["selection_bucket"]].append(row)
    for rows in pools.values():
        rows.sort(key=lambda row: (-row["priority_score"], row["sample_id"]))

    selected: list[dict] = []
    selected_counts = Counter()
    covered: set[str] = set()

    def available_buckets() -> list[str]:
        return [bucket for bucket in BUCKETS if selected_counts[bucket] < quotas.get(bucket, 0)]

    def take_best(bucket: str, required_feature: str | None = None) -> dict | None:
        pool = pools[bucket]
        eligible = [index for index, row in enumerate(pool)
                    if required_feature is None or row["priority_features"].get(required_feature)]
        if not eligible:
            return None
        index = max(eligible, key=lambda idx: (
            sum(FEATURE_WEIGHTS.get(tag, 0.0) for tag in pool[idx]["priority_reasons"]
                if tag not in covered),
            pool[idx]["priority_score"],
            pool[idx]["sample_id"],
        ))
        row = pool.pop(index)
        selected.append(row)
        selected_counts[bucket] += 1
        covered.update(row["priority_reasons"])
        return row

    # The explicit camera-distance + raw city_mm pool is small enough to
    # include in this batch without altering the source sampling proportions.
    camera_pools: dict[str, list[dict]] = defaultdict(list)
    for row in classified:
        if row["priority_features"].get("camera_distance_city_mm_any"):
            camera_pools[row["selection_bucket"]].append(row)
    for rows in camera_pools.values():
        rows.sort(key=lambda row: (-row["priority_score"], row["sample_id"]))
    overflow = {bucket: len(camera_pools[bucket]) - quotas.get(bucket, 0)
                for bucket in BUCKETS if len(camera_pools[bucket]) > quotas.get(bucket, 0)}
    if overflow:
        raise ValueError(f"camera-distance city_mm jobs do not fit first-batch bucket quotas: {overflow}")
    while any(camera_pools[bucket] and selected_counts[bucket] < quotas.get(bucket, 0)
              for bucket in BUCKETS):
        for bucket in BUCKETS:
            if not camera_pools[bucket] or selected_counts[bucket] >= quotas.get(bucket, 0):
                continue
            candidate = camera_pools[bucket].pop(0)
            index = next(index for index, row in enumerate(pools[bucket])
                         if row["sample_id"] == candidate["sample_id"])
            row = pools[bucket].pop(index)
            selected.append(row)
            selected_counts[bucket] += 1
            covered.update(row["priority_reasons"])

    # Guarantee a small number of high-value, identifiable conditions before
    # filling the representative proportional batch. This affects which rows
    # are requested first, never what action the teacher should choose.
    required_features = (
        "camera_distance_city_mm_clean", "ir_registered", "ir_registration_unknown", "ir_unavailable",
        "reference_proposal_gap", "multi_category_ordinal",
        "group_ordinal_low_target_proposals", "depth_language_city_mm",
        "pose_attribute", "left_right_ordinal", "target_reference",
    )
    for feature in required_features:
        if any(row["priority_features"].get(feature) for row in selected):
            continue
        eligible_buckets = [bucket for bucket in available_buckets()
                            if any(row["priority_features"].get(feature) for row in pools[bucket])]
        if not eligible_buckets:
            continue
        bucket = min(eligible_buckets,
                     key=lambda value: (selected_counts[value] / max(1, quotas[value]), BUCKETS.index(value)))
        take_best(bucket, feature)

    while len(selected) < batch_size:
        bucket = min(available_buckets(),
                     key=lambda value: (selected_counts[value] / max(1, quotas[value]), BUCKETS.index(value)))
        take_best(bucket)
    return selected


def prioritize(data_dir: Path, *, batch_size: int = 128) -> dict:
    jobs = read_jsonl(data_dir / "blind_teacher_jobs.jsonl")
    tasks = {str(row["id"]): row for row in read_jsonl(data_dir / "train_manifest.jsonl")}
    candidates = {str(row["id"]): row for row in read_jsonl(data_dir / "train_candidates.jsonl")}
    classified = []
    for job in jobs:
        sample_id = str(job["sample_id"])
        row = {**job, **classify(job, tasks[sample_id], candidates.get(sample_id, {}))}
        row["sample_id"] = sample_id
        classified.append(row)

    first = _first_batch(classified, batch_size)
    first_ids = {row["sample_id"] for row in first}
    first_rank = {row["sample_id"]: index for index, row in enumerate(first, 1)}
    all_rows = sorted(classified, key=lambda row: (
        0 if row["sample_id"] in first_ids else 1,
        first_rank.get(row["sample_id"], 0),
        -row["priority_score"], row["selection_bucket"], row["sample_id"],
    ))
    for rank, row in enumerate(all_rows, 1):
        row["collector_priority_rank"] = rank
        row["collector_first_batch_rank"] = first_rank.get(row["sample_id"])
        row["collector_action_policy"] = "suggest any schema-valid evidence action or finish; never force an action from priority tags"
    for rank, row in enumerate(first, 1):
        row["collector_priority_rank"] = rank
        row["collector_first_batch_rank"] = rank
        row["collector_action_policy"] = "suggest any schema-valid evidence action or finish; never force an action from priority tags"

    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                      for row in rows), encoding="utf-8")

    write(data_dir / "blind_teacher_priority_jobs.jsonl", all_rows)
    write(data_dir / f"blind_teacher_first_batch{batch_size}.jsonl", first)

    def count_features(rows: list[dict]) -> dict:
        result = Counter()
        for row in rows:
            result.update(tag for tag, enabled in row["priority_features"].items() if enabled is True)
        return dict(result)

    bucket_summary = {}
    for bucket in BUCKETS:
        bucket_rows = [row for row in all_rows if row["selection_bucket"] == bucket]
        first_rows = [row for row in first if row["selection_bucket"] == bucket]
        bucket_summary[bucket] = {
            "all_jobs": len(bucket_rows),
            "first_batch_jobs": len(first_rows),
            "all_feature_counts": count_features(bucket_rows),
            "first_batch_feature_counts": count_features(first_rows),
        }
    source_categories_available = sum(
        bool(source.get("category"))
        for candidate in candidates.values()
        for item in candidate.get("candidates", [])
        for source in item.get("sources", [])
    )
    summary = {
        "jobs_total": len(all_rows),
        "first_batch_size": len(first),
        "first_batch_bucket_quotas": {bucket: sum(row["selection_bucket"] == bucket for row in first)
                                       for bucket in BUCKETS},
        "priority_method": "GT-free query text, available modalities, depth encoding, IR registration, query parse categories/scope, and candidate role counts; greedy coverage within proportional bucket quotas",
        "depth_policy": "All explicit camera-distance queries with raw city_mm metadata are prioritized; 2 are single-condition comparisons and 33 are compound object-binding comparisons using left/right or reference descriptions. Mixed marks added binding work, not lack of depth relevance.",
        "category_gap_policy": "No semantic category gap is claimed when cached candidate source rows omit category labels; role-level missing reference proposals are tagged as a search-potential proxy",
        "candidate_source_rows_with_category": source_categories_available,
        "feature_counts_all": count_features(all_rows),
        "feature_counts_first_batch": count_features(first),
        "bucket_summary": bucket_summary,
        "action_policy": "priority tags do not constrain teacher output; each actual next action must be selected from current open tools after viewing the real state",
    }
    (data_dir / "blind_teacher_priority_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    print(json.dumps(prioritize(args.data_dir, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
