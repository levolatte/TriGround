"""Create frozen, group-disjoint training/holdout inputs for capability rebuild.

This script only joins already-existing manifests, candidate caches, and offline
labels. It never creates boxes/candidates from GT and never synthesizes actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


WORKSPACE = Path(__file__).resolve().parents[3]
OLD = WORKSPACE / "results/visual_agent/decision_rebuild_20260929"
DEFAULT_OUT = WORKSPACE / "results/visual_agent/capability_rebuild_20260929/data"
REMOTE_ROOT = "/root/autodl-tmp/rematch_20260922/results/visual_agent/capability_rebuild_20260929"
REMOTE_DATA_ROOT = "/root/autodl-tmp/rematch_20260922"
LOCAL_RGBDT_ROOT = str(OLD).replace("/", "\\")
LOCAL_WORKSPACE_ROOT = "F:\\AIC"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def map_image_paths(task: dict) -> dict:
    row = dict(task)
    images = dict(row.get("images", {}))
    for key, value in images.items():
        if isinstance(value, str) and value.lower().startswith(LOCAL_RGBDT_ROOT.lower()):
            suffix = value[len(LOCAL_RGBDT_ROOT):].replace("\\", "/")
            images[key] = REMOTE_ROOT + suffix
        elif isinstance(value, str) and value.lower().startswith((LOCAL_WORKSPACE_ROOT + "\\").lower()):
            suffix = value[len(LOCAL_WORKSPACE_ROOT):].replace("\\", "/")
            images[key] = REMOTE_DATA_ROOT + suffix
    row["images"] = images
    return row


def export(output_dir: Path, holdout_groups: int = 120, seed: int = 2031) -> dict:
    old_data = OLD / "data"
    reviewed = OLD / "reviewed"
    capability_data = DEFAULT_OUT

    main_manifest = read_jsonl(old_data / "manifest.jsonl")
    main_cloud_manifest = read_jsonl(old_data / "cloud_manifest.jsonl")
    main_candidates = read_jsonl(old_data / "candidates.jsonl")
    review_manifest = read_jsonl(reviewed / "cloud_manifest.jsonl")
    review_candidates = read_jsonl(reviewed / "candidates.jsonl")
    rgbdt_manifest = sum((read_jsonl(old_data / name) for name in (
        "rgbdt_accepted_batch01.jsonl", "rgbdt_accepted_batch02.jsonl",
        "rgbdt_accepted_batch03_partial.jsonl")), [])
    smoke_manifest = read_jsonl(capability_data / "smoke_manifest.jsonl")
    labels = read_jsonl(old_data / "private_labels_offline.jsonl")
    review_labels = read_jsonl(reviewed / "private_labels_offline.jsonl")
    prior_splits = read_jsonl(old_data / "group_splits.jsonl")
    main_cloud_by_id = {str(row["id"]): row for row in main_cloud_manifest}

    # Preserve the published/previous group holdout; only eligible rows can be
    # sampled into the new 120-group split. Existing pilot groups are not used
    # as a reason to withhold the separately accepted 40 RGBDT training tasks.
    preserved = {str(row["image_group"]) for row in prior_splits
                 if row.get("split") == "preserved_holdout"}
    smoke_ids = {str(row["id"]) for row in smoke_manifest}
    smoke_groups = {str(row["image_group"]) for row in smoke_manifest}

    # Original manifests are retained for grounding rehearsal, while the
    # candidate pool contains only rows with actual, previously generated
    # candidates. A missing cache is explicitly listed and never filled from GT.
    all_by_id: dict[str, dict] = {}
    for source, rows in (("main", main_manifest), ("reviewed", review_manifest),
                         ("rgbdt_accepted", rgbdt_manifest)):
        for raw in rows:
            row = map_image_paths(raw)
            if source == "main":
                cloud_row = main_cloud_by_id.get(str(row["id"]))
                if cloud_row:
                    row["images"] = dict(cloud_row.get("images", row.get("images", {})))
            row["data_origin"] = source
            sample_id = str(row["id"])
            if sample_id in all_by_id:
                raise ValueError(f"duplicate task id across source manifests: {sample_id}")
            all_by_id[sample_id] = row
    for raw in smoke_manifest:
        sample_id = str(raw["id"])
        if sample_id not in all_by_id:
            row = dict(raw)
            row["data_origin"] = "smoke_included_training"
            all_by_id[sample_id] = row

    all_candidates: dict[str, dict] = {}
    for raw in [*main_candidates, *review_candidates]:
        sample_id = str(raw["id"])
        if sample_id in all_candidates:
            raise ValueError(f"duplicate candidate cache id: {sample_id}")
        candidate = dict(raw)
        if sample_id in all_by_id:
            candidate["images"] = dict(all_by_id[sample_id].get("images", candidate.get("images", {})))
        for item in candidate.get("candidates", []):
            mask_path = item.get("mask_path")
            if isinstance(mask_path, str):
                mapped = map_image_paths({"images": {"mask": mask_path}})["images"]["mask"]
                item["mask_path"] = mapped
        all_candidates[sample_id] = candidate
    all_labels: dict[str, dict] = {}
    for raw in [*labels, *review_labels]:
        sample_id = str(raw["id"])
        if sample_id in all_labels:
            raise ValueError(f"duplicate offline label id: {sample_id}")
        all_labels[sample_id] = raw

    # Group membership is based on image_group, not individual query IDs.
    # Do not let rows that were previously reserved or the smoke groups enter
    # the new holdout sample.
    candidate_rows = [row for sample_id, row in all_by_id.items()
                      if sample_id in all_candidates and sample_id in all_labels
                      and str(row.get("image_group", "")) not in preserved]
    eligible_groups = sorted({str(row["image_group"]) for row in candidate_rows
                              if str(row["image_group"]) not in smoke_groups})
    if len(eligible_groups) < holdout_groups:
        raise ValueError(f"only {len(eligible_groups)} eligible image groups for {holdout_groups} holdout groups")
    rng = random.Random(seed)
    held_groups = set(rng.sample(eligible_groups, holdout_groups))

    train_rows, hold_rows, prior_hold_rows = [], [], []
    missing_candidates, missing_labels = [], []
    for sample_id, row in all_by_id.items():
        group = str(row.get("image_group", ""))
        if group in preserved:
            prior_hold_rows.append(row)
        elif group in held_groups:
            hold_rows.append(row)
        elif sample_id in all_labels:
            train_rows.append(row)
        else:
            missing_labels.append(row)

        if sample_id not in all_candidates:
            reason = "accepted_rgbdt_no_candidate_cache" if row.get("data_origin") == "rgbdt_accepted" else "missing_existing_candidate_cache"
            missing_candidates.append({"id": sample_id, "image_group": group,
                                       "source": row.get("source"), "reason": reason})

    train_rows.sort(key=lambda row: str(row["id"]))
    hold_rows.sort(key=lambda row: str(row["id"]))
    prior_hold_rows.sort(key=lambda row: str(row["id"]))
    train_ids = {str(row["id"]) for row in train_rows}
    hold_ids = {str(row["id"]) for row in hold_rows}
    candidate_train = [row for sample_id, row in all_candidates.items() if sample_id in train_ids]
    candidate_hold = [row for sample_id, row in all_candidates.items() if sample_id in hold_ids]
    train_labels = [all_labels[sample_id] for sample_id in sorted(train_ids)]
    hold_labels = [all_labels[sample_id] for sample_id in sorted(hold_ids)]

    # A compact frozen evaluation view picks exactly one labelled question per
    # held image group; the full 163-row group-disjoint pool remains available.
    held_by_group: dict[str, list[dict]] = defaultdict(list)
    for row in hold_rows:
        held_by_group[str(row["image_group"])].append(row)
    eval_rng = random.Random(seed)
    eval_rows = [eval_rng.choice(held_by_group[group]) for group in sorted(held_by_group)]
    eval_ids = {str(row["id"]) for row in eval_rows}
    eval_labels = [all_labels[sample_id] for sample_id in sorted(eval_ids)]
    eval_candidates = [row for row in candidate_hold if str(row["id"]) in eval_ids]

    # Make a GT-free first teacher batch from actual training images and
    # actual candidate provenance. These are job descriptions, not labels.
    manifest_by_id = {str(row["id"]): row for row in train_rows}
    candidates_by_id = {str(row["id"]): row for row in candidate_train}
    teacher_candidates = []
    for source in ("city",):
        pool = []
        for sample_id, candidate in candidates_by_id.items():
            task = manifest_by_id[sample_id]
            qi = candidate.get("query_info", {})
            refs = qi.get("reference_categories") or []
            relation = qi.get("relation_type", "none")
            if (task.get("source") == source and "depth_visual" in task.get("available_modalities", [])
                    and len(candidate.get("candidates", [])) >= 2 and (refs or relation not in (None, "none"))):
                pool.append((len(candidate.get("candidates", [])) + 2 * bool(refs), sample_id,
                             "city_depth_reference_or_relation"))
        teacher_candidates.extend(sorted(pool, reverse=True)[:8])

    rgbt_pool = []
    for sample_id, candidate in candidates_by_id.items():
        task = manifest_by_id[sample_id]
        if not str(task.get("source", "")).startswith("rgbt_") or "ir" not in task.get("available_modalities", []):
            continue
        modalities = {source.get("modality") for cand in candidate.get("candidates", [])
                      for source in cand.get("sources", []) if source.get("modality")}
        if len(candidate.get("candidates", [])) >= 2:
            rgbt_pool.append((len(modalities), len(candidate.get("candidates", [])), sample_id,
                              "rgbt_rgb_ir_candidate_evidence_assessment"))
    teacher_candidates.extend((cand_count + modality_count, sample_id, reason)
                              for modality_count, cand_count, sample_id, reason in
                              sorted(rgbt_pool, reverse=True)[:8])

    # Four further cases explicitly emphasize disagreement between actual
    # candidate sources; no GT or future outcome participates in this rank.
    disagreement = []
    already = {sample_id for _, sample_id, _ in teacher_candidates}
    for sample_id, candidate in candidates_by_id.items():
        if sample_id in already:
            continue
        source_count = max((len({str(src.get("modality")) for src in cand.get("sources", [])
                                if src.get("modality")}) for cand in candidate.get("candidates", [])), default=0)
        distinct_modalities = {str(src.get("modality")) for cand in candidate.get("candidates", [])
                              for src in cand.get("sources", []) if src.get("modality")}
        if source_count >= 2 or len(distinct_modalities) >= 2:
            disagreement.append((source_count, len(distinct_modalities), sample_id,
                                 "real_candidate_source_disagreement"))
    teacher_candidates.extend((source_count + modality_count, sample_id, reason)
                              for source_count, modality_count, sample_id, reason in
                              sorted(disagreement, reverse=True)[:4])
    teacher_candidates = teacher_candidates[:20]
    teacher_jobs = []
    for index, (_, sample_id, reason) in enumerate(teacher_candidates, 1):
        task = manifest_by_id[sample_id]
        candidate = candidates_by_id[sample_id]
        teacher_jobs.append({
            "job_id": f"capability_teacher_{index:02d}", "sample_id": sample_id,
            "source": task.get("source"), "image_group": task.get("image_group"),
            "selection_bucket": reason, "query": task["query"],
            "images": task.get("images", {}),
            "available_modalities": task.get("available_modalities", []),
            "candidate_count": len(candidate.get("candidates", [])),
            "candidate_cache_path": REMOTE_ROOT + "/data/train_candidates.jsonl",
            "manifest_path": REMOTE_ROOT + "/data/train_manifest.jsonl",
            "teacher_policy": "inspect the supplied images and current candidates; choose tools from observed uncertainty; do not use hidden GT",
            "label_status": "pending_teacher_decision",
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(output_dir / "train_labels.jsonl", train_labels)
    write_jsonl(output_dir / "holdout_manifest.jsonl", hold_rows)
    write_jsonl(output_dir / "holdout_labels.jsonl", hold_labels)
    write_jsonl(output_dir / "holdout_eval120_manifest.jsonl", eval_rows)
    write_jsonl(output_dir / "holdout_eval120_labels.jsonl", eval_labels)
    write_jsonl(output_dir / "holdout_eval120_candidates.jsonl", eval_candidates)
    write_jsonl(output_dir / "preserved_holdout_excluded.jsonl", prior_hold_rows)
    write_jsonl(output_dir / "train_candidates.jsonl", candidate_train)
    write_jsonl(output_dir / "holdout_candidates.jsonl", candidate_hold)
    write_jsonl(output_dir / "candidate_missing.jsonl", missing_candidates)
    write_jsonl(output_dir / "label_missing.jsonl", missing_labels)
    write_jsonl(output_dir / "teacher_jobs_batch01.jsonl", teacher_jobs)
    split_rows = ([{"image_group": group, "split": "new_holdout", "seed": seed}
                   for group in sorted(held_groups)] +
                  [{"image_group": group, "split": "preserved_holdout"}
                   for group in sorted(preserved)])
    write_jsonl(output_dir / "split_groups.jsonl", split_rows)

    counts_by_origin_split = Counter((row.get("data_origin"),
                                      "preserved_holdout" if row in prior_hold_rows else
                                      "new_holdout" if str(row["image_group"]) in held_groups else "train")
                                     for row in all_by_id.values())
    summary = {
        "seed": seed,
        "new_holdout_groups": len(held_groups),
        "new_holdout_group_ids": sorted(held_groups),
        "preserved_holdout_groups": len(preserved),
        "smoke_groups_forced_train": len(smoke_groups),
        "manifest_rows": {"all_labelled_train": len(train_rows), "new_holdout": len(hold_rows),
                          "formal_eval120": len(eval_rows), "prior_holdout_excluded": len(prior_hold_rows),
                          "missing_label_excluded": len(missing_labels)},
        "candidate_rows": {"all_train": len(candidate_train), "new_holdout": len(candidate_hold),
                           "formal_eval120": len(eval_candidates),
                           "missing_train": sum(row["id"] in train_ids for row in missing_candidates),
                           "missing_holdout": sum(row["id"] in hold_ids for row in missing_candidates)},
        "candidate_sources": dict(Counter(str(all_by_id[sample_id].get("source", "unknown"))
                                           for sample_id in train_ids if sample_id in all_candidates)),
        "manifest_origins": dict(Counter(str(row.get("data_origin", "unknown")) for row in train_rows)),
        "teacher_jobs_batch01": len(teacher_jobs),
        "origins_by_split": {f"{origin}:{split}": count for (origin, split), count in sorted(counts_by_origin_split.items())},
        "files": {name: str(output_dir / name) for name in (
            "train_manifest.jsonl", "train_labels.jsonl", "holdout_manifest.jsonl",
            "holdout_labels.jsonl", "train_candidates.jsonl", "candidate_missing.jsonl")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--holdout-groups", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()
    print(json.dumps(export(args.output_dir, args.holdout_groups, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
