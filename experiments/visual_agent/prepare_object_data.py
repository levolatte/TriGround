"""Prepare GT-free object-agent manifests and separate offline labels.

City, RGB-T GroundBench and RoboRefIt provide natural-language queries and
boxes. RGBDT500 provides annotated frames but no queries, so it produces image
annotation jobs only. Missing modalities stay absent from ``images``.
"""
from __future__ import annotations

import argparse
import csv
import json
import posixpath
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .evaluate import SHARED_RE, SHARED_SEQUENCES


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / "results/visual_agent/decision_rebuild_20260929/data"
CITY_QUERY_MARKER = "Locate the object described by this query: "
STAGES = {
    "foundation_candidate": [],
    "observation": ["foundation_candidate_output"],
    "next_action": ["actual_observation_trace"],
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def normalized_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.casefold()).strip(" \t\r\n.,;:!?")


def _rooted(root: str, relative: str) -> str:
    """Join POSIX-style dataset paths without interpreting a remote root locally."""
    root = root.replace("\\", "/").rstrip("/")
    relative = relative.replace("\\", "/").lstrip("/")
    return posixpath.normpath(posixpath.join(root, relative))


def _bbox(values, scale: float = 1.0) -> list[float]:
    box = [float(value) / scale for value in values]
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError(f"invalid normalized xyxy box: {values}")
    return box


def _city_query(row: dict) -> str:
    prompt = row["conversations"][0]["value"]
    if CITY_QUERY_MARKER not in prompt:
        raise ValueError(f"{row['id']}: unknown City query prompt")
    query = prompt.split(CITY_QUERY_MARKER, 1)[1].split("\nReturn", 1)[0].strip()
    if not query:
        raise ValueError(f"{row['id']}: empty City query")
    return query


def _city_box(row: dict) -> list[float]:
    answer = json.loads(row["conversations"][1]["value"])
    return _bbox(answer["bbox_2d"], 1000.0)


def _city_group(row: dict) -> str:
    return PurePosixPath(row["image"][0].replace("\\", "/")).stem


def _city_record(row: dict, image_root: str) -> dict:
    rgb, ir, depth_visual = row["image"]
    relative_rgb = rgb.replace("\\", "/")
    if not relative_rgb.startswith("visible/"):
        raise ValueError(f"{row['id']}: expected visible/ City RGB path")
    depth_raw = "depth/" + relative_rgb.removeprefix("visible/")
    group = _city_group(row)
    return {
        "id": str(row["id"]), "source_id": str(row["id"]), "source": "city",
        "scene_id": f"city:{group}", "image_group": f"city:{group}",
        "query": _city_query(row),
        "images": {"rgb": _rooted(image_root, rgb), "ir": _rooted(image_root, ir),
                   "depth_visual": _rooted(image_root, depth_visual),
                   "depth_raw": _rooted(image_root, depth_raw)},
        "available_modalities": ["rgb", "ir", "depth_visual", "depth_raw"],
        "depth_encoding": "city_mm",
        "depth_visual_encoding": "city_native_depth_visual",
        "ir_rgb_registration": "normalized_shared_frame",
        "registration_source": "City source prompt states the supplied RGB, infrared and depth views are aligned.",
        "_bbox": _city_box(row),
    }


def _rgbt_record(row: dict, image_root: str) -> dict:
    query = str(row["query"]).strip()
    if not query:
        raise ValueError(f"{row['id']}: empty RGBT query")
    aux_type = str(row["aux_type"]).casefold()
    if aux_type not in {"ir", "depth"}:
        raise ValueError(f"{row['id']}: unsupported aux_type {aux_type!r}")
    group = str(row.get("scene_id") or row["original_image_id"])
    images = {"rgb": _rooted(image_root, row["rgb"]),
              ("ir" if aux_type == "ir" else "depth_visual"): _rooted(image_root, row["aux"])}
    modalities = ["rgb", "ir"] if aux_type == "ir" else ["rgb", "depth_visual"]
    return {
        "id": str(row["id"]), "source_id": str(row["id"]), "source": str(row["source"]),
        "scene_id": group, "image_group": _source_key(str(row["source"]), group), "query": query,
        "images": images, "available_modalities": modalities,
        "depth_encoding": "unknown" if aux_type == "depth" else "not_available",
        "depth_visual_encoding": "unknown" if aux_type == "depth" else "not_available",
        "ir_rgb_registration": "normalized_shared_frame" if aux_type == "ir" else "not_available",
        "registration_source": (
            "https://arxiv.org/html/2512.24561v1 (RGBT-Ground reports spatially aligned RGB/thermal pairs)"
            if aux_type == "ir" else "not_applicable"),
        "_bbox": _bbox(row["bbox"]),
    }


def _robo_record(row: dict, image_root: str, depth_visual_root: str) -> dict:
    query = str(row["query"]).strip()
    if not query:
        raise ValueError(f"{row['id']}: empty RoboRefIt query")
    group = str(row.get("scene_id") or row["original_image_id"])
    depth_visual_relative = str(row["depth"]).replace("\\", "/")
    visual_relative = depth_visual_relative.replace(
        "../final_dataset/train/depth/", "../depth_visual_fixed_p01_p99/", 1)
    if visual_relative == depth_visual_relative:
        raise ValueError(f"{row['id']}: unknown RoboRefIt depth path {row['depth']}")
    return {
        "id": str(row["id"]), "source_id": str(row["id"]), "source": "roborefit",
        "scene_id": group, "image_group": _source_key("roborefit", group), "query": query,
        "images": {"rgb": _rooted(image_root, row["rgb"]),
                   "depth_raw": _rooted(image_root, row["depth"]),
                   "depth_visual": _rooted(depth_visual_root, visual_relative)},
        "available_modalities": ["rgb", "depth_visual", "depth_raw"],
        "depth_encoding": "unknown",
        "depth_visual_encoding": str(row.get("depth_visualization", "fixed_raw_p01_p99_v1")),
        "ir_rgb_registration": "not_available", "registration_source": "not_applicable",
        "_bbox": _bbox(row["bbox"]),
    }


def _source_key(source: str, scene_id: str) -> str:
    if source == "city":
        scene_id = scene_id.removeprefix("city:")
    return f"{source}:{scene_id}"


def _city_group_from_manifest(row: dict) -> str:
    images = row.get("images", {})
    if images.get("rgb"):
        return PurePosixPath(str(images["rgb"]).replace("\\", "/")).stem
    return PurePosixPath(str(row["image"][0]).replace("\\", "/")).stem


def exclusions(args) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]], dict]:
    groups: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    city412 = read_jsonl(args.city412)
    for row in city412:
        group = f"city:{_city_group_from_manifest(row)}"
        groups["city"].add(group)
        reasons["city"][group].add("city412_holdout")
    old_holdout = read_jsonl(args.old_t_holdout)
    for row in old_holdout:
        group = f"city:{_city_group_from_manifest(row)}"
        groups["city"].add(group)
        reasons["city"][group].add("old_t_holdout50")

    city_raw = read_json(args.city_train)
    city_shared_groups = {
        f"city:{_city_group(row)}" for row in city_raw
        if SHARED_RE.search(str(row["id"]))
    }
    for group in city_shared_groups:
        groups["city"].add(group)
        reasons["city"][group].add("known_shared_video_sequence")

    diagnostic_rows = read_json(args.mainline_diagnostics)
    diagnostic_source_groups: dict[str, set[str]] = defaultdict(set)
    for row in diagnostic_rows:
        source = str(row["source"])
        scene_id = str(row["scene_id"])
        bucket = "city" if source == "city" else "rgbt" if source.startswith("rgbt_groundbench") else "robo"
        group = _source_key(source, scene_id)
        groups[bucket].add(group)
        reasons[bucket][group].add("mainline_diagnostic70")
        diagnostic_source_groups[source].add(group)
    counts = {
        "city412_rows": len(city412),
        "city412_groups": len({_city_group_from_manifest(row) for row in city412}),
        "old_t_holdout_rows": len(old_holdout),
        "old_t_holdout_groups": len({_city_group_from_manifest(row) for row in old_holdout}),
        "mainline_diagnostic_rows": len(diagnostic_rows),
        "mainline_diagnostic_groups_by_source": {
            source: len(source_groups) for source, source_groups in diagnostic_source_groups.items()},
        "known_shared_video_sequences": list(SHARED_SEQUENCES),
        "known_shared_video_groups": len(city_shared_groups),
        "known_shared_video_rows": sum(1 for row in city_raw if SHARED_RE.search(str(row["id"]))),
    }
    return groups, reasons, counts


def _row_group(record: dict) -> str:
    return str(record["image_group"])


def select_unique_queries(records: list[dict], requested: int, *, max_per_group: int,
                          seed: int, reserved_queries: set[str] | None = None) -> tuple[list[dict], dict]:
    """Choose distinct normalized Query strings, with a per-image group cap."""
    reserved_queries = reserved_queries or set()
    by_group_query: dict[str, dict[str, dict]] = defaultdict(dict)
    seen_ids = set()
    for row in records:
        sample_id = str(row["id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate source sample ID {sample_id}")
        seen_ids.add(sample_id)
        key = normalized_query(row["query"])
        if key in reserved_queries:
            continue
        group = _row_group(row)
        # Exact source/group/query duplicates add no new supervision information.
        by_group_query[group].setdefault(key, row)

    query_groups: dict[str, list[str]] = defaultdict(list)
    for group, queries in by_group_query.items():
        for query_key in queries:
            query_groups[query_key].append(group)
    rng = random.Random(seed)
    tie = {query_key: rng.random() for query_key in query_groups}
    assignments: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    used_queries = set()
    # Rare Query strings claim a group first. Within each Query, put it in the
    # least-used eligible group to spread the selected set across scenes.
    ordered_queries = sorted(query_groups, key=lambda key: (len(query_groups[key]), tie[key], key))
    for query_key in ordered_queries:
        options = [group for group in query_groups[query_key]
                   if len(assignments[group]) < max_per_group]
        if not options:
            continue
        options.sort(key=lambda group: (
            len(assignments[group]), len(by_group_query[group]),
            rng.random(), group))
        group = options[0]
        assignments[group].append((query_key, by_group_query[group][query_key]))
        used_queries.add(query_key)

    groups = list(assignments)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(assignments[group])
    selected = []
    for round_index in range(max_per_group):
        for group in groups:
            if round_index < len(assignments[group]) and len(selected) < requested:
                selected.append(assignments[group][round_index][1])
        if len(selected) == requested:
            break
    selected_query_keys = {normalized_query(row["query"]) for row in selected}
    if len(selected_query_keys) != len(selected):
        raise ValueError("query selection produced repeated normalized Query text")
    counts = Counter(_row_group(row) for row in selected)
    return selected, {
        "requested_queries": requested,
        "selected_queries": len(selected),
        "shortfall": max(0, requested - len(selected)),
        "eligible_rows_after_group_exclusions": len(records),
        "eligible_unique_query_texts": len(query_groups),
        "selectable_unique_query_texts_under_group_cap": len(used_queries),
        "selected_unique_groups": len(counts),
        "max_selected_queries_per_group": max(counts.values(), default=0),
        "groups_with_multiple_selected_queries": sum(value > 1 for value in counts.values()),
    }


def load_sources(args, excluded_groups: dict[str, set[str]]) -> tuple[dict[str, list[dict]], dict]:
    city_raw = read_json(args.city_train)
    city_records = [_city_record(row, args.city_image_root) for row in city_raw]
    rgbt_records = [_rgbt_record(row, args.rgbt_image_root) for row in read_jsonl(args.rgbt_manifest)]
    robo_records = [_robo_record(row, args.robo_image_root, args.robo_depth_visual_root)
                    for row in read_jsonl(args.robo_manifest)]

    source_rows = {"city": city_records, "rgbt": rgbt_records, "robo": robo_records}
    input_stats = {}
    eligible = {}
    for source, records in source_rows.items():
        source_excluded = excluded_groups.get(source, set())
        keep = [row for row in records if row["image_group"] not in source_excluded]
        eligible[source] = keep
        input_stats[source] = {
            "input_rows": len(records),
            "input_unique_groups": len({row["image_group"] for row in records}),
            "input_unique_query_texts": len({normalized_query(row["query"]) for row in records}),
            "excluded_rows": len(records) - len(keep),
            "excluded_groups_matched": len({row["image_group"] for row in records} & source_excluded),
            "eligible_rows": len(keep),
            "eligible_unique_groups": len({row["image_group"] for row in keep}),
            "eligible_unique_query_texts": len({normalized_query(row["query"]) for row in keep}),
        }
    return eligible, input_stats


def parse_groundtruth(text: str) -> dict[str, tuple[float, float, float, float]]:
    parsed = {}
    for values in csv.reader(text.splitlines()):
        if len(values) != 5:
            raise ValueError(f"invalid RGBDT groundtruth row: {values}")
        frame, x, y, width, height = values
        parsed[frame] = (float(x), float(y), float(width), float(height))
    if len(parsed) != 10:
        raise ValueError(f"expected 10 RGBDT groundtruth rows, got {len(parsed)}")
    return parsed


def _depth_visual(depth_path: Path, visual_path: Path) -> None:
    with Image.open(depth_path) as image:
        values = np.asarray(image)
    if values.dtype != np.uint16:
        raise ValueError(f"expected uint16 RGBDT depth, got {values.dtype}: {depth_path}")
    visual = np.zeros(values.shape, dtype=np.uint8)
    valid = values > 0
    valid_values = values[valid].astype(np.float32)
    low, high = np.percentile(valid_values, [1, 99])
    if high <= low:
        visual[valid] = 128
    else:
        scaled = np.clip((valid_values - low) / (high - low), 0, 1)
        visual[valid] = 1 + np.rint(scaled * 254).astype(np.uint8)
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visual).convert("RGB").save(visual_path, format="PNG")


def prepare_rgbdt_jobs(args, output_dir: Path, seed: int) -> tuple[list[dict], list[dict], dict]:
    plan = read_json(args.rgbdt_plan)
    holdout_rows = read_jsonl(args.rgbdt_external_manifest)
    external_sequences = set(plan["external_review_sequences"])
    manifest_external_sequences = {str(row["sequence"]) for row in holdout_rows}
    if external_sequences != manifest_external_sequences:
        raise ValueError("RGBDT external sequence holdout differs between plan and manifest")
    pilot_sequences = {str(row["sequence"]) for row in plan["pilot"]}
    train_sequences = [str(value) for value in plan["remaining_train_sequences"]]
    if len(train_sequences) != len(set(train_sequences)):
        raise ValueError("duplicate RGBDT remaining train sequence")

    rng = random.Random(seed)
    rng.shuffle(train_sequences)
    selected_sequences = train_sequences[:args.rgbdt_count]
    jobs, labels = [], []
    output_root = (output_dir / "images" / "rgbdt500").resolve()
    with zipfile.ZipFile(args.rgbdt_zip) as archive:
        names = set(archive.namelist())
        for sequence in selected_sequences:
            gt_member = f"{sequence}/groundtruth.txt"
            if gt_member not in names:
                raise FileNotFoundError(f"{args.rgbdt_zip} lacks {gt_member}")
            gt_rows = parse_groundtruth(archive.read(gt_member).decode("utf-8-sig"))
            valid_frames = [frame for frame, (_, _, width, height) in gt_rows.items()
                            if width > 0 and height > 0 and
                            all(f"{sequence}/{modality}/{frame}" in names
                                for modality in ("color", "depth", "infrared"))]
            if not valid_frames:
                raise ValueError(f"RGBDT sequence {sequence} has no valid complete frame")
            frame = rng.choice(sorted(valid_frames))
            x, y, width, height = gt_rows[frame]
            paths = {}
            for modality in ("color", "infrared", "depth"):
                destination = output_root / sequence / modality / frame
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(f"{sequence}/{modality}/{frame}"))
                paths[modality] = destination
            visual_path = output_root / sequence / "depth_visual" / frame
            _depth_visual(paths["depth"], visual_path)
            with Image.open(paths["color"]) as rgb, Image.open(paths["infrared"]) as ir, Image.open(paths["depth"]) as depth:
                image_size = list(rgb.size)
                if ir.size != rgb.size or depth.size != rgb.size:
                    raise ValueError(f"RGBDT modalities are not size-aligned: {sequence}/{frame}")
                if depth.mode != "I;16":
                    raise ValueError(f"RGBDT depth is not uint16: {sequence}/{frame} {depth.mode}")
            sample_id = f"rgbdt500_{sequence}_{Path(frame).stem}"
            group_id = f"rgbdt500:{sequence}"
            image_map = {
                "rgb": str(paths["color"]), "ir": str(paths["infrared"]),
                "depth_raw": str(paths["depth"]), "depth_visual": str(visual_path),
            }
            jobs.append({
                "id": sample_id, "source_id": sample_id, "source": "rgbdt500",
                "scene_id": group_id, "image_group": group_id, "sequence": sequence,
                "frame": frame, "images": image_map,
                "available_modalities": ["rgb", "ir", "depth_visual", "depth_raw"],
                "depth_encoding": "unknown",
                "depth_visual_encoding": "per_image_raw_p01_p99_numeric_order_zero_invalid",
                "ir_rgb_registration": "unknown",
                "registration_source": "unknown",
                "task": "Write a natural Query grounded in the visible target. Do not use the private box label.",
                "status": "query_annotation_required",
            })
            labels.append({
                "id": sample_id, "source": "rgbdt500", "image_group": group_id,
                "bbox_xyxy_normalized": _bbox(
                    [x / image_size[0], y / image_size[1],
                     (x + width) / image_size[0], (y + height) / image_size[1]]),
                "bbox_xywh_pixels": [x, y, width, height], "image_size": image_size,
                "label_source": f"{args.rgbdt_zip}::{sequence}/groundtruth.txt",
            })

    inventory = {
        "requested_annotation_jobs": args.rgbdt_count,
        "selected_annotation_jobs": len(jobs),
        "shortfall": max(0, args.rgbdt_count - len(jobs)),
        "natural_query_count": 0,
        "pilot_sequences_preserved": len(pilot_sequences),
        "external_holdout_rows_preserved": len(holdout_rows),
        "external_holdout_sequences_preserved": len(external_sequences),
        "available_training_sequences": len(train_sequences),
        "selected_unique_sequences": len({row["sequence"] for row in jobs}),
        "selected_sequences": selected_sequences,
        "query_policy": "RGBDT500 has no Query; these are annotation jobs, not accepted Query records.",
    }
    return jobs, labels, inventory


def _public(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "_bbox"}


def _stage_jobs(rows: list[dict], stage: str) -> list[dict]:
    return [{"id": row["id"], "source": row["source"], "image_group": row["image_group"],
             "stage": stage, "manifest": "cloud_manifest.jsonl", "status": "pending",
             "requires": list(STAGES[stage]),
             "supervision_policy": "Create labels only from the public Query and completed real pipeline evidence."}
            for row in rows]


def _cloud_path(value: str, local_root: str, cloud_root: str) -> str:
    value = str(value).replace("\\", "/")
    local_root = local_root.replace("\\", "/").rstrip("/")
    if value.casefold().startswith((local_root + "/").casefold()):
        return _rooted(cloud_root, value[len(local_root) + 1:])
    return value


def _cloud_rows(rows: list[dict], local_root: str, cloud_root: str) -> list[dict]:
    cloud = []
    for row in rows:
        copied = dict(row)
        copied["images"] = {key: _cloud_path(path, local_root, cloud_root)
                            for key, path in row["images"].items()}
        cloud.append(copied)
    return cloud


def _group_split_rows(selected_rows: list[dict], exclusion_reasons: dict[str, dict[str, set[str]]],
                      rgbdt_plan: dict) -> list[dict]:
    splits = {}
    for row in selected_rows:
        splits[(row["source"], row["image_group"])] = {
            "source": row["source"], "image_group": row["image_group"],
            "split": "candidate_pool", "reason": "selected_for_object_decision_preparation",
        }
    for reasons_by_group in exclusion_reasons.values():
        for group, reasons in reasons_by_group.items():
            source = group.split(":", 1)[0]
            splits.setdefault((source, group), {
                "source": source, "image_group": group, "split": "preserved_holdout",
            })["reasons"] = sorted(reasons)
    for record in rgbdt_plan["pilot"]:
        group = f"rgbdt500:{record['sequence']}"
        splits[("rgbdt500", group)] = {
            "source": "rgbdt500", "image_group": group,
            "split": "preserved_pilot", "reason": "existing_pilot_sequence",
        }
    for sequence in rgbdt_plan["external_review_sequences"]:
        group = f"rgbdt500:{sequence}"
        splits[("rgbdt500", group)] = {
            "source": "rgbdt500", "image_group": group,
            "split": "preserved_holdout", "reason": "external_review_sequence",
        }
    return [splits[key] for key in sorted(splits)]


def prepare(args) -> dict:
    if args.output_dir.joinpath("inventory.json").exists():
        raise FileExistsError(f"frozen manifests already exist: {args.output_dir}")
    excluded, exclusion_reasons, exclusion_counts = exclusions(args)
    eligible, source_stats = load_sources(args, excluded)

    selections, selection_stats = {}, {}
    used_queries: set[str] = set()
    for source, requested, cap in (("city", args.city_count, 3),
                                   ("rgbt", args.rgbt_count, 1),
                                   ("robo", args.robo_count, 1)):
        selected, stats = select_unique_queries(
            eligible[source], requested, max_per_group=cap,
            seed=args.seed + {"city": 1, "rgbt": 2, "robo": 3}[source],
            reserved_queries=used_queries)
        selections[source] = selected
        selection_stats[source] = {**stats, "max_queries_per_group_policy": cap}
        used_queries.update(normalized_query(row["query"]) for row in selected)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for source in ("city", "rgbt", "robo") for row in selections[source]]
    public_rows = [_public(row) for row in selected_rows]
    source_labels = [{"id": row["id"], "source": row["source"],
                      "scene_id": row["scene_id"], "image_group": row["image_group"],
                      "bbox_xyxy_normalized": row["_bbox"],
                      "label_source": "source_dataset_annotation"}
                     for row in selected_rows]

    # Make the 2,800 real-query inputs available before touching the much larger
    # RGBDT archive, so cloud preparation can begin independently.
    write_jsonl(output_dir / "manifest.jsonl", public_rows)
    cloud_public_rows = _cloud_rows(public_rows, args.local_root, args.cloud_root)
    write_jsonl(output_dir / "cloud_manifest.jsonl", cloud_public_rows)
    write_jsonl(output_dir / "private_labels_offline.jsonl", source_labels)
    for stage in STAGES:
        write_jsonl(output_dir / f"{stage}_jobs.jsonl", _stage_jobs(public_rows, stage))
    group_splits = _group_split_rows(selected_rows, exclusion_reasons, read_json(args.rgbdt_plan))
    write_jsonl(output_dir / "group_splits.jsonl", group_splits)

    rgbdt_jobs, rgbdt_labels, rgbdt_stats = prepare_rgbdt_jobs(args, output_dir, args.seed + 4)
    write_jsonl(output_dir / "annotation_jobs.jsonl", rgbdt_jobs)
    cloud_rgbdt_jobs = _cloud_rows(rgbdt_jobs, args.local_root, args.cloud_root)
    write_jsonl(output_dir / "cloud_annotation_jobs.jsonl", cloud_rgbdt_jobs)
    write_jsonl(output_dir / "private_labels_offline.jsonl", source_labels + rgbdt_labels)
    private_labels = source_labels + rgbdt_labels

    unique_queries = {normalized_query(row["query"]) for row in public_rows}
    if len(unique_queries) != len(public_rows):
        raise ValueError("selected public manifest does not contain globally unique Query text")
    seed_candidates = read_json(args.mainline_release)
    approved_counts = seed_candidates.get("counts", {})
    inventory = {
        "seed": args.seed,
        "requested": {"city_queries": args.city_count, "rgbt_queries": args.rgbt_count,
                      "robo_queries": args.robo_count,
                      "rgbdt_annotation_jobs": args.rgbdt_count,
                      "total_requested_items": args.city_count + args.rgbt_count + args.robo_count + args.rgbdt_count},
        "actual": {
            "city_queries": len(selections["city"]),
            "rgbt_queries": len(selections["rgbt"]),
            "robo_queries": len(selections["robo"]),
            "rgbdt_annotation_jobs": len(rgbdt_jobs),
            "explicit_natural_queries": len(public_rows),
            "globally_unique_explicit_queries": len(unique_queries),
            "rows_including_queryless_rgbdt_jobs": len(public_rows) + len(rgbdt_jobs),
            "candidate_pool_rows": len(public_rows),
            "mainline_training_rows": 0,
        },
        "shortfall": {
            "city_queries": selection_stats["city"]["shortfall"],
            "rgbt_queries": selection_stats["rgbt"]["shortfall"],
            "robo_queries": selection_stats["robo"]["shortfall"],
            "rgbdt_annotation_jobs": rgbdt_stats["shortfall"],
            "explicit_queries_to_reach_3000": max(0, 3000 - len(unique_queries)),
        },
        "sources": source_stats,
        "selection": selection_stats,
        "rgbdt": rgbdt_stats,
        "exclusions": exclusion_counts,
        "exclusion_sources": {
            "city412": str(args.city412),
            "old_t_holdout50": str(args.old_t_holdout),
            "mainline70_diagnostics": str(args.mainline_diagnostics),
            "rgbdt_external_holdout": str(args.rgbdt_external_manifest),
        },
        "existing_mainline_review_seed": {
            "release": str(args.mainline_release),
            "approved_unique_train": approved_counts.get("approved_unique_train"),
            "approved_diagnostic": approved_counts.get("approved_diagnostic"),
            "included_in_new_manifest": False,
            "note": "Existing reviewed records remain separate; the 70 diagnostics are excluded by source group.",
        },
        "public_manifest_fields": sorted(public_rows[0].keys()) if public_rows else [],
        "private_label_fields": sorted(private_labels[0].keys()) if private_labels else [],
        "stage_policy": {
            "foundation_candidate": "Run real proposal generation from Query and available image modalities; evaluate against private boxes offline.",
            "observation": "Create only from generated candidates and real executed visual-tool observations.",
            "next_action": "Create only from an actual observation trace; no action is inferred from GT or Query alone.",
            "rgbdt500": "Queryless frames are annotation jobs only and are excluded from manifest.jsonl until natural Queries are written.",
            "candidate_pool": "These are independent decision-preparation candidates, not approved mainline training rows.",
        },
        "gt_free_fields": ["id", "source", "source_id", "scene_id", "image_group", "query",
                           "images", "available_modalities", "depth_encoding",
                           "depth_visual_encoding", "ir_rgb_registration", "registration_source"],
        "split_semantics": "3000 requested items means 2800 explicit Query candidates plus 200 queryless RGBDT annotation jobs; candidate_pool is not a mainline train split. Preserved holdouts are listed in group_splits.jsonl.",
    }
    inventory["group_splits"] = {
        split: sum(row["split"] == split for row in group_splits)
        for split in sorted({row["split"] for row in group_splits})
    }
    inventory["group_splits_file"] = "group_splits.jsonl"
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_docs(output_dir, inventory, args)
    return inventory


def write_docs(output_dir: Path, inventory: dict, args) -> None:
    actual = inventory["actual"]
    readme = f"""# TriGround 对象决策数据准备\n\n"""
    readme += (f"本包生成 {actual['globally_unique_explicit_queries']} 条无重复自然语言 Query，"
               f"以及 {actual['rgbdt_annotation_jobs']} 个没有 Query 的 RGBDT500 图片标注任务。"
               "查询和图片任务分开计数；未把无 Query 帧说成已标注训练题。\n\n")
    readme += "## 文件\n\n"
    readme += "- `manifest.jsonl`：2800 条 GT-free Query 输入，供真实 initial baseline、Query 解析和 candidate 阶段使用。\n"
    readme += "- `cloud_manifest.jsonl`：相同 GT-free 清单，已把本机 `F:/AIC` 路径映射到云端 `/root/autodl-tmp/rematch_20260922`。\n"
    readme += "- `private_labels_offline.jsonl`：源数据目标框，仅供离线候选覆盖和评分；不要传给模型或控制器。\n"
    readme += "- `foundation_candidate_jobs.jsonl`、`observation_jobs.jsonl`、`next_action_jobs.jsonl`：分阶段待办状态。后两阶段需要实际候选和真实观察轨迹，不预填动作标签。\n"
    readme += "- `annotation_jobs.jsonl`：RGBDT500 200 帧/序列的人工 Query 构题任务。图像已选定，任务文本为空，不含源框。对应源框只在私有标签文件中。\n"
    readme += "- `cloud_annotation_jobs.jsonl`：RGBDT 标注任务的云端图像路径版本。\n"
    readme += "- `group_splits.jsonl`：所选 candidate pool 图组和保留评测/试点图组，明确没有把 candidate pool 宣称为主线训练集。\n"
    readme += "- `images/rgbdt500/`：只含选定帧的 RGB、IR、uint16 原始 Depth 和固定可视化，不解压整个官方 ZIP。\n"
    readme += "- `inventory.json`：来源规模、留出排除、实际样本数、缺额与模态编码。\n\n"
    readme += "## 实际 Query 数\n\n"
    for source in ("city", "rgbt", "robo"):
        count = inventory["actual"][f"{source}_queries"]
        stat = inventory["sources"][source]
        readme += f"- {source}: {count} / {inventory['requested'][f'{source}_queries']}；排除后 {stat['eligible_rows']} 条、{stat['eligible_unique_groups']} 组。\n"
    readme += (f"- RGBDT500: {actual['rgbdt_annotation_jobs']} / {inventory['requested']['rgbdt_annotation_jobs']} 标注任务，"
               f"当前自然 Query 数为 0。\n\n")
    readme += "## 模态与隔离\n\n"
    readme += "City 的来源提示确认 RGB/IR/Depth 共享场景坐标，使用 `ir_rgb_registration=normalized_shared_frame`。RGBT GroundBench 的论文报告 RGB/thermal 空间配准，IR 行据此标记 `normalized_shared_frame`；RoboRefIt 与 RGBDT500 的配准状态未知。缺失模态不造占位图。City 标注的 Depth 编码为 `city_mm`；RoboRefIt 与 RGBDT500 的原始单位未知，均写 `depth_encoding=unknown`，RGBDT 可视化只按每图原始数值 P01–P99 拉伸、不作远近语义解释。\n\n"
    readme += "City412、旧 T holdout50、主线70条诊断和 RGBDT500 external holdout 均从新样本池排除，原材料未修改。RGBDT 外部清单实际覆盖 50 帧/20 个序列；取样只从 acquisition plan 标记的 280 个剩余训练序列中抽取。\n\n"
    readme += "## 复现命令\n\n"
    readme += "从 `F:/AIC` 运行：\n\n```powershell\n$env:PYTHONPATH = 'code'\npython -m experiments.visual_agent.prepare_object_data `\n"
    readme += f"  --output-dir '{args.output_dir}' `\n  --seed {args.seed}\n```\n\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    handoff = "# Object data handoff\n\n"
    handoff += f"- Prepared manifest: `{output_dir / 'manifest.jsonl'}` ({actual['explicit_natural_queries']} real, globally unique queries).\n"
    handoff += f"- Queryless RGBDT worklist: `{output_dir / 'annotation_jobs.jsonl'}` ({actual['rgbdt_annotation_jobs']} jobs).\n"
    handoff += "- Next: run real baseline and text-only query parse on `manifest.jsonl`; create candidates only after both outputs are complete. Then collect actual visual-tool observations before assigning next-action supervision.\n"
    handoff += "- RGBDT rows enter the public query manifest only after a human writes a natural query and it uniquely identifies the visible target. Never copy `bbox_xyxy_normalized` into model input.\n"
    (output_dir / "HANDOFF.md").write_text(handoff, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-train", type=Path,
                        default=ROOT / "results/gu_diagnosis_20260927/source_snapshot/city_train.json")
    parser.add_argument("--city-val", type=Path,
                        default=ROOT / "results/gu_diagnosis_20260927/source_snapshot/city_val.json")
    parser.add_argument("--city412", type=Path,
                        default=ROOT / "results/visual_agent/preparation_seed2026_stratified/city412.jsonl")
    parser.add_argument("--old-t-holdout", type=Path,
                        default=ROOT / "results/visual_agent/preparation_seed2026_stratified/t_holdout50.jsonl")
    parser.add_argument("--mainline-diagnostics", type=Path,
                        default=ROOT / "results/triground_abv_execution_20260928/deployment_600_seed2026/manifests/diagnostics/normal.json")
    parser.add_argument("--mainline-release", type=Path,
                        default=ROOT / "results/triground_abv_execution_20260928/reviewed_data/data/release_600_seed2026/release.json")
    parser.add_argument("--rgbt-manifest", type=Path,
                        default=ROOT / "results/visual_agent/decision_rebuild_20260929/source/train6000.jsonl")
    parser.add_argument("--robo-manifest", type=Path,
                        default=ROOT / "data/external/RoboRefIt/subset/converted/train.jsonl")
    parser.add_argument("--rgbdt-zip", type=Path, default=ROOT / "data/external/RGBDT500/Train.zip")
    parser.add_argument("--rgbdt-plan", type=Path, default=ROOT / "data/external/RGBDT500/acquisition_plan.json")
    parser.add_argument("--rgbdt-external-manifest", type=Path,
                        default=ROOT / "data/external/RGBDT500/external_review_manifest.jsonl")
    parser.add_argument("--city-image-root", default="/root/autodl-tmp/rematch_20260922/data/city/train")
    parser.add_argument("--rgbt-image-root", default="/root/autodl-tmp/rematch_20260922/data/external/RGBT-GroundBench/converted")
    parser.add_argument("--robo-image-root", default=str(ROOT / "data/external/RoboRefIt/subset/converted"))
    parser.add_argument("--robo-depth-visual-root", default=str(ROOT / "data/external/RoboRefIt/subset/converted"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-root", default=str(ROOT))
    parser.add_argument("--cloud-root", default="/root/autodl-tmp/rematch_20260922")
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--city-count", type=int, default=1200)
    parser.add_argument("--rgbt-count", type=int, default=1000)
    parser.add_argument("--robo-count", type=int, default=600)
    parser.add_argument("--rgbdt-count", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.city_count, args.rgbt_count, args.robo_count, args.rgbdt_count) < 0:
        raise ValueError("requested counts must be non-negative")
    result = prepare(args)
    print(json.dumps({"actual": result["actual"], "shortfall": result["shortfall"],
                      "outputs": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
