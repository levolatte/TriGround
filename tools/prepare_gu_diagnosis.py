"""Prepare CPU-only manifests for the 2026-09-27 G/U diagnosis."""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict, deque
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path, PurePosixPath
from typing import Any


SEED = 2026
MODALITIES = ("rgb", "infrared", "depth")
U_QUOTA = {"rgb": 217, "infrared": 128, "depth": 55}
QUERY_MARKER = "Locate the object described by this query: "
ALT_QUERY_MARKER = "Locate the target described by: "
DEPTH_NOTE = (
    "This depth visualization uses a fixed inverse mapping of the raw sensor values: "
    "smaller positive stored values are brighter and zero is black. "
    "Its physical distance unit is not established."
)
OUTPUT_SUFFIXES = ("_metadata.jsonl", "_provenance.json")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL records")
    return rows


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _query(row: dict[str, Any]) -> str:
    prompt = row["conversations"][0]["value"]
    marker = next((m for m in (QUERY_MARKER, ALT_QUERY_MARKER) if m in prompt), None)
    if marker is None:
        raise ValueError(f"sample {row['id']} has no supported Query marker")
    query = prompt.split(marker, 1)[1].split("\nReturn", 1)[0]
    if not query.strip():
        raise ValueError(f"sample {row['id']} has an empty Query")
    return query


def _normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _query_key(source_id: str, query: str) -> str:
    return f"{source_id}::{query}"


def _qwen_box(box: list[float]) -> list[int]:
    if len(box) != 4:
        raise ValueError(f"expected four xyxy coordinates, got {box}")
    values = [Decimal(str(value)) for value in box]
    if not all(Decimal(0) <= value <= Decimal(1) for value in values):
        raise ValueError(f"coordinates outside [0, 1]: {box}")
    if values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError(f"empty xyxy box: {box}")
    result = [int((value * 1000).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) for value in values]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"box collapsed after 0-1000 quantization: {box} -> {result}")
    return result


def _answer(box: list[float]) -> str:
    return json.dumps({"bbox_2d": _qwen_box(box)}, separators=(",", ":"))


def _canonical_prompt(query: str, target_modality: str) -> str:
    if target_modality not in MODALITIES:
        raise ValueError(f"unsupported target modality: {target_modality}")
    return (
        "<image>\n<image>\n<image>\n"
        "These are views of the same scene in this order: RGB, infrared, depth. "
        f"{DEPTH_NOTE} Locate the object described by this query: {query}\n"
        f"Return the box in the {target_modality.upper() if target_modality == 'rgb' else target_modality} image coordinates. "
        'Return only JSON in this exact form with coordinates normalized to 0-1000: '
        '{"bbox_2d":[x1,y1,x2,y2]}'
    )


def _city_prompt(query: str, inputs: tuple[str, ...]) -> str:
    if not inputs or inputs[0] != "rgb":
        raise ValueError(f"City input combination must start with RGB: {inputs}")
    descriptions = {"rgb": "RGB", "infrared": "infrared", "depth": "depth"}
    if any(modality not in descriptions for modality in inputs) or len(set(inputs)) != len(inputs):
        raise ValueError(f"unsupported City input combination: {inputs}")
    image_tokens = "\n".join("<image>" for _ in inputs)
    if inputs == ("rgb",):
        intro = "The image is the RGB view."
    else:
        intro = f"These are views of the same scene in this order: {', '.join(descriptions[m] for m in inputs)}."
        if "depth" in inputs:
            intro += " In the depth image, brighter valid pixels are nearer, darker valid pixels are farther, and black pixels are invalid."
    return (
        f"{image_tokens}\n{intro} Locate the object described by this query: {query}\n"
        "Return the box in the RGB image coordinates. Return only JSON in this exact form with coordinates normalized to 0-1000: "
        '{"bbox_2d":[x1,y1,x2,y2]}'
    )


def _city_image_path(root: str, value: str) -> str:
    raw = value.replace("\\", "/")
    path = PurePosixPath(raw)
    if path.is_absolute():
        return path.as_posix()
    return (PurePosixPath(root.replace("\\", "/")) / path).as_posix()


def _city_row(row: dict[str, Any], city_root: str, inputs: tuple[str, ...], sample_id: str | None = None) -> dict[str, Any]:
    source_paths = dict(zip(MODALITIES, row["image"]))
    image_paths = [_city_image_path(city_root, source_paths[modality]) for modality in inputs]
    sample = copy.deepcopy(row)
    sample["id"] = sample_id or row["id"]
    sample["image"] = image_paths[0] if len(image_paths) == 1 else image_paths
    sample["conversations"] = [
        {"from": "human", "value": _city_prompt(_query(row), inputs)},
        copy.deepcopy(row["conversations"][1]),
    ]
    return sample


def _round_robin_sample(
    rows: list[dict[str, Any]], count: int, group_of, rng: random.Random
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(group_of(row))].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key in keys:
        pool = groups[key].copy()
        rng.shuffle(pool)
        queues[key] = deque(pool)
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if queues[key]:
                selected.append(queues[key].popleft())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"only {len(selected)} rows available across groups; need {count}")
        rng.shuffle(keys)
    return selected


def _candidate_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates = {row["task_id"]: row for row in rows}
    if len(candidates) != len(rows):
        raise ValueError("released candidate task IDs must be unique")
    return candidates


def _raw_gt(
    native_row: dict[str, Any], metadata: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    candidate = candidates.get(native_row["id"])
    if candidate is None:
        raise ValueError(f"raw candidate missing for released task {native_row['id']}")
    sample_parts = native_row["id"].split("::")
    if len(sample_parts) < 3:
        raise ValueError(f"released sample ID is not bundle::query::modality: {native_row['id']}")
    bundle_id, sample_query_id, sample_modality = sample_parts[0], sample_parts[1], sample_parts[-1]
    if sample_modality != metadata["target_modality"] or candidate["output_modality"] != sample_modality:
        raise ValueError(f"released sample/candidate modality mismatch: {native_row['id']}")
    source_id = metadata["source_id"]
    if candidate["bundle_id"] != bundle_id or not source_id.startswith(bundle_id + "::"):
        raise ValueError(f"released source_id does not match candidate bundle: {native_row['id']}")
    query = _query(native_row)
    if query != candidate["query"]:
        raise ValueError(f"released Query differs from raw candidate: {native_row['id']}")
    box = candidate["bbox_xyxy_normalized"]
    rgb_box = candidate.get("rgb_gt_bbox_xyxy_normalized")
    if box is None or rgb_box is None:
        raise ValueError(f"raw candidate lacks target or RGB GT box: {native_row['id']}")
    answer = json.loads(native_row["conversations"][1]["value"])["bbox_2d"]
    if answer != _qwen_box(box):
        raise ValueError(f"raw candidate box does not reproduce released integer label: {native_row['id']}")
    query_key = _query_key(source_id, query)
    return {
        "bbox": box,
        "target_modality": sample_modality,
        "source_id": source_id,
        "query_key": query_key,
        "query": query,
        "bundle_id": bundle_id,
        "candidate_query_id": sample_query_id,
        "candidate_origin_query_id": candidate.get("origin_query_id", sample_query_id),
        "object_id": candidate["object_id"],
        "rgb_bbox": rgb_box,
        "visible": native_row["image"][0],
        "infrared": native_row["image"][1],
        "depth": native_row["image"][2],
    }


def _write_manifest(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    _write_json(output_dir / f"{name}.json", rows)
    _write_jsonl(output_dir / f"{name}_metadata.jsonl", metadata)
    _write_json(output_dir / f"{name}_provenance.json", provenance)


def _city_gt_record(source: dict[str, Any], source_row: dict[str, Any], city_root: str) -> dict[str, Any]:
    images = [_city_image_path(city_root, path) for path in source_row["image"]]
    return {
        "bbox": source["bbox"],
        "query": source["query"],
        "class_name": source.get("class_name"),
        "source_id": source_row["id"],
        "visible": images[0],
        "infrared": images[1],
        "depth": images[2],
        "bbox_source": "--city-gt original floating point City GT",
    }


def _new_sample(
    native_row: dict[str, Any], raw_gt: dict[str, Any], target_modality: str,
    target_box: list[float], sample_id: str,
) -> dict[str, Any]:
    sample = copy.deepcopy(native_row)
    sample["id"] = sample_id
    sample["conversations"] = [
        {"from": "human", "value": _canonical_prompt(raw_gt["query"], target_modality)},
        {"from": "gpt", "value": _answer(target_box)},
    ]
    return sample


def build(args: argparse.Namespace) -> dict[str, Any]:
    old_dir = args.old_root / "manifests" / "seed2026"
    old_g = _read_rows(old_dir / "g_train.json")
    old_u = _read_rows(old_dir / "u_train.json")
    old_meta = _read_rows(old_dir / "metadata.jsonl")
    if not (len(old_g) == len(old_u) == len(old_meta) == 1600):
        raise ValueError("old seed2026 G/U manifests and metadata must contain 1600 rows")
    if [r["id"] for r in old_g] != [r["id"] for r in old_u]:
        raise ValueError("old G/U identity order differs")
    if [r["presentation_id"] for r in old_meta] != [r["id"] for r in old_g]:
        raise ValueError("old metadata does not match manifest identity order")

    released_rows = _read_rows(args.released_jsonl)
    released_meta_rows = _read_rows(args.released_metadata)
    candidate_rows = _read_rows(args.released_candidates)
    city_rows = _read_rows(args.city_native)
    val_rows = _read_rows(args.city_val)
    city_gt = _read_object(args.city_gt)
    released_by_id = {row["id"]: row for row in released_rows}
    released_meta = {row["id"]: row for row in released_meta_rows}
    if len(released_by_id) != len(released_rows) or len(released_meta) != len(released_meta_rows):
        raise ValueError("released rows and metadata must have unique IDs")
    if set(released_by_id) != set(released_meta):
        raise ValueError("released native IDs do not match released metadata IDs")
    candidate_by_id = _candidate_index(candidate_rows)

    raw_gt_by_id: dict[str, dict[str, Any]] = {}
    raw_boxes_by_source: dict[str, dict[str, list[float]]] = defaultdict(dict)
    tasks_by_key: dict[str, dict[str, Any]] = {}
    query_by_source: dict[str, str] = {}
    for sample_id, native in released_by_id.items():
        meta = released_meta[sample_id]
        if meta.get("review_status") != "human_accepted":
            raise ValueError(f"released metadata is not human accepted: {sample_id}")
        gt = _raw_gt(native, meta, candidate_by_id)
        if gt["source_id"] in query_by_source and query_by_source[gt["source_id"]] != gt["query"]:
            raise ValueError(f"one released source_id has multiple Query strings: {gt['source_id']}")
        query_by_source[gt["source_id"]] = gt["query"]
        raw_boxes_by_source[gt["source_id"]][gt["target_modality"]] = gt["bbox"]
        previous_rgb = raw_boxes_by_source[gt["source_id"]].get("rgb")
        if previous_rgb is not None and previous_rgb != gt["rgb_bbox"]:
            raise ValueError(f"RGB GT differs across candidate rows for {gt['source_id']}")
        raw_boxes_by_source[gt["source_id"]]["rgb"] = gt["rgb_bbox"]
        gt.update({key: native["image"][i] for i, key in enumerate(("visible", "infrared", "depth"))})
        raw_gt_by_id[sample_id] = gt
        tasks_by_key.setdefault(gt["source_id"], {})[gt["target_modality"]] = {
            "native": native, "metadata": meta, "gt": gt,
        }

    if len(tasks_by_key) != 98 or len(released_rows) != 181:
        raise ValueError(f"expected 98 source Queries and 181 released tasks; got {len(tasks_by_key)} and {len(released_rows)}")
    for source_id, modalities in tasks_by_key.items():
        if "rgb" not in modalities or set(modalities) - set(MODALITIES):
            raise ValueError(f"released source Query lacks accepted RGB supervision: {source_id}")
        if len({tuple(value["native"]["image"]) for value in modalities.values()}) != 1:
            raise ValueError(f"released modal rows do not share input image paths: {source_id}")

    city_by_id = {row["id"]: row for row in city_rows}
    if len(city_by_id) != len(city_rows):
        raise ValueError("City training sample IDs must be unique")
    old_city_meta = [row for row in old_meta if row["source"] == "city"]
    old_city_ids = {row["source_task_id"] for row in old_city_meta}
    old_city_queries = {_normalized_query(row["query"]) for row in old_city_meta}
    old_new_rows = [row for row in old_meta if row["source"] != "city"]
    if len(old_city_meta) != 1200 or len(old_new_rows) != 400:
        raise ValueError(f"old schedule must have 1200 City and 400 released slots; got {len(old_city_meta)}, {len(old_new_rows)}")

    exposure_counts = Counter(row["source_task_id"] for row in old_new_rows)
    source_query_frequency: Counter[str] = Counter()
    for row in old_new_rows:
        release_meta = released_meta[row["source_task_id"]]
        source_query_frequency[release_meta["source_id"]] += 1
        if release_meta["target_modality"] != row["output_modality"]:
            raise ValueError(f"old schedule target modality mismatch: {row['source_task_id']}")
    if set(source_query_frequency) != set(tasks_by_key) or sum(source_query_frequency.values()) != 400:
        raise ValueError("old 400 slots do not cover exactly the 98 released source Queries")
    if Counter(source_query_frequency.values()) != Counter({2: 26, 3: 2, 4: 33, 5: 22, 6: 7, 7: 6, 8: 2}):
        raise ValueError("old 400 Query frequency profile differs from the audited seed2026 profile")
    if Counter(row["output_modality"] for row in old_new_rows) != Counter(U_QUOTA):
        raise ValueError("old 400 target-modality quotas do not match 217/128/55")

    released_query_texts = {_normalized_query(value) for value in query_by_source.values()}
    city_candidates = []
    seen_candidate_queries = set()
    for row in city_rows:
        query = _query(row)
        normalized = _normalized_query(query)
        if row["id"] in old_city_ids or normalized in old_city_queries or normalized in released_query_texts:
            continue
        if normalized in seen_candidate_queries:
            continue
        seen_candidate_queries.add(normalized)
        city_candidates.append(row)
    if len(city_candidates) < 98:
        raise ValueError(f"only {len(city_candidates)} unseen City Query candidates remain; need 98")
    mapped_city_rows = _round_robin_sample(city_candidates, 98, lambda row: row["image"][0], random.Random(SEED))
    source_keys = sorted(tasks_by_key)
    b_mapping: dict[str, dict[str, Any]] = {}
    mapping_rows = []
    for source_id, city_row in zip(source_keys, mapped_city_rows):
        query = query_by_source[source_id]
        city_query = _query(city_row)
        if _normalized_query(city_query) in old_city_queries or _normalized_query(city_query) == _normalized_query(query):
            raise AssertionError(f"B mapping selected an exposed or identical Query: {city_row['id']}")
        if city_row["id"] in old_city_ids:
            raise AssertionError(f"B mapping selected an old City identity: {city_row['id']}")
        frequency = source_query_frequency[source_id]
        city_group = _city_image_path(args.city_root, city_row["image"][0])
        city_bbox = json.loads(city_row["conversations"][1]["value"])["bbox_2d"]
        city_bbox_norm = [coordinate / 1000 for coordinate in city_bbox]
        mapping = {
            "new_source_query_key": _query_key(source_id, query),
            "source_id": source_id,
            "query": query,
            "city_source_id": city_row["id"],
            "city_query": city_query,
            "city_group": city_group,
            "frequency": frequency,
            "target_bbox_qwen1000": city_bbox,
            "target_bbox_xyxy_0_1_from_qwen1000": city_bbox_norm,
            "target_area_fraction_from_qwen1000": (city_bbox_norm[2] - city_bbox_norm[0]) * (city_bbox_norm[3] - city_bbox_norm[1]),
        }
        b_mapping[source_id] = {"city_row": city_row, "mapping": mapping}
        mapping_rows.append(mapping)

    new_b: list[dict[str, Any]] = []
    new_gstar: list[dict[str, Any]] = []
    new_ustar: list[dict[str, Any]] = []
    b_meta: list[dict[str, Any]] = []
    g_meta: list[dict[str, Any]] = []
    u_meta: list[dict[str, Any]] = []
    modality_exposures = Counter()
    for schedule_row in old_new_rows:
        sample_id = schedule_row["source_task_id"]
        old_native = released_by_id[sample_id]
        old_release_meta = released_meta[sample_id]
        source_id = old_release_meta["source_id"]
        target_modality = old_release_meta["target_modality"]
        gt = raw_gt_by_id[sample_id]
        presentation_id = schedule_row["presentation_id"]

        b_row = _city_row(b_mapping[source_id]["city_row"], args.city_root, MODALITIES, presentation_id)
        g_target = "rgb"
        g_row = _new_sample(old_native, gt, g_target, raw_boxes_by_source[source_id][g_target], presentation_id)
        u_row = _new_sample(old_native, gt, target_modality, gt["bbox"], presentation_id)
        new_b.append(b_row)
        new_gstar.append(g_row)
        new_ustar.append(u_row)
        modality_exposures[target_modality] += 1

        common = {
            "ordinal": schedule_row["ordinal"],
            "presentation_id": presentation_id,
            "source_id": source_id,
            "query_key": gt["query_key"],
            "query": gt["query"],
            "group": old_release_meta["group_id"],
            "source_task_id": sample_id,
            "input_modalities": ["rgb", "infrared", "depth"],
            "depth_policy": old_release_meta["depth_policy"],
            "is_new": True,
        }
        b_meta.append({
            "ordinal": schedule_row["ordinal"],
            "presentation_id": presentation_id,
            "source": "city",
            "source_id": b_mapping[source_id]["city_row"]["id"],
            "source_task_id": b_mapping[source_id]["city_row"]["id"],
            "group": b_mapping[source_id]["mapping"]["city_group"],
            "query": b_mapping[source_id]["mapping"]["city_query"],
            "query_key": b_mapping[source_id]["city_row"]["id"],
            "target_modality": "rgb",
            "output_modality": "rgb",
            "is_new": False,
            "city_source_id": b_mapping[source_id]["city_row"]["id"],
            "city_group": b_mapping[source_id]["mapping"]["city_group"],
            "frequency": b_mapping[source_id]["mapping"]["frequency"],
            "mapped_from_source_id": source_id,
            "mapped_from_query_key": _query_key(source_id, gt["query"]),
            "mapped_from_query": gt["query"],
            "mapped_from_source_task_id": sample_id,
        })
        g_meta.append({**common, "source": "rgbdt", "target_modality": "rgb", "output_modality": "rgb", "paired_condition": "Gstar"})
        u_meta.append({**common, "source": "rgbdt", "target_modality": target_modality, "output_modality": target_modality, "paired_condition": "Ustar"})

    def assemble_train(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
        result = []
        new_index = 0
        for schedule_row, old_row in zip(old_meta, old_rows):
            if schedule_row["source"] == "city":
                result.append(copy.deepcopy(old_row))
            else:
                result.append(copy.deepcopy(new_rows[new_index]))
                new_index += 1
        if new_index != 400:
            raise AssertionError(f"assembled {new_index} new rows, expected 400")
        return result

    train_b = assemble_train(old_g, new_b)
    train_gstar = assemble_train(old_g, new_gstar)
    train_ustar = assemble_train(old_u, new_ustar)
    train_meta: dict[str, list[dict[str, Any]]] = {"train_b": [], "train_gstar": [], "train_ustar": []}
    b_index = g_index = u_index = 0
    for old_row in old_meta:
        if old_row["source"] == "city":
            copied = {
                **old_row, "source": "city", "is_new": False,
                "target_modality": "rgb", "output_modality": "rgb",
                "query_key": old_row.get("query_key", f"city::{old_row['source_task_id']}"),
            }
            train_meta["train_b"].append(copy.deepcopy(copied))
            train_meta["train_gstar"].append(copy.deepcopy(copied))
            train_meta["train_ustar"].append(copy.deepcopy(copied))
        else:
            train_meta["train_b"].append(b_meta[b_index])
            train_meta["train_gstar"].append(g_meta[g_index])
            train_meta["train_ustar"].append(u_meta[u_index])
            b_index += 1
            g_index += 1
            u_index += 1

    # Select a fixed 96-row diagnostic panel: one sample from each of the 78 RGB frames,
    # then round-robin extras so frame counts differ by at most one where possible.
    city_val_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    val_ids = set()
    for row in val_rows:
        if row.get("split") != "val":
            raise ValueError(f"City validation sample has non-val split: {row['id']}")
        if row["id"] in val_ids:
            raise ValueError(f"duplicate City validation ID: {row['id']}")
        val_ids.add(row["id"])
        city_val_groups[row["image"][0]].append(row)
        if row["id"] not in city_gt:
            raise ValueError(f"City GT missing validation sample: {row['id']}")
        if city_gt[row["id"]]["query"] != _query(row):
            raise ValueError(f"City raw GT Query differs from native validation prompt: {row['id']}")
    if len(city_val_groups) != 78 or len(val_rows) < 96:
        raise ValueError(f"expected City val with 78 image groups and at least 96 rows; got {len(city_val_groups)} groups/{len(val_rows)} rows")
    city96_rows = _round_robin_sample(val_rows, 96, lambda row: row["image"][0], random.Random(SEED))
    selected_ids = [row["id"] for row in city96_rows]
    selected_gt = {
        sample_id: _city_gt_record(city_gt[sample_id], row, args.city_root)
        for sample_id, row in zip(selected_ids, city96_rows)
    }
    combos = {
        "city96_rgb": ("rgb",),
        "city96_rgb_ir": ("rgb", "infrared"),
        "city96_rgb_depth": ("rgb", "depth"),
        "city96_trimodal": MODALITIES,
    }
    city_manifests: dict[str, list[dict[str, Any]]] = {}
    city_meta: dict[str, list[dict[str, Any]]] = {}
    for name, inputs in combos.items():
        rows = []
        meta = []
        for row in city96_rows:
            sample = _city_row(row, args.city_root, inputs)
            rows.append(sample)
            meta.append({
                "id": row["id"], "source": "city_val", "group": row["image"][0],
                "query": _query(row), "query_key": row["id"], "target_modality": "rgb", "output_modality": "rgb",
                "input_modalities": list(inputs), "depth_policy": "millimeter" if "depth" in inputs else None,
            })
        city_manifests[name] = rows
        city_meta[name] = meta

    # Keep the audited eight pressure identities, but regenerate released prompts with the new template.
    old_pressure16 = _read_rows(old_dir / "pressure16.json")
    pressure2city = _read_rows(old_dir / "pressure2city.json")
    old_summary = _read_object(old_dir / "summary.json")
    pressure_source_ids = old_summary["pressure16"]["source_task_ids"]
    if len(old_pressure16) != 16 or len(pressure_source_ids) != 8 or len(pressure2city) != 2:
        raise ValueError("old pressure manifests must have 16 and 2 rows")
    if [row["id"] for row in old_pressure16[:2]] != [row["id"] for row in pressure2city]:
        raise ValueError("old pressure2city is not the first two pressure16 examples")
    pressure16 = []
    pressure16_meta = []
    for repeat in range(2):
        for representative_index, source_task_id in enumerate(pressure_source_ids):
            pressure_id = f"gu_diag_pressure_{repeat * 8 + representative_index + 1:06d}"
            if source_task_id in released_by_id:
                native = released_by_id[source_task_id]
                meta = released_meta[source_task_id]
                gt = raw_gt_by_id[source_task_id]
                sample = _new_sample(native, gt, meta["target_modality"], gt["bbox"], pressure_id)
                pressure16_meta.append({
                    "id": pressure_id, "source": "rgbdt", "source_task_id": source_task_id,
                    "source_id": meta["source_id"], "query_key": gt["query_key"], "query": gt["query"],
                    "target_modality": meta["target_modality"], "repeat": repeat + 1,
                    "representative_index": representative_index,
                })
            else:
                sample = copy.deepcopy(old_pressure16[representative_index])
                sample["id"] = pressure_id
                pressure16_meta.append({
                    "id": pressure_id, "source": "city", "source_task_id": source_task_id,
                    "source_id": source_task_id, "query_key": source_task_id,
                    "query": _query(sample), "target_modality": "rgb", "repeat": repeat + 1,
                    "representative_index": representative_index,
                })
            pressure16.append(sample)
    pressure2city_meta = []
    for source_task_id, sample in zip(pressure_source_ids[:2], pressure2city):
        pressure2city_meta.append({
            "id": sample["id"], "source": "city", "source_task_id": source_task_id,
            "source_id": source_task_id, "query_key": source_task_id, "query": _query(sample),
            "target_modality": "rgb", "output_modality": "rgb",
        })
    env8_rows = val_rows[:8]
    env8_meta = [{
        "id": row["id"], "source": "city_val", "index": index,
        "group": row["image"][0], "query": _query(row), "query_key": row["id"],
        "input_modalities": list(MODALITIES),
    } for index, row in enumerate(env8_rows)]

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_args = {
        "old_root": str(args.old_root), "released_jsonl": str(args.released_jsonl),
        "released_metadata": str(args.released_metadata), "released_candidates": str(args.released_candidates),
        "city_native": str(args.city_native), "city_root": args.city_root,
        "city_val": str(args.city_val), "city_gt": str(args.city_gt), "seed": SEED,
    }
    datasets: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    common_train_provenance = {
        "seed": SEED, "old_order_source": str(old_dir / "metadata.jsonl"),
        "old_city_rows_copied_verbatim": 1200, "new_slots": 400,
        "released_query_count": 98, "frequency_source": str(old_dir / "metadata.jsonl"),
        "raw_gt_source": str(args.released_candidates), "prompt_version": "gu_diag_v1", "inputs": source_args,
    }
    datasets["train_b"] = (train_b, train_meta["train_b"], {
        **common_train_provenance, "condition": "B: replace old new slots with unseen City Queries",
        "query_mapping": str(output_dir / "b_query_mapping.jsonl"),
    })
    datasets["train_gstar"] = (train_gstar, train_meta["train_gstar"], {
        **common_train_provenance, "condition": "Gstar: canonical prompt with RGB target boxes",
        "depth_policy": "sensor_linear_20000",
    })
    datasets["train_ustar"] = (train_ustar, train_meta["train_ustar"], {
        **common_train_provenance, "condition": "Ustar: same canonical prompt with RGB/IR/Depth target boxes",
        "target_modality_quota": dict(U_QUOTA), "depth_policy": "sensor_linear_20000",
    })
    for name in ("fit_legacy181", "fit_canonical_aux83"):
        fit_rows = []
        fit_meta = []
        for row in released_rows:
            gt = raw_gt_by_id[row["id"]]
            meta = released_meta[row["id"]]
            if name == "fit_canonical_aux83" and gt["target_modality"] == "rgb":
                continue
            item = copy.deepcopy(row)
            if name == "fit_canonical_aux83":
                item["conversations"] = [
                    {"from": "human", "value": _canonical_prompt(gt["query"], gt["target_modality"])},
                    {"from": "gpt", "value": _answer(gt["bbox"])},
                ]
            fit_rows.append(item)
            fit_meta.append({
                **meta, "query": gt["query"], "query_key": gt["query_key"],
                "target_modality": gt["target_modality"], "output_modality": gt["target_modality"],
                "bbox_source_id": gt["bundle_id"],
            })
        gt_rows = {row["id"]: {
            **{key: value for key, value in raw_gt_by_id[row["id"]].items() if key != "bbox"},
            "bbox": raw_gt_by_id[row["id"]]["bbox"],
            "bbox_source": "--released-candidates original floating point candidate box",
        } for row in fit_rows}
        if name == "fit_canonical_aux83" and len(fit_rows) != 83 or name == "fit_legacy181" and len(fit_rows) != 181:
            raise AssertionError(f"{name} has unexpected row count: {len(fit_rows)}")
        datasets[name] = (fit_rows, fit_meta, {
            "condition": "original released U prompts" if name == "fit_legacy181" else "canonical shared auxiliary prompt",
            "prompt_version": "legacy_released_u" if name == "fit_legacy181" else "gu_diag_v1",
            "raw_gt_file": str(output_dir / f"{name}_gt.json"), "raw_gt_source": str(args.released_candidates),
            "inputs": source_args,
        })
        # Assigned separately below because GT is a sidecar keyed by the unchanged sample ID.
        datasets[name + "_gt"] = ([], [], {"records": gt_rows})
    for name, rows in city_manifests.items():
        datasets[name] = (rows, city_meta[name], {
            "condition": "same 96 City validation IDs under a diagnostic input combination",
            "prompt_version": "city_diag_v1",
            "input_modalities": list(combos[name]), "raw_gt_file": str(output_dir / "city96_gt.json"),
            "raw_gt_source": str(args.city_gt), "selection_seed": SEED,
            "selection": "one sample from each of 78 RGB image groups, then round-robin to 96",
            "inputs": source_args,
        })
    datasets["pressure16"] = (pressure16, [], {
        "condition": "old pressure diagnostics: 8 representative rows repeated twice",
        "source": str(old_dir / "pressure16.json"), "inputs": source_args,
    })
    datasets["pressure2city"] = (pressure2city, pressure2city_meta, {
        "condition": "old two-City reload/evaluation subset",
        "source": str(old_dir / "pressure2city.json"), "inputs": source_args,
    })
    datasets["reload2"] = (copy.deepcopy(pressure2city), copy.deepcopy(pressure2city_meta), {
        "condition": "same two rows used for the reload comparison",
        "source": str(old_dir / "pressure2city.json"), "inputs": source_args,
    })
    datasets["env8"] = (copy.deepcopy(env8_rows), env8_meta, {
        "condition": "first eight City validation rows in source order",
        "prompt_version": "city_val_native_original",
        "source": str(args.city_val), "gt_source": str(args.city_gt), "m2_reference_checked_by": "run_gu_diagnosis.sh m2_env8_verify",
        "inputs": source_args,
    })

    for name, (rows, metadata, provenance) in datasets.items():
        if name.endswith("_gt"):
            _write_json(output_dir / f"{name}.json", provenance["records"])
        elif name == "city96_gt":
            _write_json(output_dir / "city96_gt.json", selected_gt)
        elif metadata or rows:
            _write_manifest(output_dir, name, rows, metadata, provenance)
        else:
            _write_json(output_dir / f"{name}.json", rows)
            _write_json(output_dir / f"{name}_provenance.json", provenance)
    _write_jsonl(output_dir / "b_query_mapping.jsonl", mapping_rows)
    _write_jsonl(output_dir / "b_query_mapping_metadata.jsonl", [{
        "source_id": row["source_id"], "query_key": row["new_source_query_key"],
        "city_source_id": row["city_source_id"], "city_group": row["city_group"],
        "frequency": row["frequency"], "target_bbox_qwen1000": row["target_bbox_qwen1000"],
        "target_area_fraction_from_qwen1000": row["target_area_fraction_from_qwen1000"],
    } for row in mapping_rows])
    _write_json(output_dir / "b_query_mapping_provenance.json", {
        "source": "released source Queries and full City training native manifest",
        "selection_seed": SEED,
        "exclusions": "old 1200 City IDs, old City Query text, and identical released Query text",
        "balance_group": "first image path",
        "count": len(mapping_rows),
    })
    _write_json(output_dir / "city96_gt.json", selected_gt)

    _write_json(output_dir / "city96_gt_provenance.json", {
        "source": str(args.city_gt), "selection_seed": SEED,
        "id_source": str(args.city_val), "count": len(selected_gt),
    })
    _write_manifest(output_dir, "env8", env8_rows, env8_meta, datasets["env8"][2])

    summary = {
        "seed": SEED,
        "train_rows": {"train_b": len(train_b), "train_gstar": len(train_gstar), "train_ustar": len(train_ustar)},
        "old_city_rows_preserved": 1200,
        "new_slot_rows": 400,
        "released_source_queries": len(tasks_by_key),
        "new_slot_target_modality_counts": dict(sorted(modality_exposures.items())),
        "source_query_frequency": dict(sorted(source_query_frequency.items())),
        "b_query_mapping_rows": len(mapping_rows),
        "b_city_groups": len({row["city_group"] for row in mapping_rows}),
        "b_query_mapping_frequency_sum": sum(row["frequency"] for row in mapping_rows),
        "city96_rows": len(city96_rows),
        "city96_groups": len({row["image"][0] for row in city96_rows}),
        "city96_ids_same_across_variants": all([row["id"] for row in city_manifests[name]] == selected_ids for name in combos),
        "fit_rows": {"fit_legacy181": 181, "fit_canonical_aux83": 83},
        "env8_rows": len(env8_rows),
        "pressure16_rows": len(pressure16),
        "reload2_rows": len(pressure2city),
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True, help="snapshot root containing manifests/seed2026")
    parser.add_argument("--released-jsonl", type=Path, required=True)
    parser.add_argument("--released-metadata", type=Path, required=True)
    parser.add_argument("--released-candidates", type=Path, required=True, help="original reviewed_evidence.jsonl for floating-point boxes")
    parser.add_argument("--city-native", type=Path, required=True)
    parser.add_argument("--city-root", required=True, help="POSIX City data root used in emitted image paths")
    parser.add_argument("--city-val", type=Path, required=True)
    parser.add_argument("--city-gt", type=Path, required=True, help="raw floating-point City GT keyed by validation ID")
    parser.add_argument("--output-dir", type=Path, required=True, help="must be new or empty")
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
