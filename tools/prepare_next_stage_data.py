"""Materialize C/D/T and object-evidence G/U schedules for native Qwen3-VL SFT.

External JSONL uses the existing staged records (rgb, optional infrared/aux,
optional depth, query, normalized xyxy bbox, scene_id). RGBDT500 records must
also carry review_status=human_accepted or approved_batch. The latter means the
pilot batch has passed the separately recorded human acceptance gate.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from tools.prepare_instance_candidates import confirmed_instance_task
from tools.prepare_qwen3vl_native_sft import (
    bbox_to_qwen1000,
    depth_mm_to_grayscale_rgb,
    native_prompt,
)


QUOTAS = {
    ("C", 1): {"city": 8000},
    ("C", 2): {"city": 4000},
    ("D", 1): {"city": 3200, "rgbdt": 2800, "rgbt": 1200, "robo": 800},
    ("T", 1): {"city": 3200, "rgbdt": 2800, "rgbt": 1200, "robo": 800},
    ("D", 2): {"city": 3000, "rgbdt": 1000},
    ("T", 2): {"city": 3000, "rgbdt": 1000},
    ("G", 1): {"city": 2400, "rgbdt": 3200, "rgbt": 1600, "robo": 800},
    ("U", 1): {"city": 2400, "rgbdt": 3200, "rgbt": 1600, "robo": 800},
    ("G", 2): {"city": 3000, "rgbdt": 1000},
    ("U", 2): {"city": 3000, "rgbdt": 1000},
}
LETTERS = "ABCDEF"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL records")
    if not rows:
        raise ValueError(f"{path}: empty source")
    return rows


def _path(value: str, root: Path) -> Path:
    result = (root / value).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _depth_policy(row: dict[str, Any], source: str) -> str:
    if source == "city":
        if row.get("split", "train") != "train":
            raise ValueError(f"City native sample {row['id']} is not in the training split")
        return "millimeter"
    raw = row.get("depth_policy", row.get("depth_unit"))
    if raw is None and source == "robo":
        return "visual"  # Original RoboRefIt files have no documented metric unit.
    if raw not in {"millimeter", "meter", "visual", "sensor_linear_20000"}:
        raise ValueError(f"{source}:{row['id']}: unsupported depth_policy {raw!r}")
    return raw


def _source_row(row: dict[str, Any], source: str, manifest: Path, city_root: Path) -> dict[str, Any]:
    if source == "city":
        images = row["image"]
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"City native sample {row['id']} is not trimodal")
        rgb, infrared, depth = (_path(value, city_root) for value in images)
        answer = json.loads(row["conversations"][1]["value"])["bbox_2d"]
        bbox = [float(value) / 1000 for value in answer]
        return {
            **row,
            "id": row["id"], "query": row["conversations"][0]["value"],
            "bbox": bbox, "rgb_path": rgb, "infrared_path": infrared,
            "depth_path": depth, "depth_policy": "millimeter",
            "depth_already_rendered": True, "group_id": str(rgb),
            "source": source, "conversations": row["conversations"],
        }
    if source == "rgbdt" and row.get("review_status") not in {"human_accepted", "approved_batch"}:
        raise ValueError(f"RGBDT500 row {row['id']} has not passed the human-review gate")
    if row.get("split", "train") != "train":
        raise ValueError(f"{source}:{row['id']} is not in the training split")
    rgb = _path(row["rgb"], manifest.parent)
    infrared_name = row.get("infrared", row.get("aux") if row.get("aux_type") == "ir" else None)
    infrared = _path(infrared_name, manifest.parent) if infrared_name else None
    depth = _path(row["depth"], manifest.parent) if row.get("depth") else None
    if source == "rgbdt" and (infrared is None or depth is None):
        raise ValueError(f"RGBDT500 row {row['id']} needs all three real views")
    if source == "rgbt" and (infrared is None or depth is not None):
        raise ValueError(f"RGB-T row {row['id']} needs exactly RGB and IR")
    if source == "robo" and (depth is None or infrared is not None):
        raise ValueError(f"RoboRefIt row {row['id']} needs exactly RGB and depth")
    bbox_to_qwen1000(row["bbox"])
    result = {
        **row, "source": source, "rgb_path": rgb,
        "infrared_path": infrared, "depth_path": depth,
        "depth_policy": _depth_policy(row, source) if depth else None,
        "group_id": str(row.get("merged_scene_group") or row.get("sequence_id") or row.get("scene_id") or rgb),
    }
    if not str(result["query"]).strip():
        raise ValueError(f"empty query: {source}:{row['id']}")
    return result


def _load_sources(args: argparse.Namespace, quotas: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    paths = {"city": args.city_native, "rgbdt": args.rgbdt, "rgbt": args.rgbt, "robo": args.robo}
    sources = {}
    for source in quotas:
        path = paths[source]
        if path is None:
            raise ValueError(f"{source} source is required for {args.branch} phase {args.phase}")
        sources[source] = [_source_row(row, source, path, args.city_root) for row in _read_rows(path)]
        ids = [row["id"] for row in sources[source]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{path}: duplicate sample IDs within {source}")
    return sources


def _balanced_pool(rows: list[dict[str, Any]], rng: random.Random):
    """Round-robin image/sequence groups, reshuffling each exhausted group."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    queues = {key: deque() for key in keys}
    while True:
        for key in keys:
            if not queues[key]:
                order = groups[key].copy()
                rng.shuffle(order)
                queues[key].extend(order)
            yield queues[key].popleft()
        rng.shuffle(keys)


def build_schedule(
    sources: dict[str, list[dict[str, Any]]], quotas: dict[str, int], seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tickets = [source for source, count in quotas.items() for _ in range(count)]
    rng.shuffle(tickets)
    pools = {source: _balanced_pool(rows, random.Random(seed + 101 * index))
             for index, (source, rows) in enumerate(sources.items())}
    return [next(pools[source]) for source in tickets]


def _metric_depth(path: Path, policy: str) -> np.ndarray:
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError(f"metric depth must be a 2D numeric image: {path}")
    depth = depth.astype(np.float64)
    if policy == "meter":
        depth *= 1000
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return depth


def _render_depth(row: dict[str, Any], output_dir: Path, cache: dict[tuple[Path, str], Path]) -> Path:
    source = row["depth_path"]
    policy = row["depth_policy"]
    if policy == "visual" or row.get("depth_already_rendered"):
        return source
    key = (source, policy)
    if key not in cache:
        destination = output_dir / "depth_rgb" / f"{len(cache):06d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if policy == "sensor_linear_20000":
            with Image.open(source) as image:
                raw = np.asarray(image)
            # The fixed visualization uses stored counts, not an assumed meter unit.
            if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"raw sensor depth must be a 2D integer image: {source}")
            visual = depth_mm_to_grayscale_rgb(raw)
        else:
            depth_mm = np.rint(_metric_depth(source, policy)).astype(np.int64)
            visual = depth_mm_to_grayscale_rgb(depth_mm)
        Image.fromarray(visual).save(destination)
        cache[key] = destination
    return cache[key]


def _modalities(row: dict[str, Any], output_dir: Path, cache: dict[tuple[Path, str], Path]) -> tuple[list[str], tuple[str, ...]]:
    images = [str(row["rgb_path"])]
    modalities = ["rgb"]
    if row["infrared_path"] is not None:
        images.append(str(row["infrared_path"]))
        modalities.append("infrared")
    if row["depth_path"] is not None:
        images.append(str(_render_depth(row, output_dir, cache)))
        modalities.append("depth")
    return images, tuple(modalities)


def _native_sample(row: dict[str, Any], exposure: int, output_dir: Path, cache: dict[tuple[Path, str], Path], task: str, rng: random.Random, auxiliary: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    images, modalities = _modalities(row, output_dir, cache)
    if task == "bbox":
        # City retains the exact original prompt and answer; no Query rewrite.
        if row["source"] == "city":
            prompt = row["conversations"][0]["value"]
            answer = row["conversations"][1]["value"]
        else:
            prompt = native_prompt(row["query"], modalities, row["depth_policy"] or "visual")
            answer = json.dumps({"bbox_2d": bbox_to_qwen1000(row["bbox"])}, separators=(",", ":"))
    elif task == "instance":
        instance = confirmed_instance_task(row)
        candidates = instance["candidates"].copy()
        rng.shuffle(candidates)
        lines = [f"{LETTERS[index]}: {bbox_to_qwen1000(candidate['bbox'])}" for index, candidate in enumerate(candidates)]
        answer = LETTERS[next(index for index, candidate in enumerate(candidates)
                              if candidate["object_id"] == instance["positive_object_id"])]
        prompt = (
            "\n".join("<image>" for _ in images) + "\n"
            + "Views are in this order: " + ", ".join(modalities) + ". "
            + "Given the whole scene and these candidate boxes (0-1000 coordinates), "
            + f"which box matches this query: {row['query']}\n"
            + "\n".join(lines) + "\nReturn only the candidate letter."
        )
    elif task == "nearfar":
        nearfar = _nearfar_task(row)
        if nearfar is None:
            raise ValueError(f"invalid near/far evidence: {row['id']}")
        objects = nearfar["objects"].copy()
        rng.shuffle(objects)
        answer = LETTERS[next(index for index, obj in enumerate(objects)
                              if obj["object_id"] == nearfar["near_object_id"])]
        prompt = (
            "\n".join("<image>" for _ in images) + "\n"
            + "Views are in this order: RGB, infrared, depth. In the depth view, "
            + "brighter valid pixels are nearer and black pixels are invalid. "
            + "Which candidate is closer to the camera?\n"
            + "\n".join(f"{LETTERS[index]}: {bbox_to_qwen1000(obj['bbox'])}"
                        for index, obj in enumerate(objects))
            + "\nReturn only the candidate letter."
        )
    elif task == "cross_bbox":
        if not auxiliary or auxiliary["review_status"] not in {"human_accepted", "approved_batch"}:
            raise ValueError("cross-modal box needs task acceptance")
        target_modality = auxiliary["target_modality"]
        if target_modality not in modalities or target_modality not in {"infrared", "depth"}:
            raise ValueError("cross-modal target image is missing")
        prompt = ("\n".join("<image>" for _ in images) + "\nViews are in this order: "
                  + ", ".join(modalities) + ". Locate the target described by: "
                  + auxiliary["query"] + f"\nReturn its box in the {target_modality} image, "
                  + 'using 0-1000 xyxy coordinates as {"bbox_2d":[x1,y1,x2,y2]}.')
        answer = json.dumps({"bbox_2d": bbox_to_qwen1000(auxiliary["bbox"])}, separators=(",", ":"))
    elif task == "relation":
        if not auxiliary or auxiliary["review_status"] not in {"human_accepted", "approved_batch"}:
            raise ValueError("relation task needs acceptance")
        candidates = auxiliary["candidates"].copy()
        ids = [candidate["object_id"] for candidate in candidates]
        if not 2 <= len(ids) <= 6 or len(ids) != len(set(ids)) or auxiliary["answer_object_id"] not in ids:
            raise ValueError("relation candidates must have one identified answer")
        rng.shuffle(candidates)
        lines = [f"{LETTERS[i]}: {bbox_to_qwen1000(c['bbox'])}" for i, c in enumerate(candidates)]
        answer = LETTERS[next(i for i, c in enumerate(candidates) if c["object_id"] == auxiliary["answer_object_id"])]
        prompt = ("\n".join("<image>" for _ in images) + "\nViews are in this order: "
                  + ", ".join(modalities) + ". Candidate coordinates refer to RGB (0-1000 xyxy).\n"
                  + auxiliary["query"] + "\n" + "\n".join(lines) + "\nReturn only the candidate letter.")
    else:
        raise ValueError(task)
    sample = {
        "id": f"{row['source']}:{row['id']}:presentation{exposure:05d}",
        "image": images[0] if len(images) == 1 else images,
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}],
    }
    metadata = {
        "id": sample["id"], "source": row["source"], "source_id": row["id"],
        "group_id": row["group_id"], "task": task,
        "modalities": list(modalities), "depth_policy": row["depth_policy"],
        "review_status": row.get("review_status", "source_original"),
        "target_modality": auxiliary["target_modality"] if task == "cross_bbox" else "rgb",
        "auxiliary_task_id": auxiliary.get("id") if auxiliary else None,
    }
    return sample, metadata


def _select_joint_tasks(schedule: list[dict], seed: int) -> dict[int, dict]:
    """25% relation + 15% cross-box ceilings; shortages remain direct bbox."""
    rng = random.Random(seed + 7001)
    assigned = {}
    for kind, proportion in (("relation", .25), ("cross_bbox", .15)):
        eligible = []
        for index, row in enumerate(schedule):
            tasks = [t for t in row.get("task_pool", []) if t["type"] == kind
                     and t.get("review_status") in {"human_accepted", "approved_batch"}]
            if index not in assigned and tasks:
                eligible.append((index, tasks))
        rng.shuffle(eligible)
        for index, tasks in eligible[:int(len(schedule) * proportion)]:
            assigned[index] = rng.choice(tasks)
    return assigned


def _select_rgb_augmentations(schedule: list[dict], seed: int, fraction: float) -> dict[int, tuple[str, float]]:
    if not 0 <= fraction <= .2:
        raise ValueError("RGB augmentation fraction must be between 0 and 0.2")
    eligible = [i for i, row in enumerate(schedule) if row.get("augmentation_eligible") is True
                and row.get("augmentation_review_status") in {"human_accepted", "approved_batch"}
                and (row.get("infrared_path") is not None or row.get("depth_path") is not None)]
    rng = random.Random(seed + 8001)
    rng.shuffle(eligible)
    variants = [("brightness", .35), ("brightness", .55), ("brightness", .75), ("blur", 1.), ("blur", 2.)]
    # Every selected source retains at least one unaltered presentation.
    remaining = Counter((r["source"], r["id"]) for r in schedule)
    result = {}
    for index in eligible:
        if len(result) >= int(len(schedule) * fraction):
            break
        row = schedule[index]
        key = row["source"], row["id"]
        if remaining[key] <= 1:
            continue
        remaining[key] -= 1
        result[index] = rng.choice(variants)
    return result


def _augment_rgb(path: Path, variant: tuple[str, float], output_dir: Path, cache: dict) -> Path:
    key = (path, *variant)
    if key not in cache:
        target = output_dir / "rgb_augmentation" / f"{len(cache):06d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as original:
            rgb = original.convert("RGB")
            if variant[0] == "brightness":
                modified = ImageEnhance.Brightness(rgb).enhance(variant[1])
            elif variant[0] == "blur":
                modified = rgb.filter(ImageFilter.GaussianBlur(variant[1]))
            else:
                raise ValueError("unsupported RGB-only augmentation")
            modified.save(target)
        cache[key] = target
    return cache[key]


def _interior_values(depth: np.ndarray, bbox: list[float]) -> tuple[float, float, float]:
    height, width = depth.shape
    x1, y1, x2, y2 = bbox
    pad_x, pad_y = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    xa, xb = int((x1 + pad_x) * width), int(np.ceil((x2 - pad_x) * width))
    ya, yb = int((y1 + pad_y) * height), int(np.ceil((y2 - pad_y) * height))
    values = depth[ya:yb, xa:xb]
    if values.size == 0:
        return 0, 0, float("inf")
    good = values[values > 0]
    fraction = good.size / values.size
    if good.size == 0:
        return fraction, 0, float("inf")
    median = float(np.median(good))
    iqr = float(np.percentile(good, 75) - np.percentile(good, 25))
    return fraction, median, iqr


def _nearfar_task(row: dict[str, Any]) -> dict[str, Any] | None:
    if row["depth_path"] is None or row["infrared_path"] is None or row["depth_policy"] not in {"millimeter", "meter"}:
        return None
    objects = row.get("nearfar_objects")
    if not isinstance(objects, list) or len(objects) != 2 or not all(obj.get("confirmed") is True for obj in objects):
        return None
    if len({obj["object_id"] for obj in objects}) != 2:
        return None
    boxes = [obj["bbox"] for obj in objects]
    for box in boxes:
        bbox_to_qwen1000(box)
    depth = _metric_depth(row["depth_path"], row["depth_policy"])
    with Image.open(row["rgb_path"]) as rgb_image:
        if depth.shape != (rgb_image.height, rgb_image.width):
            raise ValueError(f"depth/RGB size mismatch for numerical relation: {row['id']}")
    stats = [_interior_values(depth, box) for box in boxes]
    if any(fraction < 0.7 or median <= 0 or iqr > 0.25 * median
           for fraction, median, iqr in stats):
        return None
    near_index = 0 if stats[0][1] < stats[1][1] else 1
    far_index = 1 - near_index
    separation = stats[far_index][1] - stats[near_index][1]
    if separation <= 500 or separation <= 0.1 * stats[near_index][1]:
        return None
    return {"objects": objects, "near_object_id": objects[near_index]["object_id"], "stats_mm": stats}


def _select_auxiliary(schedule: list[dict[str, Any]], seed: int) -> dict[int, str]:
    distinct = {(row["source"], row["id"]): row for row in schedule}
    instance_rows = {row_id: row for row_id, row in distinct.items()
                     if "confirmed_objects" in row and confirmed_instance_task(row)}
    nearfar_rows = {row_id: row for row_id, row in distinct.items()
                    if "nearfar_objects" in row and _nearfar_task(row) is not None}
    for kind, rows in (("instance", instance_rows), ("nearfar", nearfar_rows)):
        groups = {row["group_id"] for row in rows.values()}
        if len(rows) < 200 or len(groups) < 40:
            raise ValueError(f"{kind} auxiliary pool needs >=200 unique questions across >=40 groups; got {len(rows)} / {len(groups)}")
    rng = random.Random(seed + 9001)
    instance_indices = [index for index, row in enumerate(schedule)
                        if (row["source"], row["id"]) in instance_rows]
    rng.shuffle(instance_indices)
    if len(instance_indices) < 800:
        raise ValueError(f"T needs 800 instance presentations in the D schedule; got {len(instance_indices)}")
    assigned = dict.fromkeys(instance_indices[:800], "instance")
    nearfar_indices = [index for index, row in enumerate(schedule)
                       if (row["source"], row["id"]) in nearfar_rows and index not in assigned]
    rng.shuffle(nearfar_indices)
    if len(nearfar_indices) < 800:
        raise ValueError(f"T needs 800 near/far presentations in the D schedule; got {len(nearfar_indices)}")
    assigned.update(dict.fromkeys(nearfar_indices[:800], "nearfar"))
    return assigned


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    quotas = dict(QUOTAS[(args.branch, args.phase)])
    if "robo" in quotas and args.robo is None and args.robo_fallback_rgbdt:
        quotas["rgbdt"] += quotas.pop("robo")
    sources = _load_sources(args, quotas)
    schedule = build_schedule(sources, quotas, args.seed + 10000 * args.phase)
    task_by_index = _select_auxiliary(schedule, args.seed) if args.branch == "T" and args.phase == 1 else {}
    paired_tasks = _select_joint_tasks(schedule, args.seed) if args.branch in {"G", "U"} and args.phase == 1 else {}
    joint_tasks = paired_tasks if args.branch == "U" else {}
    augmentation_fraction = getattr(args, "augmentation_fraction", .2) if args.branch in {"G", "U"} else 0
    augmentations = _select_rgb_augmentations(schedule, args.seed + 10000 * args.phase, augmentation_fraction)
    # A direct-query augmentation review does not approve a different relation
    # question. Exclude the same presentations from both sides of the comparison.
    augmentations = {i: variant for i, variant in augmentations.items() if i not in paired_tasks}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each phase has a different image traversal order. Numeric cache filenames
    # must not overwrite images still referenced by another phase's manifest.
    asset_dir = output_dir / f"{args.branch.lower()}_phase{args.phase}_assets"
    depth_cache: dict[tuple[Path, str], Path] = {}
    augmentation_cache = {}
    samples, metadata = [], []
    rng = random.Random(args.seed + 100000 * args.phase)
    for index, row in enumerate(schedule):
        auxiliary = joint_tasks.get(index)
        task = auxiliary["type"] if auxiliary else task_by_index.get(index, "bbox")
        if index in augmentations:
            row = {**row, "rgb_path": _augment_rgb(row["rgb_path"], augmentations[index], asset_dir, augmentation_cache)}
        sample, meta = _native_sample(row, index, asset_dir, depth_cache, task, rng, auxiliary)
        meta["augmentation"] = {"kind": augmentations[index][0], "value": augmentations[index][1]} if index in augmentations else None
        samples.append(sample)
        metadata.append(meta)
    name = f"{args.branch.lower()}_phase{args.phase}"
    annotation = output_dir / f"{name}.json"
    sidecar = output_dir / f"{name}_metadata.jsonl"
    annotation.write_text(json.dumps(samples, ensure_ascii=False) + "\n", encoding="utf-8")
    sidecar.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata), encoding="utf-8")
    report = {
        "branch": args.branch, "phase": args.phase, "seed": args.seed,
        "presentations": len(samples), "quotas": quotas,
        "actual_sources": dict(Counter(row["source"] for row in schedule)),
        "tasks": dict(Counter(row["task"] for row in metadata)),
        "auxiliary_shortfall_to_bbox": ({
            "relation": int(len(schedule) * .25) - sum(r["task"] == "relation" for r in metadata),
            "cross_bbox": int(len(schedule) * .15) - sum(r["task"] == "cross_bbox" for r in metadata),
        } if args.branch == "U" and args.phase == 1 else {}),
        "augmentation_presentations": len(augmentations),
        "augmentation_fraction_actual": len(augmentations) / len(schedule),
        "unique_queries_by_source": {source: len({row["id"] for row in rows}) for source, rows in sources.items()},
        "unique_groups_by_source": {source: len({row["group_id"] for row in rows}) for source, rows in sources.items()},
        "unique_presented_queries_by_source": {source: len({row["id"] for row in schedule if row["source"] == source}) for source in sources},
        "unique_presented_groups_by_source": {source: len({row["group_id"] for row in schedule if row["source"] == source}) for source in sources},
        "repeat_presentations_by_source": {
            source: count - len({row["id"] for row in schedule if row["source"] == source})
            for source, count in quotas.items()
        },
        "rendered_metric_depth_images": len(depth_cache),
        "annotation": str(annotation), "metadata": str(sidecar),
    }
    (output_dir / f"{name}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city-native", type=Path, required=True)
    parser.add_argument("--city-root", type=Path, required=True)
    parser.add_argument("--rgbdt", type=Path, help="Approved RGBDT500 JSONL with true RGB/IR/depth and Query")
    parser.add_argument("--rgbt", type=Path, help="RGBT-GroundBench training JSONL")
    parser.add_argument("--robo", type=Path, help="RoboRefIt training JSONL")
    parser.add_argument("--robo-fallback-rgbdt", action="store_true")
    parser.add_argument("--branch", choices=("C", "D", "T", "G", "U"), required=True)
    parser.add_argument("--augmentation-fraction", type=float, default=.2)
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
