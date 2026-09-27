"""Prepare paired G/U native SFT manifests for the same-day pilot."""
from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


MODALITIES = ("rgb", "infrared", "depth")
NEW_MODALITY_COUNTS_200 = {"rgb": 217, "infrared": 128, "depth": 55}
QUERY_MARKER = "Locate the object described by this query: "
QUERY_MARKERS = (QUERY_MARKER, "Locate the target described by: ")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON array or JSONL records")
    return rows


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _query_from_row(row: dict[str, Any], *, require_marker: bool) -> str:
    prompt = row["conversations"][0]["value"]
    marker = next((candidate for candidate in QUERY_MARKERS if candidate in prompt), None)
    if marker is None:
        if require_marker:
            raise ValueError(f"native sample {row['id']} has no supported Query marker")
        return prompt
    query = prompt.split(marker, 1)[1].split("\nReturn", 1)[0]
    if not query.strip():
        raise ValueError(f"native sample {row['id']} has an empty Query")
    return query


def _city_image_paths(row: dict[str, Any], city_root: Path) -> list[str]:
    images = row["image"]
    if not isinstance(images, list) or len(images) != 3:
        raise ValueError(f"City native sample {row['id']} is not trimodal")
    root = city_root.resolve()
    return [str(path if Path(path).is_absolute() else (root / path).resolve()) for path in images]


def _city_tasks(rows: list[dict[str, Any]], city_root: Path) -> list[dict[str, Any]]:
    tasks = []
    seen = set()
    for row in rows:
        if row.get("split", "train") != "train":
            raise ValueError(f"City native sample {row['id']} is not in the training split")
        if row["id"] in seen:
            raise ValueError(f"duplicate City sample ID: {row['id']}")
        seen.add(row["id"])
        task_row = copy.deepcopy(row)
        task_row["image"] = _city_image_paths(row, city_root)
        query = _query_from_row(row, require_marker=False)
        tasks.append({
            "row": task_row,
            "source": "city",
            "source_task_id": row["id"],
            "group": task_row["image"][0],
            "query": query,
            "task": "bbox",
            "output_modality": "rgb",
            "is_new": False,
        })
    if not tasks:
        raise ValueError("City native source is empty")
    return tasks


def _new_tasks(rows: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != 181 or len(metadata) != len(rows):
        raise ValueError(f"released RGBDT source must have 181 rows and matching metadata; got {len(rows)}")
    rows_by_id = {row["id"]: row for row in rows}
    metadata_by_id = {row["id"]: row for row in metadata}
    if len(rows_by_id) != len(rows) or len(metadata_by_id) != len(metadata):
        raise ValueError("released rows and metadata must have unique IDs")
    if set(rows_by_id) != set(metadata_by_id):
        raise ValueError("released native IDs do not match metadata IDs")

    tasks = []
    rgb_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    tasks_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    modality_counts = Counter()
    for sample_id, row in rows_by_id.items():
        meta = metadata_by_id[sample_id]
        if meta.get("review_status") != "human_accepted":
            raise ValueError(f"released metadata {sample_id} has not passed human review")
        modality = meta.get("target_modality")
        if modality not in MODALITIES:
            raise ValueError(f"released metadata {sample_id} has unsupported target_modality {modality!r}")
        query = _query_from_row(row, require_marker=True)
        source_id = str(meta.get("source_id", sample_id.rsplit("::", 1)[0]))
        task = {
            "row": row,
            "source": str(meta.get("source", "rgbdt")),
            "source_task_id": sample_id,
            "source_id": source_id,
            "group": str(meta.get("group_id", source_id)),
            "query": query,
            "task": str(meta.get("task", "bbox")),
            "output_modality": modality,
            "is_new": True,
        }
        key = (source_id, query)
        tasks_by_key[key].append(task)
        if modality == "rgb":
            if key in rgb_by_key:
                raise ValueError(f"multiple RGB tasks for released query {source_id}")
            rgb_by_key[key] = task
        modality_counts[modality] += 1
        tasks.append(task)

    expected_counts = {"rgb": 98, "infrared": 58, "depth": 25}
    if dict(modality_counts) != expected_counts:
        raise ValueError(f"released modality counts differ from expected {expected_counts}: {dict(modality_counts)}")

    for key, siblings in tasks_by_key.items():
        if key not in rgb_by_key:
            raise ValueError(f"released task has no RGB task with the same Query: {key[0]}")
        rgb_images = rgb_by_key[key]["row"]["image"]
        for task in siblings:
            if task["row"]["image"] != rgb_images:
                raise ValueError(f"released modalities do not share the same image paths: {task['source_task_id']}")
    return tasks


def _balanced_tasks(tasks: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[task["group"]].append(task)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key in group_keys:
        rows = groups[key].copy()
        rng.shuffle(rows)
        queues[key] = deque(rows)

    result = []
    while len(result) < count:
        for key in group_keys:
            if not queues[key]:
                rows = groups[key].copy()
                rng.shuffle(rows)
                queues[key].extend(rows)
            result.append(queues[key].popleft())
            if len(result) == count:
                break
        rng.shuffle(group_keys)
    return result


def _new_exposures(tasks: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    tasks_by_modality = {
        modality: sorted((task for task in tasks if task["output_modality"] == modality),
                         key=lambda task: task["source_task_id"])
        for modality in MODALITIES
    }
    supplement = []
    for modality, count in (("rgb", 21), ("infrared", 12), ("depth", 5)):
        supplement.extend(rng.sample(tasks_by_modality[modality], count))
    exposures = [task for task in tasks for _ in range(2)] + supplement
    if len(exposures) != 400 or Counter(task["output_modality"] for task in exposures) != Counter(NEW_MODALITY_COUNTS_200):
        raise AssertionError("200-step new-data quota construction is inconsistent")
    return exposures


def _presentation_rows(task: dict[str, Any], presentation_id: str, rgb_by_key: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    u_row = copy.deepcopy(task["row"])
    g_row = copy.deepcopy(task["row"])
    u_row["id"] = presentation_id
    g_row["id"] = presentation_id
    if task["is_new"] and task["output_modality"] != "rgb":
        rgb_task = rgb_by_key[(task["source_id"], task["query"])]
        g_row["conversations"] = copy.deepcopy(rgb_task["row"]["conversations"])
    return g_row, u_row


def build(args: argparse.Namespace) -> dict[str, Any]:
    released_rows = read_rows(args.released_jsonl)
    metadata_rows = read_rows(args.metadata_jsonl or (args.released_jsonl.parent / "metadata.jsonl"))
    city_rows = read_rows(args.city_native)
    new_tasks = _new_tasks(released_rows, metadata_rows)
    city_tasks = _city_tasks(city_rows, args.city_root)
    rgb_by_key = {
        (task["source_id"], task["query"]): task
        for task in new_tasks if task["output_modality"] == "rgb"
    }

    steps = args.steps
    if steps not in (200, 600):
        raise ValueError(f"steps must be 200 or 600, got {steps}")
    cycles = steps // 200
    g_samples: list[dict[str, Any]] = []
    u_samples: list[dict[str, Any]] = []
    metadata_out: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    new_modality_presentations = Counter()
    source_presentations = Counter()

    for cycle in range(cycles):
        cycle_rng = random.Random(args.seed + cycle * 1009)
        new_exposures = _new_exposures(new_tasks, cycle_rng)
        cycle_rng.shuffle(new_exposures)
        city_exposures = iter(_balanced_tasks(city_tasks, 1200, random.Random(args.seed + cycle * 2027 + 17)))
        new_index = 0
        for block in range(200):
            tickets = ["city"] * 6 + ["new"] * 2
            cycle_rng.shuffle(tickets)
            for source in tickets:
                task = next(city_exposures) if source == "city" else new_exposures[new_index]
                if source == "new":
                    new_index += 1
                ordinal = len(schedule) + 1
                presentation_id = f"gu{steps}_{ordinal:06d}"
                g_row, u_row = _presentation_rows(task, presentation_id, rgb_by_key)
                g_samples.append(g_row)
                u_samples.append(u_row)
                schedule_row = {
                    "ordinal": ordinal,
                    "presentation_id": presentation_id,
                    "source": task["source"],
                    "group": task["group"],
                    "query": task["query"],
                    "task": task["task"],
                    "output_modality": task["output_modality"],
                    "source_task_id": task["source_task_id"],
                }
                schedule.append(schedule_row)
                metadata_out.append({
                    **schedule_row,
                    "g_id": g_row["id"],
                    "u_id": u_row["id"],
                    "g_uses_rgb_supervision": bool(task["is_new"] and task["output_modality"] != "rgb"),
                })
                source_presentations[task["source"]] += 1
                if task["is_new"]:
                    new_modality_presentations[task["output_modality"]] += 1
        if new_index != 400:
            raise AssertionError(f"cycle {cycle + 1} used {new_index} new presentations, expected 400")

    representatives = []
    representatives.extend(_balanced_tasks(city_tasks, 2, random.Random(args.seed + 7001)))
    rep_rng = random.Random(args.seed + 7002)
    for modality in MODALITIES:
        candidates = sorted((task for task in new_tasks if task["output_modality"] == modality),
                            key=lambda task: task["source_task_id"])
        representatives.extend(rep_rng.sample(candidates, 2))
    pressure16 = []
    for repeat in range(2):
        for index, task in enumerate(representatives):
            presentation_id = f"gu_pressure_{repeat * len(representatives) + index + 1:06d}"
            _, u_row = _presentation_rows(task, presentation_id, rgb_by_key)
            pressure16.append(u_row)
    pressure2city = copy.deepcopy(pressure16[:2])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "g_train.json", g_samples)
    _write_json(output_dir / "u_train.json", u_samples)
    _write_json(output_dir / "pressure16.json", pressure16)
    _write_json(output_dir / "pressure2city.json", pressure2city)
    _write_jsonl(output_dir / "metadata.jsonl", metadata_out)
    _write_jsonl(output_dir / "paired_schedule.jsonl", schedule)

    expected_city = 1200 * cycles
    expected_new = 400 * cycles
    expected_total = 1600 * cycles
    expected_modality = {key: value * cycles for key, value in NEW_MODALITY_COUNTS_200.items()}
    report = {
        "seed": args.seed,
        "steps": steps,
        "presentations_per_batch": 8,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "presentations": len(schedule),
        "unique_released_tasks": len(new_tasks),
        "city_presentations": source_presentations["city"],
        "new_presentations": source_presentations.get("rgbdt", 0),
        "new_presentations_by_output_modality": dict(sorted(new_modality_presentations.items())),
        "expected": {
            "presentations": expected_total,
            "city_presentations": expected_city,
            "new_presentations": expected_new,
            "new_presentations_by_output_modality": expected_modality,
        },
        "g_auxiliary_rows_using_rgb_supervision": sum(row["g_uses_rgb_supervision"] for row in metadata_out),
        "city_unique_image_groups": len({task["group"] for task in city_tasks}),
        "city_group_source": "first image path after --city-root resolution",
        "released_image_paths_preserved": True,
        "pressure16": {
            "rows": len(pressure16),
            "representatives": 8,
            "repeats_per_representative": 2,
            "first_two_sources": ["city", "city"],
            "source_task_ids": [task["source_task_id"] for task in representatives],
            "source_counts": dict(Counter(task["source"] for task in representatives)),
            "output_modality_counts": dict(Counter(task["output_modality"] for task in representatives)),
            "new_task_output_modality_counts": dict(Counter(
                task["output_modality"] for task in representatives if task["is_new"]
            )),
        },
        "pressure2city": {"rows": len(pressure2city), "ids": [row["id"] for row in pressure2city]},
        "outputs": {
            "g_train": str(output_dir / "g_train.json"),
            "u_train": str(output_dir / "u_train.json"),
            "metadata": str(output_dir / "metadata.jsonl"),
            "paired_schedule": str(output_dir / "paired_schedule.jsonl"),
            "pressure16": str(output_dir / "pressure16.json"),
            "pressure2city": str(output_dir / "pressure2city.json"),
        },
    }
    if report["presentations"] != expected_total or report["city_presentations"] != expected_city or report["new_presentations"] != expected_new:
        raise AssertionError("source presentation quotas do not match configured steps")
    if dict(new_modality_presentations) != expected_modality:
        raise AssertionError(f"new modality quotas differ: {dict(new_modality_presentations)} != {expected_modality}")
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-jsonl", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, help="Defaults to metadata.jsonl beside --released-jsonl")
    parser.add_argument("--city-native", type=Path, required=True)
    parser.add_argument("--city-root", type=Path, required=True, help="Root used to resolve relative City image paths")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, choices=(200, 600), default=200)
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
