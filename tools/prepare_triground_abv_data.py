"""Prepare review-only A/B/V candidates, then release a human-approved schedule.

The default command writes proposals and a review atlas, never training data.
Only ``release`` accepts explicit human decisions and writes native SFT manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000, depth_mm_to_grayscale_rgb, native_prompt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/triground_abv_20260927"
DEPTH_V2_OUT = ROOT / "results/triground_abv_depthv2_20260928"
CITY = ROOT / "results/gu_diagnosis_20260927/source_snapshot/city_train.json"
CITY_RAW_ZIP = Path("F:/Downloads/qwen_generation_train_val_manifests.zip")
CITY_IMAGES = ROOT / "data/city_object_extension"
ROBO = ROOT / "data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl"
ROBO_RAW = ROOT / "data/external/RoboRefIt/raw/final_dataset/train/roborefit_train.json"
RGBT = ROOT / "results/multimodal_data_20260925/next_sources/rgbt_cloud_train6000.jsonl"
SEED = 20260927
TARGETS = {"depth_relation": 64, "robo_competition": 56, "ir_complement": 60,
           "reliability": 60, "diag_depth": 32, "diag_ir": 32, "diag_aux": 32}
KNOWN_UNRELIABLE_CITY_SCENES = {"000014_022_00000069"}
MINIMUMS = {"depth_relation": 48, "robo_competition": 42,
            "ir_complement": 40, "reliability": 45}
QUERY_MARKER = "Locate the object described by this query: "
ANCESTOR_MANIFESTS = (
    ROOT / "results/gu_diagnosis_20260927/source_snapshot/city_train.json",
    ROOT / "results/gu_pilot_20260926/manifests/seed2026/g_train.json",
    ROOT / "results/gu_pilot_20260926/manifests/seed2026/u_train.json",
    ROOT / "results/gu_diagnosis_20260927/cloud_snapshot/manifests/train_b.json",
    ROOT / "results/gu_diagnosis_20260927/cloud_snapshot/manifests/train_gstar.json",
    ROOT / "results/gu_diagnosis_20260927/cloud_snapshot/manifests/train_ustar.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def ancestor_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact-ID/path audit of locally retained manifests, not a novelty certificate."""
    diagnostic = [row for row in rows if row["split"] == "diagnostic"
                  and row["source"] != "city"]
    ancestors: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    all_paths: set[str] = set()
    basename_sources: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in ANCESTOR_MANIFESTS:
        historic = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(historic, dict):
            historic = list(historic.values())
        ids = {str(row.get("id", "")) for row in historic}
        images = {str(image).replace("\\", "/").casefold() for row in historic
                  for image in row.get("image", [])}
        for image in images:
            if "/visible/" in f"/{image}" or "/rgb/" in f"/{image}":
                basename_sources[Path(image).name].append(
                    {"manifest": str(path.resolve()), "source_image": image})
        all_ids.update(ids)
        all_paths.update(images)
        ancestors[str(path.resolve())] = {"rows": len(historic), "image_paths": len(images)}
    matches = []
    basename_matches = []
    for row in diagnostic:
        image = str(row["images"]["rgb"]).replace("\\", "/").casefold()
        if row["source_id"] in all_ids or image in all_paths:
            matches.append({"task_id": row["task_id"], "source_id": row["source_id"],
                            "exact_id": row["source_id"] in all_ids,
                            "exact_rgb_path": image in all_paths})
        if Path(image).name in basename_sources:
            basename_matches.append({"task_id": row["task_id"],
                                     "rgb_basename": Path(image).name,
                                     "historical_sources": basename_sources[Path(image).name]})
    return {"scope": "local exact source-ID and RGB path against retained City/G/U manifests",
            "diagnostic_external_tasks": len(diagnostic),
            "diagnostic_by_source": dict(Counter(row["source"] for row in diagnostic)),
            "ancestor_manifests": ancestors, "exact_matches": matches,
            "rgb_basename_matches": basename_matches,
            "historical_independence": "unproven",
            "limits": ["C/M2/R2 individual consumed-sample lists are not retained here; their City-only inputs are supported by train configs and City snapshot.",
                       "G/U/B consumed_samples.jsonl logs record presentation IDs, not source-image paths; their retained train manifests are compared above.",
                       "An RGB basename match is a lead for review, not proof of the same scene; no basename match does not exclude visual duplicates."]}


def city_query(row: dict[str, Any]) -> str:
    return row["query"]


def city_box(row: dict[str, Any]) -> list[float]:
    return list(row["bbox"])


def city_raw_labels() -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(CITY_RAW_ZIP) as archive:
        records = json.loads(archive.read("qwen_generation_train_100.json"))
    for sample_id, record in records.items():
        record["id"] = sample_id
    return records


def credible_city_class(query: str, class_name: str) -> bool:
    """Conservative lexical screen; final target identity remains a human decision."""
    head = re.split(r"\b(?:on|in|near|beside|behind|between|next to|at|under|above|with|by|from)\b",
                    query.casefold(), maxsplit=1)[0]
    if re.search(r"\b(?:legs?|wheels?|head|face|shirt|jacket|sign|license plate)\b", head):
        # These are usually parts/accessories, not the enclosing City class.
        if class_name.casefold() not in {"sign"}:
            return False
    words = {
        "person": r"\b(person|man|woman|boy|girl|pedestrian|child|people|worker|rider)\b",
        "car": r"\b(car|sedan|suv|vehicle|van|jeep|taxi)\b",
        "bicycle": r"\b(bicycle|bike|cycle)\b",
        "motorcycle": r"\b(motorcycle|motorbike)\b",
        "animal": r"\b(animal|deer|bird|dog|cat|horse|cow|goat)\b",
        "sign": r"\b(sign|signboard)\b",
        "street light": r"\b(street.?light|lamp.?post)\b",
        "garbage can": r"\b(garbage can|trash can|bin|dustbin)\b",
    }
    pattern = words.get(class_name.casefold())
    if pattern is None:
        base = re.escape(class_name.casefold().rstrip("s"))
        pattern = rf"\b{base}s?\b"
    return bool(re.search(pattern, head))


def _iou(a: list[float], b: list[float]) -> float:
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = w * h
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


def _depth_core(raw: np.ndarray, bbox: list[float]) -> dict[str, float]:
    height, width = raw.shape[:2]
    x1, y1, x2, y2 = [int(v * size) for v, size in zip(bbox, (width, height, width, height))]
    dx, dy = max(1, (x2-x1)//4), max(1, (y2-y1)//4)
    patch = raw[y1+dy:y2-dy, x1+dx:x2-dx]
    valid = patch[(patch > 0) & (patch < 20_000)]
    return {"core_valid_fraction": round(float(len(valid) / patch.size), 4) if patch.size else 0,
            "core_median_mm": round(float(np.median(valid)), 1) if len(valid) else -1}


def _base_task(*, task_id: str, bundle_id: str, scene_id: str, split: str,
               category: str, source: str, images: dict[str, str], query: str,
               bbox: list[float], object_id: str, source_id: str,
               depth_policy: str = "visual", **extra: Any) -> dict[str, Any]:
    bbox_to_qwen1000(bbox)
    return {"task_id": task_id, "bundle_id": bundle_id, "scene_id": scene_id,
            "split": split, "category": category, "source": source,
            "images": images, "proposed_query": query, "bbox": bbox,
            "target_object_id": object_id, "source_id": source_id,
            "depth_policy": depth_policy, "review_status": "draft_unapproved", **extra}


def city_candidates(output: Path = OUT, *, specs_path: Path | None = None) -> list[dict[str, Any]]:
    """Build scoped relations from reviewed source objects; keep scene splits fixed."""
    from tools.depth_relation_tasks import build_depth_tasks

    source = read_jsonl(output / "data/candidates.jsonl")
    scenes = {row["scene_id"]: row for row in source if row["source"] == "city"}
    specs_path = specs_path or Path(__file__).resolve().parents[1] / "configs/depth_relations_v2.json"
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    relations = build_depth_tasks(specs, city_raw_labels(), scenes)
    return relations + [row for row in source if row["source"] == "city"
                        and row["category"] not in {"depth_relation", "diag_depth"}]


def robo_candidates() -> list[dict[str, Any]]:
    selected = read_jsonl(ROBO)
    selected_by_stem = {Path(row["rgb"]).stem: row for row in selected}
    raw = json.loads(ROBO_RAW.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        stem = Path(row["rgb_path"].replace("\\", "/")).stem
        if stem in selected_by_stem:
            grouped[stem].append(row)
    pairs = []
    reliability = []
    for stem, rows in grouped.items():
        selected_row = selected_by_stem[stem]
        images = {"rgb": str((ROBO.parent / selected_row["rgb"]).resolve()),
                  "depth": str((ROBO.parent / selected_row["depth"]).resolve())}
        by_class: dict[str, dict[tuple[int, ...], dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_class[str(row.get("class", ""))].setdefault(tuple(row["bbox"]), row)
        options = []
        for class_name, boxes in by_class.items():
            values = list(boxes.values())
            for index, left in enumerate(values):
                for right in values[index+1:]:
                    if left["text"].strip().casefold() == right["text"].strip().casefold():
                        continue
                    a = [left["bbox"][0]/640, left["bbox"][1]/480,
                         left["bbox"][2]/640, left["bbox"][3]/480]
                    b = [right["bbox"][0]/640, right["bbox"][1]/480,
                         right["bbox"][2]/640, right["bbox"][3]/480]
                    if _iou(a, b) < .3:
                        options.append((left, right, class_name, a, b))
        if options:
            left, right, class_name, a, b = sorted(options, key=lambda x: (str(x[2]), x[0]["num"], x[1]["num"]))[0]
            pairs.append((stem, selected_row, images, left, right, class_name, a, b))
        reliability.append((stem, selected_row, images))
    pairs.sort(key=lambda item: item[0])
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    if len(pairs) < 28:
        raise ValueError(f"only {len(pairs)} Robo pair scenes; need 28")
    pair_scenes = {item[0] for item in pairs[:28]}
    train_sequences = {int(stem)//100 for stem in pair_scenes}
    reliability = [item for item in reliability if item[0] not in pair_scenes]
    rng.shuffle(reliability)
    reliability_train = []
    for item in reliability:
        sequence = int(item[0])//100
        if sequence not in train_sequences:
            reliability_train.append(item)
            train_sequences.add(sequence)
            if len(reliability_train) == 15:
                break
    diagnostic = []
    diag_sequences = set()
    for item in reliability:
        sequence = int(item[0])//100
        if sequence not in train_sequences and sequence not in diag_sequences:
            diagnostic.append(item)
            diag_sequences.add(sequence)
            if len(diagnostic) == 32:
                break
    if len(reliability_train) < 15 or len(diagnostic) < 32:
        raise ValueError(f"Robo sequence holdout yields {len(reliability_train)} reliability and {len(diagnostic)} diagnostic scenes")
    output = []
    for stem, selected_row, images, left, right, class_name, a, b in pairs[:28]:
        bundle = f"robo:{stem}:competition"
        for row, box, opponent in ((left, a, right), (right, b, left)):
            output.append(_base_task(
                task_id=f"{bundle}:{row['num']}", bundle_id=bundle,
                scene_id=selected_row["scene_id"], split="train", category="robo_competition",
                source="roborefit", images=images, query=row["text"].strip(), bbox=box,
                object_id=f"robo:{stem}:{row['num']}", source_id=f"roborefit_train_raw_{row['num']}",
                depth_policy="visual", original_query=row["text"].strip(),
                opponent_object_id=f"robo:{stem}:{opponent['num']}", opponent_original_query=opponent["text"].strip(),
                object_class=class_name, scene_category=selected_row.get("scene"),
                sequence_group=f"robo:{int(stem)//100:03d}",
                review_instruction="Source boxes are author annotations, but distinct object identity and whole Query semantics need human approval; Depth unit is unknown.",
            ))
    for category, batch in (("reliability", reliability_train), ("diag_aux", diagnostic)):
        for stem, row, images in batch:
            output.append(_base_task(
                task_id=f"robo:{stem}:{category}", bundle_id=f"robo:{stem}:{category}",
                scene_id=row["scene_id"], split="diagnostic" if category == "diag_aux" else "train",
                category=category, source="roborefit", images=images, query=row["query"],
                bbox=row["bbox"], object_id=f"robo:{stem}:{row['id']}", source_id=row["id"],
                depth_policy="visual", original_query=row["query"],
                reliability_variants=["normal", "depth_missing"],
                sequence_group=f"robo:{int(stem)//100:03d}",
            ))
    return output


def rgbt_sequence(scene_id: str) -> str:
    source, stem = scene_id.split(":", 1)
    if source in {"flir", "m3fd"}:
        digits = re.search(r"(\d+)$", stem)
        return f"{source}:{int(digits.group(1))//100:03d}"
    return f"mfad:{stem.split('-')[0][:20]}"


def rgbt_candidates(requests: Path, output: Path) -> list[dict[str, Any]]:
    source = {row["id"]: row for row in read_jsonl(RGBT)}
    requested = list(csv.DictReader(requests.open(encoding="utf-8-sig", newline="")))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requested:
        row = source[request["id"]]
        grouped[row["source"]].append(row)
    output_rows = []
    for dataset, rows in sorted(grouped.items()):
        sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            sequences[rgbt_sequence(row["scene_id"])].append(row)
        ordered_groups = list(sequences.values())
        ordered_groups.sort(key=lambda group: min(rows.index(row) for row in group))
        allocated: dict[str, list[dict[str, Any]]] = defaultdict(list)
        role = "ir_complement"
        quotas = {"ir_complement": 30, "diag_ir": 16, "reliability": 10}
        for group in ordered_groups:
            if role != "reserve_ir" and len(allocated[role]) >= quotas[role]:
                role = {"ir_complement": "diag_ir", "diag_ir": "reliability",
                        "reliability": "reserve_ir", "reserve_ir": "reserve_ir"}[role]
            allocated[role].extend(group)
        for category, pool in allocated.items():
            for row in pool:
                rgb_rel = row["rgb"].replace("../raw/", "")
                ir_rel = row["aux"].replace("../raw/", "")
                local_root = output / "review/ir_sources"
                original_root = ROOT / "data/external/RGBT-GroundBench"
                rgb = local_root / rgb_rel
                ir = local_root / ir_rel
                if not rgb.is_file():
                    rgb = original_root / rgb_rel
                if not ir.is_file():
                    ir = original_root / ir_rel
                output_rows.append(_base_task(
                    task_id=f"{row['id']}:{category}", bundle_id=f"{row['id']}:{category}",
                    scene_id=row["scene_id"], split="diagnostic" if category == "diag_ir" else "train",
                    category=category, source=row["source"],
                    images={"rgb": str(rgb), "infrared": str(ir)}, query=row["query"],
                    bbox=row["bbox"], object_id=f"{row['scene_id']}:{row['id']}",
                    source_id=row["id"], depth_policy="visual", original_query=row["query"],
                    sequence_group=rgbt_sequence(row["scene_id"]),
                    cloud_images={"rgb": rgb_rel, "infrared": ir_rel},
                    reliability_variants=["normal", "ir_missing"] if category == "reliability" else [],
                    review_instruction="For IR complement, review RGB alone before RGB+IR without GT overlay; author bbox alone does not prove IR necessity."
                ))
    return output_rows


DECISION_FIELDS = ["task_id", "category", "split", "bundle_id", "scene_id", "source_id",
                   "original_query", "proposed_query", "approved_query", "decision", "reviewer",
                   "object_binding", "full_semantics", "query_unique", "depth_order",
                   "all_same_class_checked", "depth_visible_reliable",
                   "rgb_only_unique", "ir_unique", "aux_necessary", "answer_survives_blank", "note",
                   "review_mode", "modality_judgment"]


def _thumbnail(path: Path, out: Path, bbox: list[float] | None = None) -> bool:
    if not path.is_file():
        return False
    with Image.open(path) as original:
        picture = original.convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(picture)
        width, height = picture.size
        xy = [bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height]
        draw.rectangle(xy, outline="#ff2828", width=max(2, width//200))
    picture.thumbnail((1920, 1440), Image.Resampling.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    picture.save(out, quality=88)
    return True


def render_review(rows: list[dict[str, Any]], output: Path, *,
                  refresh_decision_template: bool = True) -> dict[str, Any]:
    """Render the frozen candidates without changing decisions or training data."""
    review = output / "review"
    assets = review / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["bundle_id"]].append(row)
    bundles = []
    missing = Counter()
    for ordinal, (bundle_id, tasks) in enumerate(grouped.items()):
        item = tasks[0]
        views = {}
        for modality in ("rgb", "infrared", "depth"):
            if modality not in item["images"]:
                continue
            dest = assets / f"bundle_{ordinal:04d}_{modality}.jpg"
            if _thumbnail(Path(item["images"][modality]), dest):
                views[modality] = f"assets/{dest.name}"
            else:
                missing[modality] += 1
        review_tasks = []
        for index, task in enumerate(tasks):
            dest = assets / f"bundle_{ordinal:04d}_{index:02d}_answer.jpg"
            answer = f"assets/{dest.name}" if _thumbnail(Path(task["images"]["rgb"]), dest, task["bbox"]) else None
            review_tasks.append({**task, "answer_image": answer})
        bundles.append({"id": bundle_id, "scene": item["scene_id"],
                        "category": item["category"], "split": item["split"],
                        "source": item["source"], "views": views, "tasks": review_tasks})
    template_root = Path(__file__).with_name("review_ui")
    translations = json.loads((template_root / "queries_zh.json").read_text(encoding="utf-8"))
    for row in rows:
        if row.get("query_zh"):
            translations[row["proposed_query"].strip()] = row["query_zh"]
        if row.get("original_query_zh"):
            translations[row["original_query"].strip()] = row["original_query_zh"]
    queries = {row[key].strip() for row in rows for key in ("proposed_query", "original_query")
               if row.get(key)}
    payload = {"version": "abv-review-v2", "fields": DECISION_FIELDS, "bundles": bundles,
               "translations": {query: translations[query] for query in sorted(queries)
                                if query in translations}}
    revision_path = output / "data/review_revision.json"
    if revision_path.is_file():
        payload["revision"] = json.loads(revision_path.read_text(encoding="utf-8"))
    template = (template_root / "shell.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    (review / "index.html").write_text(template.replace("__REVIEW_DATA__", encoded), encoding="utf-8")
    for name in ("review.css", "review.js"):
        shutil.copyfile(template_root / name, review / name)
    if refresh_decision_template:
        with (review / "decisions_blank.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in DECISION_FIELDS})
    return {"bundles": len(bundles), "tasks": len(rows), "missing_view_files": dict(missing),
            "html": str((review / "index.html").resolve()),
            "decisions_blank": str((review / "decisions_blank.csv").resolve())}


def revise_depth(source_output: Path = OUT, output: Path = DEPTH_V2_OUT,
                 specs_path: Path | None = None) -> dict[str, Any]:
    """Write a new review package; leave the previous package and decisions intact."""
    if output.resolve() == source_output.resolve() or (output.exists() and any(output.iterdir())):
        raise FileExistsError("Depth revision needs a new output directory; existing review is preserved")
    source = read_jsonl(source_output / "data/candidates.jsonl")
    old_depth = [row for row in source if row["category"] in {"depth_relation", "diag_depth"}]
    unchanged = [row for row in source if row["category"] not in {"depth_relation", "diag_depth"}]
    new_depth = [row for row in city_candidates(source_output, specs_path=specs_path)
                 if row["category"] in {"depth_relation", "diag_depth"}]
    for row in new_depth:
        row["predecessor_task_ids"] = [old["task_id"] for old in old_depth
                                       if old["source_id"] == row["source_id"]]
    if not new_depth:
        raise ValueError("Depth revision requires at least one reviewed proposal")
    rows = new_depth + unchanged
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate proposal task ID")
    scene_splits: dict[str, set[str]] = defaultdict(set)
    grouped_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scene_splits[row["scene_id"]].add(row["split"])
        group = row.get("sequence_group") or (
            f"city_location:{row['location_group']}" if row.get("location_group") else None)
        if group:
            grouped_splits[group].add(row["split"])
    if any(len(splits) != 1 for splits in scene_splits.values()):
        raise ValueError("scene crossed train/diagnostic")
    if any(len(splits) != 1 for splits in grouped_splits.values()):
        raise ValueError("location/sequence crossed train/diagnostic")
    source_bundles = list(dict.fromkeys(row["bundle_id"] for row in source))
    revision = {
        "source_package_key": f"aic-abv-review-v2:{len(source_bundles)}:{source_bundles[0]}:{source_bundles[-1]}",
        "source_output": str(source_output.resolve()),
        "unchanged_task_ids": [row["task_id"] for row in unchanged],
        "retired_depth_task_ids": [row["task_id"] for row in old_depth],
        "new_depth_task_ids": [row["task_id"] for row in new_depth],
        "note": "新版深度题另审；未改题目的已有记录可沿用。旧包和旧记录保留。",
    }
    write_jsonl(output / "data/candidates.jsonl", rows)
    (output / "data/review_revision.json").write_text(
        json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    specs_path = specs_path or Path(__file__).resolve().parents[1] / "configs/depth_relations_v2.json"
    shutil.copyfile(specs_path, output / "data/depth_specs.json")
    overlap_path = source_output / "data/ancestor_overlap.json"
    if overlap_path.is_file():
        shutil.copyfile(overlap_path, output / "data/ancestor_overlap.json")
    else:
        (output / "data/ancestor_overlap.json").write_text(
            json.dumps(ancestor_overlap(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atlas = render_review(rows, output)
    summary = {"status": "draft_unapproved", "targets": TARGETS,
               "candidate_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
               "depth_relation_types": dict(Counter(row["relation_type"] for row in new_depth)),
               "depth_scenes": len({row["scene_id"] for row in new_depth}),
               "unchanged_tasks": len(unchanged), "retired_depth_tasks": len(old_depth),
               "new_depth_tasks": len(new_depth), "review": atlas, "release_created": False}
    (output / "data/candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def prepare(output: Path = DEPTH_V2_OUT) -> dict[str, Any]:
    return revise_depth(OUT, output)


def _yes(row: dict[str, str], field: str) -> bool:
    return row.get(field, "").strip().casefold() == "yes"


def _approved(candidate: dict[str, Any], decision: dict[str, str]) -> bool:
    if decision.get("decision", "").strip().casefold() != "approve":
        return False
    # Direct review is the user's current workflow: a keep click confirms the
    # Query/box is usable, without claiming a blinded modality experiment.
    if decision.get("review_mode") == "quick":
        return True
    if not decision.get("reviewer", "").strip():
        raise ValueError(f"approved task needs reviewer: {candidate['task_id']}")
    if not all(_yes(decision, field) for field in ("object_binding", "full_semantics", "query_unique")):
        raise ValueError(f"approved task lacks object/Query verification: {candidate['task_id']}")
    category = candidate["category"]
    if category in {"depth_relation", "diag_depth"} and not all(
        _yes(decision, field) for field in ("depth_order", "aux_necessary",
                                          "all_same_class_checked", "depth_visible_reliable")
    ):
        raise ValueError(f"depth task lacks verified order/necessity: {candidate['task_id']}")
    if category in {"depth_relation", "diag_depth", "ir_complement", "diag_ir"} and not decision.get("note", "").strip():
        raise ValueError(f"modality-dependent task needs written visual evidence: {candidate['task_id']}")
    if category in {"ir_complement", "diag_ir"} and not (
        decision.get("rgb_only_unique", "").strip().casefold() == "no"
        and _yes(decision, "ir_unique") and _yes(decision, "aux_necessary")
    ):
        raise ValueError(f"IR task lacks blinded complement evidence: {candidate['task_id']}")
    if category in {"reliability", "diag_aux"} and not (
        _yes(decision, "rgb_only_unique") and _yes(decision, "answer_survives_blank")
    ):
        raise ValueError(f"reliability task is not RGB-sufficient: {candidate['task_id']}")
    return True


def _requires_pair(row: dict[str, Any]) -> bool:
    if row["category"] == "robo_competition":
        return True
    if row["category"] not in {"depth_relation", "diag_depth"}:
        return False
    if row.get("construction_version") == "depth_relations_v2":
        return row.get("pair_required") is True
    return True


def accepted_candidates(candidates: list[dict[str, Any]], decisions_path: Path) -> list[dict[str, Any]]:
    with decisions_path.open(encoding="utf-8-sig", newline="") as stream:
        decisions = list(csv.DictReader(stream))
    by_id = {row["task_id"]: row for row in decisions}
    if len(by_id) != len(decisions):
        raise ValueError("duplicate decision task ID")
    candidates_by_id = {row["task_id"]: row for row in candidates}
    unknown = set(by_id) - set(candidates_by_id)
    if unknown:
        raise ValueError(f"unknown decision task IDs: {sorted(unknown)[:5]}")
    accepted = []
    for candidate in candidates:
        decision = by_id.get(candidate["task_id"], {})
        if _approved(candidate, decision):
            row = dict(candidate)
            row["query"] = decision.get("approved_query", "").strip() or candidate["proposed_query"]
            if not row["query"]:
                raise ValueError(f"empty approved Query: {row['task_id']}")
            row["reviewer"] = decision.get("reviewer", "").strip()
            row["human_note"] = decision.get("note", "")
            row["review_mode"] = decision.get("review_mode") or "structured"
            row["modality_judgment"] = decision.get("modality_judgment", "")
            row["review_status"] = "human_quick_approved" if row["review_mode"] == "quick" else "human_approved"
            row["approval_evidence"] = {field: decision.get(field, "") for field in DECISION_FIELDS
                                        if field not in {"task_id", "category", "split", "bundle_id",
                                                         "scene_id", "source_id", "original_query", "proposed_query"}}
            accepted.append(row)
    # Legacy direction pairs and explicit v2 counterfactual pairs are indivisible.
    pair_key = lambda row: (row["category"], row["bundle_id"])
    all_pairs = Counter(pair_key(row) for row in candidates if _requires_pair(row))
    passed_pairs = Counter(pair_key(row) for row in accepted if _requires_pair(row))
    accepted = [row for row in accepted if not _requires_pair(row)
                or (all_pairs[pair_key(row)] == 2 and passed_pairs[pair_key(row)] == 2)]
    return accepted


def _atomic_groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups = []
    for row in rows:
        if _requires_pair(row):
            pairs[row["bundle_id"]].append(row)
        else:
            groups.append([row])
    for bundle_id, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError(f"incomplete required pair: {bundle_id}")
        groups.append(pair)
    return sorted(groups, key=lambda group: (group[0]["bundle_id"], group[0]["task_id"]))


def _take_groups(rows: list[dict[str, Any]], limit: int, *, balanced: bool = False) -> list[dict[str, Any]]:
    remaining = _atomic_groups(rows)
    chosen = []
    scene_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    while remaining and len(chosen) < limit:
        fitting = [group for group in remaining if len(group) <= limit - len(chosen)]
        if not fitting:
            break
        if balanced:
            group = min(fitting, key=lambda item: (
                scene_counts[item[0]["scene_id"]],
                family_counts[item[0].get("relation_type") or "legacy"],
                item[0]["bundle_id"], item[0]["task_id"]))
        else:
            group = fitting[0]
        chosen.extend(group)
        scene_counts[group[0]["scene_id"]] += 1
        family_counts[group[0].get("relation_type") or "legacy"] += len(group)
        remaining.remove(group)
    return chosen


def _select_release(accepted: list[dict[str, Any]], steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if steps not in {400, 600}:
        raise ValueError("A/B/V supports exactly 400 or 600 steps")
    # Both horizons use the same approved unique-task pool. The 400-step run
    # reduces presentations per task, never lowers the human-review gate.
    quotas = {"depth_relation": 64, "robo_competition": 56, "ir_complement": 60,
              "reliability_city": 30, "reliability_rgbt": 15, "reliability_robo": 15}
    sorted_rows = sorted(accepted, key=lambda row: (row["bundle_id"], row["task_id"]))
    selected = []
    for category in ("depth_relation", "robo_competition"):
        rows = [row for row in sorted_rows if row["category"] == category]
        selected.extend(_take_groups(rows, quotas[category], balanced=category == "depth_relation"))
    selected.extend([row for row in sorted_rows if row["category"] == "ir_complement"][:quotas["ir_complement"]])
    for source, key in (("city", "reliability_city"), ("roborefit", "reliability_robo")):
        selected.extend([row for row in sorted_rows if row["category"] == "reliability"
                         and row["source"] == source][:quotas[key]])
    selected.extend([row for row in sorted_rows if row["category"] == "reliability"
                     and row["source"].startswith("rgbt_")][:quotas["reliability_rgbt"]])
    diagnostics = []
    for category in ("diag_depth", "diag_ir", "diag_aux"):
        rows = [row for row in sorted_rows if row["category"] == category]
        diagnostics.extend(_take_groups(rows, TARGETS[category], balanced=category == "diag_depth"))
    counts = Counter(row["category"] for row in selected)
    diag_counts = Counter(row["category"] for row in diagnostics)
    below = {key: (counts[key], minimum) for key, minimum in MINIMUMS.items()
             if counts[key] < minimum}
    if below:
        raise ValueError(f"human-approved training minimums not met: {below}; no release written")
    return selected, diagnostics, dict(quotas)


def _old_city_scene(row: dict[str, Any]) -> str:
    return Path(row["image"][0]).stem


def _draw_old(rows: list[dict[str, Any]], count: int, *, counters: Counter[str],
              scene_counters: Counter[str], scene_cap: int, rng: random.Random,
              excluded_scenes: set[str] = frozenset()) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = _old_city_scene(row)
        if scene not in excluded_scenes:
            groups[scene].append(row)
    scenes = sorted(groups)
    rng.shuffle(scenes)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < count:
        progressed = False
        for scene in scenes:
            if scene_counters[scene] >= scene_cap:
                continue
            candidate = next((row for row in groups[scene] if counters[row["id"]] < 2), None)
            if candidate is None:
                continue
            result.append(candidate)
            counters[candidate["id"]] += 1
            scene_counters[scene] += 1
            progressed = True
            if len(result) == count:
                break
        if not progressed:
            raise ValueError(f"City source exhausted at {len(result)}/{count} with scene cap {scene_cap}")
        rng.shuffle(scenes)
    return result


def _old_sample(row: dict[str, Any], position: int) -> dict[str, Any]:
    sample = json.loads(json.dumps(row))
    root = "/root/autodl-tmp/rematch_20260922/data/city/train/"
    sample["image"] = [root + value.replace("\\", "/") for value in row["image"]]
    sample["id"] = f"abv:old:{position:04d}:{row['id']}"
    sample["origin_query_id"] = row["id"]
    sample["task_category"] = "old_city"
    sample["scene_id"] = f"city:{_old_city_scene(row)}"
    return sample


def _blank_image(path: Path, assets: Path) -> str:
    with Image.open(path) as image:
        size = image.size
    out = assets / f"blank_{size[0]}x{size[1]}.png"
    if not out.is_file():
        Image.new("RGB", size, (0, 0, 0)).save(out)
    return str(out.resolve())


def _native_task(row: dict[str, Any], sample_id: str, missing: tuple[str, ...],
                 assets: Path) -> dict[str, Any]:
    modalities = tuple(modality for modality in ("rgb", "infrared", "depth") if modality in row["images"])
    images = []
    for modality in modalities:
        path = Path(row["images"][modality])
        if not path.is_file():
            raise FileNotFoundError(f"approved task image missing: {path}")
        images.append(_blank_image(path, assets) if modality in missing else str(path.resolve()))
    prompt = native_prompt(row["query"], modalities, row["depth_policy"], missing_modalities=missing)
    answer = json.dumps({"bbox_2d": bbox_to_qwen1000(row["bbox"])}, separators=(",", ":"))
    return {"id": sample_id, "split": "train" if row["split"] == "train" else "diagnostic",
            "source": row["source"], "scene_id": row["scene_id"], "task_category": row["category"],
            "source_task_id": row["task_id"], "missing_modalities_actual": list(missing),
            "intervention_applied": bool(missing), "image": images,
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]}


def diagnostic_gt_record(row: dict[str, Any]) -> dict[str, Any]:
    """Raw float GT in the format shared by evaluator and independent report."""
    return {"bbox": row["bbox"], "visible": row["images"]["rgb"],
            "infrared": row["images"].get("infrared", ""),
            "depth": row["images"].get("depth", ""),
            "query": row["query"], "scene_id": row["scene_id"],
            "source_id": row["source_id"], "category": row["category"],
            "target_object_id": row["target_object_id"]}


def _task_presentations(row: dict[str, Any], assets: Path, count: int) -> list[dict[str, Any]]:
    if not 0 <= count <= 8:
        raise ValueError("each new natural Query may appear at most eight times")
    if row["category"] != "reliability":
        variants = [("normal", ())] * count
    elif row["source"] == "city":
        cycle = [("normal", ()), ("ir_missing", ("infrared",)),
                 ("depth_missing", ("depth",)), ("both_missing", ("infrared", "depth"))]
        variants = [cycle[index % 4] for index in range(count)]
    elif row["source"] == "roborefit":
        variants = [("normal", ())] * ((count+1)//2) + [("depth_missing", ("depth",))] * (count//2)
    else:
        variants = [("normal", ())] * ((count+1)//2) + [("ir_missing", ("infrared",))] * (count//2)
    return [_native_task(row, f"abv:new:{row['task_id']}:{variant}:{index}", missing, assets)
            for index, (variant, missing) in enumerate(variants)]


def _relation_presentations(rows: list[dict[str, Any]], positions: int,
                            assets: Path) -> list[dict[str, Any]]:
    groups = _atomic_groups(rows)
    counts = [0] * len(groups)
    remaining = positions
    while remaining:
        eligible = [index for index, group in enumerate(groups)
                    if counts[index] < 8 and len(group) <= remaining]
        if not eligible:
            break
        index = min(eligible, key=lambda item: (counts[item], item))
        counts[index] += 1
        remaining -= len(groups[index])
    return [sample for group, count in zip(groups, counts, strict=True)
            for row in group for sample in _task_presentations(row, assets, count)]


def _schedule(accepted: list[dict[str, Any]], steps: int, assets: Path,
              output: Path, seed: int = 2026) -> dict[str, list[dict[str, Any]]]:
    city = json.loads(CITY.read_text(encoding="utf-8"))
    diag_locations = {row["location_group"] for row in read_jsonl(output / "data/candidates.jsonl")
                      if row["category"] == "diag_depth"}
    city = [row for row in city if _old_city_scene(row).split("_")[0] not in diag_locations]
    city_new_scenes = {row["scene_id"].split(":", 1)[1] for row in accepted if row["source"] == "city"}
    rng = random.Random(seed)
    counters: Counter[str] = Counter()
    scene_counters: Counter[str] = Counter()
    common_count = steps * 8 * 3 // 5
    common = _draw_old(city, common_count, counters=counters, scene_counters=scene_counters,
                       scene_cap=6, rng=rng, excluded_scenes=city_new_scenes)
    extra = _draw_old(city, steps*8-common_count, counters=counters,
                      scene_counters=scene_counters, scene_cap=11, rng=rng)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key, categories in (("relation", {"depth_relation", "robo_competition"}),
                            ("ir", {"ir_complement"}), ("reliability", {"reliability"})):
        tasks = [row for row in accepted if row["category"] in categories]
        if key == "relation" and tasks:
            depth = [row for row in tasks if row["category"] == "depth_relation"]
            robo = [row for row in tasks if row["category"] == "robo_competition"]
            group = []
            for pool, target_600 in ((depth, 512), (robo, 448)):
                positions = min(target_600, len(pool)*8) * steps // 600
                if pool and any(_requires_pair(row) for row in pool):
                    positions -= positions % 2
                group.extend(_relation_presentations(pool, positions, assets))
        else:
            target_600 = 480
            presentations = min(target_600, len(tasks)*8) * steps // 600
            base, remainder = divmod(presentations, len(tasks)) if tasks else (0, 0)
            group = [sample for index, row in enumerate(tasks)
                     for sample in _task_presentations(row, assets, base + (index < remainder))]
        rng.shuffle(group)
        queues[key] = deque(group)
    A, B = [], []
    common_index = extra_index = 0
    for block in range(steps//10):
        for local in range(80):
            position = block*80+local
            if local < 48:
                old = _old_sample(common[common_index], position)
                common_index += 1
                A.append(old)
                B.append(json.loads(json.dumps(old)))
            else:
                old = _old_sample(extra[extra_index], position)
                extra_index += 1
                A.append(old)
                key = "relation" if local < 64 else "ir" if local < 72 else "reliability"
                B.append(queues[key].popleft() if queues[key] else json.loads(json.dumps(old)))
    assert common_index == common_count and extra_index == steps*8-common_count
    return {"A": A, "B": B, "V": json.loads(json.dumps(B))}


def release(decisions: Path, *, steps: int = 600, seed: int = 2026,
            output: Path = OUT) -> dict[str, Any]:
    if seed not in {2026, 2027}:
        raise ValueError("release seed must be one of the planned 2026/2027 seeds")
    candidates = read_jsonl(output / "data/candidates.jsonl")
    accepted = accepted_candidates(candidates, decisions)
    train, diagnostic, quotas = _select_release(accepted, steps)
    if not diagnostic:
        raise ValueError("at least one real diagnostic case is required by the evaluator")
    train_scenes = {row["scene_id"] for row in train}
    if train_scenes & {row["scene_id"] for row in diagnostic}:
        raise ValueError("scene overlap between approved train and diagnostic")
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for row in train + diagnostic:
        if row.get("sequence_group"):
            sequence_splits[row["sequence_group"]].add(row["split"])
    if any(len(splits) > 1 for splits in sequence_splits.values()):
        raise ValueError("RGBT/Robo sequence crossed train and diagnostic")
    ready = output / "data" / f"release_{steps}_seed{seed}"
    if ready.exists():
        raise FileExistsError(f"frozen release directory already exists: {ready}")
    for row in train + diagnostic:
        for image in row["images"].values():
            if not Path(image).is_file():
                raise FileNotFoundError(f"approved source image missing: {row['task_id']}: {image}")
    assets = ready / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    schedules = _schedule(train, steps, assets, output, seed)
    manifests = {}
    for arm, rows in schedules.items():
        path = ready / f"{arm}.json"
        path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
        manifests[arm] = str(path.resolve())
    diagnostic_files = {}
    scene_map, class_map, gt_rows = {}, {}, {}
    for condition in ("normal", "ir_missing", "depth_missing", "both_missing"):
        samples = []
        for row in diagnostic:
            modalities = {m for m in ("infrared", "depth") if m in row["images"]}
            requested = {"normal": set(), "ir_missing": {"infrared"},
                         "depth_missing": {"depth"}, "both_missing": {"infrared", "depth"}}[condition]
            missing = tuple(m for m in ("infrared", "depth") if m in modalities & requested)
            sample_id = f"abv:diag:{row['task_id']}"
            sample = _native_task(row, sample_id, missing, assets)
            sample["requested_condition"] = condition
            samples.append(sample)
            if condition == "normal":
                scene_map[sample_id] = row.get("location_group") or row.get("sequence_group") or row["scene_id"]
                class_map[sample_id] = {"diag_depth": "depth", "diag_ir": "ir",
                                        "diag_aux": "rgb_sufficient"}[row["category"]]
                gt_rows[sample_id] = diagnostic_gt_record(row)
        path = ready / "diagnostics" / f"{condition}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(samples, ensure_ascii=False) + "\n", encoding="utf-8")
        diagnostic_files[condition] = str(path.resolve())
    gt_path, scene_path, class_path = (ready / "diagnostics/gt.json",
                                       ready / "diagnostics/scene_map.json",
                                       ready / "diagnostics/class_map.json")
    gt_path.write_text(json.dumps(gt_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scene_path.write_text(json.dumps(scene_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    class_path.write_text(json.dumps(class_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {"approved_unique_train": len(train), "approved_diagnostic": len(diagnostic),
              "review_modes": dict(Counter(row["review_mode"] for row in train + diagnostic)),
              "modality_judgments": dict(Counter(row["modality_judgment"] or "unrated" for row in train + diagnostic)),
              "train_by_category": dict(Counter(row["category"] for row in train)),
              "diagnostic_by_category": dict(Counter(row["category"] for row in diagnostic)),
              "presentations_by_arm": {arm: len(rows) for arm, rows in schedules.items()},
              "old_city_BV": sum(row["task_category"] == "old_city" for row in schedules["B"]),
              "new_BV": sum(row["task_category"] != "old_city" for row in schedules["B"]),
              "targets": quotas,
              "diagnostic_category_interpretable": {
                  category: sum(row["category"] == category for row in diagnostic) >= 16
                  for category in ("diag_depth", "diag_ir", "diag_aux")},
              "diagnostic_small_n_rule": "fewer than 16 is case-by-case only; it does not block training"}
    audit_files = {"accepted_reviews": "accepted_reviews.jsonl",
                   "scene_exposure": "scene_exposure.json",
                   "ancestor_overlap": "ancestor_overlap.json"}
    payload = {"status": "ready", "steps": steps, "seed": seed, "manifests": manifests,
               "diagnostics": diagnostic_files, "gt_manifest": str(gt_path.resolve()),
               "scene_map": str(scene_path.resolve()), "class_map": str(class_path.resolve()),
               "counts": counts, "audit_files": audit_files,
               "path_roots": {"city_old_cloud": "/root/autodl-tmp/rematch_20260922/data/city/train/",
                              "new_review_assets_local": str((output / "review").resolve()),
                              "rgbt_cloud": "/root/autodl-tmp/rematch_20260922/data/external/RGBT-GroundBench/raw/"}}
    review_path = ready / "accepted_reviews.jsonl"
    selected_train_ids = {row["task_id"] for row in train}
    selected_diagnostic_ids = {row["task_id"] for row in diagnostic}
    write_jsonl(review_path, [dict(row,
                                  selected_for_training=row["task_id"] in selected_train_ids,
                                  selected_for_diagnostic=row["task_id"] in selected_diagnostic_ids)
                             for row in accepted])
    exposure = {arm: {"by_scene": dict(sorted(Counter(row["scene_id"] for row in rows).items())),
                      "by_category": dict(sorted(Counter(row["task_category"] for row in rows).items()))}
                for arm, rows in schedules.items()}
    exposure_path = ready / "scene_exposure.json"
    exposure_path.write_text(json.dumps(exposure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(output / "data/ancestor_overlap.json", ready / audit_files["ancestor_overlap"])
    release_path = ready / "release.json"
    release_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**payload, "release_file": str(release_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "revise-depth", "review", "release"))
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--steps", type=int, choices=(400, 600), default=600)
    parser.add_argument("--seed", type=int, choices=(2026, 2027), default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-output", type=Path, default=OUT)
    parser.add_argument("--depth-specs", type=Path)
    args = parser.parse_args()
    output = args.output or (DEPTH_V2_OUT if args.command in {"prepare", "revise-depth"} else OUT)
    if args.command in {"prepare", "revise-depth"}:
        result = revise_depth(args.source_output, output, args.depth_specs)
    elif args.command == "review":
        result = render_review(read_jsonl(output / "data/candidates.jsonl"), output,
                               refresh_decision_template=False)
    else:
        if args.decisions is None:
            parser.error("release requires --decisions from actual human review")
        result = release(args.decisions, steps=args.steps, seed=args.seed, output=output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
