"""Build paired TriGround T/M SFT schedules and independent probes.

The prior human-reviewed ABV release supplies the RGB-box tasks. IR-read boxes
are taken only from a separate review file; its absence produces a preview with
an explicit deficit. No inferred RGB box is ever used as an IR target.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000, depth_mm_to_grayscale_rgb, native_prompt
from tools.prepare_triground_abv_data import CITY, diagnostic_gt_record, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
ABV = ROOT / "results/triground_abv_execution_20260928/reviewed_data/data/release_600_seed2026"
OUTPUT = ROOT / "results/triground_tm_20260929/data"
CITY_LOCAL = ROOT / "data/city_object_extension"
CITY_CLOUD = "/root/autodl-tmp/rematch_20260922/data/city/train/"
HOLDOUT_LOCATIONS = {"000003", "000005", "000016"}
IR_QUERY_MARKER = "Locate the object described by this query: "

QUOTAS_600 = {
    "depth_read_normal": 120, "depth_read_invalid": 30,
    "depth_joint": 150, "ir_read_normal": 160, "ir_read_blank": 40,
    "ir_joint": 200, "competition": 250, "reliability_normal": 50,
    "reliability_ir_low": 100, "reliability_depth_low": 100,
}
QUOTAS_400 = {
    "depth_read_normal": 80, "depth_read_invalid": 20,
    "depth_joint": 100, "ir_read_normal": 107, "ir_read_blank": 26,
    "ir_joint": 134, "competition": 166, "reliability_normal": 33,
    "reliability_ir_low": 67, "reliability_depth_low": 67,
}
QUOTAS = {600: QUOTAS_600, 400: QUOTAS_400}


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path.resolve())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return str(path.resolve())


def _answer_bbox(box: list[float] | None) -> dict[str, Any]:
    return {"bbox_2d": None if box is None else bbox_to_qwen1000(box)}


def _sample(sample_id: str, *, split: str, source: str, scene_id: str,
            category: str, source_task_id: str, images: list[str],
            modalities: list[str], query: str, prompt: str,
            answer: dict[str, Any] | str, task_type: str,
            coordinate_system: str = "qwen_0_1000", **extra: Any) -> dict[str, Any]:
    if len(images) != len(modalities) or prompt.count("<image>") != len(images):
        raise ValueError(f"{sample_id}: image/prompt/modality slots disagree")
    target = json.dumps(answer, ensure_ascii=False, separators=(",", ":")) if isinstance(answer, dict) else answer
    return {
        "id": sample_id, "split": split, "source": source, "scene_id": scene_id,
        "task_category": category, "source_task_id": source_task_id,
        "image": images, "modalities": modalities, "query": query,
        "task_type": task_type, "coordinate_system": coordinate_system,
        "expected_answer": answer,
        "missing_modalities_actual": [],
        "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": target}],
        **extra,
    }


def _city_path(relative: str) -> str:
    relative = relative.replace("\\", "/")
    local = CITY_LOCAL / relative
    return str(local.resolve()) if local.is_file() else CITY_CLOUD + relative


def _old_city(row: dict[str, Any], sample_id: str) -> dict[str, Any]:
    prompt = row["conversations"][0]["value"]
    query = prompt.split(IR_QUERY_MARKER, 1)[1].split("\n", 1)[0]
    answer = json.loads(row["conversations"][1]["value"])
    stem = Path(row["image"][0]).stem
    return _sample(
        sample_id, split="train", source="city", scene_id=f"city:{stem}",
        category="old_city", source_task_id=row["id"],
        images=[_city_path(relative) for relative in row["image"]],
        modalities=["rgb", "ir", "depth"], query=query, prompt=prompt,
        answer=answer, task_type="rgb_bbox", origin_query_id=row["id"],
        location_group=stem.split("_")[0],
    )


def _draw_city(rows: list[dict[str, Any]], count: int, rng: random.Random,
               used: Counter[str]) -> list[dict[str, Any]]:
    scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenes[Path(row["image"][0]).stem].append(row)
    scene_ids = sorted(scenes)
    rng.shuffle(scene_ids)
    for group in scenes.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        progressed = False
        for scene in scene_ids:
            row = next((item for item in scenes[scene] if used[item["id"]] < 2), None)
            if row is None:
                continue
            selected.append(row)
            used[row["id"]] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f"old City exhausted at {len(selected)}/{count}")
        rng.shuffle(scene_ids)
    return selected


def _modalities(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(m for m in ("rgb", "infrared", "depth") if m in row["images"])


def _blank_image(path: Path, assets: Path) -> str:
    with Image.open(path) as image:
        width, height = image.size
    output = assets / f"blank_{width}x{height}.png"
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (width, height)).save(output)
    return str(output.resolve())


def _low_ir_image(path: Path, assets: Path, task_id: str) -> str:
    output = assets / "low_ir" / f"{task_id.replace(':', '_')}.png"
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            altered = ImageEnhance.Contrast(image.convert("RGB")).enhance(0.15)
            altered.filter(ImageFilter.GaussianBlur(radius=2.5)).save(output)
    return str(output.resolve())


def _bbox_task(row: dict[str, Any], sample_id: str, assets: Path,
               *, low: str | None = None, split: str = "train",
               condition: str = "normal") -> dict[str, Any]:
    names = _modalities(row)
    paths = []
    for name in names:
        path = Path(row["images"][name])
        if not path.is_file():
            raise FileNotFoundError(f"{row['task_id']}: {path}")
        if name == low == "infrared" and condition == "reliability_ir_low":
            paths.append(_low_ir_image(path, assets, row["task_id"]))
        else:
            paths.append(_blank_image(path, assets) if name == low else str(path.resolve()))
    # Do not tell the model that an auxiliary image was degraded.
    prompt = native_prompt(row["query"], names, row["depth_policy"])
    sample = _sample(
        sample_id, split=split, source=row["source"], scene_id=row["scene_id"],
        category=row["category"], source_task_id=row["task_id"], images=paths,
        modalities=["ir" if n == "infrared" else n for n in names],
        query=row["query"], prompt=prompt, answer=_answer_bbox(row["bbox"]),
        task_type="rgb_bbox", requested_condition=condition,
        degraded_modality="ir" if low == "infrared" else low,
        source_id=row["source_id"],
    )
    sample["missing_modalities_actual"] = ([] if condition == "reliability_ir_low" else
                                           (["ir" if low == "infrared" else low] if low else []))
    return sample


def _depth_crop(raw: np.ndarray, region: list[float]) -> Image.Image:
    height, width = raw.shape
    x1, y1, x2, y2 = region
    left, top = int(x1 * width), int(y1 * height)
    right, bottom = max(left + 1, int(np.ceil(x2 * width))), max(top + 1, int(np.ceil(y2 * height)))
    crop = raw[top:bottom, left:right]
    if not crop.size:
        raise ValueError(f"empty depth ROI: {region}")
    return Image.fromarray(depth_mm_to_grayscale_rgb(crop))


def _depth_pair(row: dict[str, Any], assets: Path, rng: random.Random,
                *, invalid: bool, sample_id: str, split: str,
                reverse: bool | None = None) -> dict[str, Any]:
    relation = row["relation"]
    target_id = relation["target_id"]
    others = relation["competitors"]
    evidence = row["depth_evidence"]["by_source_id"]
    other_id = relation.get("reference_id") or min(
        others, key=lambda item: (abs(evidence[item]["raw_median_mm"] - evidence[target_id]["raw_median_mm"]), item))
    target = evidence[target_id]
    other = evidence[other_id]
    if target["region_source"] != "manual_subject_interior" or other["region_source"] != "manual_subject_interior":
        raise ValueError(f"non-manual ROI in {row['task_id']}")
    ordered = [(target_id, target), (other_id, other)]
    if reverse is None:
        rng.shuffle(ordered)
    else:
        ordered.sort(key=lambda item: item[0], reverse=reverse)
    with Image.open(row["images"]["depth_raw"]) as source:
        raw = np.asarray(source)
    crops = [_depth_crop(raw, item["region"]) for _, item in ordered]
    canvas_size = (max(crop.width for crop in crops), max(crop.height for crop in crops))
    images = []
    for (object_id, _), crop in zip(ordered, crops, strict=True):
        path = assets / "depth_crops" / f"{row['task_id'].replace(':', '_')}_{object_id.replace(':', '_')}_{canvas_size[0]}x{canvas_size[1]}.png"
        if not path.is_file():
            canvas = Image.new("RGB", canvas_size)
            canvas.paste(crop, ((canvas_size[0]-crop.width)//2, (canvas_size[1]-crop.height)//2))
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path)
        images.append(str(path.resolve()))
    blank_side = None
    if invalid:
        blank_side = rng.randrange(2)
        images[blank_side] = _blank_image(Path(images[blank_side]), assets)
    label = "unknown" if invalid else ("A_nearer" if ordered[0][1]["raw_median_mm"] < ordered[1][1]["raw_median_mm"] else "B_nearer")
    prompt = (
        "<image>\n<image>\nThese are two depth crops, A then B, from the same scene. "
        "The fixed gray scale makes brighter valid pixels nearer; black pixels are invalid. "
        "Which subject is nearer to the camera? "
        "Return exactly A_nearer, B_nearer, or unknown when either crop has no valid depth."
    )
    sample = _sample(
        sample_id, split=split, source=row["source"], scene_id=row["scene_id"],
        category="depth_read_invalid" if invalid else "depth_read_normal",
        source_task_id=row["task_id"], images=images,
        modalities=["depth", "depth"], query=row["query"], prompt=prompt,
        answer=label, task_type="depth_relation", coordinate_system="categorical",
        roi_source_ids=[item[0] for item in ordered], invalid_side=blank_side,
        location_group=row["location_group"],
        reading_scope="manual_subject_pair_only", uses_oracle_regions=True,
        full_relation=dict(relation),
    )
    return sample


def _ir_read(row: dict[str, Any], review: dict[str, Any], assets: Path,
             *, blank: bool, sample_id: str, split: str) -> dict[str, Any]:
    ir = Path(row["images"]["infrared"])
    if not ir.is_file():
        raise FileNotFoundError(f"{row['task_id']}: {ir}")
    image = _blank_image(ir, assets) if blank else str(ir.resolve())
    query = review["ir_query"]
    prompt = (
        "<image>\nThis is an infrared image. Locate the object described by this query: "
        f"{query}\nReturn only JSON in this exact form with coordinates normalized to 0-1000: "
        '{"bbox_2d":[x1,y1,x2,y2]}; use {"bbox_2d":null} if the target cannot be seen.'
    )
    sample = _sample(
        sample_id, split=split, source=row["source"], scene_id=row["scene_id"],
        category="ir_read_blank" if blank else "ir_read_normal",
        source_task_id=row["task_id"], images=[image], modalities=["ir"],
        query=query, prompt=prompt,
        answer=_answer_bbox(None if blank else review["ir_bbox"]),
        task_type="ir_bbox", source_id=row["source_id"],
        ir_review_split=review["split"],
        ir_gt_bbox=None if blank else list(review["ir_bbox"]),
    )
    sample["missing_modalities_actual"] = ["ir"] if blank else []
    return sample


def _review_map(path: Path | None, accepted: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    reviews = read_jsonl(path)
    result = {}
    for review in reviews:
        task_id = review["task_id"]
        if task_id not in accepted or task_id in result:
            raise ValueError(f"IR review has missing/duplicate source task: {task_id}")
        if review["split"] not in {"train", "diagnostic"}:
            raise ValueError(f"invalid IR review split: {task_id}")
        if review["accepted"]:
            if not review["ir_query"].strip():
                raise ValueError(f"empty reviewed IR query: {task_id}")
            bbox_to_qwen1000(review["ir_bbox"])
            if "infrared" not in accepted[task_id]["images"]:
                raise ValueError(f"IR source absent: {task_id}")
            result[task_id] = review
    return result


def _diagnostic_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    if path.suffix == ".jsonl":
        data = read_jsonl(path)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    ids = [item if isinstance(item, str) else item["task_id"] for item in data]
    if len(ids) > 16 or len(ids) != len(set(ids)):
        raise ValueError("IR diagnostic list must have at most 16 unique task IDs")
    return ids


def _allocate_atomic(rows: list[dict[str, Any]], target: int,
                     used: Counter[str], rng: random.Random,
                     *, group_key: str = "task_id") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    entries = [sorted(group, key=lambda row: row["task_id"]) for group in groups.values()]
    rng.shuffle(entries)
    result = []
    while len(result) < target:
        possible = [group for group in entries if len(group) <= target-len(result)
                    and all(used[row["task_id"]] < 8 for row in group)]
        if not possible:
            break
        group = min(possible, key=lambda item: (max(used[row["task_id"]] for row in item),
                                                sum(used[row["task_id"]] for row in item),
                                                entries.index(item)))
        result.extend(group)
        used.update(row["task_id"] for row in group)
    return result


def _balanced_extras(extra: dict[str, list[dict[str, Any]]],
                     fallback: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    queues = {key: list(value) for key, value in extra.items() if value}
    if fallback:
        queues["city_fallback"] = list(fallback)
    slots = []
    for category, samples in queues.items():
        rng.shuffle(samples)
        for index in range(len(samples)):
            slots.append(((index + rng.random()) / len(samples), category))
    slots.sort()
    counters: Counter[str] = Counter()
    result = []
    for _, category in slots:
        result.append(queues[category][counters[category]])
        counters[category] += 1
    return result


def _category_rows(accepted: list[dict[str, Any]], ir_reviews: dict[str, dict[str, Any]],
                   diag_ids: list[str]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], set[str]]:
    diagnostic = [row for row in accepted if row.get("selected_for_diagnostic")
                  and row["category"] in {"diag_ir", "diag_aux", "diag_depth"}]
    depth = [row for row in accepted if row["split"] == "train" and row["category"] == "depth_relation"]
    held = [row for row in depth if row["location_group"] in HOLDOUT_LOCATIONS]
    diagnostic.extend(held)
    if len(held) != 17 or len({row["scene_id"] for row in held}) != 8:
        raise ValueError(f"Depth holdout changed: {len(held)} tasks, {len({row['scene_id'] for row in held})} scenes")
    by_id = {row["task_id"]: row for row in accepted}
    for task_id in diag_ids:
        if task_id not in ir_reviews or ir_reviews[task_id]["split"] != "diagnostic":
            raise ValueError(f"IR diagnostic lacks approved diagnostic review: {task_id}")
        if task_id not in {row["task_id"] for row in diagnostic}:
            diagnostic.append(by_id[task_id])
    diag_scenes = {row["scene_id"] for row in diagnostic}
    diag_locations = {row.get("location_group") for row in diagnostic if row.get("location_group")}
    diag_sequences = {row.get("sequence_group") for row in diagnostic if row.get("sequence_group")}
    train = [row for row in accepted if row.get("selected_for_training")
             and row["scene_id"] not in diag_scenes
             and row.get("location_group") not in diag_locations
             and row.get("sequence_group") not in diag_sequences]
    if any(row["scene_id"] in diag_scenes for row in train):
        raise AssertionError("diagnostic scene in train")
    pools = {
        "depth": [row for row in train if row["category"] == "depth_relation"],
        "ir_joint": [row for row in train if row["category"] == "ir_complement"],
        "ir_read": [row for row in train if row["task_id"] in ir_reviews and ir_reviews[row["task_id"]]["split"] == "train"],
        "competition": [row for row in train if row["category"] == "robo_competition"],
        "reliability": [row for row in train if row["category"] == "reliability"],
    }
    return pools, diagnostic, diag_locations


def prepare(*, steps: int = 600, seed: int = 2028,
            ir_review: Path | None = None, ir_diagnostics: Path | None = None,
            output: Path = OUTPUT, abv: Path = ABV,
            city_path: Path = CITY, refresh: bool = False) -> dict[str, Any]:
    quotas = QUOTAS[steps]
    accepted = read_jsonl(abv / "accepted_reviews.jsonl")
    by_id = {row["task_id"]: row for row in accepted}
    reviews = _review_map(ir_review, by_id)
    diag_ids = _diagnostic_ids(ir_diagnostics)
    pools, diagnostic, diag_locations = _category_rows(accepted, reviews, diag_ids)
    release_dir = output / f"{'release' if ir_review else 'preview'}_{steps}_seed{seed}"
    if ir_review and release_dir.exists() and not refresh:
        raise FileExistsError(release_dir)
    assets = release_dir / "assets"
    rng = random.Random(seed)
    used: Counter[str] = Counter()
    requested = dict(quotas)
    extra: dict[str, list[dict[str, Any]]] = {}

    depth = pools["depth"]
    if len(depth) != 39 or len({row["scene_id"] for row in depth}) != 17:
        raise ValueError(f"Depth train split changed: {len(depth)} tasks")
    depth = [dict(row, paired_key=row["bundle_id"] if row["pair_required"] else row["task_id"])
             for row in depth]
    depth_subject_pairs = [row for row in depth if row["relation_type"] != "middle"]
    if len(depth_subject_pairs) != 32:
        raise ValueError(f"Depth subject-pair count changed: {len(depth_subject_pairs)}")
    for name, count in (("depth_read_invalid", quotas["depth_read_invalid"]),
                        ("depth_read_normal", quotas["depth_read_normal"]),
                        ("depth_joint", quotas["depth_joint"])):
        chosen = _allocate_atomic(depth if name == "depth_joint" else depth_subject_pairs,
                                  count, used, rng, group_key="paired_key")
        extra[name] = [(_bbox_task(row, f"tm:{name}:{i}:{row['task_id']}", assets)
                        if name == "depth_joint" else
                        _depth_pair(row, assets, rng, invalid=name.endswith("invalid"),
                                    sample_id=f"tm:{name}:{i}:{row['task_id']}", split="train"))
                       for i, row in enumerate(chosen)]
    for name, count in (("ir_read_normal", quotas["ir_read_normal"]),
                        ("ir_read_blank", quotas["ir_read_blank"])):
        chosen = _allocate_atomic(pools["ir_read"], count, used, rng)
        extra[name] = [_ir_read(row, reviews[row["task_id"]], assets,
                                blank=name.endswith("blank"),
                                sample_id=f"tm:{name}:{i}:{row['task_id']}", split="train")
                       for i, row in enumerate(chosen)]
    chosen = _allocate_atomic(pools["ir_joint"], quotas["ir_joint"], used, rng)
    extra["ir_joint"] = [_bbox_task(row, f"tm:ir_joint:{i}:{row['task_id']}", assets)
                         for i, row in enumerate(chosen)]
    chosen = _allocate_atomic(pools["competition"], quotas["competition"], used, rng, group_key="bundle_id")
    extra["competition"] = [_bbox_task(row, f"tm:competition:{i}:{row['task_id']}", assets)
                            for i, row in enumerate(chosen)]
    for name, low in (("reliability_normal", None), ("reliability_ir_low", "infrared"),
                      ("reliability_depth_low", "depth")):
        candidates = [row for row in pools["reliability"] if low is None or low in row["images"]]
        chosen = _allocate_atomic(candidates, quotas[name], used, rng)
        extra[name] = [_bbox_task(row, f"tm:{name}:{i}:{row['task_id']}", assets,
                                  low=low, condition=name)
                       for i, row in enumerate(chosen)]
    deficits = {name: requested[name]-len(extra[name]) for name in requested}

    old_city = json.loads(city_path.read_text(encoding="utf-8"))
    diag_scenes = {row["scene_id"].split(":", 1)[1] for row in diagnostic if row["source"] == "city"}
    old_city = [row for row in old_city
                if Path(row["image"][0]).stem.split("_")[0] not in diag_locations
                and Path(row["image"][0]).stem not in diag_scenes]
    city_used: Counter[str] = Counter()
    common = _draw_city(old_city, steps*8, rng, city_used)
    t_used = city_used.copy()
    m_used = city_used.copy()
    t_extra = _draw_city(old_city, steps*2, rng, t_used)
    fallback_count = sum(deficits.values())
    m_fallback = _draw_city(old_city, fallback_count, rng, m_used)
    common_samples = [_old_city(row, f"tm:common:{i:04d}:{row['id']}") for i, row in enumerate(common)]
    t_samples = [_old_city(row, f"tm:T:extra:{i:04d}:{row['id']}") for i, row in enumerate(t_extra)]
    m_fallback_samples = [_old_city(row, f"tm:M:fallback:{i:04d}:{row['id']}") for i, row in enumerate(m_fallback)]
    m_extras = _balanced_extras(extra, m_fallback_samples, rng)
    if len(m_extras) != steps*2:
        raise AssertionError(f"M extra count {len(m_extras)}")
    T, M = [], []
    for step in range(steps):
        start = step*8
        T.extend(common_samples[start:start+8] + t_samples[step*2:step*2+2])
        M.extend(common_samples[start:start+8] + m_extras[step*2:step*2+2])

    manifests = {"T": _write_json(release_dir / "T.json", T),
                 "M": _write_json(release_dir / "M.json", M)}
    diagnostics = {}
    gt, scene_map, class_map = {}, {}, {}
    for condition, low_set in (("normal", set()), ("ir_missing", {"infrared"}),
                               ("depth_missing", {"depth"}),
                               ("both_missing", {"infrared", "depth"})):
        samples = []
        for row in diagnostic:
            # An intervention affects only an auxiliary view actually present.
            low = next((name for name in ("infrared", "depth") if name in low_set and name in row["images"]), None)
            sample_id = f"tm:diag:{row['task_id']}"
            sample = _bbox_task(row, sample_id, assets, low=low, split="diagnostic", condition=condition)
            if len(low_set) == 2 and all(name in row["images"] for name in low_set):
                sample["image"] = [_blank_image(Path(row["images"][name]), assets) if name in low_set
                                   else image for name, image in zip(_modalities(row), sample["image"], strict=True)]
                sample["degraded_modality"] = "ir+depth"
                sample["missing_modalities_actual"] = ["ir", "depth"]
            samples.append(sample)
            if condition == "normal":
                gt[sample_id] = diagnostic_gt_record(row)
                scene_map[sample_id] = row.get("location_group") or row.get("sequence_group") or row["scene_id"]
                class_map[sample_id] = (
                    "depth" if row["category"] in {"diag_depth", "depth_relation"}
                    else "ir" if row["category"] == "diag_ir" or row["task_id"] in diag_ids
                    else "rgb_sufficient"
                )
        diagnostics[condition] = _write_json(release_dir / "diagnostics" / f"{condition}.json", samples)
    gt_path = _write_json(release_dir / "diagnostics/gt.json", gt)
    scene_path = _write_json(release_dir / "diagnostics/scene_map.json", scene_map)
    class_path = _write_json(release_dir / "diagnostics/class_map.json", class_map)

    depth_probe_rows = [row for row in diagnostic if row["category"] in {"diag_depth", "depth_relation"}]
    depth_probe = [_depth_pair(row, assets, rng, invalid=False,
                               sample_id=f"tm:probe:depth:{row['task_id']}", split="diagnostic")
                   for row in depth_probe_rows]
    depth_probe += [_depth_pair(row, assets, rng, invalid=True,
                                sample_id=f"tm:probe:depth_invalid:{row['task_id']}", split="diagnostic")
                    for row in depth_probe_rows]
    ir_probe = []
    for task_id in diag_ids:
        row = by_id[task_id]
        for blank in (False, True):
            ir_probe.append(_ir_read(row, reviews[task_id], assets, blank=blank,
                                     sample_id=f"tm:probe:ir:{'blank' if blank else 'normal'}:{task_id}",
                                     split="diagnostic"))
    probes = {"depth_read": _write_json(release_dir / "probes/depth_read.json", depth_probe),
              "ir_read": _write_json(release_dir / "probes/ir_read.json", ir_probe)}
    # One actual M presentation per trained mother. T uses paired positions.
    selected_positions = []
    seen_mothers: set[str] = set()
    for position, sample in enumerate(M):
        if sample["task_category"] == "old_city" or sample["source_task_id"] in seen_mothers:
            continue
        seen_mothers.add(sample["source_task_id"])
        selected_positions.append(position)
    train_probes = {"M": _write_json(release_dir / "probes/train_M_unique.json", [M[i] for i in selected_positions]),
                    "T": _write_json(release_dir / "probes/train_T_paired.json", [T[i] for i in selected_positions])}
    source_reviews = _write_jsonl(release_dir / "source_reviews.jsonl", accepted)
    ir_review_copy = _write_jsonl(release_dir / "ir_reviews.jsonl",
                                  read_jsonl(ir_review) if ir_review else [])
    depth_bundles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in depth:
        depth_bundles[row["bundle_id"]].append(row)
    counts = {
        "presentations": {"T": len(T), "M": len(M)},
        "common_city": len(common_samples), "T_extra_city": len(t_samples),
        "M_extra_city_fallback": fallback_count,
        "M_extra_actual": {name: len(items) for name, items in extra.items()},
        "M_mother_max": max(used.values()) if used else 0,
        "training_probe_mothers": len(selected_positions),
        "depth_bundle_max_exposure_gap": max(
            (max(used[row["task_id"]] for row in group)-min(used[row["task_id"]] for row in group)
             for group in depth_bundles.values()), default=0),
        "depth_required_pair_max_exposure_gap": max(
            (max(used[row["task_id"]] for row in group)-min(used[row["task_id"]] for row in group)
             for group in depth_bundles.values() if group[0]["pair_required"]), default=0),
        "depth_train_tasks": len(depth), "depth_train_scenes": len({row["scene_id"] for row in depth}),
        "depth_subject_pair_tasks": len(depth_subject_pairs),
        "depth_heldout_tasks": 17, "depth_heldout_scenes": 8,
        "diagnostic_tasks": len(diagnostic), "depth_probe_rows": len(depth_probe),
        "ir_probe_rows": len(ir_probe), "reviewed_ir_train": len(pools["ir_read"]),
    }
    city_scene_exposure = {
        arm: dict(sorted(Counter(row["scene_id"] for row in samples
                                 if row["source"] == "city").items()))
        for arm, samples in (("T", T), ("M", M))
    }
    exposure_path = _write_json(release_dir / "city_scene_exposure.json", city_scene_exposure)
    counts["city_scenes"] = {
        arm: {"unique": len(exposure), "max_presentations": max(exposure.values()),
              "min_presentations": min(exposure.values()),
              "total_presentations": sum(exposure.values())}
        for arm, exposure in city_scene_exposure.items()
    }
    payload = {
        "status": "ready" if ir_review else "provisional",
        "steps": steps, "seed": seed, "manifests": manifests,
        "diagnostics": diagnostics, "gt_manifest": gt_path,
        "scene_map": scene_path, "class_map": class_path,
        "probes": probes, "training_probes": train_probes,
        "city_scene_exposure": exposure_path,
        "quotas": quotas, "deficits": deficits, "counts": counts,
        "sources": {"abv_release": str(abv.resolve()), "abv_accepted_reviews": str((abv / "accepted_reviews.jsonl").resolve()),
                    "source_reviews": source_reviews, "ir_review": str(ir_review.resolve()) if ir_review else None,
                    "ir_reviews_retained": ir_review_copy,
                    "ir_diagnostics": str(ir_diagnostics.resolve()) if ir_diagnostics else None,
                    "city_source": str(city_path.resolve()), "city_cloud_root": CITY_CLOUD},
    }
    payload["release_file"] = str((release_dir / "release.json").resolve())
    _write_json(release_dir / "release.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(400, 600), default=600)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--ir-review", type=Path)
    parser.add_argument("--ir-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="refresh after a reviewed metadata correction")
    args = parser.parse_args()
    print(json.dumps(prepare(steps=args.steps, seed=args.seed, ir_review=args.ir_review,
                             ir_diagnostics=args.ir_diagnostics, output=args.output,
                             refresh=args.refresh),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
