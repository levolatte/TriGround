"""Build reviewed grounding indexes and a fixed four-arm presentation schedule.

The default formal build requires restored City media and an explicit official-test
frame exclusion report. ``--local-pool`` builds a labelled nonformal RGBDT/Robo
smoke pool while those City inputs are unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from tools.prepare_qwen3vl_native_sft import native_prompt, render_depth


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
AUX = DATA / "grounding_aux_round1_20261025"
CITY_ZIP = ROOT / "docs/research/2026-10-01-structure-pro-audit/package/reference/original_city_labels.zip"
CITY_DEV = ROOT / "docs/research/2026-10-24-label-fix/city_val_corrected.json"
CITY_CORRECTIONS = ROOT / "docs/research/2026-10-24-label-fix/corrections.json"
ROBO = DATA / "external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl"


@dataclass(frozen=True)
class Sources:
    city_zip: Path
    city_dev: Path
    city_corrections: Path
    rgbdt_inventory: Path
    pairs: Path
    external_dev: Path
    robo: Path
    bundle: Path
    robo_media_dir: Path


def source_paths(data_root: Path, source_dir: Path | None = None) -> Sources:
    if source_dir is None:
        return Sources(CITY_ZIP, CITY_DEV, CITY_CORRECTIONS,
                       AUX / "inventory/rgbdt_train.jsonl",
                       AUX / "bundle/annotations/pairs.jsonl",
                       AUX / "annotations/external_grounding_final.jsonl",
                       ROBO, AUX / "bundle", ROBO.parent)
    return Sources(source_dir / "original_city_labels.zip",
                   source_dir / "city_val_corrected.json",
                   source_dir / "corrections.json",
                   source_dir / "rgbdt_train.jsonl",
                   source_dir / "pairs.jsonl",
                   source_dir / "external_grounding_final.jsonl",
                   source_dir / "train_depth_visual_fixed_p01_p99.jsonl",
                   data_root / "grounding_aux_round1_20261025/bundle",
                   data_root / "external/RoboRefIt/subset/converted")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def box(value: list[float]) -> list[float]:
    result = [float(x) for x in value]
    if len(result) != 4 or not all(math.isfinite(x) and 0 <= x <= 1 for x in result):
        raise ValueError(f"invalid normalized box: {value}")
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError(f"empty box: {value}")
    return result


def iou(a: list[float], b: list[float]) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def relative_image(path: Path, data_root: Path) -> str:
    path = (path if path.is_absolute() else data_root / path).resolve()
    return path.relative_to(data_root.resolve()).as_posix()


def sample(*, sample_id: str, source: str, group_id: str, query: str,
           images: list[str], modalities: list[str], depth_policy: str | None,
           bbox: list[float], label_source: str, negative_boxes: list[list[float]] | None = None,
           negative_kind: list[str] | None = None, **extra: object) -> dict:
    if images[0] == "" or len(images) != len(modalities):
        raise ValueError(f"{sample_id}: images/modalities mismatch")
    prompt = native_prompt(query, tuple(modalities), depth_policy or "visual")
    return {
        "id": sample_id, "source": source, "group_id": group_id,
        "images": images, "modalities": modalities, "depth_policy": depth_policy,
        "messages": [{"role": "user", "content": prompt}],
        "solution": box(bbox), "negative_boxes": negative_boxes or [],
        "negative_kind": negative_kind or [], "label_source": label_source,
        **extra,
    }


def rgbdt_key(row: dict) -> tuple[str, str, str]:
    return (str(row["scene_id"]), str(row["image_id"]), row["query"].strip().casefold().rstrip("?. "))


def make_rgbdt_and_pairs(data_root: Path, missing: set[str],
                         sources: Sources | None = None) -> tuple[list[dict], list[dict]]:
    sources = sources or source_paths(data_root)
    inventory = read_jsonl(sources.rgbdt_inventory)
    pairs = read_jsonl(sources.pairs)
    merged: dict[tuple[str, str, str], dict] = {}
    revisions: list[dict] = []
    for row in inventory:
        assert row["split"] == "train"
        images = [relative_image(Path(p), data_root) for p in row["images"]]
        merged[rgbdt_key(row)] = sample(
            sample_id=row["id"], source="rgbdt", group_id=f"rgbdt:{row['scene_id']}",
            query=row["query"], images=images, modalities=row["modalities"],
            depth_policy=row["depth_policy"], bbox=row["bbox"],
            label_source=row["label_source"], scene_id=str(row["scene_id"]),
            image_id=str(row["image_id"]), source_ids=[row["id"]], category="regular",
        )
    for row in pairs:
        if row["split"] != "train":
            continue
        if row["review_status"] != "accepted" or row["choice"] not in {"A", "B"}:
            raise ValueError(f"{row['id']}: pair has no accepted A/B judgment")
        answer = 0 if row["choice"] == "A" else 1
        positive = box(row["candidates"][answer])
        target = box(row["bbox"])
        if positive != target:
            raise ValueError(f"{row['id']}: accepted candidate disagrees with GT")
        negative = box(row["candidates"][1 - answer])
        if iou(target, negative) >= 0.9:
            raise ValueError(f"{row['id']}: negative nearly equals GT")
        images = [relative_image(sources.bundle / p, data_root) for p in row["images"]]
        source = "rgbdt" if row["source"] == "rgbdt500" else "robo"
        group_id = (f"rgbdt:{row['scene_id']}" if source == "rgbdt"
                    else f"robo:{row['scene_id']}")
        new = sample(
            sample_id=row["id"], source=source, group_id=group_id,
            query=row["query"], images=images, modalities=row["modalities"],
            depth_policy=row["depth_policy"], bbox=target,
            negative_boxes=[negative], negative_kind=[row["category"]],
            label_source="reviewed_pair", scene_id=str(row["scene_id"]),
            image_id=str(row["image_id"]), source_ids=[row["id"]], category=row["category"],
        )
        if source == "robo":
            # Robo pair queries already occur in the 2,000-row primary manifest.
            # Merge them in make_robo, keyed by image and Query.
            continue
        key = rgbdt_key(row)
        old = merged.get(key)
        if old is None:
            merged[key] = new
            continue
        if old["solution"] != target:
            revisions.append({"id": old["id"], "review_id": row["id"],
                              "old_box": old["solution"], "reviewed_box": target})
        if old["label_source"] == "reviewed_pair" and old["solution"] != target:
            raise ValueError(f"{row['id']}: conflicting reviewed GT for one Query")
        old["solution"] = target
        old["label_source"] = "reviewed_pair"
        old["images"] = images
        old["source_ids"].append(row["id"])
        old["category"] = row["category"] if old["category"] == "regular" else old["category"]
        if negative not in old["negative_boxes"]:
            old["negative_boxes"].append(negative)
            old["negative_kind"].append(row["category"])
    return list(merged.values()), revisions


def make_robo(data_root: Path, missing: set[str], sources: Sources | None = None) -> list[dict]:
    sources = sources or source_paths(data_root)
    primary = read_jsonl(sources.robo)
    pairs = [r for r in read_jsonl(sources.pairs)
             if r["split"] == "train" and r["source"] == "roborefit"]
    merged = {}
    for row in primary:
        images = [relative_image((sources.robo_media_dir / row[k]), data_root) for k in ("rgb", "depth")]
        key = (row["scene_id"], row["query"].strip().casefold().rstrip("?. "))
        merged[key] = sample(
            sample_id=row["id"], source="robo", group_id=f"robo:{row['scene_id']}",
            query=row["query"], images=images, modalities=["rgb", "depth"],
            depth_policy="visual", bbox=row["bbox"], label_source="roborefit_source",
            scene_id=row["scene_id"], image_id=Path(row["rgb"]).stem,
            source_ids=[row["id"]], category="regular",
            depth_visualization=row["depth_visualization"], depth_unit=row["depth_unit"],
        )
    for row in pairs:
        key = (row["scene_id"], row["query"].strip().casefold().rstrip("?. "))
        if row["review_status"] != "accepted" or row["choice"] not in {"A", "B"}:
            raise ValueError(f"{row['id']}: pair has no accepted A/B judgment")
        answer = 0 if row["choice"] == "A" else 1
        target = box(row["bbox"])
        if box(row["candidates"][answer]) != target:
            raise ValueError(f"{row['id']}: accepted candidate disagrees with GT")
        negative = box(row["candidates"][1 - answer])
        if iou(target, negative) >= 0.9:
            raise ValueError(f"{row['id']}: negative nearly equals GT")
        images = [relative_image(sources.bundle / p, data_root) for p in row["images"]]
        if key not in merged:
            merged[key] = sample(
                sample_id=row["id"], source="robo", group_id=f"robo:{row['scene_id']}",
                query=row["query"], images=images, modalities=row["modalities"],
                depth_policy="visual", bbox=target, label_source="reviewed_pair",
                negative_boxes=[negative], negative_kind=[row["category"]],
                scene_id=row["scene_id"], image_id=row["image_id"],
                source_ids=[row["id"]], category=row["category"],
                depth_visualization="fixed_raw_p01_p99_v1", depth_unit="unknown",
            )
            continue
        old = merged[key]
        if iou(old["solution"], target) < 0.8:
            raise ValueError(f"{row['id']}: reviewed Robo GT conflicts with primary GT")
        old["solution"] = target
        old["images"] = images
        old["label_source"] = "reviewed_pair"
        old["negative_boxes"].append(negative)
        old["negative_kind"].append(row["category"])
        old["source_ids"].append(row["id"])
        old["category"] = row["category"]
    return list(merged.values())


def city_frame(row: dict) -> str:
    return Path(row["visible"].replace("\\", "/")).stem


def city_image_paths(row: dict, city_root: Path, output: Path,
                     data_root: Path, missing: set[str], rendered: dict[str, str]) -> list[str]:
    source_dir = city_root / "target_v2"
    visible = (source_dir / row["visible"]).resolve()
    infrared = (source_dir / row["infrared"]).resolve()
    depth = (source_dir / row["depth"]).resolve()
    rgb = relative_image(visible, data_root)
    ir = relative_image(infrared, data_root)
    if depth.is_file() and str(depth) not in rendered:
        destination = output / "depth_rgb" / f"{city_frame(row)}.png"
        render_depth(depth, destination)
        rendered[str(depth)] = relative_image(destination, data_root)
    depth_rel = (rendered[str(depth)] if str(depth) in rendered else
                 (output / "depth_rgb" / f"{city_frame(row)}.png").relative_to(data_root).as_posix())
    return [rgb, ir, depth_rel]


def make_city(city_root: Path, data_root: Path, output: Path,
              official_exclusions: Path, missing: set[str],
              sources: Sources | None = None) -> tuple[list[dict], list[dict], dict]:
    sources = sources or source_paths(data_root)
    exclusion = json.loads(official_exclusions.read_text(encoding="utf-8"))
    if "excluded_frame_keys" not in exclusion or not exclusion["excluded_frame_keys"]:
        raise ValueError("official City exclusion JSON needs nonempty excluded_frame_keys")
    test_frames = set(exclusion["excluded_frame_keys"])
    with zipfile.ZipFile(sources.city_zip) as archive:
        train = json.loads(archive.read("qwen_generation_train_100.json"))
    all_val = json.loads(sources.city_dev.read_text(encoding="utf-8"))
    correction = json.loads(sources.city_corrections.read_text(encoding="utf-8"))
    deferred = {r["id"] for r in correction["cases"] if r["status"] in {"deferred", "partial"}}
    reliable_dev = {key: value for key, value in all_val.items() if key not in deferred}
    if len(reliable_dev) != 400:
        raise ValueError(f"expected reliable City400 development source, got {len(reliable_dev)}")
    dev = {key: row for key, row in reliable_dev.items() if city_frame(row) not in test_frames}
    dev_frames = {city_frame(row) for row in all_val.values()}
    rendered: dict[str, str] = {}
    dev_rows = [sample(sample_id=key, source="city", group_id=f"city:{city_frame(row)}",
                       query=row["query"], images=city_image_paths(row, city_root, output, data_root, missing, rendered),
                       modalities=["rgb", "infrared", "depth"], depth_policy="millimeter",
                       bbox=row["bbox"], label_source="city_dev_corrected", category="dev")
                for key, row in dev.items()]
    rows = []
    removed = Counter({"official_test_dev": len(reliable_dev) - len(dev)})
    for key, row in train.items():
        frame = city_frame(row)
        if frame in test_frames:
            removed["official_test"] += 1
            continue
        if frame in dev_frames:
            removed["city_dev"] += 1
            continue
        rows.append(sample(sample_id=key, source="city", group_id=f"city:{frame}",
                           query=row["query"], images=city_image_paths(row, city_root, output, data_root, missing, rendered),
                           modalities=["rgb", "infrared", "depth"], depth_policy="millimeter",
                           bbox=row["bbox"], label_source="city_original_float", category="regular"))
    return rows, dev_rows, dict(removed)


def make_external_dev(data_root: Path, missing: set[str],
                      sources: Sources | None = None) -> tuple[list[dict], set[str]]:
    sources = sources or source_paths(data_root)
    rows = read_jsonl(sources.external_dev)
    accepted = [r for r in rows if r["status"] in {"accepted", "corrected"}]
    if len(accepted) != 94:
        raise ValueError(f"expected reviewed external94, got {len(accepted)}")
    result = [sample(sample_id=r["id"], source="rgbdt", group_id=f"rgbdt:{r['scene_id']}",
                     query=r["final_query"], images=[relative_image(Path(p), data_root) for p in r["images"]],
                     modalities=r["modalities"], depth_policy=r["depth_policy"],
                     bbox=r["final_bbox"], label_source="external_reviewed", category="dev")
              for r in accepted]
    aux_eval = [r for r in read_jsonl(sources.pairs) if r["split"] == "eval"]
    excluded = {r["group_id"] for r in result} | {f"rgbdt:{r['scene_id']}" for r in aux_eval}
    return result, excluded


def holdout_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    by_source: dict[str, list[str]] = defaultdict(list)
    for key, members in groups.items():
        by_source[members[0]["source"]].append(key)
    held = set()
    for source, keys in sorted(by_source.items()):
        keys.sort()
        random.Random(seed + {"city": 11, "rgbdt": 23, "robo": 37}[source]).shuffle(keys)
        held.update(keys[:max(1, round(len(keys) * 0.1))])
    train = [r for r in rows if r["group_id"] not in held]
    holdout = [r for r in rows if r["group_id"] in held]
    return train, holdout, {s: len(set(keys) & held) for s, keys in by_source.items()}


def balanced_pool(rows: list[dict], seed: int):
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if not groups:
        raise ValueError("required sampling pool is empty")
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    queues = {key: deque() for key in keys}
    while True:
        for key in keys:
            if not queues[key]:
                members = groups[key].copy()
                rng.shuffle(members)
                queues[key].extend(members)
            yield queues[key].popleft()
        rng.shuffle(keys)


def sampling_pools(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "regular_city": [r for r in rows if r["source"] == "city"],
        "regular_rgbdt": [r for r in rows if r["source"] == "rgbdt"],
        "regular_robo": [r for r in rows if r["source"] == "robo"],
        "wrong_object": [r for r in rows if "wrong_object" in r["negative_kind"]],
        "extent": [r for r in rows if "extent" in r["negative_kind"]],
    }


def schedule(rows: list[dict], presentations: int, seed: int, local_pool: bool) -> list[dict]:
    if presentations <= 0 or presentations % (4 if local_pool else 20):
        raise ValueError("presentations must be a positive multiple of 4 (local) or 20 (formal)")
    if local_pool:
        quotas = {"regular_rgbdt": presentations // 4, "regular_robo": presentations // 4,
                  "wrong_object": presentations // 4, "extent": presentations // 4}
    else:
        quotas = {"regular_city": presentations * 2 // 5,
                  "regular_rgbdt": presentations // 20, "regular_robo": presentations // 20,
                  "wrong_object": presentations // 4, "extent": presentations // 4}
    pools = sampling_pools(rows)
    streams = {}
    for index, (name, count) in enumerate(quotas.items()):
        if count and not pools[name]:
            raise ValueError(f"required sampling pool is empty: {name}")
        streams[name] = balanced_pool(pools[name], seed + 1009 * (index + 1))
    tickets = [name for name, count in quotas.items() for _ in range(count)]
    random.Random(seed).shuffle(tickets)
    return [{**next(streams[name]), "occurrence_id": f"presentation:{i:06d}",
             "sampling_pool": name} for i, name in enumerate(tickets)]


def coverage(rows: list[dict], presentations: list[dict], local_pool: bool) -> dict:
    by_source = {}
    for source in sorted({r["source"] for r in rows}):
        eligible = [r for r in rows if r["source"] == source]
        shown = [r for r in presentations if r["source"] == source]
        eligible_ids = {r["id"] for r in eligible}
        shown_ids = {r["id"] for r in shown}
        eligible_groups = {r["group_id"] for r in eligible}
        shown_groups = {r["group_id"] for r in shown}
        by_source[source] = {
            "eligible_unique_queries": len(eligible_ids),
            "presented_unique_queries": len(shown_ids),
            "unseen_queries": len(eligible_ids - shown_ids),
            "eligible_groups": len(eligible_groups),
            "presented_groups": len(shown_groups),
            "all_groups_covered": eligible_groups <= shown_groups,
            "presentations": len(shown),
            "repeat_exposures": len(shown) - len(shown_ids),
        }
    pools = sampling_pools(rows)
    names = (("regular_rgbdt", "regular_robo", "wrong_object", "extent") if local_pool
             else ("regular_city", "regular_rgbdt", "regular_robo", "wrong_object", "extent"))
    by_pool = {}
    for name in names:
        eligible = pools[name]
        shown = [r for r in presentations if r["sampling_pool"] == name]
        eligible_ids = {r["id"] for r in eligible}
        shown_ids = {r["id"] for r in shown}
        eligible_groups = {r["group_id"] for r in eligible}
        shown_groups = {r["group_id"] for r in shown}
        by_pool[name] = {
            "eligible_unique_queries": len(eligible_ids),
            "presented_unique_queries": len(shown_ids),
            "unseen_queries": len(eligible_ids - shown_ids),
            "eligible_groups": len(eligible_groups),
            "presented_groups": len(shown_groups),
            "all_groups_covered": eligible_groups <= shown_groups,
            "presentations": len(shown),
            "repeat_exposures": len(shown) - len(shown_ids),
        }
    return {"by_source": by_source, "by_pool": by_pool,
            "all_source_groups_covered": all(item["all_groups_covered"] for item in by_source.values())}


def assert_cpu_only() -> None:
    """This data builder must not inherit an already initialized CUDA context."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("grounding data preparation is CPU-only; CUDA was already initialized")


def filter_excluded_groups(rows: list[dict], excluded_group_ids: set[str]) -> tuple[list[dict], int]:
    kept = [row for row in rows if row["group_id"] not in excluded_group_ids]
    return kept, len(rows) - len(kept)


def final_missing_media(rows: list[dict], data_root: Path,
                        city_media_root: Path | None = None) -> set[str]:
    missing = set()
    for row in rows:
        for image in row["images"]:
            path = data_root / image
            if not path.is_file():
                missing.add(str(path))
        if row["source"] == "city":
            frame = row["group_id"].split(":", 1)[1]
            raw_depth = city_media_root / "depth" / f"{frame}.png"
            if not raw_depth.is_file():
                missing.add(str(raw_depth))
    return missing


def build(*, output: Path, data_root: Path = DATA, city_media_root: Path | None = None,
          city_test_exclusions: Path | None = None, local_pool: bool = False,
          presentations: int | None = None, seed: int = 2026,
          source_dir: Path | None = None,
          official_test_excluded_groups: Path | None = None) -> dict:
    assert_cpu_only()
    if presentations is None:
        presentations = 1920 if local_pool else 19200
    if local_pool and (city_media_root is not None or city_test_exclusions is not None):
        raise ValueError("--local-pool uses RGBDT/Robo only; omit City inputs")
    if not local_pool and (city_media_root is None or city_test_exclusions is None):
        raise ValueError("formal build requires --city-media-root and --city-test-exclusions")
    output = output.resolve()
    data_root = data_root.resolve()
    output.relative_to(data_root)
    output.mkdir(parents=True, exist_ok=True)
    sources = source_paths(data_root, source_dir)
    missing: set[str] = set()
    rgbdt, revisions = make_rgbdt_and_pairs(data_root, missing, sources)
    robo = make_robo(data_root, missing, sources)
    external_dev, excluded_rgbdt = make_external_dev(data_root, missing, sources)
    external_source_reviewed = len(external_dev)
    # All 94 original development groups remain forbidden in training, including
    # groups removed from the published development index below.
    rgbdt = [r for r in rgbdt if r["group_id"] not in excluded_rgbdt]
    city, city_dev, removed_city = (make_city(city_media_root, data_root, output, city_test_exclusions, missing, sources)
                                    if city_media_root is not None and city_test_exclusions is not None
                                    else ([], [], {}))
    city_source_reliable = len(city_dev) + removed_city.get("official_test_dev", 0)
    excluded_group_ids = (set(json.loads(official_test_excluded_groups.read_text(encoding="utf-8"))["excluded_group_ids"])
                          if official_test_excluded_groups is not None else set())
    city, removed_city_train = filter_excluded_groups(city, excluded_group_ids)
    rgbdt, removed_rgbdt_train = filter_excluded_groups(rgbdt, excluded_group_ids)
    robo, removed_robo_train = filter_excluded_groups(robo, excluded_group_ids)
    city_dev, removed_city_dev = filter_excluded_groups(city_dev, excluded_group_ids)
    external_dev, removed_external_dev = filter_excluded_groups(external_dev, excluded_group_ids)
    group_removed_rows = {
        "pre_holdout_train": {"city": removed_city_train, "rgbdt": removed_rgbdt_train,
                              "robo": removed_robo_train},
        "city400_dev": {"city": removed_city_dev},
        "external94_dev": {"rgbdt": removed_external_dev},
    }
    missing.update(final_missing_media(city + rgbdt + robo + city_dev + external_dev,
                                       data_root, city_media_root))
    (output / "missing_media.json").write_text(
        json.dumps(sorted(missing), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if missing:
        raise FileNotFoundError(f"{len(missing)} media files missing; see {output / 'missing_media.json'}")
    train, holdout, held_groups = holdout_split(city + rgbdt + robo, seed)
    presentations_rows = schedule(train, presentations, seed, local_pool)
    diagnostic_by_id = {}
    for row in presentations_rows:
        diagnostic_by_id.setdefault(row["id"], row)
    diagnostic = list(diagnostic_by_id.values())[:128]
    mode = "nonformal_local_pool" if local_pool else "formal"
    for name, rows in (("train_index", train), ("holdout", holdout),
                       ("schedule", presentations_rows), ("diagnostic128", diagnostic),
                       ("external94_dev", external_dev),
                       ("city400_dev", city_dev)):
        write_jsonl(output / f"{name}.jsonl", [{**row, "build_mode": mode} for row in rows])
    report = {
        "mode": mode,
        "seed": seed, "data_root": str(data_root),
        "source_rows_before_holdout": dict(Counter(r["source"] for r in city + rgbdt + robo)),
        "train_rows": dict(Counter(r["source"] for r in train)),
        "holdout_rows": dict(Counter(r["source"] for r in holdout)),
        "holdout_groups": held_groups,
        "presentation_counts": dict(Counter(r["sampling_pool"] for r in presentations_rows)),
        "unique_presented_ids": len({r["id"] for r in presentations_rows}),
        "unique_presented_groups": len({r["group_id"] for r in presentations_rows}),
        "diagnostic_queries": len(diagnostic),
        "schedule_coverage": coverage(train, presentations_rows, local_pool),
        "arm_schedule_shared": True,
        "cpu_only": True,
        "missing_media_count": 0,
        "reviewed_revisions": revisions,
        "city_excluded": removed_city,
        "excluded_group_ids": sorted(excluded_group_ids),
        "official_test_group_removed_rows": group_removed_rows,
        "dev_rows": {"city_source_reliable": city_source_reliable,
                     "city_official_safe": len(city_dev),
                     "external_source_reviewed": external_source_reviewed,
                     "external_actual": len(external_dev)},
        "city_dev_groups": len({r["group_id"] for r in city_dev}),
        "grouping_limit": "Robo physical-scene IDs unavailable; isolated by image. City isolated by image frame; RGBDT by known sequence.",
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--city-media-root", type=Path)
    parser.add_argument("--city-test-exclusions", type=Path)
    parser.add_argument("--official-test-excluded-groups", type=Path,
                        help="JSON with excluded_group_ids from exact official-test media audit")
    parser.add_argument("--source-dir", type=Path,
                        help="portable source annotation directory; media stays under --data-root")
    parser.add_argument("--local-pool", action="store_true")
    parser.add_argument("--presentations", type=int,
                        help="default: 19200 formal, 1920 nonformal local pool")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(build(output=args.output, data_root=args.data_root,
                           city_media_root=args.city_media_root,
                           city_test_exclusions=args.city_test_exclusions,
                           local_pool=args.local_pool, source_dir=args.source_dir,
                           official_test_excluded_groups=args.official_test_excluded_groups,
                           presentations=args.presentations, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
