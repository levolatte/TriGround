"""Merge real grounding sources into scene-disjoint ms-swift SFT manifests.

Example (all paths remain local until copied to the training host)::

    python tools/prepare_semifinal_sft.py \
      --source roborefit=../data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl \
      --val-ratio 0.1 --output-dir ../tmp/structural_20261023/sft_data

Inputs may be normalized query/bbox manifests, native conversations, or swift
messages. --weight controls source-pool presentation counts, not unique samples.
Exclusions accept source+scene_id, source+id, or exact RGB paths from manifests.
Use explicit source+scene_id exclusions when assets were relocated; filenames
alone are never treated as global scene identities.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_pretrained_grounder import extract_query, parse_generated_bbox
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000, native_prompt, render_depth


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [dict(row, id=row.get("id", key)) for key, row in payload.items()]
    return payload


def assignments(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"expected a unique NAME=VALUE: {value}")
        result[name] = item
    return result


def resolve_image(value: str, root: Path) -> str:
    path = Path(value.replace("\\", "/"))
    return str((root / path).resolve())


def normalize_record(row: dict, pool: str, root: Path, depth_dir: Path,
                     media_modes: dict, rendered: dict,
                     modalities: tuple[str, ...] | None = None,
                     depth_policy: str | None = None) -> dict:
    if row.get("split", "train").lower() not in {"train", "training"}:
        raise ValueError(f"{row.get('id')}: only training splits may enter a source pool")
    source = str(row.get("source") or pool)
    policy = depth_policy or row.get("depth_policy", "visual")
    if "messages" in row or "conversations" in row:
        if "messages" in row:
            messages = [dict(message) for message in row["messages"]]
        else:
            roles = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = [{"role": roles[m["from"]], "content": m["value"]}
                        for m in row["conversations"]]
        images = row.get("images", row.get("image"))
        if isinstance(images, str):
            images = [images]
        images = [resolve_image(path, root) for path in images]
        user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
        query = row.get("query") or extract_query(user_text)
        views = tuple(row.get("modalities") or modalities or ())
        if not views:
            if len(images) == 1:
                views = ("rgb",)
            elif len(images) == 3:
                views = ("rgb", "infrared", "depth")
            elif "RGB, infrared" in user_text:
                views = ("rgb", "infrared")
            elif "RGB, depth" in user_text:
                views = ("rgb", "depth")
            else:
                raise ValueError(f"{row['id']}: supply --modalities {pool}=rgb,depth or rgb,infrared")
        if messages[-1]["role"] != "assistant":
            raise ValueError(f"{row['id']}: SFT requires a final assistant target")
        bbox = parse_generated_bbox(messages[-1]["content"])
        if bbox is None:
            raise ValueError(f"{row['id']}: no valid final 0--1000 bbox target")
    else:
        query = row["query"]
        paths = {"rgb": row.get("rgb", row.get("visible"))}
        for view in ("infrared", "depth"):
            if row.get(view):
                paths[view] = row[view]
        if row.get("aux"):
            paths[{"ir": "infrared", "depth": "depth"}[row["aux_type"]]] = row["aux"]
        views = tuple(view for view in ("rgb", "infrared", "depth") if view in paths)
        images = [resolve_image(paths[view], root) for view in views]
        if "depth" in views and policy in {"millimeter", "sensor_linear_20000"}:
            index = views.index("depth")
            original = images[index]
            if original not in rendered:
                destination = depth_dir / f"{len(rendered):08d}.png"
                render_depth(Path(original), destination)
                rendered[original] = str(destination.resolve())
            images[index] = rendered[original]
        bbox = row["bbox"]
        messages = [
            {"role": "user", "content": native_prompt(query, views, policy)},
            {"role": "assistant", "content": json.dumps(
                {"bbox_2d": bbox_to_qwen1000(bbox)}, separators=(",", ":"))},
        ]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{row['id']}: empty query")
    if not views or views[0] != "rgb" or len(set(views)) != len(views):
        raise ValueError(f"{row['id']}: expected unique views starting with RGB")
    if any(view not in {"rgb", "infrared", "depth"} for view in views):
        raise ValueError(f"{row['id']}: unknown modalities {views}")
    placeholders = sum(m["content"].count("<image>") for m in messages)
    if placeholders != len(images) or len(images) != len(views):
        raise ValueError(f"{row['id']}: image placeholders, paths and modalities disagree")
    for path in images:
        if path not in media_modes:
            with Image.open(path) as image:
                media_modes[path] = image.mode
        if media_modes[path] in {"I", "I;16", "I;16B", "I;16L", "F"}:
            raise ValueError(f"{path}: raw depth needs an explicit rendering policy before SFT")
    result = {
        "messages": messages, "images": images, "id": str(row["id"]),
        "source": source, "scene_id": str(row.get("scene_id") or images[0]),
        "query": query, "modalities": list(views), "bbox": bbox,
    }
    for key in ("class_name", "origin_query_id", "original_image_id", "depth_unit",
                "depth_visualization", "capability", "task_category"):
        if key in row:
            result[key] = row[key]
    if "depth" in views:
        result["depth_policy"] = policy
    return result


def exclusion_keys(paths: list[Path], root: Path | None = None) -> tuple[set, set, set]:
    scenes, ids, images = set(), set(), set()
    for path in paths:
        for row in read_records(path):
            if row.get("source") and row.get("scene_id"):
                scenes.add((str(row["source"]), str(row["scene_id"])))
            if row.get("source") and row.get("id"):
                ids.add((str(row["source"]), str(row["id"])))
            rgb = row.get("rgb", row.get("visible"))
            if rgb is None:
                rgb = row.get("images", row.get("image"))
                if isinstance(rgb, list):
                    rgb = rgb[0]
            if rgb:
                images.add(resolve_image(rgb, root or path.parent))
            elif not row.get("source") or not (row.get("scene_id") or row.get("id")):
                raise ValueError(f"{path}: exclusion needs source+scene_id/id or an RGB path")
    return scenes, ids, images


def build(sources: dict[str, Path], output: Path, *, weights: dict[str, float] | None = None,
          image_roots: dict[str, Path] | None = None, depth_policies: dict[str, str] | None = None,
          modalities: dict[str, tuple[str, ...]] | None = None, val_ratio: float = 0.1,
          train_samples: int | None = None, seed: int = 2026,
          exclude: list[Path] | None = None, exclude_root: Path | None = None) -> dict:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    for settings in (weights, image_roots, depth_policies, modalities):
        if settings and set(settings) - set(sources):
            raise ValueError(f"unknown source names: {set(settings) - set(sources)}")
    pools, media_modes, rendered = {}, {}, {}
    excluded_scenes, excluded_ids, excluded_images = exclusion_keys(exclude or [], exclude_root)
    for name, path in sources.items():
        root = (image_roots or {}).get(name, path.parent)
        pools[name] = [normalize_record(
            row, name, root, output / "depth_visual", media_modes, rendered,
            (modalities or {}).get(name), (depth_policies or {}).get(name))
            for row in read_records(path)]
    for rows in pools.values():
        for row in rows:
            if (row["source"], row["id"]) in excluded_ids or row["images"][0] in excluded_images:
                excluded_scenes.add((row["source"], row["scene_id"]))
    counts_before = {name: len(rows) for name, rows in pools.items()}
    pools = {name: [row for row in rows
                    if (row["source"], row["scene_id"]) not in excluded_scenes]
             for name, rows in pools.items()}
    grouped = defaultdict(set)
    for rows in pools.values():
        for row in rows:
            grouped[row["source"]].add(row["scene_id"])
    rng = random.Random(seed)
    val_scenes = set()
    for source in sorted(grouped):
        scenes = sorted(grouped[source])
        rng.shuffle(scenes)
        val_scenes.update((source, scene) for scene in scenes[:round(len(scenes) * val_ratio)])
    train_pools, val = {}, []
    for name, rows in pools.items():
        train_pools[name] = []
        for row in rows:
            is_val = (row["source"], row["scene_id"]) in val_scenes
            (val if is_val else train_pools[name]).append(dict(row, split="val" if is_val else "train"))
    if train_samples is None and not weights:
        quotas = {name: len(rows) for name, rows in train_pools.items()}
    else:
        total = train_samples if train_samples is not None else sum(map(len, train_pools.values()))
        if total <= 0:
            raise ValueError("train_samples must be positive")
        w = {name: (weights or {}).get(name, 1.0) for name in sources}
        if any(not math.isfinite(x) or x < 0 for x in w.values()) or sum(w.values()) <= 0:
            raise ValueError("weights must be finite, nonnegative and have positive sum")
        raw = {name: total * value / sum(w.values()) for name, value in w.items()}
        quotas = {name: math.floor(value) for name, value in raw.items()}
        for name in sorted(raw, key=lambda name: (-(raw[name] - quotas[name]), name))[:total - sum(quotas.values())]:
            quotas[name] += 1
    train = []
    for name, rows in train_pools.items():
        remaining = quotas[name]
        if remaining and not rows:
            raise ValueError(f"{name}: no training rows remain after exclusions and scene split")
        while remaining:
            cycle = rows.copy()
            rng.shuffle(cycle)
            take = min(remaining, len(cycle))
            train.extend(cycle[:take])
            remaining -= take
    rng.shuffle(train)
    if not train:
        raise ValueError("no training rows")
    report = {"seed": seed, "val_ratio": val_ratio, "source_pools": {},
              "train_presentations": len(train), "val_rows": len(val),
              "train_unique_ids": len({(r["source"], r["id"]) for r in train}),
              "train_scenes": len({(r["source"], r["scene_id"]) for r in train}),
              "val_scenes": len(val_scenes), "rendered_depth_images": len(rendered),
              "train_modalities": dict(Counter("+".join(r["modalities"]) for r in train)),
              "path_policy": "absolute local paths; rebase before copying to another host"}
    for name in sources:
        report["source_pools"][name] = {
            "input": str(sources[name]), "input_rows": counts_before[name],
            "excluded_rows": counts_before[name] - len(pools[name]),
            "train_pool_rows": len(train_pools[name]), "train_presentations": quotas[name],
            "val_rows": len(pools[name]) - len(train_pools[name]),
        }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if len({row["id"] for row in val}) != len(val):
        raise ValueError("val_gt.json requires unique query IDs across sources; namespace input IDs")
    (output / "val_gt.json").write_text(
        json.dumps({row["id"]: row["bbox"] for row in val}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--weight", action="append", default=[], metavar="NAME=WEIGHT")
    parser.add_argument("--image-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--depth-policy", action="append", default=[], metavar="NAME=POLICY")
    parser.add_argument("--modalities", action="append", default=[], metavar="NAME=rgb,depth")
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--exclude-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = build(
        {name: Path(path).resolve() for name, path in assignments(args.source).items()},
        args.output_dir.resolve(), weights={n: float(w) for n, w in assignments(args.weight).items()},
        image_roots={n: Path(p).resolve() for n, p in assignments(args.image_root).items()},
        depth_policies=assignments(args.depth_policy),
        modalities={n: tuple(m.split(",")) for n, m in assignments(args.modalities).items()},
        val_ratio=args.val_ratio, train_samples=args.train_samples, seed=args.seed,
        exclude=args.exclude, exclude_root=args.exclude_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
