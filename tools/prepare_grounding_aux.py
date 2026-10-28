"""Build paired bbox/aux SFT schedules from human-reviewed candidate pairs.

Each of 480 pair presentations contributes four consecutive slots. With an
eight-row training batch, two presentations form one paired A/B step. Keep
DataLoader shuffling disabled for the intended 240-step comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000, native_prompt


POOLS = (("rgbdt500", "wrong_object"), ("roborefit", "wrong_object"),
         ("rgbdt500", "extent"))
CHOICES = {"A", "B", "both", "neither", "unknown"}
MODALITIES = {"rgb", "infrared", "depth"}
DEPTH_POLICIES = {"visual", "millimeter", "sensor_linear_20000"}
REQUIRED = {"id", "source", "scene_id", "image_id", "query", "images",
            "modalities", "depth_policy", "bbox", "candidates", "choice",
            "reason", "category", "split", "evidence", "review_status",
            "reviewer", "review_note"}
PRESENTATIONS_PER_POOL = 160
MAX_PAIRS_PER_POOL = 80


def read_pairs(paths: list[Path], image_root: Path | None = None) -> list[dict]:
    """Validate the external annotation boundary once and resolve RGB media."""
    rows = []
    ids = set()
    modes = {}
    for manifest in paths:
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{manifest}:{line_number}: missing fields {sorted(missing)}")
            key = (row["source"], str(row["id"]))
            if key in ids:
                raise ValueError(f"duplicate pair id: {key}")
            ids.add(key)
            if (row["source"], row["category"]) not in POOLS:
                raise ValueError(f"{key}: unsupported source/category")
            if row["split"] not in {"train", "eval"}:
                raise ValueError(f"{key}: split must be train or eval")
            if row["review_status"] != "accepted" or not row["reviewer"]:
                raise ValueError(f"{key}: only accepted, reviewed pairs may enter")
            if not all(isinstance(row[field], str) and row[field].strip()
                       for field in ("id", "scene_id", "image_id", "query", "reason")):
                raise ValueError(f"{key}: id, scene_id, image_id, query, and reason must be nonempty")
            if row["choice"] not in CHOICES:
                raise ValueError(f"{key}: invalid choice {row['choice']!r}")
            views = row["modalities"]
            images = row["images"]
            if (not isinstance(views, list) or not views or views[0] != "rgb"
                    or len(set(views)) != len(views) or set(views) - MODALITIES
                    or not isinstance(images, list) or len(images) != len(views)):
                raise ValueError(f"{key}: images/modalities must align and start with RGB")
            expected_views = ["rgb", "infrared", "depth"] if row["source"] == "rgbdt500" else ["rgb", "depth"]
            if views != expected_views:
                raise ValueError(f"{key}: source requires modalities {expected_views}")
            if row["depth_policy"] not in DEPTH_POLICIES:
                raise ValueError(f"{key}: unsupported depth policy")
            bbox_to_qwen1000(row["bbox"])
            if not isinstance(row["candidates"], list) or len(row["candidates"]) != 2:
                raise ValueError(f"{key}: expected two candidate boxes")
            for candidate in row["candidates"]:
                bbox_to_qwen1000(candidate)
            resolved = []
            for value in images:
                path = Path(value)
                if not path.is_absolute():
                    if image_root is None:
                        raise ValueError(f"{key}: relative image requires --image-root")
                    path = image_root / path
                path = path.resolve()
                if path not in modes:
                    with Image.open(path) as image:
                        modes[path] = image.mode
                if modes[path] != "RGB":
                    raise ValueError(f"{key}: media must be 8-bit RGB, got {modes[path]}: {path}")
                resolved.append(str(path))
            rows.append(dict(row, images=resolved))
    if not rows:
        raise ValueError("no reviewed pairs")
    train_scenes = {(r["source"], str(r["scene_id"])) for r in rows if r["split"] == "train"}
    eval_scenes = {(r["source"], str(r["scene_id"])) for r in rows if r["split"] == "eval"}
    overlap = train_scenes & eval_scenes
    if overlap:
        raise ValueError(f"train/eval scene overlap: {sorted(overlap)[:5]}")
    return rows


def aux_prompt(query: str, modalities: tuple[str, ...], depth_policy: str,
               candidates: list[list[int]]) -> str:
    """Ask for a short candidate judgment without inserting annotation metadata."""
    names = {"rgb": "RGB", "infrared": "infrared", "depth": "depth"}
    prefix = "\n".join("<image>" for _ in modalities)
    if modalities == ("rgb",):
        views = "The image is the RGB view."
    else:
        views = "These are aligned views of the same scene in this order: "
        views += ", ".join(names[view] for view in modalities) + "."
    if "depth" in modalities and depth_policy == "millimeter":
        views += " Brighter valid depth pixels are nearer; black pixels are invalid."
    elif "depth" in modalities and depth_policy == "sensor_linear_20000":
        views += (" This depth visualization uses a fixed inverse mapping of raw sensor values:"
                  " smaller positive stored values are brighter and zero is black."
                  " Its physical distance unit is not established.")
    return (
        f"{prefix}\n{views} For the query below, judge whether each candidate box"
        " tightly locates the complete visible target object. Coordinates are"
        " 0-1000 xyxy in the RGB view.\n"
        f"Query: {query}\nA: {candidates[0]}\nB: {candidates[1]}\n"
        "Return only short JSON with choice and reason. Choose A or B if one box"
        " is correct, both if both are correct, neither if both are clearly wrong,"
        " and unknown if the images cannot establish the answer."
    )


def _pair_id(row: dict) -> str:
    return f"{row['source']}:{row['id']}"


def _swapped_choice(choice: str, swap: bool) -> str:
    if swap and choice in {"A", "B"}:
        return "B" if choice == "A" else "A"
    return choice


def _sample(row: dict, slot_id: str, channel: str, swap: bool = False) -> dict:
    modalities = tuple(row["modalities"])
    if channel == "box":
        prompt = native_prompt(row["query"], modalities, row["depth_policy"])
        answer = {"bbox_2d": bbox_to_qwen1000(row["bbox"])}
    else:
        candidates = [bbox_to_qwen1000(candidate) for candidate in row["candidates"]]
        if swap:
            candidates.reverse()
        prompt = aux_prompt(row["query"], modalities, row["depth_policy"], candidates)
        answer = {"choice": _swapped_choice(row["choice"], swap), "reason": row["reason"]}
    return {
        "id": f"{slot_id}:{channel}", "slot_id": slot_id, "pair_id": _pair_id(row),
        "query": row["query"], "channel": channel, "category": row["category"],
        "images": row["images"],
        "messages": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False,
                                                                 separators=(",", ":"))}],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def build(paths: list[Path], output_dir: Path, *, seed: int = 2026,
          image_root: Path | None = None) -> dict:
    rows = read_pairs(paths, image_root)
    rng = random.Random(seed)
    train = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    presentations = []
    pool_report = {}
    for source, category in POOLS:
        pool = sorted((row for row in train if (row["source"], row["category"]) == (source, category)),
                      key=lambda row: _pair_id(row))
        if not pool:
            raise ValueError(f"no accepted training pairs for {source}/{category}")
        rng.shuffle(pool)
        selected = pool[:MAX_PAIRS_PER_POOL]
        chosen = []
        while len(chosen) < PRESENTATIONS_PER_POOL:
            cycle = selected.copy()
            rng.shuffle(cycle)
            chosen.extend(cycle[:PRESENTATIONS_PER_POOL - len(chosen)])
        swaps = [False] * (PRESENTATIONS_PER_POOL // 2) + [True] * (PRESENTATIONS_PER_POOL // 2)
        rng.shuffle(swaps)
        presentations.extend((row, (source, category), swap)
                             for row, swap in zip(chosen, swaps))
        pool_report[f"{source}/{category}"] = {
            "available_reviewed_pairs": len(pool), "selected_distinct_pairs": len(selected),
            "unique_images": len({row["images"][0] for row in selected}),
            "unique_sequences": len({row["scene_id"] for row in selected}),
            "presentations": PRESENTATIONS_PER_POOL,
            "repeated_presentations": PRESENTATIONS_PER_POOL - len(selected),
            "swapped_presentations": sum(swaps),
        }
    rng.shuffle(presentations)
    arm_a, arm_b, audit = [], [], []
    for presentation_id, (row, pool, swap) in enumerate(presentations):
        unit = []
        for position in range(4):
            slot_id = f"p{presentation_id:03d}s{position}"
            a = _sample(row, slot_id, "box")
            channel_b = "aux" if position == 3 else "box"
            b = _sample(row, slot_id, channel_b, swap if channel_b == "aux" else False)
            unit.append((a, b, {
                "slot_id": slot_id, "presentation_id": presentation_id,
                "pair_id": _pair_id(row), "source": row["source"],
                "category": row["category"], "scene_id": row["scene_id"],
                "image_id": row["image_id"], "channel_b": channel_b,
                "swapped": swap if channel_b == "aux" else False,
                "canonical_choice": row["choice"], "presented_choice":
                    _swapped_choice(row["choice"], swap) if channel_b == "aux" else None,
                "evidence": row["evidence"], "reviewer": row["reviewer"],
                "review_note": row["review_note"],
            }))
        for a, b, record in unit:
            arm_a.append(a)
            arm_b.append(b)
            audit.append(record)
    aux_eval = []
    eval_swaps = [False] * ((len(eval_rows) + 1) // 2) + [True] * (len(eval_rows) // 2)
    rng.shuffle(eval_swaps)
    for row, swap in zip(eval_rows, eval_swaps):
        sample = _sample(row, f"eval:{_pair_id(row)}", "aux", swap)
        sample["messages"] = sample["messages"][:1]
        sample["expected"] = {"choice": _swapped_choice(row["choice"], swap), "reason": row["reason"]}
        aux_eval.append(sample)
    summary = {
        "seed": seed, "steps": 240, "batch_size": 8,
        "arm_a": {"rows": len(arm_a), "box": len(arm_a), "aux": 0},
        "arm_b": {"rows": len(arm_b), "box": sum(r["channel"] == "box" for r in arm_b),
                  "aux": sum(r["channel"] == "aux" for r in arm_b)},
        "pools": pool_report, "eval_pairs": len(aux_eval),
        "eval_choices": dict(Counter(row["expected"]["choice"] for row in aux_eval)),
        "ordering": "shuffled four-row units; B=[box,box,box,aux]; two units per eight-row step; DataLoader shuffle=false",
        "train_unique_images": len({row["images"][0] for row, _, _ in presentations}),
        "train_unique_sequences": len({(row["source"], row["scene_id"]) for row, _, _ in presentations}),
        "eval_unique_images": len({row["images"][0] for row in eval_rows}),
        "eval_unique_sequences": len({(row["source"], row["scene_id"]) for row in eval_rows}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("A", arm_a), ("B", arm_b), ("aux_eval", aux_eval),
                       ("paired_slots", audit)):
        _write_jsonl(output_dir / f"{name}.jsonl", data)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(build(args.pairs, args.output_dir, seed=args.seed,
                           image_root=args.image_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
