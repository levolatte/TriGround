"""Prepare a fixed, one-time cross-scene IR/depth donor diagnostic.

This tool reads only the frozen normal native manifest, its scene map and image
headers. It never reads ground truth or predictions and never chooses boxes.
Run the resulting manifests only for an already shortlisted model, then compare
each shuffled result with normal predictions restricted to the SAME eligible IDs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import re

from PIL import Image


MODALITIES = ("infrared", "depth")
FILENAMES = {"infrared": "ir", "depth": "depth"}
ORDER = re.compile(r"in this order:\s*([^.]+)\.")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _modalities(row: dict) -> tuple[str, ...]:
    images = row["image"]
    prompt = next(item["value"] for item in row["conversations"] if item["from"] == "human")
    if prompt.count("<image>") != len(images):
        raise ValueError(f"{row['id']}: prompt/image slot count differs")
    if len(images) == 1:
        return ("rgb",)
    match = ORDER.search(prompt)
    if not match:
        raise ValueError(f"{row['id']}: native prompt lacks explicit modality order")
    names = tuple(item.strip().lower() for item in match.group(1).split(","))
    if (names not in (("rgb", "infrared"), ("rgb", "depth"),
                      ("rgb", "infrared", "depth")) or len(names) != len(images)):
        raise ValueError(f"{row['id']}: unsupported native modality order {names}")
    return names


def _image_size(value: str, data_root: Path | None, cache: dict[str, tuple[int, int]]) -> tuple[int, int]:
    if value not in cache:
        path = Path(value)
        if not path.is_absolute():
            if data_root is None:
                raise ValueError(f"relative image requires --data-root: {value}")
            path = data_root / path
        with Image.open(path) as image:
            cache[value] = image.size
    return cache[value]


def prepare(normal_path: Path, scene_map_path: Path, output_dir: Path, *,
            seed: int = 2026, data_root: Path | None = None,
            allow_size_mismatch: bool = False) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"shuffle output already exists: {output_dir}")
    rows = read_json(normal_path)
    scene_map = read_json(scene_map_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("normal diagnostic must be a nonempty native JSON list")
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(scene_map):
        raise ValueError("normal IDs must be unique and exactly match the scene map")
    if any(not isinstance(scene, str) or not scene for scene in scene_map.values()):
        raise ValueError("scene map has an empty/non-string scene")
    modalities = {sample_id: _modalities(row) for sample_id, row in by_id.items()}
    sizes: dict[str, tuple[int, int]] = {}
    results = {}
    summaries = {}
    for modality in MODALITIES:
        rng = random.Random(seed + MODALITIES.index(modality))
        candidates = {}
        skipped = {}
        for sample_id in sorted(by_id):
            row = by_id[sample_id]
            names = modalities[sample_id]
            if modality not in names:
                skipped[sample_id] = {"reason": "modality_not_present",
                                      "actual_modalities": list(names)}
                continue
            if modality in row.get("missing_modalities_actual", ()) or row.get("intervention_applied"):
                skipped[sample_id] = {"reason": "not_a_real_normal_modality"}
                continue
            slot = names.index(modality)
            path = row["image"][slot]
            candidates[sample_id] = {
                "slot": slot, "path": path,
                "size": _image_size(path, data_root, sizes),
                "scene": scene_map[sample_id],
            }
        assignments = {}
        for sample_id in sorted(candidates):
            recipient = candidates[sample_id]
            donors = [(donor_id, donor) for donor_id, donor in candidates.items()
                      if donor["scene"] != recipient["scene"]
                      and donor["path"] != recipient["path"]]
            same_size = [(donor_id, donor) for donor_id, donor in donors
                         if donor["size"] == recipient["size"]]
            pool = same_size or (donors if allow_size_mismatch else [])
            if not pool:
                skipped[sample_id] = {
                    "reason": "no_cross_scene_same_size_donor" if donors else "no_cross_scene_donor",
                    "recipient_size": list(recipient["size"]),
                    "cross_scene_candidates": len(donors),
                }
                continue
            donor_id, donor = rng.choice(sorted(pool, key=lambda item: item[0]))
            assignments[sample_id] = {
                "donor_id": donor_id, "recipient_scene": recipient["scene"],
                "donor_scene": donor["scene"], "slot": recipient["slot"],
                "recipient_path": recipient["path"], "donor_path": donor["path"],
                "recipient_size": list(recipient["size"]), "donor_size": list(donor["size"]),
                "same_size": recipient["size"] == donor["size"],
            }
        eligible_ids = [str(row["id"]) for row in rows if str(row["id"]) in assignments]
        shuffled = []
        for sample_id in eligible_ids:
            changed = deepcopy(by_id[sample_id])
            entry = assignments[sample_id]
            changed["image"][entry["slot"]] = entry["donor_path"]
            shuffled.append(changed)
        prefix = FILENAMES[modality]
        results[prefix] = {
            "manifest": shuffled,
            "donor_map": {
                "modality": modality, "seed": seed + MODALITIES.index(modality),
                "source_normal": str(normal_path), "source_scene_map": str(scene_map_path),
                "eligible_recipient_ids": eligible_ids,
                "assignments": assignments, "skipped": skipped,
                "allow_size_mismatch": allow_size_mismatch,
                "comparison_rule": "Compare against normal predictions restricted to exactly eligible_recipient_ids; never compare with the full normal denominator.",
                "usage_rule": "Run once only for the shortlisted model; never select a bbox from the shuffled prediction.",
            },
        }
        summaries[prefix] = {
            "modality": modality, "eligible_recipient_ids": eligible_ids,
            "eligible_count": len(eligible_ids), "skipped_count": len(skipped),
            "size_mismatch_count": sum(not item["same_size"] for item in assignments.values()),
        }
    output_dir.mkdir(parents=True)
    for prefix, result in results.items():
        (output_dir / f"{prefix}_shuffle.json").write_text(
            json.dumps(result["manifest"], ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output_dir / f"{prefix}_donors.json").write_text(
            json.dumps(result["donor_map"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "source_normal": str(normal_path), "seed": seed, "diagnostics": summaries,
        "comparison_rule": "For each modality, recompute normal on exactly that modality's eligible_recipient_ids. The full normal denominator is invalid for this comparison.",
        "usage_rule": "One-time diagnostic after shortlisting; do not use shuffled outputs to pick or modify boxes.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--allow-size-mismatch", action="store_true")
    args = parser.parse_args()
    summary = prepare(args.normal, args.scene_map, args.output_dir, seed=args.seed,
                      data_root=args.data_root, allow_size_mismatch=args.allow_size_mismatch)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
