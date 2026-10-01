"""Freeze GT-free City manifests for the visual-agent experiments.

Raw train/val inputs may contain labels. Only the separate optional training
scoring file retains labels; every inference JSONL is deliberately label-free.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath


SEED = 2026
SHARED_SEQUENCES = (
    "000002_025", "000002_041", "000002_043", "000002_074",
    "000013_016", "000014_025", "000014_052",
)
MAIN_DIAGNOSTIC_GROUPS = ("000013_004_00000238", "000610", "001179")
SHARED_RE = re.compile(r"(?<!\d)(?:" + "|".join(SHARED_SEQUENCES) + r")(?:_suppl)?_")
QUERY_MARKER = "Locate the object described by this query: "


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def image_group(row: dict) -> str:
    return str(PurePosixPath(row["image"][0]).name)


def query(row: dict) -> str:
    prompt = row["conversations"][0]["value"]
    if QUERY_MARKER not in prompt:
        raise ValueError(f"{row['id']}: unknown query prompt")
    return prompt.split(QUERY_MARKER, 1)[1].split("\nReturn", 1)[0].strip()


def inference_row(row: dict, data_root: str) -> dict:
    rgb, ir, depth_visual = row["image"]
    raw = str(PurePosixPath("depth") / PurePosixPath(rgb).relative_to("visible"))
    return {
        "id": str(row["id"]), "query": query(row),
        "images": {key: str(PurePosixPath(data_root) / value)
                   for key, value in zip(("rgb", "ir", "depth_visual", "depth_raw"),
                                         (rgb, ir, depth_visual, raw))},
        "depth_encoding": "city_mm",
    }


def eligible_train(train: list[dict]) -> tuple[list[dict], dict]:
    shared = [r for r in train if SHARED_RE.search(str(r["id"]))]
    diagnostic = [r for r in train if image_group(r).removesuffix(".png") in MAIN_DIAGNOSTIC_GROUPS]
    shared_ids, diagnostic_ids = {r["id"] for r in shared}, {r["id"] for r in diagnostic}
    if shared_ids & diagnostic_ids:
        raise ValueError("known shared sequence and diagnostic group exclusions overlap")
    excluded = shared_ids | diagnostic_ids
    kept = [r for r in train if r["id"] not in excluded]
    return kept, {
        "shared_sequence_queries": len(shared),
        "shared_sequence_groups": len({image_group(r) for r in shared}),
        "main_diagnostic_queries": len(diagnostic),
        "main_diagnostic_groups": len({image_group(r) for r in diagnostic}),
        "kept_queries": len(kept), "kept_groups": len({image_group(r) for r in kept}),
        "shared_sequences": list(SHARED_SEQUENCES),
        "main_diagnostic_image_groups": list(MAIN_DIAGNOSTIC_GROUPS),
    }


def select_dev_groups(val: list[dict], count: int = 16, seed: int = SEED) -> list[dict]:
    groups = sorted({image_group(r) for r in val})
    if len(groups) < count:
        raise ValueError(f"only {len(groups)} dev image groups")
    chosen = set(random.Random(seed).sample(groups, count))
    return [r for r in val if image_group(r) in chosen]


def selection_tags(row: dict) -> list[str]:
    """Text-only selection hints; no claim of image/depth quality or GT result."""
    q = query(row).casefold()
    tags = []
    if any(word in q for word in ("left", "right", "behind", "beside", "between", "front", "near", "far")):
        tags.append("relation_text")
    if any(word in q for word in ("first", "second", "third", "fourth", "last", "leftmost", "rightmost")):
        tags.append("ordinal_text")
    if any(word in q for word in ("part", "wheel", "window", "door", "head", "hand", "group", "pair")):
        tags.append("part_or_group_text")
    if any(word in q for word in ("two", "both", "other", "another", "same", "first", "second", "third", "fourth")):
        tags.append("competition_text")
    if any(word in q for word in ("red", "blue", "white", "black", "green", "yellow", "dark", "bright", "striped", "plaid")):
        tags.append("appearance_text")
    if any(word in q for word in ("nearer", "farther", "closest", "farthest", "closer to the camera")):
        tags.append("camera_depth_relation_text")
    return tags or ["general_text"]


def select_distinct_starts(rows: list[dict], count: int, seed: int) -> list[dict]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[image_group(row)].append(row)
    rng = random.Random(seed)
    groups = sorted(by_group)
    if len(groups) < count:
        raise ValueError(f"only {len(groups)} eligible image groups for {count} starts")
    rng.shuffle(groups)
    return [rng.choice(by_group[group]) for group in groups[:count]]


def select_debug32(rows: list[dict], seed: int = SEED) -> list[tuple[dict, str]]:
    """Select text-proxy strata for inspection; visual-quality strata remain unverified."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    intents = (("keep_part_group_to_check", "part_or_group_text", 4),
               ("same_class_competition_to_check", "competition_text", 8),
               ("spatial_relation_to_check", "relation_text", 8),
               ("rgb_ir_quality_to_check", "appearance_text", 6),
               ("depth_validity_to_check", "camera_depth_relation_text", 6))
    selected, seen = [], set()
    for intent, tag, quota in intents:
        matching = [r for r in shuffled if tag in selection_tags(r) and image_group(r) not in seen]
        fallback = [r for r in shuffled if image_group(r) not in seen]
        for row in (matching + fallback):
            group = image_group(row)
            if group in seen:
                continue
            seen.add(group)
            selected.append((row, intent if row in matching else intent + "_text_proxy_unavailable"))
            if sum(label.startswith(intent) for _, label in selected) == quota:
                break
    if len(selected) != 32:
        raise ValueError(f"debug selection yielded {len(selected)} rather than 32 starts")
    return selected


def baseline_map(path: Path) -> dict[str, list[float]]:
    result = {}
    for row in read_jsonl(path):
        sample_id = str(row["id"])
        if sample_id in result:
            raise ValueError(f"duplicate C baseline ID: {sample_id}")
        result[sample_id] = row.get("bbox", row.get("prediction"))
    return result


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "inventory.json").exists():
        raise FileExistsError(f"frozen manifests already exist: {out}")
    train, val = read_json(args.train), read_json(args.val)
    if len({r["id"] for r in train}) != len(train) or len({r["id"] for r in val}) != len(val):
        raise ValueError("duplicate raw sample IDs")
    eligible, exclusion = eligible_train(train)
    dev = select_dev_groups(val)
    debug = select_debug32(eligible)
    train32 = [r for r, _ in debug]
    used32 = {image_group(r) for r in train32}
    t250 = select_distinct_starts([r for r in eligible if image_group(r) not in used32], 250, SEED + 1)
    holdout_groups = set(random.Random(SEED + 2).sample(sorted({image_group(r) for r in t250}), 50))
    t_train = [r for r in t250 if image_group(r) not in holdout_groups]
    t_holdout = [r for r in t250 if image_group(r) in holdout_groups]
    cohorts = {"city412": val, "dev16groups": dev, "train32": train32,
               "t250": t250, "t_train200": t_train, "t_holdout50": t_holdout}
    for name, rows in cohorts.items():
        write_jsonl(out / f"{name}.jsonl", [inference_row(row, args.data_root) for row in rows])
    baseline = baseline_map(args.c_baseline)
    val_ids = {str(r["id"]) for r in val}
    if set(baseline) != val_ids:
        raise ValueError("C baseline IDs do not equal full development IDs")
    for name, rows in (("city412", val), ("dev16groups", dev)):
        write_jsonl(out / f"{name}_c_baseline.jsonl",
                    [{"id": str(r["id"]), "bbox": baseline[str(r["id"])]} for r in rows])
    write_jsonl(out / "train32_selection_tags.jsonl",
                [{"id": str(r["id"]), "selection_tags": selection_tags(r),
                  "selection_intent": intent,
                  "image_evidence_verified": False,
                  "note": "Text-only hint; IR ambiguity and Depth validity require real tool inspection."}
                 for r, intent in debug])
    # An optional train GT is only exported to a separate offline scoring folder.
    if args.train_gt:
        gt = read_json(args.train_gt)
        scoring = out / "scoring"
        scoring.mkdir(exist_ok=True)
        write_json(scoring / "train32_gt.json", {r["id"]: gt[r["id"]] for r in train32})
        write_json(scoring / "t250_gt.json", {r["id"]: gt[r["id"]] for r in t250})
    inventory = {
        "seed": SEED, "source_train": str(args.train), "source_val": str(args.val),
        "source_c_baseline": str(args.c_baseline), "data_root": args.data_root,
        "exclusion": exclusion,
        "cohorts": {name: {"queries": len(rows), "groups": len({image_group(r) for r in rows}),
                            "ids": [str(r["id"]) for r in rows]}
                    for name, rows in cohorts.items()},
        "selection_note": "train32 selection intents use Query text proxies; actual same-class competition, RGB/IR ambiguity, Depth validity and KEEP correctness remain unverified.",
        "gt_free_fields": ["id", "query", "images", "depth_encoding"],
    }
    write_json(out / "inventory.json", inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "val", "c-baseline", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-gt", type=Path,
                        help="Optional object-keyed training GT; written only below output-dir/scoring")
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps({"exclusion": result["exclusion"],
                      "cohorts": {k: {"queries": v["queries"], "groups": v["groups"]}
                                  for k, v in result["cohorts"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
