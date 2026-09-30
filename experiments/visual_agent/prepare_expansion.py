"""Prepare a GT-free, group-separated expansion manifest from City train.

Selection tags are Query-text proxies, not claims about RGB/IR/Depth validity.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import prepare


TAG_ORDER = ("camera_distance_text", "ordinal_text", "identity_text")
REFERENCE = re.compile(
    r"\b(?:left|right)\s+of\b|\b(?:next\s+to|beside|between|behind|above|below|in\s+front\s+of)\b"
)
EXPLICIT_CAMERA = re.compile(
    r"\b(?:foreground|midground|background|frontmost|backmost)\b|"
    r"\b(?:front|middle|back)\s+(?:ground|layer|of\s+the\s+scene)\b|"
    r"\b(?:closer|closest|nearest|nearer|farther|farthest|furthest|further|near|far|toward)"
    r"\s+(?:(?:to|from)\s+)?(?:the\s+)?(?:camera|viewer)\b"
)
BARE_CAMERA = re.compile(r"\b(?:closest|nearest|farthest|furthest)\b")
NON_CAMERA_DISTANCE = re.compile(
    r"\b(?:closest|nearest|farthest|furthest|closer|nearer|farther|further)\b"
    r".{0,40}?\b(?:to|from|than)\s+(?!(?:the\s+)?(?:camera|viewer)\b)(?:the\s+)?\w+"
)
ORDINAL = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|"
    r"leftmost|rightmost|frontmost|backmost|\d+(?:st|nd|rd|th))\b"
)
IDENTITY = re.compile(
    r"\b(?:red|blue|green|yellow|white|black|brown|gray|grey|orange|purple|pink|"
    r"silver|gold|teal|beige|striped|plaid|patterned|wearing|shirt|coat|jacket|"
    r"dress|hat|helmet|logo|text|letter|number|same|matching|identical|holding)\b"
)


def image_basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def tags_for_query(query: str) -> list[str]:
    text = query.casefold()
    reference = REFERENCE.search(text)
    target = text[:reference.start()] if reference else text
    comparison = NON_CAMERA_DISTANCE.search(target)
    if comparison:
        target = target[:comparison.start()]
    tags = []
    if EXPLICIT_CAMERA.search(target) or BARE_CAMERA.search(target):
        tags.append("camera_distance_text")
    if ORDINAL.search(text):
        tags.append("ordinal_text")
    if IDENTITY.search(text):
        tags.append("identity_text")
    return tags or ["general_text"]


def select_rows(rows: list[dict], count: int, seed: int) -> list[tuple[dict, int, list[str]]]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[prepare.image_group(row)].append(row)
    rng = random.Random(seed)
    groups = sorted(by_group)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(by_group[group])
    selected = []
    tag_counts = Counter()

    def choose(group: str, round_number: int):
        candidates = by_group[group]
        if not candidates:
            return
        row = max(candidates, key=lambda item: sum(
            1 / (1 + tag_counts[tag]) for tag in tags_for_query(prepare.query(item))
            if tag in TAG_ORDER))
        candidates.remove(row)
        tags = tags_for_query(prepare.query(row))
        selected.append((row, round_number, tags))
        tag_counts.update(tag for tag in tags if tag in TAG_ORDER)

    for round_number in (1, 2):
        for group in groups:
            if len(selected) == count:
                return selected
            choose(group, round_number)
    raise ValueError(f"only {len(selected)} eligible queries with at most two per image group; need {count}")


def prepare_expansion(args):
    train = prepare.read_json(args.train)
    if len({str(row["id"]) for row in train}) != len(train):
        raise ValueError("duplicate raw train sample IDs")
    eligible, exclusion = prepare.eligible_train(train)
    existing_groups = {
        image_basename(row["images"]["rgb"])
        for manifest in args.existing_manifests
        for row in prepare.read_jsonl(manifest)
    }
    available = [row for row in eligible if prepare.image_group(row) not in existing_groups]
    selected = select_rows(available, args.count, args.seed)
    rows = [row for row, _, _ in selected]
    selected_groups = {prepare.image_group(row) for row in rows}
    tag_counts = Counter(tag for _, _, tags in selected for tag in tags)
    inventory = {
        "source_train": str(args.train),
        "existing_manifests": [str(path) for path in args.existing_manifests],
        "data_root": args.data_root,
        "seed": args.seed,
        "requested_queries": args.count,
        "selected_queries": len(rows),
        "selected_independent_image_groups": len(selected_groups),
        "selection_round_counts": dict(Counter(round_number for _, round_number, _ in selected)),
        "selected_text_proxy_counts": {tag: tag_counts[tag] for tag in (*TAG_ORDER, "general_text")},
        "exclusion": {
            **exclusion,
            "existing_manifest_image_groups": len(existing_groups),
            "existing_manifest_eligible_image_groups_removed": (
                len({prepare.image_group(row) for row in eligible})
                - len({prepare.image_group(row) for row in available})
            ),
            "existing_manifest_queries_removed_after_eligibility": len(eligible) - len(available),
            "available_queries": len(available),
            "available_image_groups": len({prepare.image_group(row) for row in available}),
        },
        "selection_note": "Query text proxies only; RGB/IR identity evidence and Depth validity were not checked.",
        "gt_free_fields": ["id", "query", "images", "depth_encoding"],
    }
    out = args.output_dir
    if (out / "inventory.json").exists():
        raise FileExistsError(f"expansion manifest already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    prepare.write_jsonl(out / "manifest.jsonl", [prepare.inference_row(row, args.data_root) for row in rows])
    prepare.write_jsonl(out / "selection_tags.jsonl", [
        {"id": str(row["id"]), "image_group": prepare.image_group(row),
         "selection_round": round_number, "selection_tags": tags,
         "image_evidence_verified": False,
         "note": "Query-text proxy only; modalities and target correctness are unverified."}
        for row, round_number, tags in selected
    ])
    prepare.write_json(out / "inventory.json", inventory)
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--existing-manifests", type=Path, action="append", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2029)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    result = prepare_expansion(args)
    print(json.dumps({key: result[key] for key in
                      ("selected_queries", "selected_independent_image_groups", "selected_text_proxy_counts")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
