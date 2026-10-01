"""Prepare GT-free candidate-union and auxiliary-content replay inputs."""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def keyed(rows: list[dict], name: str) -> dict[str, dict]:
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{name}: duplicate IDs")
    return result


def union(args):
    manifest = keyed(read_jsonl(args.manifest), "manifest")
    fixed = keyed(read_jsonl(args.fixed_cache), "fixed cache")
    dynamic = keyed(read_jsonl(args.dynamic_predictions), "dynamic predictions")
    if set(manifest) != set(fixed) or set(manifest) != set(dynamic):
        raise ValueError("candidate union requires the same complete ID set")
    rows, additions = [], 0
    for sample_id in manifest:
        source = fixed[sample_id]
        pred = dynamic[sample_id]
        old = source["candidates"]
        public = pred["final_candidates"]
        by_id = {str(c["id"]): c for c in public}
        if len(by_id) != len(public) or "KEEP" not in by_id:
            raise ValueError(f"{sample_id}: duplicate dynamic candidate ID")
        if source["c_bbox"] != by_id["KEEP"]["bbox"]:
            raise ValueError(f"{sample_id}: dynamic KEEP changed C initial bbox")
        old_ids = {str(c["id"]) for c in old if not c["is_baseline"]}
        converted = []
        for candidate in old:
            key = "KEEP" if candidate["is_baseline"] else str(candidate["id"])
            updated = by_id.get(key)
            if updated is None or updated["bbox"] != candidate["bbox"] or updated["role"] != candidate["role"]:
                raise ValueError(f"{sample_id}: dynamic pool changed or dropped old ID {candidate['id']}")
            merged = copy.deepcopy(candidate)
            merged["sources"] = updated.get("sources", merged.get("sources", []))
            merged["possible_same_object_ids"] = updated.get("possible_same_object_ids", [])
            converted.append(merged)
        new_public = [c for c in public if str(c["id"]) not in old_ids | {"KEEP"}]
        for item in new_public:
            if not str(item["id"]).isdigit():
                raise ValueError(f"{sample_id}: new candidate ID is not numeric")
            converted.append({"id": int(item["id"]), "role": item["role"], "bbox": item["bbox"],
                              "is_baseline": False, "sources": item.get("sources", []),
                              "possible_same_object_ids": item.get("possible_same_object_ids", []),
                              "depth": {"status": "pending", "reason": "new_replay_candidate"},
                              "mask_path": None})
        if len({c["id"] for c in converted}) != len(converted):
            raise ValueError(f"{sample_id}: converted candidate ID collision")
        additions += len(new_public)
        row = dict(source)
        row["candidates"] = converted
        row["replay_provenance"] = {"kind": "candidate_union", "source_run": str(args.dynamic_predictions)}
        rows.append(row)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_cache, rows)
    meta = {"kind": "candidate_union", "queries": len(rows), "new_candidate_entries": additions,
            "candidate_cache": str(args.output_cache),
            "interpretation": "Run static/fixed-pool mechanisms with this cache and search disabled. Candidate pools were frozen from a prior dynamic run; this is an offline replay, not independent discovery."}
    args.output_cache.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def permute(args):
    rows = read_jsonl(args.manifest)
    by_group = {}
    for row in rows:
        group = row["images"]["rgb"]
        by_group.setdefault(group, row)
    groups = sorted(by_group)
    if len(groups) < args.count:
        raise ValueError(f"only {len(groups)} available image groups")
    chosen = random.Random(args.seed).sample(groups, args.count)
    panel = [by_group[group] for group in chosen]
    donors = panel[1:] + panel[:1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "correct.jsonl", panel)
    mapping = []
    variants = {"ir_swapped": [], "depth_swapped": []}
    for row, donor in zip(panel, donors):
        if row["depth_encoding"] != donor["depth_encoding"]:
            raise ValueError("depth permutation requires matching encodings")
        mapping.append({"id": row["id"], "donor_id": donor["id"]})
        ir = copy.deepcopy(row)
        ir["images"]["ir"] = donor["images"]["ir"]
        variants["ir_swapped"].append(ir)
        depth = copy.deepcopy(row)
        for key in ("depth_raw", "depth_visual"):
            depth["images"][key] = donor["images"][key]
        variants["depth_swapped"].append(depth)
    for name, variant in variants.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", variant)
    if args.candidate_cache:
        cache = keyed(read_jsonl(args.candidate_cache), "candidate cache")
        if not {r["id"] for r in panel} <= set(cache):
            raise ValueError("candidate cache lacks permutation panel IDs")
        fixed = [cache[r["id"]] for r in panel]
        write_jsonl(args.output_dir / "correct_candidates.jsonl", fixed)
        write_jsonl(args.output_dir / "ir_swapped_candidates.jsonl", fixed)
        cleared = copy.deepcopy(fixed)
        for row in cleared:
            for candidate in row["candidates"]:
                candidate["depth"] = {"status": "pending", "reason": "depth_content_permuted_remeasure"}
                candidate["mask_path"] = None
        write_jsonl(args.output_dir / "depth_swapped_candidates.jsonl", cleared)
    meta = {"kind": "auxiliary_content_permutation", "seed": args.seed,
            "queries": len(panel), "image_groups": len(chosen), "mapping": mapping,
            "candidate_pool": "fixed original candidates; IR-sourced boxes remain from original IR",
            "depth_cache": "cleared in depth_swapped_candidates.jsonl; live measurement must use swapped raw Depth",
            "interpretation": "Matched content sensitivity only. Keep the same IDs and denominator, including no-call cases; this is not strict RGB-only ablation."}
    (args.output_dir / "permutation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("union")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fixed-cache", type=Path, required=True)
    p.add_argument("--dynamic-predictions", type=Path, required=True)
    p.add_argument("--output-cache", type=Path, required=True)
    p = sub.add_parser("permute")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--candidate-cache", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    result = union(args) if args.command == "union" else permute(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
