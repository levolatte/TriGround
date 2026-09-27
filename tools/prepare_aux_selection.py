"""Freeze GT-free inference inputs for the auxiliary-selection experiment."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path, PurePosixPath


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def query_from_prompt(prompt):
    marker = "Locate the object described by this query: "
    return prompt.split(marker, 1)[1].split("\nReturn", 1)[0].strip()


def inference_row(row, data_root):
    rgb, ir, visual = row["image"]
    raw = str(PurePosixPath("depth") / PurePosixPath(rgb).relative_to("visible"))
    prompt = row["conversations"][0]["value"]
    return {
        "id": str(row["id"]), "query": query_from_prompt(prompt),
        "images": {key: str(PurePosixPath(data_root) / value)
                   for key, value in zip(("rgb", "ir", "depth_visual", "depth_raw"), (rgb, ir, visual, raw))},
        "depth_encoding": "city_mm",
    }


def select_training_smoke(rows, seed=2026):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected, seen = [], set()
    for i in indices:
        group = tuple(rows[i]["image"])
        if group in seen:
            continue
        seen.add(group)
        selected.append(rows[i])
        if len(selected) == 16:
            return selected
    raise ValueError("fewer than 16 distinct training image groups")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # Manifests are frozen once. A rerun must not replace inputs underneath a running job.
    if (out / "manifest_inventory.json").exists():
        raise FileExistsError(f"already frozen: {out}")
    train, val, gt = read_json(args.train), read_json(args.val), read_json(args.gt)
    c_rows = read_rows(args.c_predictions)
    c_by_id = {row["id"]: row for row in c_rows}
    assert len(val) == 412 and len(gt) == 412 and len(c_rows) == len(c_by_id) == 412
    assert {row["id"] for row in val} == set(gt) == set(c_by_id)
    smoke = select_training_smoke(train)
    assert not ({row["id"] for row in smoke} & set(gt))
    env_ids = [row["id"] for row in c_rows[:8]]
    fixed96 = read_json(args.fixed96)
    ids96 = set(fixed96) if isinstance(fixed96, dict) else {row["id"] for row in fixed96}
    assert len(ids96) == 96 and ids96 <= set(gt)
    cohorts = {"train16": smoke, "city412": val,
               "env8": [row for row in val if row["id"] in env_ids],
               "city96": [row for row in val if row["id"] in ids96]}
    counts = {}
    for name, rows in cohorts.items():
        inputs = [inference_row(row, args.data_root) for row in rows]
        assert len({row["id"] for row in inputs}) == len(inputs)
        dump_rows(out / f"{name}.jsonl", inputs)
        dump(out / f"{name}_baseline_prompts.json", {row["id"]: row["conversations"][0]["value"] for row in rows})
        counts[name] = {"queries": len(inputs), "groups": len({row["images"]["rgb"] for row in inputs}),
                        "unique_files": len({p for row in inputs for p in row["images"].values()})}
    dump_rows(out / "c_baseline.jsonl", [{"id": row["id"], "bbox": row["prediction"]} for row in c_rows])
    dump_rows(out / "c_report_predictions.jsonl", [{"id": row["id"], "prediction": row["prediction"],
                                                   "parsed": row["parsed"], "raw_text": row["raw_text"]} for row in c_rows])
    dump(out / "env8_expected.json", {key: {field: c_by_id[key].get(field) for field in
         ("prediction", "raw_text", "image_grid_thw", "prompt")} for key in env_ids})
    # Only the scoring process reads the files in this directory.
    (out / "scoring").mkdir(exist_ok=True)
    dump(out / "scoring" / "city412_gt.json", gt)
    dump(out / "scoring" / "city96_gt.json", {key: gt[key] for key in ids96})
    inventory = {"seed": 2026, "cohorts": counts, "data_root": args.data_root,
                 "inference_fields": ["id", "query", "images", "depth_encoding"],
                 "baseline": "C1500", "c_expected_hits": 292,
                 "train16_ids": [row["id"] for row in smoke], "env8_ids": env_ids,
                 "city96_ids": [row["id"] for row in cohorts["city96"]]}
    dump(out / "manifest_inventory.json", inventory)
    print(json.dumps(inventory, ensure_ascii=False, indent=2))


def verify_env(expected, actual):
    expected = read_json(expected)
    actual = {row["id"]: row for row in read_rows(actual)}
    assert set(expected) == set(actual), "environment8 IDs differ"
    differences = []
    for key, reference in expected.items():
        for field in ("prediction", "raw_text", "image_grid_thw", "prompt"):
            if actual[key].get(field) != reference[field]:
                differences.append({"id": key, "field": field})
    if differences:
        raise ValueError(f"C environment does not reproduce: {differences}")
    print("C_ENV8_MATCH prediction/raw_text/grid/prompt 8/8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("train", "val", "gt", "c-predictions", "fixed96", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--data-root", required=True)
    p = sub.add_parser("verify-env")
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        verify_env(args.expected, args.actual)


if __name__ == "__main__":
    main()
