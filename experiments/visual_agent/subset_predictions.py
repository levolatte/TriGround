"""Select a frozen, GT-free prediction subset for matched offline scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def select_predictions(manifest, predictions):
    ids = [str(row["id"]) for row in _read_jsonl(manifest)]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{manifest}: empty manifest or duplicate ID")
    source = {}
    for row in _read_jsonl(predictions):
        sample_id = str(row["id"])
        if sample_id in source:
            raise ValueError(f"{predictions}: duplicate prediction ID {sample_id}")
        source[sample_id] = row
    missing = set(ids) - source.keys()
    if missing:
        raise ValueError(f"{predictions}: missing {len(missing)} frozen IDs: {sorted(missing)[:5]}")
    return [source[sample_id] for sample_id in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = select_predictions(args.manifest, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                   for row in rows), encoding="utf-8")
    print(json.dumps({"selected": len(rows), "available": len(_read_jsonl(args.predictions)),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
