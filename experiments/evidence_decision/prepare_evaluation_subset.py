"""Materialize exact-manifest prediction subsets for strict full-denominator scoring."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def subset_predictions(manifest_rows: list[dict], prediction_rows: list[dict]) -> tuple[list[dict], dict]:
    ids = [str(row["id"]) for row in manifest_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest contains duplicate IDs")
    if not ids:
        raise ValueError("manifest is empty")
    predictions = {}
    for row in prediction_rows:
        sample_id = str(row["id"])
        if sample_id in predictions:
            raise ValueError(f"predictions contain duplicate ID {sample_id}")
        predictions[sample_id] = row
    expected = set(ids)
    selected = [predictions[sample_id] if sample_id in predictions else {
        "id": manifest_rows[index]["id"], "bbox": None, "prediction_missing": True,
    } for index, sample_id in enumerate(ids)]
    summary = {"expected": len(ids), "present": len(expected & predictions.keys()),
               "missing": len(expected - predictions.keys()),
               "ignored_outside_manifest": len(predictions.keys() - expected)}
    return selected, summary


def _parse_run(spec: str) -> tuple[str, Path]:
    name, separator, path = spec.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path:
        raise ValueError(f"invalid --run {spec!r}; use NAME=predictions.jsonl")
    return name, Path(path)


def run(args) -> dict:
    manifest_rows = read_jsonl(args.manifest)
    specs = [_parse_run(spec) for spec in args.run]
    names = [name for name, _ in specs]
    if not specs or len(set(names)) != len(names):
        raise ValueError("provide at least one uniquely named --run")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, source in specs:
        rows, summary = subset_predictions(manifest_rows, read_jsonl(source))
        output = args.output_dir / f"{name}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8")
        outputs[name] = {**summary, "source": str(source), "output": str(output)}
    return {"manifest": str(args.manifest), "runs": outputs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="NAME=predictions.jsonl; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run(parser.parse_args(argv)), ensure_ascii=False))


if __name__ == "__main__":
    main()
