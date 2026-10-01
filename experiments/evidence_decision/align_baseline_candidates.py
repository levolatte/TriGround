"""Align a frozen candidate cache's C box to a fresh direct-C prediction run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _box(value, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name}: expected normalized xyxy")
    result = [float(number) for number in value]
    if not all(math.isfinite(number) for number in result):
        raise ValueError(f"{name}: non-finite coordinate")
    if not (0 <= result[0] < result[2] <= 1 and 0 <= result[1] < result[3] <= 1):
        raise ValueError(f"{name}: invalid normalized xyxy {value}")
    return result


def _index(rows: list[dict], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in indexed:
            raise ValueError(f"{label}: duplicate ID {sample_id}")
        indexed[sample_id] = row
    if not indexed:
        raise ValueError(f"{label}: empty input")
    return indexed


def align_candidates(candidate_rows: list[dict], baseline_rows: list[dict], *,
                     model: str, adapter: str, prediction_source: str) -> list[dict]:
    candidates_by_id = _index(candidate_rows, "candidate cache")
    baseline_by_id = _index(baseline_rows, "direct-C predictions")
    if candidates_by_id.keys() != baseline_by_id.keys():
        missing = candidates_by_id.keys() - baseline_by_id.keys()
        extra = baseline_by_id.keys() - candidates_by_id.keys()
        raise ValueError(f"ID coverage differs: missing_C={len(missing)}, extra_C={len(extra)}")

    source_metadata = {
        "source": "fresh direct-C reference_holdout predictions",
        "prediction_file": prediction_source,
        "model": model,
        "adapter": adapter,
    }
    aligned = []
    for row in candidate_rows:
        sample_id = str(row["id"])
        prediction = baseline_by_id[sample_id]
        c_box = _box(prediction.get("bbox", prediction.get("prediction")),
                     f"direct-C {sample_id}.bbox")
        result = dict(row)
        result["c_bbox"] = c_box
        result["frozen_c_baseline_source"] = source_metadata
        items = [candidate for candidate in row["candidates"] if candidate.get("is_baseline") is True]
        if len(items) != 1:
            raise ValueError(f"{sample_id}: expected exactly one baseline candidate")
        baseline = dict(items[0])
        if baseline.get("role") != "target":
            raise ValueError(f"{sample_id}: baseline candidate must have target role")
        baseline["bbox"] = c_box
        sources = baseline.get("sources")
        if not isinstance(sources, list):
            raise ValueError(f"{sample_id}: baseline candidate has no source list")
        c_sources = [source for source in sources if source.get("modality") == "c"]
        if len(c_sources) != 1:
            raise ValueError(f"{sample_id}: expected exactly one C source on baseline candidate")
        baseline_sources = [dict(source) for source in sources]
        c_index = next(index for index, source in enumerate(sources) if source.get("modality") == "c")
        baseline_sources[c_index]["bbox"] = c_box
        baseline["sources"] = baseline_sources
        result["candidates"] = [baseline if candidate is items[0] else candidate
                                 for candidate in row["candidates"]]
        aligned.append(result)
    return aligned


def run(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    aligned = align_candidates(
        read_jsonl(args.candidates), read_jsonl(args.baseline_predictions),
        model=args.model, adapter=args.adapter,
        prediction_source=str(args.baseline_predictions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned),
                           encoding="utf-8")
    return {"rows": len(aligned), "output": str(args.output),
            "model": args.model, "adapter": args.adapter}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args(argv)), ensure_ascii=False))


if __name__ == "__main__":
    main()
