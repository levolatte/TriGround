"""Package single-pass predictions; preserve every source field except bbox."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_reward import parse_bbox  # noqa: E402
from tools.quant_grounding_postprocess import normalize_bbox_completion  # noqa: E402


def make_submission(source: dict, predictions: list[dict], answer_mode: str = "short",
                    format_policy: str = "strict") -> dict:
    by_id = {row["id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != set(source):
        raise ValueError("Prediction IDs must match every official Query exactly once")
    result = deepcopy(source)
    for sample_id, row in by_id.items():
        if len(row["samples"]) != 1:
            raise ValueError(f"{sample_id}: submission requires one answer, not G8 selection")
        sample = row["samples"][0]
        if sample["finish_reason"] not in {"stop", "eos"}:
            raise ValueError(f"{sample_id}: incomplete/truncated generation")
        text = normalize_bbox_completion(sample["text"], answer_mode, format_policy)
        box = parse_bbox(text, answer_mode)
        if box is None:
            raise ValueError(f"{sample_id}: invalid answer; cannot invent a replacement box")
        result[sample_id]["bbox"] = box
    for sample_id, original in source.items():
        actual = result[sample_id]
        if set(actual) != set(original) | {"bbox"}:
            raise AssertionError(f"{sample_id}: submission schema changed")
        if any(actual[key] != value for key, value in original.items() if key != "bbox"):
            raise AssertionError(f"{sample_id}: source fields changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--answer-mode", choices=["short", "thinking"], default="short")
    parser.add_argument("--format-policy", choices=["strict", "punctuation", "single_box"], default="strict")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.queries.read_text(encoding="utf-8-sig"))
    predictions = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = make_submission(source, predictions, args.answer_mode, args.format_policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "predictions.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.output_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(path, "predictions.json")
    print(json.dumps({"queries": len(result), "format_policy": args.format_policy,
                      "json": str(path), "zip": str(args.output_dir / "submission.zip")}))


if __name__ == "__main__":
    main()
