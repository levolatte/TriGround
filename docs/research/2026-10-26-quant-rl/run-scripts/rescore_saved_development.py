"""CPU-only punctuation rescore from complete saved development labels and text.

Run from the code checkout with --scored ORIGINAL.jsonl --summary SUMMARY.json
--raw RAW.jsonl --expected-queries N --output-dir NEW_DIRECTORY.
The original strict scores must reproduce before the new policy is applied.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.eval_quant_grounding import read_jsonl, score_sample, summarize, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original = read_jsonl(args.scored)
    raw = read_jsonl(args.raw)
    original_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    assert len(original) == len(raw) == args.expected_queries
    assert original_summary["overall"]["queries"] == args.expected_queries
    assert len({row["id"] for row in original}) == args.expected_queries
    by_id = {row["id"]: row for row in raw}
    assert len(by_id) == args.expected_queries
    assert set(by_id) == {row["id"] for row in original}
    answer_mode = original_summary["answer_mode"]
    reward_mode = original_summary["reward_mode"]
    rescored = []
    repairs = []
    for row in original:
        assert len(row["samples"]) == len(by_id[row["id"]]["samples"]) == 1
        sample = row["samples"][0]
        raw_sample = by_id[row["id"]]["samples"][0]
        assert all(sample[key] == raw_sample[key] for key in ("text", "finish_reason"))
        # Saved development rows do not carry negative boxes. This utility is
        # specifically for the frozen development sets with no negative reward.
        assert not row["negative_kind"]
        assert sample["score"]["max_negative_iou"] is None
        strict = score_sample(raw_sample, row, answer_mode, reward_mode)
        assert strict == sample, f"{row['id']}: original strict score changed"
        repaired = score_sample(raw_sample, row, answer_mode, reward_mode, "punctuation")
        updated = deepcopy(row)
        updated["samples"] = [repaired]
        rescored.append(updated)
        if repaired["format_repaired"]:
            repairs.append({
                "id": row["id"], "text": sample["text"],
                "normalized_text": repaired["normalized_text"],
                "bbox": repaired["bbox"], "iou": repaired["iou"],
                "strict_hit": sample["hit"], "postprocessed_hit": repaired["hit"],
            })
    strict_summary = summarize(original)
    for key, value in original_summary["overall"].items():
        assert strict_summary["overall"][key] == value, f"strict summary differs: {key}"
    summary = summarize(rescored)
    summary.update({
        "answer_mode": answer_mode, "reward_mode": reward_mode,
        "format_policy": "punctuation", "samples_per_query": 1,
        "resource": {"rescore_only": True, "new_generations": 0},
        "input_files": {"scored": str(args.scored), "raw": str(args.raw),
                        "summary": str(args.summary)},
        "strict_overall": strict_summary["overall"], "repairs": repairs,
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.output_dir / "predictions.jsonl", rescored)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "overall": summary["overall"],
                      "repairs": repairs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
