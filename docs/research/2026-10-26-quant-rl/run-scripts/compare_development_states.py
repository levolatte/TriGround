"""CPU-only paired comparison of warmup50/A59/B59/C29/C550 development answers.

python docs/research/2026-10-26-quant-rl/run-scripts/compare_development_states.py
    --input-root docs/research/2026-10-26-quant-rl/evidence/punctuation
    --output-prefix /path/to/five-state-paired-development
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.eval_quant_grounding import read_jsonl, score_sample, summarize


STATES = {"warmup50": "warmup50", "A59": "a-final", "B59": "b-final",
          "C29": "c-matched", "C550": "c-final"}
SPLITS = {"city378": 378, "external88": 88}
POLICIES = ("strict", "punctuation")


def metrics(samples):
    count = len(samples)
    return {"queries": count, "hits": sum(sample["hit"] for sample in samples),
            "acc05": sum(sample["hit"] for sample in samples) / count,
            "miou": sum(effective_iou(sample) for sample in samples) / count,
            "invalid": sum(not sample["score"]["valid"] for sample in samples),
            "truncated": sum(sample["score"]["truncated"] for sample in samples)}


def effective_iou(sample):
    # A complete-set mean: invalid/unterminated answers contribute zero.
    return sample["iou"] if sample["score"]["valid"] else 0.0


def load_state(directory, expected):
    rows = read_jsonl(directory / "predictions.jsonl")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (len(rows) != expected or len({row["id"] for row in rows}) != expected
            or any(len(row["samples"]) != 1 for row in rows)):
        raise ValueError(f"{directory}: expected {expected} unique single-answer rows")
    if summary["answer_mode"] != "short" or summary["format_policy"] != "punctuation":
        raise ValueError(f"{directory}: expected short answers with punctuation scoring")
    by_policy = {policy: {} for policy in POLICIES}
    for row in rows:
        sample = row["samples"][0]
        if row["negative_kind"] or sample["score"]["max_negative_iou"] is not None:
            raise ValueError(f"{directory}/{row['id']}: development rows must have no negative boxes")
        for policy in POLICIES:
            by_policy[policy][row["id"]] = score_sample(
                sample, row, "short", summary["reward_mode"], policy)
        if by_policy["punctuation"][row["id"]] != sample:
            raise ValueError(f"{directory}/{row['id']}: saved punctuation score differs from recomputation")
    recomputed_overall = summarize(rows)["overall"]
    for key, value in summary["overall"].items():
        if recomputed_overall[key] != value:
            raise ValueError(f"{directory}: saved overall summary differs at {key}")
    return {"rows": {row["id"]: row for row in rows}, "samples": by_policy,
            "metrics": {policy: metrics(list(samples.values())) for policy, samples in by_policy.items()},
            "input_directory": str(directory)}


def compare_states(before, after):
    before_rows, after_rows = before["rows"], after["rows"]
    if before_rows.keys() != after_rows.keys():
        raise ValueError("Paired states have different ID sets")
    for query_id in before_rows:
        for key in ("solution", "source", "group_id", "negative_kind"):
            if before_rows[query_id][key] != after_rows[query_id][key]:
                raise ValueError(f"{query_id}: paired states differ at {key}")
    pairs = {}
    for policy in POLICIES:
        per_query = []
        for query_id in before_rows:
            left, right = before["samples"][policy][query_id], after["samples"][policy][query_id]
            left_iou, right_iou = effective_iou(left), effective_iou(right)
            per_query.append({"id": query_id,
                              "before_hit": left["hit"], "after_hit": right["hit"],
                              "before_iou": left_iou, "after_iou": right_iou,
                              "iou_diff": right_iou - left_iou,
                              "rescue": right["hit"] and not left["hit"],
                              "regression": left["hit"] and not right["hit"]})
        rescue_ids = [row["id"] for row in per_query if row["rescue"]]
        regression_ids = [row["id"] for row in per_query if row["regression"]]
        pairs[policy] = {"queries": len(per_query), "rescues": len(rescue_ids),
                         "regressions": len(regression_ids),
                         "net_hits": len(rescue_ids) - len(regression_ids),
                         "mean_iou_diff": sum(row["iou_diff"] for row in per_query) / len(per_query),
                         "rescue_ids": rescue_ids, "regression_ids": regression_ids,
                         "per_query": per_query}
    return pairs


def markdown(report):
    lines = ["# 五状态开发集逐题比较", "",
             "严格口径与标点规范化均从同一原始回答复算；命中要求答案有效且IoU≥0.5。",
             "mIoU使用全体题目为分母，无效或截断答案的IoU记为0。逐题明细见同名JSON。",
             "C29为232次新增呈现，A59/B59为236次，是近似同曝光对照；C550为4400次。", ""]
    for split, result in report["splits"].items():
        lines += [f"## {split}", "",
                  "| 状态 | 口径 | 命中 | ACC@0.5 | mIoU | 无效 | 截断 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for state, values in result["states"].items():
            for policy in POLICIES:
                item = values["metrics"][policy]
                label = "严格" if policy == "strict" else "标点规范化"
                lines.append(f"| {state} | {label} | {item['hits']}/{item['queries']} | "
                             f"{item['acc05']:.2%} | {item['miou']:.6f} | "
                             f"{item['invalid']} | {item['truncated']} |")
        lines += ["", "以下方向均为后状态相对前状态。", "",
                  "| 前→后 | 口径 | 救回 | 退步 | 净增命中 | 平均IoU差 |",
                  "|---|---|---:|---:|---:|---:|"]
        for pair in result["comparisons"].values():
            for policy in POLICIES:
                item = pair[policy]
                label = "严格" if policy == "strict" else "标点规范化"
                lines.append(f"| {pair['before']}→{pair['after']} | {label} | "
                             f"{item['rescues']} | {item['regressions']} | {item['net_hits']:+d} | "
                             f"{item['mean_iou_diff']:+.6f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/punctuation")
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    paths = {(state, split): args.input_root / f"{stem}-{split}"
             for state, stem in STATES.items() for split in SPLITS}
    missing = [str(directory / name) for directory in paths.values()
               for name in ("predictions.jsonl", "summary.json") if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Incomplete five-state results:\n" + "\n".join(missing))
    report = {"metric": "first-answer IoU >= 0.5 with valid termination",
              "miou_policy": "all queries; invalid or truncated answers contribute zero IoU",
              "normalized_policy": "punctuation", "splits": {}}
    split_ids = []
    for split, expected in SPLITS.items():
        loaded = {state: load_state(paths[state, split], expected) for state in STATES}
        pairs = {}
        for before, after in combinations(STATES, 2):
            pairs[f"{after}_vs_{before}"] = {
                "before": before, "after": after, **compare_states(loaded[before], loaded[after])}
        split_ids.append(set(loaded["warmup50"]["rows"]))
        report["splits"][split] = {
            "states": {state: {"metrics": value["metrics"], "input_directory": value["input_directory"]}
                       for state, value in loaded.items()},
            "validation": {"identical_id_sets": True, "identical_ground_truth_and_metadata": True,
                           "single_answer_per_query": True, "saved_scores_reproduced": True},
            "comparisons": pairs}
    if split_ids[0] & split_ids[1]:
        raise ValueError("City and external development ID sets overlap")
    prefix = args.output_prefix or args.input_root / "five-state-paired-development"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(prefix.with_suffix('.json')), "markdown": str(prefix.with_suffix('.md'))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
