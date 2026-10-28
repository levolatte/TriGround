"""CPU-only counterfactuals over recorded B59 G8 answers; no new model sampling."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "2026-10-26-quant-rl/evidence/b-final-reward-metrics.jsonl"


def unequal_fraction(values, size):
    subsets = list(combinations(values, size))
    return sum(max(xs) - min(xs) > 1e-12 for xs in subsets) / len(subsets)


def main():
    groups = [json.loads(line)["items"] for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    assert len(groups) == 236 and all(len(g) == 8 for g in groups)
    assert len({g[0]["occurrence_id"] for g in groups}) == len(groups)
    rows = []
    for group in groups:
        assert len({x["occurrence_id"] for x in group}) == 1
        rewards = [x["reward"] for x in group]
        # This is a new reward hypothesis, not a recomputation of the official score.
        smooth = [(float(x["hit"]) + 0.2 * x["iou"]) if x["valid"] else -0.2 for x in group]
        counts = Counter(x["generated_text"] for x in group)
        p2 = unequal_fraction(rewards, 2)
        p4 = unequal_fraction(rewards, 4)
        p8 = unequal_fraction(rewards, 8)
        # Stop after 2 if unequal, otherwise inspect 4, then 8. Same permutation
        # nesting makes E[N] = 2 + 2*P(equal first 2) + 4*P(equal first 4).
        adaptive_n = 2 + 2 * (1 - p2) + 4 * (1 - p4)
        rows.append({
            "id": group[0]["id"], "occurrence_id": group[0]["occurrence_id"],
            "source": group[0]["source"], "unique_texts": len(counts),
            "all_hit": all(x["hit"] for x in group),
            "all_wrong": not any(x["hit"] for x in group),
            "original_signal_g2": p2, "original_signal_g4": p4, "original_signal_g8": p8,
            "smooth_signal_g2": unequal_fraction(smooth, 2),
            "smooth_signal_g4": unequal_fraction(smooth, 4),
            "smooth_signal_g8": unequal_fraction(smooth, 8),
            "adaptive_2_4_8_expected_answers": adaptive_n,
            "generated_tokens": [x["generated_tokens"] for x in group],
            "reward_range": max(rewards) - min(rewards),
            "iou_range": max(x["iou"] or 0.0 for x in group) - min(x["iou"] or 0.0 for x in group),
        })
    total = len(groups) * 8
    summary = {
        "source": str(SOURCE), "groups": len(groups), "answers": total,
        "unique_queries": len({r["id"] for r in rows}),
        "unique_texts_within_group": sum(r["unique_texts"] for r in rows),
        "exact_text_duplicate_fraction": 1 - sum(r["unique_texts"] for r in rows) / total,
        "groups_with_duplicate_text": sum(r["unique_texts"] < 8 for r in rows),
        "mean_generated_tokens": mean(n for r in rows for n in r["generated_tokens"]),
        "all_hit_groups": sum(r["all_hit"] for r in rows),
        "all_wrong_groups": sum(r["all_wrong"] for r in rows),
        "subsampling": [{
            "g": g,
            "original_expected_signal_groups": sum(r[f"original_signal_g{g}"] for r in rows),
            "original_signal_group_probability": mean(r[f"original_signal_g{g}"] for r in rows),
            "original_signal_groups_per_100_answers": 100 * mean(r[f"original_signal_g{g}"] for r in rows) / g,
            "smooth_expected_signal_groups": sum(r[f"smooth_signal_g{g}"] for r in rows),
            "smooth_signal_group_probability": mean(r[f"smooth_signal_g{g}"] for r in rows),
        } for g in (2, 4, 8)],
        "adaptive_2_4_8_expected_answers": sum(r["adaptive_2_4_8_expected_answers"] for r in rows),
        "adaptive_2_4_8_answer_saving_fraction": 1 - mean(r["adaptive_2_4_8_expected_answers"] for r in rows) / 8,
        "limitations": [
            "Exact enumeration of subsets of already recorded G8; not new on-policy G2/G4 training.",
            "Signal means unequal rewards, not measured gradient magnitude, accuracy gain or convergence.",
            "Smooth reward replaces 0.2*min(IoU,0.5) with 0.2*IoU; it changes the objective.",
            "Duplicate text is a screening upper bound, not verified identical response token IDs.",
            "Adaptive stopping changes the estimator; expected sampling savings exclude launch/prefill costs.",
            "All metrics concern original B59, not the live continuation after step 59.",
        ],
    }
    (HERE / "evidence/rollout-counterfactual.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (HERE / "evidence/rollout-counterfactual-groups.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
