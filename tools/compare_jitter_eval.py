"""配对比较两次评测：总命中 + 抖动靶心的机制指标。

为什么不能只看总命中：City378 命中数的标准误约 8.5 题，而 1600→3200 的整条无抖动轨迹
只在 311~313 之间摆动。抖动预期的收益（+3~+6 题）完全淹没在这个噪声里。

所以主判据是**机制指标**：抖动针对的实测事实是"近失题的框系统性偏小"（尺寸比中位 0.79）。
若抖动真的生效，同一批近失题的平均尺寸比应向 1.0 移动、平均 IoU 应上升。
近失集由**基线**定义，两次运行都在这个固定集合上算，构成严格配对。

用法：
    python tools/compare_jitter_eval.py \
        --baseline <无抖动 predictions.jsonl> --candidate <抖动 predictions.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import read_jsonl

NEAR_LOW, NEAR_HIGH = 0.3, 0.5


def load(path: Path) -> dict[str, dict]:
    records = {}
    for row in read_jsonl(path):
        sample = row["samples"][0]
        bbox, solution = sample.get("bbox"), row["solution"]
        if bbox is None:
            continue
        predicted = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        target = (solution[2] - solution[0]) * (solution[3] - solution[1])
        records[row["id"]] = {
            "iou": sample["iou"], "hit": bool(sample["hit"]),
            "size_ratio": (predicted / target) ** 0.5 if target > 0 and predicted > 0 else None,
        }
    return records


def summarise(records: dict[str, dict], ids: list[str]) -> dict:
    picked = [records[i] for i in ids if i in records]
    ratios = [r["size_ratio"] for r in picked if r["size_ratio"] is not None]
    return {
        "queries": len(picked),
        "hits": sum(1 for r in picked if r["hit"]),
        "mean_iou": round(statistics.mean(r["iou"] for r in picked), 4) if picked else None,
        "mean_size_ratio": round(statistics.mean(ratios), 4) if ratios else None,
        "median_size_ratio": round(statistics.median(ratios), 4) if ratios else None,
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    shared = [i for i in base if i in cand]
    # 某一侧若出现无有效框的题，load 会跳过它。比较必须落在交集上，并明确报出丢弃数，
    # 不能悄悄丢——那会让两侧分母不同、比较失真。
    dropped = {"baseline_only": sorted(set(base) - set(shared)),
               "candidate_only": sorted(set(cand) - set(shared))}
    if not shared:
        raise ValueError("两次评测没有共同题目")

    near = [i for i in shared if NEAR_LOW <= base[i]["iou"] < NEAR_HIGH]
    hard = [i for i in shared if base[i]["iou"] < NEAR_LOW]
    all_ids = shared

    report = {"queries": len(shared), "dropped": dropped,
              "bands": {"all": {}, "near_miss": {"range": [NEAR_LOW, NEAR_HIGH], "defined_by": "baseline"},
                        "hard": {"range": [0.0, NEAR_LOW], "defined_by": "baseline"}}}
    for label, ids in (("all", all_ids), ("near_miss", near), ("hard", hard)):
        report["bands"][label].update({"baseline": summarise(base, ids),
                                       "candidate": summarise(cand, ids)})

    b_near, c_near = report["bands"]["near_miss"]["baseline"], report["bands"]["near_miss"]["candidate"]
    b_all, c_all = report["bands"]["all"]["baseline"], report["bands"]["all"]["candidate"]
    delta = {"hits": c_all["hits"] - b_all["hits"],
             "near_miss_mean_iou": round((c_near["mean_iou"] or 0) - (b_near["mean_iou"] or 0), 4),
             "near_miss_mean_size_ratio": round((c_near["mean_size_ratio"] or 0) - (b_near["mean_size_ratio"] or 0), 4),
             "near_miss_hits": c_near["hits"] - b_near["hits"]}
    report["delta"] = delta

    # 过闸条件：尺寸比明显上移（抖动靶心），且总命中不下降。
    report["verdict"] = (
        "adopt" if delta["near_miss_mean_size_ratio"] >= 0.03 and delta["hits"] >= 0 else "reject")
    report["verdict_reason"] = (
        "近失题尺寸比上移且总命中未降" if report["verdict"] == "adopt" else
        "尺寸比上移不足" if delta["near_miss_mean_size_ratio"] < 0.03 else "总命中下降")

    if args.out_json:
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2), flush=True)
