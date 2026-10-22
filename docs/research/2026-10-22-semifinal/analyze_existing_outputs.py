"""只读研究：预测分歧与坐标量化；不修改提交，不由无真值数据推断精度。"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000
from tools.verify_acc05_dev import box_is_valid, iou_xyxy, load_gt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "arm2_k1": "results/submission_ck500_arm2_20261004/predictions.json",
        "arm2_k2": "results/submission_k20_20261004/predictions.json",
        "arm4_k2": "results/submission_arm4_ck600_k20_20261005/predictions.json",
        "arm5_k2": "results/submission_arm5_ck700_k20_20261005/predictions.json",
    }
    arms = {name: json.loads((args.workspace / path).read_text(encoding="utf-8"))
            for name, path in paths.items()}
    reference = arms["arm2_k2"]
    ids = list(reference)
    for name, arm in arms.items():
        assert set(arm) == set(ids), name
        assert all(box_is_valid(arm[sid]["bbox"]) for sid in ids), name
        assert all({k: v for k, v in arm[sid].items() if k != "bbox"}
                   == {k: v for k, v in reference[sid].items() if k != "bbox"}
                   for sid in ids), name

    pairs = {}
    for a, b in itertools.combinations(arms, 2):
        overlaps = [iou_xyxy(arms[a][sid]["bbox"], arms[b][sid]["bbox"]) for sid in ids]
        pairs[f"{a}__{b}"] = {
            "prediction_pairs": len(ids),
            "mean_prediction_iou": statistics.mean(overlaps),
            "median_prediction_iou": statistics.median(overlaps),
            "prediction_iou_lt_0.5": sum(v < 0.5 for v in overlaps),
            "prediction_iou_zero": sum(v == 0 for v in overlaps),
            "identical_boxes": sum(arms[a][sid]["bbox"] == arms[b][sid]["bbox"] for sid in ids),
        }

    groups = Counter(tuple(row[k] for k in ("visible", "infrared", "depth"))
                     for row in reference.values())
    gt_path = (REPO / "docs/audit-20260927/evidence/aux_selection_refine_20260927/"
               "manifests/scoring/city412_gt.json")
    gt = load_gt(gt_path)
    overlaps = []
    collapsed = 0
    for box in gt.values():
        try:
            rounded = [v / 1000 for v in bbox_to_qwen1000(box)]
        except ValueError:
            collapsed += 1
            continue
        overlaps.append(iou_xyxy(box, rounded))

    # 合成几何例子，非任何比赛样本：正确候选与错误实例的平均可以同时不命中。
    first = [0.10, 0.20, 0.20, 0.40]
    second = [0.50, 0.20, 0.60, 0.40]
    mean_box = [(a + b) / 2 for a, b in zip(first, second)]
    report = {
        "purpose": "只读预测分歧与量化诊断；没有官方真值，不是精度/集成收益评估",
        "input_files": paths,
        "query_count": len(ids),
        "image_triples": len(groups),
        "queries_per_triple_histogram": dict(sorted(Counter(groups.values()).items())),
        "queries_in_multi_query_triples": sum(n for n in groups.values() if n > 1),
        "pairs": pairs,
        "city412_gt_quantization_only": {
            "gt_file": str(gt_path.relative_to(REPO)).replace("\\", "/"),
            "records": len(gt),
            "collapsed_after_quantization": collapsed,
            "iou_lt_0.5_after_quantization": sum(v < 0.5 for v in overlaps),
            "minimum_iou": min(overlaps),
            "median_iou": statistics.median(overlaps),
            "meaning": "假定已知真框时0-1000离散网格的损失，不衡量模型生成数字的能力",
        },
        "synthetic_mean_box_example": {
            "candidate_1": first, "candidate_2": second, "coordinate_mean": mean_box,
            "mean_iou_to_candidate_1": iou_xyxy(mean_box, first),
            "mean_iou_to_candidate_2": iou_xyxy(mean_box, second),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
