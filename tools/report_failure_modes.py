"""按**失败模式**分解逐题结果，供"小目标桶是否退化"的判据使用。

## 为什么需要它

总体命中数会掩盖结构性问题。计划的副判据是"**小目标桶命中数不得下降**"——
若总体涨而小目标跌，那是用易题换难题，不接受。只看总数看不出来。

## 失败模式的判据（与建立基线时一致）

对每个未命中，看预测框与 GT 的**交集占较小框的比例**：

* `overlap >= 0.3` -> **定位不精**：盖住了目标，但框不够准（IoU < 0.5）
* 否则             -> **指错目标**：基本指到别处去了
* 没有可解析预测    -> **无预测**

这两类的成因完全不同：前者是精度问题（分辨率/坐标回归），后者是判别问题
（在相似候选里选错）。混在一起看会得出错误的改进方向。

## 用法

    python tools/report_failure_modes.py --gt <city412_gt.json> \
        --arm baseline=<predictions.jsonl> --arm trained=<predictions.jsonl>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.verify_acc05_dev import load_gt  # noqa: E402

# 尺寸分桶（归一化面积）。与建立基线时使用的口径保持一致。
BUCKETS = [(0.0, 0.002, "极小 <0.2%"), (0.002, 0.01, "小 0.2-1%"),
           (0.01, 0.05, "中 1-5%"), (0.05, 1.01, "大 >5%")]
NEAR_MISS_OVERLAP = 0.3


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(pred: list[float], ref: list[float]) -> float:
    """交集占**较小框**的比例，用来区分"盖住了但不准"与"指到别处"。"""
    x1, y1 = max(pred[0], ref[0]), max(pred[1], ref[1])
    x2, y2 = min(pred[2], ref[2]), min(pred[3], ref[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    smaller = min((pred[2] - pred[0]) * (pred[3] - pred[1]),
                  (ref[2] - ref[0]) * (ref[3] - ref[1]))
    return inter / smaller if smaller > 0 else 0.0


def load_predictions(path: pathlib.Path) -> dict[str, list[float] | None]:
    """读预测。两种字段名都要认：探针/HF 用 `prediction`，vLLM 提交路径用 `bbox`。"""
    out: dict[str, list[float] | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row["prediction"] if "prediction" in row else row.get("bbox")
        out[row["id"]] = [float(v) for v in value] if value else None
    return out


def classify(pred: list[float] | None, ref: list[float]) -> str:
    if pred is None:
        return "无预测"
    if iou_xyxy(pred, ref) >= 0.5:
        return "命中"
    return "定位不精" if overlap_ratio(pred, ref) >= NEAR_MISS_OVERLAP else "指错目标"


def bucket_of(ref: list[float]) -> str:
    area = (ref[2] - ref[0]) * (ref[3] - ref[1])
    for low, high, label in BUCKETS:
        if low <= area < high:
            return label
    return BUCKETS[-1][2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=pathlib.Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=predictions.jsonl")
    parser.add_argument("--out-json", type=pathlib.Path)
    args = parser.parse_args()

    gt = load_gt(args.gt)
    results: dict[str, dict[str, dict[str, int]]] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm 需要 NAME=path 形式，收到 {spec!r}")
        predictions = load_predictions(pathlib.Path(path))
        table: dict[str, dict[str, int]] = {}
        for sample_id, ref in gt.items():
            row = table.setdefault(bucket_of(ref), {})
            row[classify(predictions.get(sample_id), ref)] = \
                row.get(classify(predictions.get(sample_id), ref), 0) + 1
        results[name] = table

    labels = [label for _, _, label in BUCKETS]
    columns = ["命中", "指错目标", "定位不精", "无预测"]
    for name, table in results.items():
        print(f"\n=== {name} ===")
        header = f"{'尺寸桶':>14s} {'题数':>5s}" + "".join(f"{c:>10s}" for c in columns)
        print(header)
        for label in labels:
            row = table.get(label)
            if not row:
                continue
            total = sum(row.values())
            cells = "".join(f"{row.get(c, 0):>10d}" for c in columns)
            print(f"{label:>14s} {total:>5d}{cells}")
        total_all = sum(sum(row.values()) for row in table.values())
        hits_all = sum(row.get("命中", 0) for row in table.values())
        print(f"{'合计':>14s} {total_all:>5d}{hits_all:>10d}  "
              f"ACC {hits_all / total_all:.4f}")

    # 判据：总体涨但小目标跌 = 用易题换难题
    if len(results) >= 2:
        names = list(results)
        base, cand = results[names[0]], results[names[1]]
        tiny = labels[0]
        base_tiny = base.get(tiny, {}).get("命中", 0)
        cand_tiny = cand.get(tiny, {}).get("命中", 0)
        base_all = sum(row.get("命中", 0) for row in base.values())
        cand_all = sum(row.get("命中", 0) for row in cand.values())
        print(f"\n=== 判决（{names[0]} -> {names[1]}）===")
        print(f"  总体命中   {base_all} -> {cand_all}  ({cand_all - base_all:+d})")
        print(f"  极小桶命中 {base_tiny} -> {cand_tiny}  ({cand_tiny - base_tiny:+d})")
        if cand_all > base_all and cand_tiny < base_tiny:
            print("  ⚠️ 总体涨但极小目标桶退化 —— 判为'用易题换难题'，不接受")
        elif cand_all > base_all and cand_tiny >= base_tiny:
            print("  ✅ 总体提升且极小目标桶未退化")
        else:
            print("  总体未提升")

    if args.out_json:
        args.out_json.write_text(
            json.dumps({"buckets": results, "labels": labels, "columns": columns},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写出 {args.out_json}")


if __name__ == "__main__":
    main()
