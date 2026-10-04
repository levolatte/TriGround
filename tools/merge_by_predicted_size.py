"""按**基线预测框的大小**分路，把两套预测合并成一份提交。

## 为什么需要它

超原生放大实验（`--upscale`）在 City412 上的分桶结果是：

    尺寸桶        题数   k=1.0（基线）  k=1.25（放大）
    极小 <0.2%     143        75             82      <- 放大明显更好
    小 0.2-1%      147       124            122
    中 1-5%         80        71             70
    大 >5%          42        42             42
    总数           412       312            316

**放大只对极小目标有益**，其他桶持平或略降。所以全局放大把极小桶的 +7 抵消掉了大半。

## 规则

逐题看**基线**预测框的归一化面积：小于阈值 → 取放大版；否则取基线版。
实测（City412，阈值 0.005-0.008 是一个宽阔平台）：

    纯基线 312  ->  纯放大 316  ->  分路 **320~321**

上界参考（每次都能选对的那个）是 328，所以分路拿到了约 56% 的可分空间。

## 为什么这个结构是安全的

它**从不无条件采用放大版**：只在模型自己认为目标小的那些题上替换。
若官方测试集的目标普遍偏大（实测代理 RGBDT500 的极小目标只占 11.5%，
远低于 City412 的 34.7%），分路会**自动退化为基线**——而基线正是已得 0.7870 的配置。
下行风险被结构性地限制住。

## 用法

    python tools/merge_by_predicted_size.py \
        --base <k1.0>/predictions.jsonl --upscaled <k1.25>/predictions.jsonl \
        --queries <official>/queries/queries.json --out-dir <merged 目录>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import load_prediction_rows  # noqa: E402
from tools.predict_native_submission import (  # noqa: E402
    load_queries,
    package_submission,
)

# 平台中心。0.005 给 321、0.006~0.008 给 320——差 1 题不值得取单点峰值，
# 取平台内更靠中间的值更不容易是开发集过拟合。
DEFAULT_AREA_THRESHOLD = 0.006


def box_area(bbox: list[float]) -> float:
    """归一化 xyxy 的面积。

    用归一化坐标而非像素：两套预测都归一到 0-1000，面积口径自动一致，
    不会因为放大倍数不同而算错。
    """
    return abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))


def index_rows(rows: list[dict[str, Any]], path: pathlib.Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}: 有记录缺少 id")
        if "bbox" not in row:
            raise ValueError(f"{sample_id}: {path} 的记录缺少 bbox")
        if sample_id in indexed:
            raise ValueError(f"{path}: id 重复 {sample_id}")
        indexed[sample_id] = row
    return indexed


def route(
    source_ids: list[str],
    base: dict[str, dict[str, Any]],
    upscaled: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """逐题分路。返回 (选中记录, 计数)。"""
    chosen: dict[str, dict[str, Any]] = {}
    stats = {"base": 0, "upscaled": 0, "missing_in_upscaled": 0}
    for sample_id in source_ids:
        base_row = base.get(sample_id)
        if base_row is None:
            raise ValueError(f"{sample_id}: 基线预测缺失——分路必须两套都完整")
        upscaled_row = upscaled.get(sample_id)
        if upscaled_row is None:
            # 放大版缺这一题：回落基线并计数。**不静默丢题**，因为丢题会让提交少记录。
            chosen[sample_id] = base_row
            stats["missing_in_upscaled"] += 1
            stats["base"] += 1
            continue
        if box_area(base_row["bbox"]) < threshold:
            chosen[sample_id] = upscaled_row
            stats["upscaled"] += 1
        else:
            chosen[sample_id] = base_row
            stats["base"] += 1
    return chosen, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=pathlib.Path, required=True,
                        help="基线（k=1.0）的 predictions.jsonl")
    parser.add_argument("--upscaled", type=pathlib.Path, required=True,
                        help="放大（k=1.25）的 predictions.jsonl")
    parser.add_argument("--queries", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expected-query-count", type=int, default=5690)
    parser.add_argument("--area-threshold", type=float, default=DEFAULT_AREA_THRESHOLD,
                        help="基线预测框的归一化面积低于此值时取放大版。"
                             "0=全部走基线；>=1=全部走放大版（两端都是恒等变换，可用于自检）。")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    source = load_queries(args.queries, args.expected_query_count)
    base = index_rows(load_prediction_rows(args.base), args.base)
    upscaled = index_rows(load_prediction_rows(args.upscaled), args.upscaled)
    print(f"基线 {len(base)} 条 / 放大 {len(upscaled)} 条 / 官方查询 {len(source)} 条")

    chosen, stats = route(list(source), base, upscaled, args.area_threshold)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = args.out_dir / "predictions.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for sample_id in source:
            handle.write(json.dumps(chosen[sample_id], ensure_ascii=False) + "\n")

    json_path, zip_path = package_submission(source, chosen, args.out_dir)

    summary = {
        "area_threshold": args.area_threshold,
        "base": str(args.base), "upscaled": str(args.upscaled),
        "queries": str(args.queries),
        "total": len(source),
        "took_upscaled": stats["upscaled"],
        "took_base": stats["base"],
        "missing_in_upscaled": stats["missing_in_upscaled"],
        "predictions_json": str(json_path), "submission_zip": str(zip_path),
    }
    (args.out_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
