"""汇总探针结果并给出判决表。

用法（云端）：
    python scripts/summarize_probe.py <probe_root> [--gt <city412_gt.json>]

对每个 `pixels_*/summary.json` 打印命中、ACC、mIoU、解析失败、平均输入 token 与峰值显存；
并（若提供 --gt）对每个档位跑一次准确率与误差分类，输出 `other_instance`（错误实例数）等
决定性的分桶计数——总分被噪声支配，分桶才是判据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_acc05_dev import box_is_valid, group_of, iou_xyxy, load_gt  # noqa: E402

BASELINE_OTHER_INSTANCE = 30


def summarize_setting(summary: dict) -> str:
    return (f"{summary['max_pixels']:>12d} {summary['hits']:6d} {summary['acc_0.5']:9.4f} "
            f"{summary['mean_iou']:8.4f} {summary['parse_failures']:11d} "
            f"{summary.get('mean_input_tokens', 0):12.1f} "
            f"{summary['gpu_peak_allocated_bytes'] / 2**30:9.2f}")


def taxonomy_for(predictions: Path, gt: dict) -> dict:
    """分桶统计。

    刻意把"错误实例"分成两档，与项目既有的 30 题口径保持一致：
      other_instance  —— 预测框与同图另一实例的 IoU >= 0.5（确定选错对象）
      suspected       —— 与同图另一实例的 IoU 在 0.25~0.5（疑似选错）
    合并两档会与历史数字（30）不可比，因此分开。
    """
    by_group: dict[str, list[str]] = {}
    for sample_id in gt:
        by_group.setdefault(group_of(sample_id), []).append(sample_id)
    buckets = {"other_instance": 0, "suspected": 0, "near_miss": 0,
               "drift": 0, "gross_miss": 0, "invalid": 0}
    hits = 0
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row["id"]
        target = gt[sample_id]
        prediction = row.get("prediction")
        if not box_is_valid(prediction):
            buckets["invalid"] += 1
            continue
        own = iou_xyxy([float(v) for v in prediction], target)
        if own >= 0.5:
            hits += 1
            continue
        peers = [p for p in by_group[group_of(sample_id)] if p != sample_id]
        peer_best = max((iou_xyxy([float(v) for v in prediction], gt[p]) for p in peers), default=0.0)
        if peer_best >= 0.5 and peer_best > own:
            buckets["other_instance"] += 1
        elif peer_best >= 0.25:
            buckets["suspected"] += 1
        elif own >= 0.4:
            buckets["near_miss"] += 1
        elif own >= 0.25:
            buckets["drift"] += 1
        else:
            buckets["gross_miss"] += 1
    return {"hits": hits, **buckets}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_root", type=Path)
    parser.add_argument("--gt", type=Path)
    args = parser.parse_args()

    summaries = sorted(args.probe_root.glob("pixels_*/summary.json"),
                       key=lambda path: int(path.parent.name.split("_")[1]))
    if not summaries:
        raise SystemExit(f"没有找到任何 pixels_*/summary.json：{args.probe_root}")

    print(f"{'max_pixels':>12s} {'hits':>6s} {'ACC@0.5':>9s} {'mIoU':>8s} "
          f"{'parse_fail':>11s} {'mean_in_tok':>12s} {'peak_GiB':>9s}")
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    for summary in loaded:
        print(summarize_setting(summary))

    if args.gt is None:
        return
    gt = load_gt(args.gt)
    print(f"\n{'max_pixels':>12s} {'hits':>6s} {'other_inst':>11s} {'suspected':>10s} "
          f"{'near_miss':>10s} {'drift':>7s} {'gross':>7s} {'invalid':>8s}")
    rows = []
    for summary, path in zip(loaded, summaries):
        predictions = path.parent / "predictions.jsonl"
        if not predictions.exists():
            print(f"{summary['max_pixels']:12d}   (缺少 predictions.jsonl)")
            continue
        counts = taxonomy_for(predictions, gt)
        rows.append((summary["max_pixels"], counts))
        print(f"{summary['max_pixels']:12d} {counts['hits']:6d} {counts['other_instance']:11d} "
              f"{counts['suspected']:10d} {counts['near_miss']:10d} {counts['drift']:7d} "
              f"{counts['gross_miss']:7d} {counts['invalid']:8d}")

    if rows:
        base_pixels, base = rows[0]
        print(f"\n以本机最低档 {base_pixels} 为同硬件基线（hits={base['hits']}）逐档对比：")
        for pixels, counts in rows[1:]:
            print(f"  {pixels:>9d}: hits {counts['hits'] - base['hits']:+d}, "
                  f"other_instance {counts['other_instance'] - base['other_instance']:+d}, "
                  f"suspected {counts['suspected'] - base['suspected']:+d}, "
                  f"near_miss {counts['near_miss'] - base['near_miss']:+d}")
    print("\n注意：历史 A=296 是在 4090+torch2.8 上测的；换到本机后同一权重实测约 293"
          "（prompt/token/grid 完全一致，差异是 <0.001 的坐标抖动）。因此必须用本机最低档做基线。")
    print(f"历史口径参考：other_instance（严格档）= {BASELINE_OTHER_INSTANCE}")


if __name__ == "__main__":
    main()
