"""逐题比对两个 predictions.jsonl，用于加速路径的等价性闸门。

官方测试集没有真值，因此不能用 `verify_acc05_dev.py`（它需要 GT 算 ACC）。
这里做的是**两条推理路径之间**的比对：同一个模型、同一个 prompt，只换推理引擎。

判据（与计划一致）：
  * 一致率 >= 99.5%（即 200 条里最多 1 条不同）
  * 不一致的条目要逐条列出，人工判定是否为 IoU 卡 0.5 的边界抖动
  * 兜底（fallback）数量差异也要报出来——兜底变多意味着某条路径在丢答案

用法：
    python scripts/compare_predictions.py <a.jsonl> <b.jsonl> [--label-a A] [--label-b B]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_acc05_dev import iou_xyxy  # noqa: E402

CONSISTENCY_FLOOR = 0.995


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def box_of(row: dict) -> list[float] | None:
    value = row.get("bbox", row.get("prediction"))
    return [float(v) for v in value] if isinstance(value, list) and len(value) == 4 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="坐标容差；0 表示要求逐位相同")
    args = parser.parse_args()

    left, right = load(args.a), load(args.b)
    shared = sorted(set(left) & set(right))
    print(f"{args.label_a}: {len(left)} 条   {args.label_b}: {len(right)} 条   可比: {len(shared)}")

    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    if only_left or only_right:
        print(f"仅在 {args.label_a}: {len(only_left)}   仅在 {args.label_b}: {len(only_right)}")

    identical = 0
    differing: list[tuple[str, list[float], list[float], float]] = []
    for sample_id in shared:
        box_a, box_b = box_of(left[sample_id]), box_of(right[sample_id])
        # 两边都没解析出框时是**一致**的（都走了兜底），不能算差异。
        # 只有一边有框才算真实差异——那意味着某条路径在丢答案。
        if box_a is None and box_b is None:
            identical += 1
            continue
        if box_a is None or box_b is None:
            differing.append((sample_id, box_a or [], box_b or [], 0.0))
            continue
        gap = max(abs(x - y) for x, y in zip(box_a, box_b))
        if gap <= args.tolerance:
            identical += 1
        else:
            differing.append((sample_id, box_a, box_b, iou_xyxy(box_a, box_b)))

    consistency = identical / len(shared) if shared else 0.0
    print(f"逐位一致: {identical}/{len(shared)} = {consistency * 100:.2f}%")

    fallback_a = sum(bool(left[s].get("fallback")) for s in shared)
    fallback_b = sum(bool(right[s].get("fallback")) for s in shared)
    print(f"兜底数: {args.label_a} {fallback_a}   {args.label_b} {fallback_b}")

    if differing:
        print(f"\n不一致 {len(differing)} 条（最多列出 20）：")
        for sample_id, box_a, box_b, iou in differing[:20]:
            print(f"  {sample_id[-32:]:34s} {args.label_a}={box_a} {args.label_b}={box_b} "
                  f"两者IoU={iou:.4f}")
        print("\n说明：坐标抖动在 0.001 量级、两者 IoU 接近 1 的属于数值噪声；"
              "若出现 IoU 很低或兜底状态不同，则是真实差异，必须查因。")

    ok = consistency >= CONSISTENCY_FLOOR
    print("\n" + ("GATE2_PASS" if ok else "GATE2_FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
