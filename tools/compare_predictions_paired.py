"""在指定题集上对两份预测做银标准配对比较。

比比较绝对分数更准：配对只看同一批题上的输赢分布，把银标准自身的难度与偏好消掉。

用法（比较兄弟上下文切片与 0.8267 那包在**同一批题**上的表现）：

    python tools/compare_predictions_paired.py \
        --baseline results/submission_semifinal_arm2_k20_20261028/predictions.json \
        --candidate .work/sib_slice_predictions.jsonl \
        --ids .work/sib_slice_400img.json

`--ids` 给一个 queries.json 结构的文件即可（只用到它的键），这样切片题集与两个包
的交集由它唯一确定，不依赖各自的覆盖范围。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

PROXY = Path(r"F:\AIC\results\semifinal_astra_annotations_20261031\queries_queries.json")


def read_any(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return {row["id"]: row for row in
                (json.loads(line) for line in
                 path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def box_of(row: dict) -> list[float] | None:
    for key in ("bbox", "solution", "bbox_2d"):
        value = row.get(key)
        if isinstance(value, list) and len(value) == 4:
            return value
    predictions = row.get("predictions")
    if predictions:
        value = predictions[0].get("bbox")
        if isinstance(value, list) and len(value) == 4:
            return value
    return None


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True,
                        help="queries.json 结构的文件，取它的键作为题集")
    parser.add_argument("--label", default="候选")
    args = parser.parse_args()

    truth = {k: v["bbox"] for k, v in
             json.loads(PROXY.read_text(encoding="utf-8")).items()}
    # `--ids` 接受三种形式：queries.json 那样的对象（取键）、id 列表，
    # 或 predictions.jsonl（取每行 id）——后者最顺手，直接拿候选预测文件当题集。
    raw_ids = args.ids.read_text(encoding="utf-8-sig")
    try:
        parsed = json.loads(raw_ids)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        wanted = set(parsed)
    else:
        wanted = {row["id"] for row in
                  (json.loads(line) for line in raw_ids.splitlines() if line.strip())}
    base, cand = read_any(args.baseline), read_any(args.candidate)

    ids = sorted(wanted & set(truth) & set(base) & set(cand))
    missing_base = len(wanted - set(base))
    missing_cand = len(wanted - set(cand))
    print(f"题集 {len(wanted)}  可比 {len(ids)}"
          f"  （基线缺 {missing_base}，候选缺 {missing_cand}）\n")
    if not ids:
        return 1

    def hit(preds, q):
        return iou(box_of(preds.get(q, {})), truth[q]) >= 0.5

    base_hits = sum(1 for q in ids if hit(base, q))
    cand_hits = sum(1 for q in ids if hit(cand, q))
    only_cand = sum(1 for q in ids if hit(cand, q) and not hit(base, q))
    only_base = sum(1 for q in ids if hit(base, q) and not hit(cand, q))

    print(f"  基线    {base_hits}/{len(ids)} = {base_hits / len(ids):.4f}")
    print(f"  {args.label}  {cand_hits}/{len(ids)} = {cand_hits / len(ids):.4f}")
    print(f"\n  配对：{args.label}独赢 {only_cand}，基线独赢 {only_base}，净 {only_cand - only_base:+d}")

    discordant = only_cand + only_base
    if discordant:
        z = (only_cand - only_base) / math.sqrt(discordant)
        p = math.erfc(abs(z) / math.sqrt(2))
        se = math.sqrt(discordant) / len(ids)
        print(f"  净差 {(only_cand - only_base) / len(ids) * 100:+.2f}pp，"
              f"标准误约 {se * 100:.2f}pp，z = {z:.2f}，p ≈ {p:.4f}")
        need = 1.96 * math.sqrt(discordant)
        print(f"  **{'显著' if p < 0.05 else '不显著'}**"
              f"（不一致 {discordant} 题，需净差超过 {need:.0f} 题）")
        # 全量外推：切片若 1866 题，5,691 是 3.05 倍
        scale = 5691 / len(ids)
        print(f"\n  外推到全量 5,691 题：约 {(only_cand - only_base) * scale:+.0f} 题"
              f"（{(only_cand - only_base) / len(ids) * 100:+.2f}pp）")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
