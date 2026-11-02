"""把新包与 0.8267 那包在银标准上做逐题配对比较（比比较绝对分数更准）。

绝对分数的比较会带上银标准对不同配置的偏差差异；配对比较只看两包在同一批题上的
输赢分布，把银标准自身的难度与偏好消掉。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

PROXY = Path(r"F:\AIC\results\semifinal_astra_annotations_20261031\queries_queries.json")
BEST = Path(r"F:\AIC\results\submission_semifinal_arm2_k20_20261028\predictions.jsonl")
NEW = Path(r"F:\AIC\results\submission_semifinal_correction1600_k20_nf4_20261030\predictions.json")


def read_any(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return {row["id"]: row for row in
                (json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def box(row: dict) -> list[float] | None:
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
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def main() -> None:
    truth = {k: v["bbox"] for k, v in
             json.loads(PROXY.read_text(encoding="utf-8")).items()}
    old, new = read_any(BEST), read_any(NEW)
    ids = sorted(set(truth) & set(old) & set(new))
    print(f"共同题数 {len(ids)}\n")

    def hit(preds, q):
        return iou(box(preds.get(q, {})), truth[q]) >= 0.5

    old_hits = sum(1 for q in ids if hit(old, q))
    new_hits = sum(1 for q in ids if hit(new, q))
    only_new = sum(1 for q in ids if hit(new, q) and not hit(old, q))
    only_old = sum(1 for q in ids if hit(old, q) and not hit(new, q))

    print(f"  0.8267 包（BF16 2x）  {old_hits}/{len(ids)} = {old_hits/len(ids):.4f}")
    print(f"  本次包（NF4  2x）     {new_hits}/{len(ids)} = {new_hits/len(ids):.4f}")
    print(f"\n  配对：本次独赢 {only_new} 题，0.8267 独赢 {only_old} 题，净 {only_new-only_old:+d}")

    discordant = only_new + only_old
    if discordant:
        # McNemar 精确二项检验（双侧）
        k = min(only_new, only_old)
        # 正态近似给出标准误；精确 p 值用二项分布
        se = math.sqrt(discordant) / len(ids)
        diff = (only_new - only_old) / len(ids)
        # 正态近似 p
        z = (only_new - only_old) / math.sqrt(discordant) if discordant else 0.0
        from math import erfc, sqrt
        p_norm = erfc(abs(z) / sqrt(2)) if discordant else 1.0
        print(f"  净差 {diff*100:+.2f}pp，标准误约 {se*100:.2f}pp，z = {z:.2f}，p ≈ {p_norm:.3f}")
        print(f"  **{'显著' if p_norm < 0.05 else '不显著'}**（不一致 {discordant} 题，"
              f"需净差超过 {1.96*math.sqrt(discordant):.0f} 题才显著）")

    # 分布：本次包在“0.8267 那包失分”的题里救回多少
    old_miss = [q for q in ids if not hit(old, q)]
    rescued = sum(1 for q in old_miss if hit(new, q))
    print(f"\n  0.8267 包失分 {len(old_miss)} 题，本次救回其中 {rescued} 题")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
