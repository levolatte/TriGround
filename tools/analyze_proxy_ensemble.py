"""用 gpt-6-astra 的银标准标注，测算跨运行选择策略能吃到多少 oracle 收益。

**标注不是官方真值**（包内 summary 写明 human_verified=false、source=gpt-6-astra）。
它对我们模型有 +1.7pp 的系统性偏好，且会把 5 题量级的差异判反，
所以只用于比较 >1pp 的差异，不能用来预测官方分数。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

RESULTS = Path(r"F:\AIC\results")
PROXY = RESULTS / "semifinal_astra_annotations_20261031" / "queries_queries.json"

RUNS = {
    "A_bf16_2x": (RESULTS / "submission_semifinal_arm2_k20_20261028" / "predictions.jsonl", 0.8267),
    "B_corr1600": (RESULTS / "submission_semifinal_correction1600_20261029" / "predictions.json", 0.8023),
    "C_arm2_nf4": (RESULTS / "submission_semifinal_arm2_nf4_20261030" / "predictions.json", 0.8014),
}
BEST = "A_bf16_2x"


def read_any(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return {row["id"]: row for row in
                (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_box(row: dict) -> list[float] | None:
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


def iou(first: list[float] | None, second: list[float] | None) -> float:
    if not first or not second:
        return 0.0
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = ((first[2] - first[0]) * (first[3] - first[1])
             + (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / union if union > 0 else 0.0


def medoid(boxes: list[list[float] | None]) -> list[float] | None:
    """选与其他候选平均 IoU 最高的那个——不需要真值。"""
    valid = [box for box in boxes if box]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    scores = [sum(iou(candidate, other) for other in valid if other is not candidate) / (len(valid) - 1)
              for candidate in valid]
    return valid[scores.index(max(scores))]


def main() -> None:
    proxy = json.loads(PROXY.read_text(encoding="utf-8"))
    truth = {key: value["bbox"] for key, value in proxy.items()}

    predicted = {}
    for name, (path, _official) in RUNS.items():
        rows = read_any(path)
        predicted[name] = {qid: extract_box(row) for qid, row in rows.items()}

    ids = sorted(set(truth) & set(predicted["A_bf16_2x"])
                 & set(predicted["B_corr1600"]) & set(predicted["C_arm2_nf4"]))
    total = len(ids)
    print(f"共同题数 {total}\n")

    def hit(qid: str, box: list[float] | None) -> bool:
        return iou(box, truth[qid]) >= 0.5

    single = {name: sum(1 for q in ids if hit(q, predicted[name][q])) for name in predicted}
    print("=== 单份命中 ===")
    for name, count in single.items():
        print(f"  {name:<12} {count:>5}  {count / total:.4f}   官方 {RUNS[name][1]:.4f}")

    oracle = sum(1 for q in ids if any(hit(q, predicted[n][q]) for n in predicted))
    best_single = max(single.values())
    print(f"\n=== Oracle（每题挑对的那份，需要知道答案，不可实现）===")
    print(f"  {oracle:>5}  {oracle / total:.4f}   +{oracle - best_single} 题 "
          f"({(oracle - best_single) / total * 100:+.2f}pp)")

    print("\n=== 可实现的无监督策略 ===")
    # 1. 多数票：任意两份 IoU>=0.5 认同就采纳
    def majority(qid: str) -> list[float] | None:
        names = list(predicted)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                first, second = predicted[names[i]][qid], predicted[names[j]][qid]
                if first and second and iou(first, second) >= 0.5:
                    return first
        return predicted[BEST][qid]

    # 2. Medoid：三份里与其余两份平均 IoU 最高
    def medoid_pick(qid: str) -> list[float] | None:
        return medoid([predicted[n][qid] for n in predicted])

    # 3. 只在分歧题上用 medoid，一致时用最强单份
    def medoid_on_disagree(qid: str) -> list[float] | None:
        boxes = [predicted[n][qid] for n in predicted]
        agree = any(boxes[i] and boxes[j] and iou(boxes[i], boxes[j]) >= 0.5
                    for i in range(3) for j in range(i + 1, 3))
        return medoid(boxes) if not agree else predicted[BEST][qid]

    for name, picker in (("多数票", majority), ("三份 medoid", medoid_pick),
                         ("分歧时 medoid", medoid_on_disagree)):
        count = sum(1 for q in ids if hit(q, picker(q)))
        print(f"  {name:<14} {count:>5}  {count / total:.4f}   "
              f"相对 {BEST} {count - single[BEST]:+d} 题")

    print("\n=== 机会在哪：各份互补性 ===")
    for name in predicted:
        if name == BEST:
            continue
        rescued = sum(1 for q in ids if not hit(q, predicted[BEST][q]) and hit(q, predicted[name][q]))
        lost = sum(1 for q in ids if hit(q, predicted[BEST][q]) and not hit(q, predicted[name][q]))
        print(f"  {name:<12} 能救 {rescued:>4} 题，会损 {lost:>4} 题，净 {rescued - lost:+d}")

    print("\n=== 分歧题里两份都错的占比（决定重排序天花板）===")
    both_wrong = sum(1 for q in ids
                     if not hit(q, predicted["A_bf16_2x"][q]) and not hit(q, predicted["B_corr1600"][q])
                     and not hit(q, predicted["C_arm2_nf4"][q]))
    print(f"  三份全错 {both_wrong} 题 / {total} = {both_wrong / total:.1%}"
          f"  ← 这部分重排序无能为力")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
