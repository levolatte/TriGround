"""用 gpt-6-astra 银标准给一份提交包打分，并与三个已知官方分数的包对比定位。

**这个脚本不预测官方分数。** 银标准的来源是另一个模型（gpt-6-astra）的生成标注，
包内 summary 写明 `human_verified: false`。实测它对我们的模型有约 +1.7pp 的系统偏好，
且会把官方差 0.0009（5 题）的两轮判反方向。

所以它的正确用法是：**把新包与已知官方分数的包放在同一个银标准上排位**，
用相对位置判断新包属于哪一档，而不是看绝对数值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROXY = Path(r"F:\AIC\results\semifinal_astra_annotations_20261031\queries_queries.json")

# 三个已知官方分数的包，作为银标准上的标尺。
REFERENCE = [
    ("arm2_k20  BF16 + 2x  ", 0.8267,
     Path(r"F:\AIC\results\submission_semifinal_arm2_k20_20261028\predictions.jsonl")),
    ("arm2_nf4  NF4 + 低分辨率", 0.8014,
     Path(r"F:\AIC\results\submission_semifinal_arm2_nf4_20261030\predictions.json")),
    ("corr1600  NF4 + 低分辨率", 0.8023,
     Path(r"F:\AIC\results\submission_semifinal_correction1600_20261029\predictions.json")),
]


def read_any(path: Path) -> dict:
    if path.suffix == ".jsonl":
        return {row["id"]: row for row in
                (json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())}
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def score(predictions: dict, truth: dict, ids: list[str]) -> tuple[int, int]:
    hits = sum(1 for q in ids if iou(extract_box(predictions.get(q, {})), truth[q]) >= 0.5)
    return hits, len(ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    proxy = json.loads(PROXY.read_text(encoding="utf-8"))
    truth = {key: value["bbox"] for key, value in proxy.items()}

    print(f"银标准 {len(truth)} 题（源自 gpt-6-astra，human_verified=false）\n")
    rows = []
    for name, official, path in REFERENCE:
        if not path.is_file():
            print(f"  跳过（缺文件）: {name}")
            continue
        ids = sorted(set(truth) & set(read_any(path)))
        hits, total = score(read_any(path), truth, ids)
        rows.append((name, official, hits, total, hits / total))

    ids = sorted(set(truth) & set(read_any(args.candidate)))
    hits, total = score(read_any(args.candidate), truth, ids)
    rows.append(("**本次**  NF4 + 2x  ", None, hits, total, hits / total))

    print(f"{'包':<24} {'官方分':>8}  {'银标准命中':>12}  银标准准确率")
    print("-" * 68)
    for name, official, hits, total, acc in rows:
        official_text = f"{official:.4f}" if official is not None else "无"
        print(f"{name:<24} {official_text:>8}  {hits:>5}/{total:<6}  {acc:.4f}")

    print("\n判读：")
    best = max((r for r in rows if r[1] is not None), key=lambda r: r[4])
    candidate = rows[-1]
    if candidate[4] >= best[4] - 0.005:
        print("  与已知最好包同档或更好 → 掉分可归因于分辨率，量化代价很小")
    else:
        low = min((r for r in rows if r[1] is not None and r[4] < best[4] - 0.01),
                  key=lambda r: r[4], default=None)
        if low is not None and candidate[4] <= low[4] + 0.005:
            print("  与 NF4 低分辨率包同档 → NF4 本身有代价，与分辨率无关")
        else:
            print("  介于两者之间 → 分辨率与量化各有贡献")

    if args.out_json:
        args.out_json.write_text(json.dumps(
            [{"name": n, "official": o, "hits": h, "total": t, "proxy_acc": round(a, 4)}
             for n, o, h, t, a in rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已写出 {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
