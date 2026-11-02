"""在 500 题测试集切片上比较新模型与 0.8267 那包——回答"测试分布上有没有提升"。

为什么需要这一步：City378 是城市场景开发集，而测试集与 RGBDT500 同源、
新训练数据又主要是 RGBDT500。开发集很可能低估新数据的收益。
这一步直接在测试分布上量，用 gpt-6-astra 的银标准做裁判，并且**配对比较**
（同一批题、同一个裁判），把裁判自身的偏差消掉。

切片选取是确定性的（固定随机种子），这样两个模型评的是完全相同的 500 题。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

RESULTS = Path(r"F:\AIC\results")
PROXY = RESULTS / "semifinal_astra_annotations_20261031" / "queries_queries.json"
OFFICIAL_BEST = RESULTS / "submission_semifinal_arm2_k20_20261028" / "predictions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="输出切片清单与评测用 dataset（绝对路径，云端可读）")
    parser.add_argument("--candidate", type=Path, default=None,
                        help="待评的新模型 predictions.jsonl（未给则只生成切片）")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    proxy = json.loads(PROXY.read_text(encoding="utf-8"))
    ids = sorted(proxy)
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(ids, min(args.slice_size, len(ids))))
    print(f"代理总题数 {len(ids)}，切片 {len(picked)} 题（seed={args.seed}）", flush=True)

    # 评测脚本要的 dataset 格式：id + images + messages + solution
    # 这里只写 id 与相对图像路径，实际评测直接复用原有的 unlabeled.jsonl 更稳妥。
    (args.out_dir / "slice_ids.json").write_text(
        json.dumps(picked, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

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

    truth = {key: proxy[key]["bbox"] for key in picked}
    best = read_any(OFFICIAL_BEST)
    best_hits = sum(1 for q in picked if iou(extract_box(best.get(q, {})), truth[q]) >= 0.5)
    print(f"参照：0.8267 那包在这 500 题上命中 {best_hits}/{len(picked)} = {best_hits / len(picked):.4f}",
          flush=True)

    summary = {"slice_size": len(picked), "seed": args.seed,
               "reference_arm2_k20_hits": best_hits,
               "reference_arm2_k20_acc": round(best_hits / len(picked), 4)}

    if args.candidate and args.candidate.is_file():
        candidate = read_any(args.candidate)
        missing = [q for q in picked if q not in candidate]
        if missing:
            print(f"WARNING: 候选缺 {len(missing)} 题（推理未完成到该切片）", flush=True)
        common = [q for q in picked if q in candidate]
        cand_hits = sum(1 for q in common if iou(extract_box(candidate[q]), truth[q]) >= 0.5)
        base_on_common = sum(1 for q in common if iou(extract_box(best.get(q, {})), truth[q]) >= 0.5)
        print(f"共同题数 {len(common)}")
        print(f"  新模型     {cand_hits}/{len(common)} = {cand_hits / len(common):.4f}")
        print(f"  0.8267 包  {base_on_common}/{len(common)} = {base_on_common / len(common):.4f}")
        # 配对：只看不一致的题
        only_new = sum(1 for q in common
                       if iou(extract_box(candidate[q]), truth[q]) >= 0.5
                       and iou(extract_box(best.get(q, {})), truth[q]) < 0.5)
        only_best = sum(1 for q in common
                        if iou(extract_box(candidate[q]), truth[q]) < 0.5
                        and iou(extract_box(best.get(q, {})), truth[q]) >= 0.5)
        print(f"  配对：新模型独赢 {only_new} 题，0.8267 独赢 {only_best} 题，净 {only_new - only_best:+d}")
        # 净差的标准误（McNemar 近似）
        discordant = only_new + only_best
        if discordant:
            import math
            se = math.sqrt(discordant) / len(common)
            print(f"  净差 {(only_new - only_best) / len(common) * 100:+.2f}pp，标准误约 {se * 100:.2f}pp")
            print(f"  **{'显著' if abs(only_new - only_best) > 1.96 * math.sqrt(discordant) else '不显著'}**"
                  f"（|净差| {'>' if abs(only_new - only_best) > 1.96 * math.sqrt(discordant) else '<='} 1.96√不一致数）")
        summary.update({"candidate_hits": cand_hits, "common": len(common),
                        "only_new": only_new, "only_best": only_best,
                        "net": only_new - only_best})

    (args.out_dir / "slice_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
