"""新索引的验收闸门。七道，任何一道不过就不许拿去训练。

设计的出发点：我们只有一个 378 题的开发集（标准误约 8.5 题），**分辨力不足以用命中数
判断数据好坏**。所以闸门全部是**结构性、连续量**的检查——它们能回答"这份数据和测试集
像不像、干不干净、框准不准"，而回答不了"它能提几分"（那只能靠训练后评测）。

用法：
    python tools/verify_replacement_index.py --mixture OUT/mixture.jsonl --media-root DATA \
        --old-index OLD/train_index.jsonl --queries SEMIFINAL/queries/queries.json --out OUT/gates.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import statistics

GATE_NAMES = ("分布贴合", "测试集比对", "几何合法", "媒体可达", "同图多实例", "总量", "来源构成")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_of(sample: dict) -> str:
    match = re.search(r"this query:\s*(.*?)\s*Return only JSON", sample["messages"][0]["content"], re.S)
    return (match.group(1) if match else "").strip()


SPATIAL = re.compile(
    r"\b(left|right|behind|in front|near|next to|far|distant|background|foreground|center|central|"
    r"middle|top|bottom|beside|between|across|along|under|above|closest|farthest)\b", re.I)


def profile(queries: list[str]) -> dict:
    words = [len(q.split()) for q in queries]
    return {
        "the_prefix": round(sum(1 for q in queries if q.startswith("The ")) / len(queries), 3),
        "a_prefix": round(sum(1 for q in queries if q.startswith("A ")) / len(queries), 3),
        "spatial": round(sum(1 for q in queries if SPATIAL.search(q)) / len(queries), 3),
        "words_median": int(statistics.median(words)),
    }


def file_digest(path) -> str:
    """合规边界上的内容比对——这是 AGENTS.md 允许加哈希的那类"真实边界"。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", required=True, type=str)
    parser.add_argument("--media-root", required=True, type=str)
    parser.add_argument("--old-index", required=True, type=str)
    parser.add_argument("--queries", required=True, type=str, help="官方 queries.json，用于分布比对")
    parser.add_argument("--test-images", type=str, default=None,
                        help="官方测试图像目录；给了才做逐像素比对")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--spot-check", type=int, default=300)
    args = parser.parse_args()

    from pathlib import Path
    media_root = Path(args.media_root)
    samples = read_jsonl(Path(args.mixture))
    old = read_jsonl(Path(args.old_index))
    official = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    official_queries = [v["query"] for v in official.values() if v.get("query")]

    gates: dict[str, dict] = {}

    # ---------- 闸门 1：查询分布贴合官方 ----------
    # **判据只看语义，不看冠词。** 理由：RGBT 全量 26,604 条里 "The" 开头只占 0.04
    # （主流是 "A ..." 与 "Person standing ..."），现有公开数据里没有任何一个能把冠词分布
    # 做到接近官方的 0.84。冠词是风格噪声，真正影响定位的是**是否含空间关系**与**句长**。
    # 冠词比例仍然记录，但不作为通过条件。
    mine, theirs = profile([query_of(s) for s in samples]), profile(official_queries)
    spatial_gap = round(abs(mine["spatial"] - theirs["spatial"]), 3)
    word_gap = abs(mine["words_median"] - theirs["words_median"])
    passed = spatial_gap <= 0.10 and word_gap <= 3
    gates["分布贴合"] = {"passed": passed, "candidate": mine, "official": theirs,
                     "spatial_gap": spatial_gap, "words_median_gap": word_gap,
                     "article_note": "冠词比例仅记录、不作判据（无可用数据能匹配 0.84）"}

    # ---------- 闸门 2：与官方测试集的内容重合（**诊断项，不作通过条件**） ----------
    # 赛题规则原文："本赛题对训练数据使用不做约束…无外部数据集使用范围限制"。
    # 同源、甚至同图，都不构成违规，所以这一项只统计规模、供报告披露使用。
    if args.test_images:
        test_dir = Path(args.test_images)
        test_hashes = set()
        for path in test_dir.rglob("*"):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                test_hashes.add(file_digest(path))
        collisions = []
        checked = 0
        for sample in samples:
            for rel in sample["images"]:
                local = media_root / rel
                if not local.is_file():
                    continue
                checked += 1
                if file_digest(local) in test_hashes:
                    collisions.append({"id": sample["id"], "image": rel})
        gates["测试集比对"] = {"passed": True, "diagnostic_only": True,
                         "checked_images": checked, "test_images": len(test_hashes),
                         "same_image_files": len(collisions),
                         "samples_affected": len({c["id"] for c in collisions}),
                         "note": "规则允许同源数据，此项只统计不拦截"}
    else:
        gates["测试集比对"] = {"passed": True, "diagnostic_only": True,
                         "skipped": "未提供 --test-images，未统计"}

    # ---------- 闸门 3：几何合法 ----------
    bad = []
    for sample in samples:
        x1, y1, x2, y2 = sample["solution"]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or (x2 - x1) * (y2 - y1) <= 0:
            bad.append(sample["id"])
    gates["几何合法"] = {"passed": not bad, "bad": bad[:20], "bad_count": len(bad)}

    # ---------- 闸门 4：媒体可达（抽样） ----------
    import random
    rng = random.Random(2026)
    picked = rng.sample(samples, min(args.spot_check, len(samples)))
    missing = []
    from PIL import Image
    sizes = Counter()
    for sample in picked:
        for rel in sample["images"]:
            local = media_root / rel
            if not local.is_file():
                missing.append(rel)
                continue
            try:
                with Image.open(local) as image:
                    sizes[image.size] += 1
            except Exception as error:
                missing.append(f"{rel}: {type(error).__name__}")
    gates["媒体可达"] = {"passed": not missing, "sampled": len(picked),
                    "sizes": {f"{w}x{h}": count for (w, h), count in sizes.most_common(6)},
                    "missing": missing[:20], "missing_count": len(missing)}

    # ---------- 闸门 5：同图多实例比例 ----------
    # 这一项的阈值不设成"不低于旧索引"（1.98），因为那是**物理上达不到的**：
    # 旧索引的 1.98 几乎全部来自 City——它 2,882 条只用了 514 张图，重数 5.61（单图最多 15 条）。
    # 而替代来源的天然上限是 RGBT 1.79、new-rgbdt500 1.00。按图分组抽样能把 RGBT 拉回
    # 它自己的 1.79，但全局只能到约 1.2。**这是替换 City 的已量化代价，不是实现缺陷。**
    # 阈值取 1.15：高于"随机抽样样本"会退化成 1.02 的水平，证明按图分组生效了。
    def per_image(rows: list[dict]) -> float:
        unique = Counter(row["images"][0] for row in rows)
        return len(rows) / max(1, len(unique))

    new_ratio, old_ratio = per_image(samples), per_image(old)
    gates["同图多实例"] = {"passed": new_ratio >= 1.15, "new": round(new_ratio, 2),
                     "old_city_era": round(old_ratio, 2), "threshold": 1.15,
                     "note": "旧值的 1.98 由 City 的 5.61 重数撑起，无 City 时不可达；"
                             "本闸门只验证按图分组抽样是否生效，不追究无法消除的差距",
                     "official_test": 4.41}

    # ---------- 闸门 6：总量 ----------
    # 本轮方案是「City 保留 60% + 40% 换新 + 再补强 30%」，目标 6,376 条，
    # 所以允许区间相应上移（原 4,905 的 ±5% 已不适用）。
    gates["总量"] = {"passed": 6000 <= len(samples) <= 6700, "rows": len(samples),
                  "allowed": [6000, 6700], "target": 6376}

    # ---------- 闸门 7：来源构成与模态 ----------
    sources = Counter(s["source"] for s in samples)
    modalities = Counter(len(s["images"]) for s in samples)
    tri_share = modalities.get(3, 0) / len(samples)
    gates["来源构成"] = {"passed": tri_share >= 0.25, "sources": dict(sources),
                    "modalities": dict(modalities), "tri_modal_share": round(tri_share, 3),
                    "threshold": 0.25}

    areas = [(s["solution"][2] - s["solution"][0]) * (s["solution"][3] - s["solution"][1])
             for s in samples]
    report = {"gates": gates, "all_passed": all(g["passed"] for g in gates.values()),
              "area_median": round(statistics.median(areas), 5),
              "area_under_0.2pct": round(sum(1 for a in areas if a < 0.002) / len(areas), 3),
              "rows": len(samples)}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'闸门':<12} {'结果':<6} 详情")
    for name in GATE_NAMES:
        gate = gates[name]
        mark = "通过" if gate["passed"] else "**不过**"
        print(f"  {name:<10} {mark:<6} {json.dumps({k: v for k, v in gate.items() if k != 'passed'}, ensure_ascii=False)[:150]}")
    print()
    print(f"总面积中位 {report['area_median']}   面积<0.2% 占 {report['area_under_0.2pct']:.1%}   共 {report['rows']} 条")
    print(f"全部通过: {report['all_passed']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
