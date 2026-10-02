"""核验官方测试集：每条查询引用的三模态图像是否都存在、可读、尺寸一致。

动机：`official_test_rematch_20260924` 里 visible 有 2005 个文件，而 infrared / depth
各只有 2000 个 —— 这个不对称必须在跑 5690 条推理之前查清。若某条查询引用了不存在的
红外或深度图，推理会在中途抛错，浪费整轮 GPU 时间。

用法（云端）：
    python scripts/verify_official_test.py --root <测试集根目录>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-images", type=int, default=40,
                        help="抽查可读性与尺寸的图片数量（0 表示全查，较慢）")
    args = parser.parse_args()

    queries_path = args.root / "queries" / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    print(f"查询数: {len(queries)}")

    keys = Counter()
    sizes: Counter = Counter()
    for record in queries.values():
        keys[tuple(sorted(record))] += 1
    print(f"字段组合: {dict(keys)}")

    missing: list[str] = []
    for query_id, record in queries.items():
        for modality in ("visible", "infrared", "depth"):
            relative = record.get(modality)
            if not relative:
                missing.append(f"{query_id}.{modality} 字段为空")
                continue
            if not (args.root / relative).exists():
                missing.append(f"{query_id}.{modality} -> {relative}")
    print(f"缺失引用: {len(missing)}")
    for item in missing[:10]:
        print(f"  {item}")

    # 实际文件数与查询引用数的对比：可以看出多余的图有没有被引用
    referenced: dict[str, set[str]] = {m: set() for m in ("visible", "infrared", "depth")}
    for record in queries.values():
        for modality in referenced:
            if record.get(modality):
                referenced[modality].add(record[modality])
    print("\n模态        查询引用   目录文件数")
    for modality in ("visible", "infrared", "depth"):
        count = len(list((args.root / "Images" / modality).glob("*")))
        print(f"  {modality:<10} {len(referenced[modality]):>8}   {count:>10}")

    # 抽查图像可读性与尺寸，并统计尺寸是否一致
    step = max(1, len(queries) // args.sample_images) if args.sample_images else 1
    checked = 0
    for index, record in enumerate(queries.values()):
        if args.sample_images and index % step:
            continue
        for modality in ("visible", "infrared", "depth"):
            path = args.root / record[modality]
            if not path.exists():
                continue
            try:
                with Image.open(path) as image:
                    sizes[image.size] += 1
            except OSError as error:
                print(f"  无法读取 {path}: {error}")
            checked += 1
        if args.sample_images and checked >= args.sample_images * 3:
            break
    print(f"\n抽查 {checked} 张，尺寸分布: {dict(sizes)}")

    problems = bool(missing)
    print("\n" + ("OFFICIAL_TEST_BROKEN" if problems else "OFFICIAL_TEST_OK"))


if __name__ == "__main__":
    main()
