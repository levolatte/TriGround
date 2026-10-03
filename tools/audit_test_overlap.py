"""审计训练侧图像与官方测试集的内容重叠，产出按帧的剔除清单。

## 为什么需要这个工具

规则严禁把测试集数据用于训练。实测发现训练侧确实混入了测试集帧：
City 有 114 张可见光与测试集**逐字节相同**，队友新传的 RGBDT500_prepared 有
161/992 样本重叠。这些必须从训练数据里剔除。

## 为什么必须按"帧"而不是按"文件"剔除

同一帧在训练侧可能有多个副本：原始深度 `depth/003353.png`（uint16）和
渲染后的 `depth_rgb/003353.png`（RGB）**内容不同、哈希不同**，但来自同一帧。
只删被查出的那个文件，渲染副本会漏网，污染依旧存在。

因此工具输出**帧键（frame key）**集合，由构建器按键剔除该帧的全部模态。

## 为什么必须按内容而不是按文件名

实测文件名匹配只查出 42 张 City 可见光，内容匹配查出 **114 张**——**低估 2.7 倍**。
文件名会因重新编码、重命名而失配，只有内容哈希才是可靠的判据。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Iterable

# 各来源的模态后缀。去掉后才得到"帧键"。
# City：`003353.png` -> `003353`
# RGBDT500_prepared：`train_001_00000020_color.png` -> `train_001_00000020`
# RoboRefIt：`0000000.png` -> `0000000`
MODALITY_SUFFIXES = ("_color", "_depth", "_infrared", "_nir", "_ir", "_rgb", "_visible")


def frame_key(path: pathlib.Path) -> str:
    """把一张图映射到它所属的帧键。

    渲染副本（`depth_rgb/003353.png`）与原始图（`depth/003353.png`）的 stem 相同，
    因此天然归到同一帧——这正是我们要的：任一模态被污染，整帧剔除。
    """
    stem = path.stem
    for suffix in MODALITY_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def md5_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def iter_images(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        yield from root.rglob(pattern)


def hash_tree(root: pathlib.Path, label: str) -> dict[str, list[pathlib.Path]]:
    """返回 {内容哈希: [路径]}，用于反向查重叠。"""
    table: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    count = 0
    for image in iter_images(root):
        table[md5_file(image)].append(image)
        count += 1
    print(f"  {label}: {count} 张图，{len(table)} 个唯一哈希")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-root", type=pathlib.Path, required=True,
                        help="官方测试集图像根目录（递归扫描）")
    parser.add_argument("--scan", type=pathlib.Path, action="append", required=True,
                        help="要审计的训练侧目录，可重复")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="输出的剔除清单 JSON")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=== 测试集哈希 ===")
    test_table = hash_tree(args.test_root, "官方测试集")

    excluded_stems: set[str] = set()
    excluded_paths: list[str] = []
    report: dict[str, dict[str, int]] = {}

    for scan_root in args.scan:
        if not scan_root.is_dir():
            raise SystemExit(f"训练侧目录不存在: {scan_root}")
        print(f"\n=== 审计 {scan_root} ===")
        total = 0
        hit = 0
        for image in iter_images(scan_root):
            total += 1
            if md5_file(image) in test_table:
                hit += 1
                excluded_paths.append(str(image))
                excluded_stems.add(frame_key(image))
        share = 100 * hit / total if total else 0.0
        print(f"  重叠 {hit} / {total}  = {share:.1f}%")
        report[str(scan_root)] = {"checked": total, "overlapping": hit}

    payload = {
        "test_root": str(args.test_root),
        "excluded_frame_keys": sorted(excluded_stems),
        "excluded_paths": sorted(excluded_paths),
        "per_scan": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== 结论 ===")
    print(f"  重叠文件: {len(excluded_paths)}")
    print(f"  受影响帧: {len(excluded_stems)}")
    print(f"  已写出:   {args.out}")


if __name__ == "__main__":
    main()
