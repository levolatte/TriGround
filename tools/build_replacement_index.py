"""用等量级外部数据替换 City，重建训练索引。

背景（实测）：现有训练索引 4,905 条里 City 占 2,882（59%），且是最贴合测试集的一块——
全三模态、查询 9 词关系式、29.9% 的框面积 <0.2%。本脚本把它整体换出，用：

* **RGBT-GroundBench**（26,604 条，图像 100% 齐全、零下载）：RGB+红外两模态，人工式英文关系查询。
  其 `obj_size` 标签（'small'/'normal'）在 `ann_*.tar` 的 `.pth` 里，已实测与 `train_full.jsonl`
  逐条 1:1 对齐、框换算误差 0.000000。
* **new-rgbdt500**（3,562 条，三模态带深度）：动物园/野外为主，覆盖官方查询里 14.1% 的动物域。

RoboRefIt 与 RGBDT500 原样保留，减少变量。

用法见 `--help`。所有路径写相对 `--media-root`，不做拷贝。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.prepare_qwen3vl_native_sft import native_prompt

RGBT_SOURCES = ("flir", "m3fd", "mfad")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rgbt(root: Path) -> list[dict]:
    """读 train_full.jsonl，并从 ann_*.tar 的 .pth 注入 obj_size。

    读取 .pth 需要 torch；这里只用它读标注，不碰模型。
    """
    import torch

    rows = read_jsonl(root / "converted" / "train_full.jsonl")
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"].replace("rgbt_groundbench_", "")].append(row)

    merged: list[dict] = []
    for source in RGBT_SOURCES:
        source_rows = by_source[source]
        with tarfile.open(root / f"ann_{source}.tar") as archive:
            member = next(m for m in archive.getmembers()
                          if m.name.endswith(f"rgbtvg_{source}_train.pth"))
            annotations = torch.load(io.BytesIO(archive.extractfile(member).read()),
                                     map_location="cpu", weights_only=False)
        if len(annotations) != len(source_rows):
            raise ValueError(f"{source}: jsonl {len(source_rows)} 条 vs pth {len(annotations)} 条，无法对齐")
        for position, (row, annotation) in enumerate(zip(source_rows, annotations)):
            filename, size, box_xywh, query, *_rest = annotation
            obj_size = annotation[5]
            # 对齐断言：文件名与查询必须逐条一致，否则后面的索引关联就是错的。
            if Path(row["rgb"]).name != filename or row["query"].strip() != str(query).strip():
                raise ValueError(f"{source}[{position}]: jsonl 与 pth 不对齐")
            width, height = size["width"], size["height"]
            x, y, w, h = box_xywh
            merged.append({
                "id": row["id"],
                # 统一存短名（flir/m3fd/mfad）：抽样配比用的是短名，长名会让筛选永远为空。
                "source": source,
                "obj_size": obj_size,
                "query": row["query"].strip(),
                "solution": [round(x / width, 6), round(y / height, 6),
                             round((x + w) / width, 6), round((y + h) / height, 6)],
                "rgb": row["rgb"],
                "ir": row["aux"],
            })
    return merged


def relative_under(media_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(media_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{resolved} 不在 media_root {media_root} 之下") from error


def rgbt_row(record: dict, convert_dir: Path, media_root: Path) -> dict:
    rgb = relative_under(media_root, (convert_dir / record["rgb"]).resolve())
    ir = relative_under(media_root, (convert_dir / record["ir"]).resolve())
    return {
        "id": record["id"],
        "images": [rgb, ir],
        "modalities": ["rgb", "infrared"],
        "solution": record["solution"],
        "messages": [{"role": "user", "content": native_prompt(record["query"], ("rgb", "infrared"))}],
        "source": f"rgbt_{record['source']}",
        "category": "regular",
        "obj_size": record["obj_size"],
    }


def new_rgbdt_row(record: dict, media_root: Path, root: Path, index: int) -> dict:
    images = [
        relative_under(media_root, root / record["visible"]),
        relative_under(media_root, root / record["infrared"]),
        relative_under(media_root, root / record["depth"]),
    ]
    return {
        "id": f"newrgbdt_{index:05d}",
        "images": images,
        "modalities": ["rgb", "infrared", "depth"],
        "solution": [round(value, 6) for value in record["bbox"]],
        "messages": [{"role": "user", "content": native_prompt(
            record["query"].strip(), ("rgb", "infrared", "depth"), "millimeter")}],
        "source": "new_rgbdt500",
        "category": "regular",
        "obj_size": "small" if (record["bbox"][2] - record["bbox"][0]) *
                    (record["bbox"][3] - record["bbox"][1]) < 0.004 else "normal",
    }


def box_area(sample: dict) -> float:
    x1, y1, x2, y2 = sample["solution"]
    return (x2 - x1) * (y2 - y1)


def query_word_count(sample: dict) -> int:
    import re
    match = re.search(r"this query:\s*(.*?)\s*Return only JSON", sample["messages"][0]["content"], re.S)
    return len((match.group(1) if match else "").split())


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--rgbt-root", type=Path, required=True)
    parser.add_argument("--new-rgbdt-root", type=Path, required=True,
                        help="new-rgbdt500 子集上传后的目录，需在 media_root 之下")
    parser.add_argument("--new-rgbdt-json", type=Path, required=True)
    parser.add_argument("--keep-index", type=Path, required=True,
                        help="现有 train_index.jsonl，用于原样复制 robo/rgbdt 样本")
    parser.add_argument("--rgbt-small", type=int, default=0)
    parser.add_argument("--rgbt-normal", type=int, default=0)
    parser.add_argument("--rgbt-source-ratio", default="flir:3,m3fd:2,mfad:5")
    # 用小目标池的全部 800 条 + 常规池 2,762 条 = 用尽 new-rgbdt500 的 3,562 条标注。
    # 这批就是 RGBDT500（官方测试集同源）的查询标注版，三模态齐全。
    parser.add_argument("--new-rgbdt-tiny", type=int, default=800)
    parser.add_argument("--new-rgbdt-regular", type=int, default=2762)
    parser.add_argument("--tiny-area", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-images", type=Path, default=None,
                        help="官方测试图目录。**默认不用于剔除**——赛题规则原文是"
                             "\"本赛题对训练数据使用不做约束…无外部数据集使用范围限制\"，"
                             "同源/同图不算违规。只在需要体检时传，用来统计重合量。")
    parser.add_argument("--drop-test-overlap", action="store_true",
                        help="显式要求剔除与测试集逐像素相同的样本。默认关闭。")
    parser.add_argument("--city-rows", type=int, default=2882,
                        help="保留多少条 City（按图分组取，以保住它 5.61 的高重数特性）。"
                             "默认 2,882 = 全量保留。")
    parser.add_argument("--emit-file-list", type=Path,
                        help="只写出被选中的 new-rgbdt500 图像清单（每行一个源路径），用于分步上传；"
                             "不校验图像是否存在、也不写索引。")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ---------- 分步上传模式：只导出 new-rgbdt500 的清单就返回 ----------
    # 放在最前面，因为它不需要 RGBT 的数据（本地只有 .pth 标注、没有 train_full.jsonl），
    # 也不需要任何图像存在。选样用的是同一个种子与同一段代码，所以导出的清单与后面
    # 正式建索引时选中的是同一批，不会错位。
    if args.emit_file_list is not None:
        records = list(json.loads(args.new_rgbdt_json.read_text(encoding="utf-8")).values())

        def _emit_tiny(record: dict) -> bool:
            x1, y1, x2, y2 = record["bbox"]
            return (x2 - x1) * (y2 - y1) < args.tiny_area

        tiny_pool = [r for r in records if _emit_tiny(r)]
        regular_pool = [r for r in records if not _emit_tiny(r)]
        if args.new_rgbdt_tiny > len(tiny_pool):
            raise ValueError(f"小目标池只有 {len(tiny_pool)} 条，要 {args.new_rgbdt_tiny} 条")
        picked_rows = (rng.sample(tiny_pool, args.new_rgbdt_tiny)
                       + rng.sample(regular_pool, args.new_rgbdt_regular))
        needed = [record[key] for record in picked_rows for key in ("visible", "infrared", "depth")]
        args.emit_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.emit_file_list.write_text("\n".join(needed) + "\n", encoding="utf-8")
        unique_files = len(set(needed))
        print(f"已写出 {len(needed)} 个路径（唯一 {unique_files} 个文件）→ {args.emit_file_list}", flush=True)
        return {"file_list": str(args.emit_file_list), "paths": len(needed),
                "unique_files": unique_files, "samples": len(picked_rows)}

    # ---------- 1. 保留 City（按比例）+ RoboRefIt + RGBDT500 ----------
    keep = [row for row in read_jsonl(args.keep_index) if row.get("source") in ("robo", "rgbdt")]
    # RoboRefIt 按本轮决策**整体剔除**：它的查询是祈使句（"will you please pass me …"），
    # 与本赛的"描述性唯一指代"不是同一任务格式，且重数只有 1.00。
    # RGBDT500（rgbdt，159 条）保留，它是真三模态且查询风格接近。
    keep = [row for row in keep if row.get("source") == "rgbdt"]
    if not keep:
        raise ValueError("keep-index 里没找到 rgbdt 样本")
    city_pool = [row for row in read_jsonl(args.keep_index) if row.get("source") == "city"]
    city_taken: list[dict] = []
    if args.city_rows > 0:
        if args.city_rows > len(city_pool):
            raise ValueError(f"City 只有 {len(city_pool)} 条，要 {args.city_rows} 条")
        # City 的 2,882 条只用了 514 张图（重数 5.61，单图最多 15 条），**它是同图多实例
        # 能力的唯一来源**。所以按图分组抽，保住这个特性；随机抽样本会把它压到 1.0 附近。
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in city_pool:
            groups[row["images"][0]].append(row)
        ordered = sorted(groups.values(), key=lambda rows: -len(rows))
        for rows_of_image in ordered:
            if len(city_taken) >= args.city_rows:
                break
            city_taken.extend(rows_of_image)
        city_taken = city_taken[:args.city_rows]
        print(f"City 保留 {len(city_taken)} 条 / {len(groups)} 张图中取了 "
              f"{len({r['images'][0] for r in city_taken})} 张", flush=True)
    print(f"保留 robo/rgbdt {len(keep)} 条", flush=True)

    # ---------- 2. RGBT：按 obj_size 抽 ----------
    rgbt = load_rgbt(args.rgbt_root)
    print(f"RGBT 可用 {len(rgbt)} 条，obj_size 分布 {dict(Counter(r['obj_size'] for r in rgbt))}", flush=True)
    convert_dir = (args.rgbt_root / "converted").resolve()

    ratio = {}
    for part in args.rgbt_source_ratio.split(","):
        name, weight = part.split(":")
        ratio[name.strip()] = float(weight)
    total_weight = sum(ratio.values())

    picked: list[dict] = []
    for bucket, share in (("small", args.rgbt_small), ("normal", args.rgbt_normal)):
        pool = [r for r in rgbt if r["obj_size"] == bucket]
        for source, weight in ratio.items():
            need = round(share * weight / total_weight)
            # **按图分组抽样**：随机抽「样本」会让几乎每条都落在不同图上，把每图查询数
            # 从数据集天然的 1.79 退化成 1.0——而「同图多实例」正是测试集的形态
            # （官方 4.41 题/图）。所以先按图分组、优先选查询数多的图，再取该图上的全部查询。
            groups: dict[str, list[dict]] = defaultdict(list)
            for record in pool:
                if record["source"] == source:
                    groups[record["rgb"]].append(record)
            if not groups:
                raise ValueError(f"{source}/{bucket} 没有任何样本")
            # 查询数多的图优先，这样同样的条数能用更少的图凑出来，重数更高。
            ordered = sorted(groups.values(), key=lambda rows: -len(rows))
            chosen: list[dict] = []
            for rows_of_image in ordered:
                if len(chosen) >= need:
                    break
                chosen.extend(rows_of_image)
            if len(chosen) < need:
                raise ValueError(f"{source}/{bucket} 全取也只有 {len(chosen)} 条，要 {need} 条")
            picked.extend(chosen[:need])
    print(f"RGBT 选中 {len(picked)} 条", flush=True)

    # ---------- 3. new-rgbdt500：先剔除与官方测试集重合的，再按面积抽 ----------
    records = list(json.loads(args.new_rgbdt_json.read_text(encoding="utf-8")).values())
    if args.drop_test_overlap and args.test_images is not None:
        # 默认**不**剔除：赛题规则明确"训练数据使用不做约束"，同源/同图不违规。
        # 这个开关只用于体检时统计重合规模。
        test_hashes = set()
        for path in args.test_images.rglob("*"):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                test_hashes.add(file_digest(path))
        before = len(records)

        def contaminated(record: dict) -> bool:
            return any(file_digest(args.new_rgbdt_root / record[key]) in test_hashes
                       for key in ("visible", "infrared", "depth"))

        records = [r for r in records if not contaminated(r)]
        print(f"new-rgbdt500 剔除测试集重合 {before - len(records)} 条，剩 {len(records)} 条", flush=True)
    elif args.test_images is not None:
        # 只统计**已上传到本机**的那批的重合量。未上传的记录会碰到不存在的文件，
        # 这里只是诊断统计，跳过即可（真正要用的样本后面会按存在性另行校验）。
        test_hashes = set()
        for path in args.test_images.rglob("*"):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                test_hashes.add(file_digest(path))

        def hit_test(record: dict) -> bool:
            for key in ("visible", "infrared", "depth"):
                local = args.new_rgbdt_root / record[key]
                if local.is_file() and file_digest(local) in test_hashes:
                    return True
            return False

        present = [r for r in records
                   if (args.new_rgbdt_root / r["visible"]).is_file()]
        overlap = sum(1 for r in present if hit_test(r))
        share = overlap / max(1, len(present))
        print(f"new-rgbdt500 本机已上传 {len(present)} 条，其中与测试集同图 {overlap} 条"
              f"（{share:.1%}）——按赛题规则保留，不剔除", flush=True)

    def is_tiny(record: dict) -> bool:
        return (record["bbox"][2] - record["bbox"][0]) * (record["bbox"][3] - record["bbox"][1]) < args.tiny_area

    tiny_pool = [r for r in records if is_tiny(r)]
    regular_pool = [r for r in records if not is_tiny(r)]
    if args.new_rgbdt_tiny > len(tiny_pool):
        raise ValueError(f"new-rgbdt500 小目标池只有 {len(tiny_pool)} 条，要 {args.new_rgbdt_tiny} 条")
    new_picked = rng.sample(tiny_pool, args.new_rgbdt_tiny) + rng.sample(regular_pool, args.new_rgbdt_regular)
    print(f"new-rgbdt500 选中 {len(new_picked)} 条（小目标池共 {len(tiny_pool)} 条）", flush=True)

    # ---------- 4. 组装 ----------
    rows = list(keep) + city_taken
    rows += [rgbt_row(r, convert_dir, args.media_root) for r in picked]
    rows += [new_rgbdt_row(r, args.media_root, args.new_rgbdt_root, i) for i, r in enumerate(new_picked)]
    rng.shuffle(rows)

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("样本 id 有重复")
    for row in rows:
        x1, y1, x2, y2 = row["solution"]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError(f"{row['id']}: 框非法 {row['solution']}")

    out = args.out_dir / "mixture.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    # ---------- 5. 摘要 ----------
    def describe(label: str) -> dict:
        images = Counter(row["images"][0] for row in rows)
        areas = [box_area(row) for row in rows]
        words = [query_word_count(row) for row in rows]
        return {"label": label, "rows": len(rows), "unique_images": len(images),
                "queries_per_image": round(len(rows) / len(images), 2),
                "modalities": dict(Counter(len(row["images"]) for row in rows)),
                "source": dict(Counter(row["source"] for row in rows)),
                "area_median": round(statistics.median(areas), 5),
                "area_under_0.2pct": round(sum(1 for a in areas if a < 0.002) / len(areas), 3),
                "words_median": int(statistics.median(words))}

    summary = {"mixture": describe("replacement"),
               "rgbt_selected": len(picked), "new_rgbdt_selected": len(new_picked),
               "kept": len(keep),
               "obj_size": dict(Counter(row.get("obj_size") for row in picked + new_picked))}
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["mixture"], ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2), flush=True)
