"""把 RGBT-GroundBench 转成与 City 同构的 swift SFT 样本。

## 为什么要有这个脚本

RGBT-GroundBench 补的是我方数据**完全没有**的东西：长查询与空间关系措辞。
实测 39,442 条查询里 **23.85% 超过 100 字符**、57.4% 含 "near"、42.5% 含 "under"，
而我们的 City 训练集**一条超过 100 字符的都没有**。
它**不补**物体类别（实际只有 person/car/truck/bicycle 四类，比 City 的 12 类更窄），
也**不补序数词**（39,442 条里 first/second/third 合计仅 9 条）。

## 两个已由调研实测确认、写错就静默出错的细节

1. **bbox 是 `[x, y, w, h]` 绝对像素，不是 xyxy。** 调研方做过边界校验：
   23,000 条样本按 xywh 解释 0 条越界，按 xyxy 解释则大量越界。
2. **类别 id 每个源数据集独立**，跨源复用会得到错误类别名（FLIR 的 3 是 person，
   在 MFAD 里未必）。因此每个源各带一份映射表。

## 深度缺失怎么处理（关键设计）

RGBT 只有 RGB + 红外，**没有深度**。我们**不用**两模态模板（那会引入第二套 prompt，
推理时用不上），而是沿用仓库既有约定：喂一张**全黑深度占位图**，并在 prompt 里显式说明
`missing_modalities=("depth",)` —— `prepare_qwen3vl_native_sft.native_prompt` 会插入
"The depth view(s) are unavailable and shown as all-black placeholders; ignore them as
scene evidence."。这样**推理模板保持不变**，模型学到的是"深度可能是黑的、要忽略"。

## 用法

    python tools/convert_rgbt_groundbench.py \
        --root /root/autodl-tmp/rematch_20260922/data/external/RGBT-GroundBench \
        --out /root/sprint_qwen36/rgbt_swift.jsonl --limit-per-source 0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import zipfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.prepare_qwen3vl_native_sft import native_prompt  # noqa: E402

# native_prompt 的 depth_policy 参数用字符串字面量，源码里没有导出常量。
DEPTH_POLICY_MM = "millimeter"

# 每个源一套映射：类别 id 不可跨源复用。
CLASS_NAMES: dict[str, dict[int, str]] = {
    "flir": {1: "truck", 2: "car", 3: "person", 4: "bicycle"},
    "m3fd": {1: "people", 2: "car", 3: "bus", 4: "motorcycle", 5: "lamp", 6: "truck"},
    "mfad": {2: "car", 3: "truck"},
}
# 每个源的标注目录与训练分片名（由 `tar -tf ann_*.tar` 实测确认，
# 目录前缀是 `rgbtvg_` 而不是 `ann_`，文件名带 `_train`）。
ANN_LAYOUT = {
    "flir": ("rgbtvg_flir", "rgbtvg_flir_train.pth"),
    "m3fd": ("rgbtvg_m3fd", "rgbtvg_m3fd_train.pth"),
    "mfad": ("rgbtvg_mfad", "rgbtvg_mfad_train.pth"),
}
NORMALIZE = 1000.0
# 明确的退化模板，混进训练只会教模型输出废话
DEGENERATE_MARKERS = ("No ", "is present", "There is no")


def read_annotations(path: pathlib.Path) -> list[Any]:
    """读取 torch ZIP 归档里的标注 —— 不需要安装 torch。

    `.pth` 是 torch 的 ZIP 归档，成员名形如 `flir_train/data.pkl`，
    但前缀随发布方式变化，所以按后缀匹配而不是写死。
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        target = next((n for n in names if n.endswith("data.pkl")), None)
        if target is None:
            raise ValueError(f"{path}: 找不到 data.pkl，成员为 {names[:6]}")
        return pickle.loads(archive.read(target))


def is_degenerate(query: str) -> bool:
    return any(marker in query for marker in DEGENERATE_MARKERS)


def convert(root: pathlib.Path, out: pathlib.Path, limit_per_source: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"by_source": {}, "skipped": {}}

    for source, (directory, member) in ANN_LAYOUT.items():
        ann_path = root / f"ann_{source}.tar"
        if not ann_path.is_file():
            stats["skipped"][source] = "标注 tar 不存在"
            continue
        # 标注必须先解包：tar 内部是 rgbtvg_<source>/rgbtvg_<source>_train.pth
        pth = root / directory / member
        if not pth.is_file():
            stats["skipped"][source] = (
                f"未解包或分片名不符，期望 {pth}；"
                f"请先 `tar -xf ann_{source}.tar -C {root}`")
            continue

        image_root = root / f"data_{source}" / "image_data" / source
        rgb_root, ir_root = image_root / "rgb", image_root / "ir"
        if not rgb_root.is_dir() or not ir_root.is_dir():
            stats["skipped"][source] = f"图像目录缺失: {rgb_root} / {ir_root}"
            continue

        mapping = CLASS_NAMES[source]
        kept = skipped_degenerate = skipped_missing_image = skipped_unknown_class = 0
        for row in read_annotations(pth):
            filename, size, box, query, class_id = row[0], row[1], row[2], row[3], row[4]
            if is_degenerate(str(query)):
                skipped_degenerate += 1
                continue
            if class_id not in mapping:
                skipped_unknown_class += 1
                continue
            rgb_path, ir_path = rgb_root / filename, ir_root / filename
            if not rgb_path.is_file() or not ir_path.is_file():
                skipped_missing_image += 1
                continue
            height, width = float(size["height"]), float(size["width"])
            x, y, w, h = (float(v) for v in box)
            # ⚠️ 源标注是 xywh 绝对像素，必须换算式转成归一化 xyxy
            xyxy = (x / width, y / height, (x + w) / width, (y + h) / height)
            if not (0.0 <= xyxy[0] < xyxy[2] <= 1.0 and 0.0 <= xyxy[1] < xyxy[3] <= 1.0):
                skipped_unknown_class += 1
                continue
            quantized = [round(v * NORMALIZE) for v in xyxy]
            # 只用既有的三模态模板 + 显式缺失说明，避免引入第二套 prompt
            prompt = native_prompt(str(query), ("rgb", "infrared", "depth"),
                                   DEPTH_POLICY_MM, missing_modalities=("depth",))
            records.append({
                "id": f"rgbt:{source}:{pathlib.Path(filename).stem}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant",
                     "content": json.dumps({"bbox_2d": quantized}, separators=(",", ":"))},
                ],
                "images": [str(rgb_path), str(ir_path)],  # 深度由训练侧按缺失变体补黑图
                "class_name": mapping[class_id],
                "source": f"rgbt_{source}",
                "missing_modalities": ["depth"],
            })
            kept += 1
            if limit_per_source and kept >= limit_per_source:
                break
        stats["by_source"][source] = {
            "kept": kept, "degenerate": skipped_degenerate,
            "unknown_class": skipped_unknown_class, "missing_image": skipped_missing_image,
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["total"] = len(records)
    stats["out"] = str(out)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--limit-per-source", type=int, default=0)
    args = parser.parse_args()
    stats = convert(args.root, args.out, args.limit_per_source)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["total"] == 0:
        raise SystemExit("没有产出任何样本，检查上面的 skipped 原因")


if __name__ == "__main__":
    main()
