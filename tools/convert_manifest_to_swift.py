"""把项目的原生 SFT 清单（conversations 格式）转换成 ms-swift 可读的标准格式。

为什么要转换：冲刺换到 Qwen3.6-27B 后，训练框架从 Qwen3-VL 专用的 qwen-vl-finetune
换成 ms-swift（官方支持 Qwen3.6）。两者的数据格式不同：

    旧（A.json 一条）：
        {"id": "abv:old:0000:city_...", "image": ["/abs/visible/x.png", ...],
         "conversations": [{"from": "human", "value": "<image>\\n<image>\\n<image>\\n...原文..."},
                           {"from": "gpt", "value": "{\\"bbox_2d\\":[801,111,869,781]}"}]}

    新（ms-swift 标准，逐行 jsonl）：
        {"messages": [{"role": "user", "content": "<image>\\n<image>\\n<image>\\n...原文..."},
                      {"role": "assistant", "content": "{\\"bbox_2d\\":[801,111,869,781]}"}],
         "images": ["/abs/visible/x.png", "/abs/infrared/x.png", "/abs/depth_rgb/x.png"]}

关键约束：**保序、保重数、文本逐字不变**。A.json 的 4800 条是 4800 次呈现（对应 660 个
唯一图组），不是 4800 道独立题；抽样/重复信息本身是配方的一部分，不能在这里去重或重排。

用法：
    python tools/convert_manifest_to_swift.py \
        --input  <A.json> \
        --output <out.jsonl> \
        [--data-root <相对路径的根>] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import parse_generated_bbox  # noqa: E402

IMAGE_PLACEHOLDER = "<image>"


def is_absolute_path(value: str) -> bool:
    """跨平台判断绝对路径。

    清单由 Linux 侧生成（`/root/autodl-tmp/...`），而本脚本也会在 Windows 上跑；
    Windows 的 `Path.is_absolute()` 对 `/root/...` 返回 False，会误判成相对路径。
    """
    return value.startswith("/") or Path(value).is_absolute()


def image_paths(record: dict, data_root: Path | None) -> list[str]:
    raw = record.get("image")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{record.get('id')}: image 必须是非空列表")
    resolved = []
    for value in raw:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{record.get('id')}: image 元素必须是非空字符串")
        if is_absolute_path(value):
            resolved.append(value)
            continue
        if data_root is None:
            raise ValueError(f"{record.get('id')}: {value} 是相对路径，但未提供 --data-root")
        resolved.append(str(data_root / value))
    return resolved


def convert_record(record: dict, data_root: Path | None) -> dict:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{record.get('id')}: 需要恰好一轮 human/gpt conversations")
    human, assistant = conversations
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError(f"{record.get('id')}: conversations 顺序必须是 human 然后 gpt")
    prompt = human.get("value")
    answer = assistant.get("value")
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise ValueError(f"{record.get('id')}: conversations 的 value 必须是字符串")
    images = image_paths(record, data_root)
    if prompt.count(IMAGE_PLACEHOLDER) != len(images):
        raise ValueError(
            f"{record.get('id')}: <image> 占位符 {prompt.count(IMAGE_PLACEHOLDER)} 个，"
            f"但 image 有 {len(images)} 个"
        )
    target = parse_generated_bbox(answer)
    if target is None:
        raise ValueError(f"{record.get('id')}: 答案里没有合法的 0--1000 bbox：{answer!r}")
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": images,
        "id": record.get("id"),
    }
    for key in ("task_category", "scene_id", "origin_query_id", "class_name", "split"):
        if key in record:
            row[key] = record[key]
    return row


def convert(input_path: Path, output_path: Path, data_root: Path | None,
            limit: int = 0) -> dict:
    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{input_path} 必须是 JSON 列表")
    if limit:
        records = records[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenes = set()
    image_counts = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = convert_record(record, data_root)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            image_counts[len(row["images"])] += 1
            if row.get("scene_id"):
                scenes.add(row["scene_id"])

    return {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(records),
        "unique_scene_ids": len(scenes),
        "images_per_record": dict(sorted(image_counts.items())),
        "data_root": str(data_root) if data_root else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        help="清单里是相对路径时使用；绝对路径（如 A.json）不需要")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert(args.input.resolve(), args.output.resolve(),
                     args.data_root.resolve() if args.data_root else None, args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
