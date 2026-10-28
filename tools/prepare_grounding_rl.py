"""项目 SFT JSON / Swift JSONL -> GRPO messages + images + 独立 solution。

优先 --gt 提供 id -> 原始归一化 xyxy（或含 bbox 的记录）映射；
也接受项目 SFT 清单中保留的 bbox 原浮点归一化 xyxy 字段。
只有量化 SFT 答案时须显式 --allow-quantized-targets，每行记录近似标签来源。
保持图像顺序、提示文字和重复样本；不渲染深度、不更改云端绝对图像路径。
只能转换许可的训练/验证数据，官方测试数据不得进入训练。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.convert_manifest_to_swift import convert_record as native_to_swift  # noqa: E402
from tools.convert_manifest_to_swift import image_paths  # noqa: E402
from tools.grounding_reward import parse_short_bbox, validate_solution  # noqa: E402


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_targets(path: Path) -> dict[str, list[float]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    targets = {}
    for key, value in raw.items():
        box = value["bbox"] if isinstance(value, dict) else value
        validate_solution(box)
        targets[key] = [float(v) for v in box]
    return targets


def convert_record(
    record: dict,
    data_root: Path | None = None,
    targets: dict[str, list[float]] | None = None,
    gt_id_key: str = "id",
    allow_quantized_targets: bool = False,
) -> dict:
    if "conversations" in record:
        native = record
        if isinstance(record["image"], str):
            native = {**record, "image": [record["image"]]}
        swift_row = native_to_swift(native, data_root)
    else:
        swift_row = record
    sample_id = record["id"]
    messages = swift_row["messages"]
    roles = [m["role"] for m in messages]
    allowed_roles = (["user", "assistant"], ["system", "user", "assistant"],
                     ["user"], ["system", "user"])
    if roles not in allowed_roles:
        raise ValueError(f"{sample_id}: expected a single user turn with optional system/answer")
    if any(not isinstance(m["content"], str) for m in messages):
        raise ValueError(f"{sample_id}: messages content must be text with <image> placeholders")
    answer = messages[-1]["content"] if roles[-1] == "assistant" else None
    prompts = messages[:-1] if answer is not None else messages
    images = image_paths({"id": sample_id, "image": swift_row["images"]}, data_root)
    if sum(m["content"].count("<image>") for m in prompts) != len(images):
        raise ValueError(f"{sample_id}: image placeholder count differs from images")

    if targets is not None:
        target_id = record[gt_id_key]
        if target_id not in targets:
            raise ValueError(f"{sample_id}: no original GT for {gt_id_key}={target_id!r}")
        solution = targets[target_id]
        validate_solution(solution)
        solution_source = "original_gt_map"
    elif "bbox" in record:
        # 项目 prepare_semifinal_sft 保留 bbox 原精度；不是 assistant 的量化坐标。
        solution = record["bbox"]
        validate_solution(solution)
        target_id = sample_id
        solution_source = "original_record_bbox"
    else:
        if not allow_quantized_targets:
            raise ValueError("provide --gt or a normalized bbox field, or explicitly use --allow-quantized-targets")
        solution = parse_short_bbox(answer) if answer is not None else None
        if solution is None:
            raise ValueError(f"{sample_id}: SFT answer is not one valid bbox_2d JSON object")
        target_id = sample_id
        solution_source = "quantized_sft_bbox_2d_1000"

    # 只传白名单元信息，原 assistant、bbox、target 等答案字段不进入模型提示。
    row = {
        "id": sample_id,
        "messages": [{"role": m["role"], "content": m["content"]} for m in prompts],
        "images": images,
        "solution": list(solution),
        "solution_source": solution_source,
        "solution_id": target_id,
    }
    for key in ("source", "split", "scene_id", "sequence_id", "origin_query_id", "query", "class_name",
                "capability", "modalities", "missing_modalities", "depth_policy", "weak_label"):
        if key in record:
            row[key] = record[key]
    return row


def convert(
    input_path: Path,
    output_path: Path,
    data_root: Path | None = None,
    gt_path: Path | None = None,
    gt_id_key: str = "id",
    allow_quantized_targets: bool = False,
) -> dict:
    targets = load_targets(gt_path) if gt_path else None
    records = load_records(input_path)
    rows = [convert_record(rec, data_root, targets, gt_id_key, allow_quantized_targets)
            for rec in records]
    if not rows:
        raise ValueError("input contains no training records")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "gt": str(gt_path) if gt_path else None,
        "gt_id_key": gt_id_key,
        "records": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "images_per_record": dict(Counter(len(row["images"]) for row in rows)),
        "solution_sources": dict(Counter(row["solution_source"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--gt", type=Path, help="id -> 原浮点归一化 xyxy，或含 bbox 的记录")
    parser.add_argument("--gt-id-key", default="id", help="与 GT 映射匹配的源记录键名")
    parser.add_argument("--allow-quantized-targets", action="store_true")
    args = parser.parse_args()
    report = convert(args.input, args.output, args.data_root, args.gt, args.gt_id_key,
                     args.allow_quantized_targets)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
