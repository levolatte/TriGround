"""复赛提交文件的合规预检。

官方要求（contest/ PDF 第 177–204 行）：
  * 输出为 `{QueryID: {visible, infrared, depth, query, bbox:[x1,y1,x2,y2]}}`；
  * **"不可修改除 bbox 字段外的其他字段，否则将导致评测程序匹配失败"**；
  * bbox 归一化到 [0,1]；
  * 反向坐标（x1>=x2、y1>=y2）、越界、NaN、空框 → **直接判无效预测**；
  * 提交前需压成 zip。

本脚本只用标准库，能在本地对已生成的 predictions.json 做提交前把关：
逐字段比对官方 queries.json，任何非 bbox 字段被改动都会报错。

用法：
    python tools/check_submission_format.py --queries <queries.json> \
        --submission <predictions.json> [--zip <submission.zip>]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

REQUIRED_KEYS = ("visible", "infrared", "depth", "query")


def check_box(box: object) -> str | None:
    """返回 None 表示合法，否则返回原因。规则直接照抄赛题 PDF。"""
    if not isinstance(box, list) or len(box) != 4:
        return "空框或长度不为 4"
    for value in box:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return "含非数值"
        if math.isnan(value) or math.isinf(value):
            return "含 NaN/Inf"
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 >= x2 or y1 >= y2:
        return "反向坐标（x1>=x2 或 y1>=y2）"
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return "坐标越界（不在 [0,1]）"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    submission = json.loads(args.submission.read_text(encoding="utf-8"))

    problems: list[str] = []

    missing = [key for key in queries if key not in submission]
    extra = [key for key in submission if key not in queries]
    if missing:
        problems.append(f"缺少 {len(missing)} 个 QueryID，例如 {missing[:3]}")
    if extra:
        problems.append(f"多出 {len(extra)} 个非官方 QueryID，例如 {extra[:3]}")

    if list(submission) != list(queries):
        problems.append("QueryID 顺序与官方不一致（评测按顺序匹配时有风险）")

    field_changed: list[str] = []
    key_missing: list[str] = []
    bad_boxes: list[tuple[str, str]] = []
    for query_id, reference in queries.items():
        record = submission.get(query_id)
        if record is None:
            continue
        if set(record) != set(reference) | {"bbox"}:
            field_changed.append(query_id)
            continue
        for key in REQUIRED_KEYS:
            if record.get(key) != reference.get(key):
                key_missing.append(f"{query_id}.{key}")
        reason = check_box(record.get("bbox"))
        if reason:
            bad_boxes.append((query_id, reason))

    if field_changed:
        problems.append(f"{len(field_changed)} 条记录的字段集合被改动，例如 {field_changed[:3]}")
    if key_missing:
        problems.append(f"{len(key_missing)} 个非 bbox 字段内容被改动，例如 {key_missing[:3]}")
    if bad_boxes:
        problems.append(f"{len(bad_boxes)} 个 bbox 会被判无效，例如 {bad_boxes[:3]}")

    zip_entries: list[str] = []
    if args.zip_path:
        with zipfile.ZipFile(args.zip_path) as archive:
            zip_entries = archive.namelist()
            if zip_entries != ["predictions.json"]:
                problems.append(f"zip 内应只有单个 predictions.json，实际为 {zip_entries}")

    print(f"官方查询数      : {len(queries)}")
    print(f"提交记录数      : {len(submission)}")
    print(f"字段被改动的记录: {len(field_changed)}")
    print(f"非 bbox 字段改动: {len(key_missing)}")
    print(f"无效 bbox       : {len(bad_boxes)}")
    if args.zip_path:
        print(f"zip 内容        : {zip_entries}")

    if problems:
        print("\n不合格：")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nSUBMISSION_FORMAT_OK")


if __name__ == "__main__":
    main()
