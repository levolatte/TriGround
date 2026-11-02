"""把开发集 jsonl 的 `solution` 字段转成 GT JSON，供两个验证工具使用。

## 为什么需要它

`tools/ensemble_grounding.py` 与 `tools/verify_acc05_dev.py` 都通过
`load_gt` 读 `{id: bbox}` 或 `{id: {..., bbox}}` 形态的 JSON，而开发集是
jsonl、答案字段叫 `solution`。本工具只做这一处格式转换。

**不做数据校验，也不做兜底**：缺 `solution`、框非法、id 重复都直接报错。开发集是
裁判，它出错必须立刻暴露，不能静默跳过某一题——跳过会让分母变小、分数虚高。

## 用法

    python tools/dev_gt_from_jsonl.py --input city400_dev.jsonl --output city400_dev_gt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_acc05_dev import box_is_valid  # noqa: E402


def convert(rows: list[dict]) -> dict[str, list[float]]:
    gt: dict[str, list[float]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("每条记录必须有非空 id")
        if sample_id in gt:
            raise ValueError(f"id 重复: {sample_id}")
        solution = row.get("solution")
        if not isinstance(solution, (list, tuple)) or len(solution) != 4:
            raise ValueError(f"{sample_id}: solution 必须是长度 4 的序列")
        box = [float(value) for value in solution]
        if not box_is_valid(box):
            raise ValueError(f"{sample_id}: 真值框非法 {box}")
        gt[sample_id] = box
    if not gt:
        raise ValueError("输入为空")
    return gt


def main() -> dict[str, list[float]]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    gt = convert(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gt, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "queries": len(gt)}, ensure_ascii=False), flush=True)
    return gt


if __name__ == "__main__":
    main()
