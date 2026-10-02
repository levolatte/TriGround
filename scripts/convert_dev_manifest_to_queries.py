"""把开发清单（city_val.json）转成官方查询格式，使两条推理路径能在有真值的集合上比 ACC。

为什么需要：官方测试集没有真值，`compare_predictions.py` 只能做**逐题逐位**比对，
而"逐位一致"对两个不同推理引擎而言是过严的判据——贪心解码上的微小数值差异会翻转
个别 token，落到不同的框，但这不代表准确率变差。真正该比的是 ACC@0.5。

而官方测试集格式是 `{QueryID: {visible, infrared, depth, query}}`，开发清单则是
`[{id, image: "<list 的字符串>", conversations: [...]}]`，因此需要一次转换。

用法：
    python scripts/convert_dev_manifest_to_queries.py \
        --manifest <city_val.json> --out <city_val_queries.json>
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

PROMPT_PREFIX = "Locate the object described by this query: "
PROMPT_SUFFIX = "\nReturn only JSON in this exact form"


def extract_query(conversations: list[dict]) -> str:
    human = next(item for item in conversations if item.get("from") == "human")
    value = human["value"]
    if PROMPT_PREFIX not in value:
        raise ValueError("human prompt does not contain the expected query prefix")
    tail = value.split(PROMPT_PREFIX, 1)[1]
    if PROMPT_SUFFIX in tail:
        tail = tail.split(PROMPT_SUFFIX, 1)[0]
    return tail.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("expected the dev manifest to be a list of records")

    queries: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["id"]
        # image 字段是 list 的字符串表示（历史产物），需要 literal_eval 还原
        images = ast.literal_eval(row["image"]) if isinstance(row["image"], str) else row["image"]
        if len(images) != 3:
            raise ValueError(f"{sample_id}: expected three image paths, got {len(images)}")
        queries[sample_id] = {
            "visible": images[0],
            "infrared": images[1],
            "depth": images[2],
            "query": extract_query(row["conversations"]),
        }

    args.out.write_text(json.dumps(queries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"转换 {len(queries)} 条 -> {args.out}")
    first = next(iter(queries.values()))
    print(f"样例: {first}")


if __name__ == "__main__":
    main()
