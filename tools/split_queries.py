"""把官方 queries.json 按 QueryID 顺序切成 N 份，支持按机器速度加权。

用法：python tools/split_queries.py <queries.json> <输出目录> <N> [加权比例,如 1,1,1,0.5]
写出 shard0.json ... shardN-1.json，每份都是同一格式的官方查询对象。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.predict_native_submission import load_queries


def main() -> None:
    queries = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    shards = int(sys.argv[3])
    if len(sys.argv) > 4:
        weights = [float(value) for value in sys.argv[4].split(",")]
    else:
        weights = [1.0] * shards
    if len(weights) != shards:
        raise ValueError(f"{len(weights)} weights for {shards} shards")
    total_weight = sum(weights)
    source = load_queries(queries, None)
    items = list(source.items())
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = []
    assigned = 0
    for index, weight in enumerate(weights):
        if index == shards - 1:
            counts.append(len(items) - assigned)
        else:
            counts.append(int(round(len(items) * weight / total_weight)))
            assigned += counts[-1]
    if sum(counts) != len(items):
        raise ValueError(f"counts {counts} sum {sum(counts)} != {len(items)}")

    cursor = 0
    for index, count in enumerate(counts):
        chunk = items[cursor:cursor + count]
        cursor += count
        path = out_dir / f"shard{index}.json"
        path.write_text(json.dumps(dict(chunk), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"shard{index}: {len(chunk):5d} ids  {chunk[0][0]} .. {chunk[-1][0]}  -> {path}")
    if cursor != len(items):
        raise ValueError(f"cursor {cursor} != {len(items)}")
    print(f"TOTAL {len(items)} across {shards} shards, weights {weights}")


if __name__ == "__main__":
    main()
