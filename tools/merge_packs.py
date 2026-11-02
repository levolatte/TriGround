"""合并多机分片推理的 predictions.jsonl，并用官方校验 + 官方打包函数出提交包。

用法：python tools/merge_packs.py <queries.json> <data_root> <modality> <输出目录> <shard 输出目录>...

校验全部复用官方路径（validate_rows / package_submission），本脚本只做拼接，
不重新实现任何验证逻辑。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import load_prediction_rows
from tools.predict_native_submission import (
    PROMPTS,
    load_queries,
    package_submission,
    validate_rows,
    valid_box,
)


def main() -> None:
    queries = Path(sys.argv[1]).resolve()
    data_root = Path(sys.argv[2]).resolve()
    modality = sys.argv[3]
    out_dir = Path(sys.argv[4]).resolve()
    shard_dirs = [Path(value).resolve() for value in sys.argv[5:]]

    source = load_queries(queries, None)
    rows: list[dict] = []
    for shard_dir in shard_dirs:
        path = shard_dir / "predictions.jsonl"
        part = load_prediction_rows(path)
        print(f"{path}: {len(part)} rows")
        rows.extend(part)
    print(f"concatenated {len(rows)} rows, source {len(source)} ids")

    rows_by_id = validate_rows(rows, source, data_root, queries, modality,
                               prompt_builder=PROMPTS[modality])
    missing = set(source) - set(rows_by_id)
    if missing:
        raise ValueError(f"{len(missing)} Query IDs missing, e.g. {sorted(missing)[:5]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, zip_path = package_submission(source, rows_by_id, out_dir)

    # 出包后再独立自检三件事（提交边界，值得重复验证）
    packed_text = json_path.read_text(encoding="utf-8")
    submission = json.loads(packed_text)
    if len(submission) != len(source):
        raise ValueError(f"packed {len(submission)} != source {len(source)}")
    invalid = [key for key, value in submission.items() if not valid_box(value["bbox"])]
    if invalid:
        raise ValueError(f"{len(invalid)} invalid boxes, e.g. {invalid[:5]}")
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        inner = archive.read("predictions.json")
    if names != ["predictions.json"]:
        raise ValueError(f"ZIP entries {names}")
    if inner != packed_text.encode("utf-8"):
        raise ValueError("ZIP inner JSON differs byte-wise from predictions.json")
    fallbacks = sum(1 for row in rows_by_id.values() if row["fallback"])
    print(f"PACKED {len(submission)} ids, {len(invalid)} invalid boxes, "
          f"fallbacks {fallbacks}, zip {zip_path.stat().st_size} bytes")
    print(f"JSON {json_path}  ZIP {zip_path}")


if __name__ == "__main__":
    main()
