"""打包兄弟上下文（sibling context）版全量预测，产出官方格式 submission.zip。

为什么不复用 `predict_submission_vllm.py --package-only`：它调用的 `validate_rows`
硬校验 `row["prompt"]` **逐字等于** `PROMPTS[modality](query)`（无兄弟上下文的标准提示）。
sibling 版记录的是「前缀句 + 标准提示」（见 `_trimodal_prompt(query, siblings)`），
必然被判 mismatch，与预测质量无关。

这里保留**除提示以外**的全部校验：ID 覆盖与顺序、query、image_paths、bbox 合法、
`raw_text` 与 bbox 一致、`parsed`/`fallback` 互斥。提示改为后缀匹配——sibling 提示
必须逐字以标准提示结尾，防止出现与本题无关的提示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import parse_generated_bbox  # noqa: E402
from tools.predict_native_submission import (  # noqa: E402
    FALLBACK_BOX,
    PROMPTS,
    image_paths,
    load_queries,
    package_submission,
    summarize,
    valid_box,
)


SIBLING_ANCHOR = "Locate the object described by this query: "


def validate_rows(rows, source, data_root: Path, queries: Path, modality: str,
                  sibling_context: bool) -> dict:
    indexed: dict = {}
    for line_number, row in enumerate(rows, 1):
        sample_id = str(row.get("id"))
        if sample_id not in source:
            raise ValueError(f"predictions.jsonl line {line_number}: unknown ID {sample_id}")
        if sample_id in indexed:
            raise ValueError(f"predictions.jsonl line {line_number}: duplicate ID {sample_id}")
        record = source[sample_id]
        if row.get("query") != record["query"]:
            raise ValueError(f"{sample_id}: query mismatch")
        if row.get("image_paths") != image_paths(record, data_root, queries):
            raise ValueError(f"{sample_id}: image_paths mismatch")
        expected_prompt = PROMPTS[modality](record["query"])
        prompt = row.get("prompt")
        if sibling_context:
            # sibling 版把「兄弟查询」那句插在 `Locate the object...` **之前**，不是在提示末尾。
            # 所以按分隔符切成三段：前缀一致 + 尾段一致 = 中间只多了那句兄弟查询。
            head, sep, tail = expected_prompt.partition(SIBLING_ANCHOR)
            if not sep:
                raise ValueError(f"standard prompt lost its {SIBLING_ANCHOR!r} anchor")
            if not isinstance(prompt, str) or not (prompt.startswith(head)
                                                   and prompt.endswith(sep + tail)):
                raise ValueError(f"{sample_id}: sibling prompt mismatch around inserted sentence")
        elif prompt != expected_prompt:
            raise ValueError(f"{sample_id}: prompt mismatch")
        if not valid_box(row.get("bbox")):
            raise ValueError(f"predictions.jsonl line {line_number}: invalid bbox")
        if not isinstance(row.get("parsed"), bool) or not isinstance(row.get("fallback"), bool):
            raise ValueError(f"predictions.jsonl line {line_number}: missing parse status")
        if row["fallback"] == row["parsed"]:
            raise ValueError(f"predictions.jsonl line {line_number}: inconsistent parse status")
        if row["fallback"] and row["bbox"] != FALLBACK_BOX:
            raise ValueError(f"{sample_id}: inconsistent fallback box")
        parsed_box = parse_generated_bbox(row.get("raw_text", ""))
        if row["parsed"] != (parsed_box is not None):
            raise ValueError(f"{sample_id}: raw text and parse status differ")
        if parsed_box is not None and row["bbox"] != parsed_box:
            raise ValueError(f"{sample_id}: bbox differs from raw text")
        indexed[sample_id] = row
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-query-count", type=int, required=True)
    parser.add_argument("--modality", choices=PROMPTS, default="trimodal")
    parser.add_argument("--sibling-context", action="store_true")
    parser.add_argument("--predictions", type=Path,
                        help="默认取 output-dir/predictions.jsonl")
    args = parser.parse_args()

    rows_path = args.predictions or (args.output_dir / "predictions.jsonl")
    source = load_queries(args.queries, args.expected_query_count)
    rows = [json.loads(line) for line in
            rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows_by_id = validate_rows(rows, source, args.data_root, args.queries,
                               args.modality, args.sibling_context)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, zip_path = package_submission(source, rows_by_id, args.output_dir)
    ordered = [rows_by_id[sample_id] for sample_id in source]
    summary = {
        **summarize(ordered, len(source), 0, 0),
        "queries": str(args.queries),
        "data_root": str(args.data_root),
        "modality": args.modality,
        "sibling_context": bool(args.sibling_context),
        "predictions_jsonl": str(rows_path),
        "predictions_json": str(json_path),
        "submission_zip": str(zip_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
