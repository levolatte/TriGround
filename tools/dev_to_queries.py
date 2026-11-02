"""把开发集 jsonl 转成官方 queries 格式，使开发集能走与提交完全同构的推理路径。

## 为什么需要它

`predict_submission_vllm.py` 是产出 0.8267 的那份工具，它支持 `--upscale`、`--samples`、
`--temperature`，也就是自一致性采样投票的完整能力。但它只吃官方 queries 格式与官方目录布局。

而开发集是 jsonl、答案在 `solution` 字段、提示词已经拼在 `messages` 里。**两者的差异只是
外壳，核心信息（三张图路径 + 查询文本）完全一致。**

用这条路径而不是 `eval_quant_grounding.py` 的理由：**后者没有 `--upscale`**。它把文件路径
直接交给 ms-swift，放大只能靠预先生成放大图——既不干净也慢。而预测工具在 `build_requests`
里用 PIL 现放大，与提交时逐字相同。

## 提示词一致性检查

预测工具会**从查询文本重新生成提示词**，不使用 jsonl 里存的 `messages`。所以必须确认
两者的提示词是同一套模板，否则开发集上的结论不能外推到提交。

本工具把 jsonl 里的提示词与 `PROMPTS["trimodal"](query)` 逐条比对，**不一致就报错**，
并把查询文本从 `messages` 里抽出来（jsonl 没有独立的 query 字段）。

## 用法

    python tools/dev_to_queries.py --dataset city400_dev.jsonl --out-queries dev_queries.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.predict_native_submission import IMAGE_FIELDS, PROMPTS  # noqa: E402

QUERY_PATTERN = re.compile(r"this query:\s*(.*?)\s*Return only JSON", re.S)


def extract_query(content: str) -> str:
    match = QUERY_PATTERN.search(content)
    if match is None:
        raise ValueError(f"提示词里找不到查询文本: {content[:120]!r}")
    return match.group(1).strip()


def convert(rows: list[dict], modality: str = "trimodal") -> dict[str, dict[str, str]]:
    template = PROMPTS[modality]
    source: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("每条记录必须有非空 id")
        if sample_id in source:
            raise ValueError(f"id 重复: {sample_id}")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"{sample_id}: 需要恰好三张图（rgb/infrared/depth），实际 {images}")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[0].get("role") != "user":
            raise ValueError(f"{sample_id}: 缺少 user 消息")
        content = messages[0].get("content")
        query = extract_query(content)
        expected = template(query)
        if content != expected:
            # 不静默放过：提示词不同意味着开发集测的不是提交时那条路径，结论不可外推。
            raise ValueError(
                f"{sample_id}: 存下的提示词与 {modality} 模板不一致；"
                f"开发集不能用于评估提交路径（可能是别的模ality 或别的 depth_policy）")
        source[sample_id] = {"query": query,
                             **{key: images[index] for index, key in enumerate(IMAGE_FIELDS)}}
    if not source:
        raise ValueError("输入为空")
    return source


def main() -> dict[str, dict[str, str]]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-queries", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "trimodal"), default="trimodal")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    source = convert(rows, args.modality)
    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    args.out_queries.write_text(json.dumps(source, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
    print(json.dumps({"out_queries": str(args.out_queries), "queries": len(source)},
                     ensure_ascii=False), flush=True)
    return source


if __name__ == "__main__":
    main()
