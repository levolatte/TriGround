"""把「一题多样本」的原始生成结果转成集成聚合器认的 candidates 格式。

## 为什么需要它

自一致性采样投票的链路是：

    一题采样 k 个回答  →  聚合成一个框  →  与单样本基线配对比较

中间这一步由 `tools/ensemble_grounding.py` 完成，它认的输入是
`{id, candidates: [{bbox, ...}]}`（一个成员多个候选）。而三个生成工具写出的都是
`{id, samples: [{text, ...}]}`。本工具就是这两者之间的唯一转换点。

**不为聚合器另写一套逻辑**：`ensemble_grounding.py` 已经实现 6 种聚合
（mean/median/medoid/consensus/cluster_medoid/wbf）、oracle 上界与 net_hits 统计，
并且明确不消费模型自报的置信度（`valid_boxes` 的注释）。自一致性正需要这种纯几何共识。

## 解析失败的样本怎么办

**丢弃样本，但保留题目。** 一道题 k 个样本全部解析失败时，它的 `candidates` 是空列表，
`ensemble_grounding` 会把它记为 `all_members_invalid` 并拒绝打包正式提交——这是正确的
失败方式（宁可拒绝生成提交包，也不静默塞一个兜底框）。

解析统计会打印出来，其中"全失败题数"是思考模式实验的格式率探针依据。

## 用法

    python tools/make_candidates.py --input raw.jsonl --output cands.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quant_grounding_reward import box_is_valid, parse_bbox  # noqa: E402


def sample_box(sample: dict, answer_mode: str) -> list[float] | None:
    """优先用上游已解析的 bbox，否则从原始文本解析。

    上游（eval_quant_grounding）只存 text；本仓库的预测工具在改造后会同时存 bbox。
    两条路都要认，否则同一份数据换个来源就要改代码。
    """
    box = sample.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        values = [float(v) for v in box]
        return values if box_is_valid(values) else None
    text = sample.get("text")
    return parse_bbox(text, answer_mode=answer_mode) if isinstance(text, str) else None


def convert(rows: list[dict], answer_mode: str) -> tuple[list[dict], dict]:
    out: list[dict] = []
    total_samples = 0
    parsed_samples = 0
    empty_questions: list[str] = []
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("每条记录必须有非空 id")
        samples = row.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"{sample_id}: 缺少 samples")
        candidates = []
        for index, sample in enumerate(samples):
            total_samples += 1
            box = sample_box(sample, answer_mode)
            if box is None:
                continue
            parsed_samples += 1
            candidates.append({"bbox": box, "sample_index": index,
                               "raw_text": sample.get("text")})
        if not candidates:
            empty_questions.append(sample_id)
        out.append({"id": sample_id, "candidates": candidates})
    stats = {"questions": len(out), "samples": total_samples,
             "parsed_samples": parsed_samples,
             "sample_parse_rate": round(parsed_samples / total_samples, 4) if total_samples else None,
             "questions_without_any_candidate": len(empty_questions),
             "question_success_rate": round((len(out) - len(empty_questions)) / len(out), 4) if out else None,
             "empty_examples": empty_questions[:10]}
    return out, stats


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answer-mode", choices=("short", "thinking"), default="short")
    parser.add_argument("--out-stats", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"{args.input}: 空文件")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("输入里有重复 id")
    converted, stats = convert(rows, args.answer_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
                           encoding="utf-8")
    if args.out_stats is not None:
        args.out_stats.parent.mkdir(parents=True, exist_ok=True)
        args.out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


if __name__ == "__main__":
    main()
