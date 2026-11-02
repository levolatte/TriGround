"""比较 vLLM 引擎选项在同一批题目上的速度与输出一致性。

背景：`build_llm_options` 里的 `enforce_eager` / `max_num_batched_tokens` /
`enable_prefix_caching` 原本是一次 15 分钟探针的临时设置，后来被硬编码成规范。
本脚本用来实测它们到底值不值得改。

一次调用只跑一个变体（每个变体都要重建引擎，同进程反复建容易留下显存碎片）：

    python tools/benchmark_vllm_engine_options.py \
        --run-dir <已完成运行的根目录> --baseline-raw <其 full/predictions_raw.jsonl> \
        --variant cuda_graph --limit 16 --output-dir <输出目录>

判据沿用 2026-10-29 那次批量实测确立的标准：**原文完全一致**且提速至少 15% 才可采用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.eval_quant_grounding import positive_int, read_jsonl
from tools.predict_vllm_quant_submission import build_llm_options, run_chunks

# 变体只动引擎构造参数，不动分辨率、提示、采样与适配器。
VARIANTS = {
    "baseline": {},
    "cuda_graph": {"enforce_eager": False},
    "big_batch": {"max_num_batched_tokens": 32768},
    "both": {"enforce_eager": False, "max_num_batched_tokens": 32768},
}
SPEEDUP_THRESHOLD = 1.15


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--limit", type=positive_int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.run_dir / "full/run_config.json").read_text(encoding="utf-8"))
    dataset = read_jsonl(args.run_dir / "full/unlabeled.jsonl")[:args.limit]
    if not dataset:
        raise ValueError("no queries to benchmark")

    options = build_llm_options(reference, reference["max_num_seqs"])
    options.update(VARIANTS[args.variant])
    for key, value in VARIANTS[args.variant].items():
        if options[key] != value:
            raise ValueError(f"variant override did not take effect: {key}")

    chunk_args = SimpleNamespace(output_dir=args.output_dir / "full",
                                 data_root=Path(reference["data_root"]),
                                 resume=False, chunk_size=reference["chunk_size"],
                                 upscale=reference["upscale"])
    chunk_args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    processor = AutoProcessor.from_pretrained(reference["model"])
    started = time.monotonic()
    llm = LLM(**options)
    loading_seconds = time.monotonic() - started
    sampling = SamplingParams(temperature=0, max_tokens=reference["max_tokens"], seed=reference["seed"])
    lora = LoRARequest("grounding", 1, str(reference["adapter"]))
    resource = run_chunks(chunk_args, dataset, [], llm, processor, sampling, lora)

    produced = read_jsonl(chunk_args.output_dir / "predictions_raw.jsonl")
    wanted = {row["id"] for row in dataset}
    baseline_rows = [row for row in read_jsonl(args.baseline_raw) if row["id"] in wanted]
    baseline = {row["id"]: row["samples"][0]["text"] for row in baseline_rows}
    missing = [row["id"] for row in produced if row["id"] not in baseline]
    if missing:
        raise ValueError(f"baseline has no answer for {len(missing)} benchmarked queries: {missing[:3]}")
    differs = [row["id"] for row in produced if row["samples"][0]["text"] != baseline[row["id"]]]

    # 基线耗时按同批题在基线 raw 里记录的每题秒数求和，避免用整轮平均值。
    baseline_seconds = sum(row["samples"][0]["seconds"] for row in baseline_rows)
    seconds = resource["generation_seconds"]
    speedup = baseline_seconds / seconds if seconds else 0.0

    accepted = not differs and speedup >= SPEEDUP_THRESHOLD
    report = {
        "variant": args.variant,
        "overrides": VARIANTS[args.variant],
        "limit": len(dataset),
        "llm_options": options,
        "loading_seconds": loading_seconds,
        "generation_seconds": seconds,
        "baseline_generation_seconds": baseline_seconds,
        "speedup": round(speedup, 4),
        "text_matches": len(produced) - len(differs),
        "text_differs": len(differs),
        "differing_ids": differs[:10],
        "decision": "adopt" if accepted else "reject",
        "reason": ("outputs identical and faster than threshold" if accepted else
                   "outputs differ" if differs else
                   f"speedup {speedup:.2f} below {SPEEDUP_THRESHOLD}"),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2), flush=True)
