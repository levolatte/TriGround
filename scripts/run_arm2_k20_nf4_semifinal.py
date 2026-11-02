"""Arm2 ck500 的 NF4 + 2 倍放大完整推理：分离「量化代价」与「纠错收益」。

**这是预置弹药，不是当前任务。** 只有在 10-31 提交纠错1600 的包、发现官方分不高于 0.8267
之后才需要启动——那时它才能回答"掉分是因为 NF4 还是因为纠错训练"。

与 `run_correction1600_k20_nf4_semifinal.py` 的唯一区别是适配器换成复赛的 Arm2 ck500，
其余三项分辨率设置、输入清单、模型、格式策略、打包流程完全一致，因此两份结果可直接对比。

复赛口径下 Arm2 ck500 + BF16 + 2 倍放大是半决赛已知最好的 0.8267。适配器实测位于
`/root/autodl-tmp/runs/retained_baselines/q36_arm2_checkpoint-500`（注意不在 `quant_rl_20261026`
项目目录下），内容为 adapter_config.json + adapter_model.safetensors(933,974,032 字节)
+ additional_config.json + args.json，r32/alpha64/dropout0.05。

参考运行必须是 `runs/semifinal_correction1600_20261029`——它有 4 段 resume_history，
且当前在跑的任务用的就是它；用它才能保证除适配器外与在跑的那批完全同口径。

用法（云端）：
    /root/vllm_env/bin/python scripts/run_arm2_k20_nf4_semifinal.py \
        --reference-run  /root/autodl-tmp/quant_rl_20261026/runs/semifinal_correction1600_20261029 \
        --adapter        /root/autodl-tmp/runs/retained_baselines/q36_arm2_checkpoint-500 \
        --output-dir     /root/autodl-tmp/quant_rl_20261026/runs/semifinal_arm2_k20_nf4_20261031
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.predict_vllm_quant_submission import main as predict, build_llm_options
from tools.predict_quant_submission import package_predictions, load_queries
from tools.eval_quant_grounding import read_jsonl

CHANGES = {"upscale": 2.0, "max_pixels": 8294400, "max_length": 32768}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    ref = json.loads((args.reference_run / "full/run_config.json").read_text())
    previous = [x for x in ref["resume_history"] if x.get("engine") == "vllm"][-1]
    config = {**ref, **CHANGES}
    expected = deepcopy(previous["options"])
    expected["max_model_len"] = CHANGES["max_length"]
    expected["mm_processor_kwargs"]["max_pixels"] = CHANGES["max_pixels"]
    assert build_llm_options(config, 8) == expected, "engine options drifted from the reference run"

    manifest = {"reference_run": str(args.reference_run), "adapter": str(args.adapter),
                "comparison": "correction1600 run swapped back to the rematch Arm2 ck500 adapter",
                "changes": {k: [ref[k], v] for k, v in CHANGES.items()},
                "adapter_change": [ref["adapter"], str(args.adapter)],
                "llm_options": expected, "chunk_size": previous["chunk_size"],
                "max_tokens": ref["max_tokens"], "seed": ref["seed"], "temperature": 0,
                "format_policy": ref["format_policy"], "resampling": "PIL LANCZOS",
                "automatic_parameter_fallback": False}
    (args.output_dir / "parameter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    state = {"status": "running", "stage": "loading", "pid": os.getpid(), "expected_queries": 5691}

    def save(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        (args.output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        argv = ["--dataset", str(args.reference_run / "full/unlabeled.jsonl"),
                "--output-dir", str(args.output_dir / "full"), "--adapter", str(args.adapter),
                "--max-num-seqs", "8", "--chunk-size", str(previous["chunk_size"]), "--upscale", "2.0"]
        for key in ("model", "data_root", "min_pixels", "max_pixels", "max_length", "max_tokens", "seed"):
            argv += ["--" + key.replace("_", "-"), str(config[key])]
        summary = predict(argv)
        full = args.output_dir / "full"
        actual = json.loads((full / "run_config.json").read_text())
        assert actual["llm_options"] == expected and actual["upscale"] == 2.0
        save(stage="packaging")
        raw = read_jsonl(full / "predictions_raw.jsonl")
        source = load_queries(Path(ref["queries"]), 5691)
        assert [row["id"] for row in raw] == list(source)
        summary.update(package_predictions(source, raw, full, ref["format_policy"]))
        summary["parameter_manifest"] = str(args.output_dir / "parameter_manifest.json")
        (full / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        save(status="completed", stage="completed", completed_queries=len(raw),
             submission=str(full / "submission.zip"))
    except Exception as error:
        save(status="failed", stage="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
