"""Summarize available fixed-cohort results; never launch an optional experiment."""
import argparse
import csv
import json
from pathlib import Path
import time

from tools.compare_grounding_runs import compare, load_rows


MODES = ("rgb_baseline", "rgb_ir", "rgb_depth", "rgb_ir_depth")


def screening_decision(native, control, treatment):
    if any(value is None for value in (native, control, treatment)):
        return {"status": "pending", "next": "complete_mandatory", "significance_claim": False}
    net = treatment["hits"] - control["hits"]
    meets = net >= 3 and treatment["hits"] >= native["hits"]
    return {"status": "screening_only", "net_hits": net,
            "next": "paired_c4_t4_seed2027" if meets else "w4",
            "parse_noninferior": treatment["parse_rate"] >= control["parse_rate"],
            "significance_claim": False}


def summarize(run_dir):
    run_dir = Path(run_dir).resolve()
    results = run_dir / "results"
    results.mkdir(exist_ok=True)
    rows = []
    for path in sorted(results.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if "diagnostic_metadata" not in report:
            continue
        n = report["diagnostic_metadata"]["samples"]
        for mode in MODES:
            if mode in report:
                metric = report[mode]
                rows.append({"experiment": path.stem, "mode": mode,
                             "samples": n, "hits": round(metric["acc_0.5"] * n),
                             **{k: metric[k] for k in ("acc_0.5", "mean_iou", "parse_rate")}})
    metrics_file = run_dir / "outputs/c4/metrics.jsonl"
    if metrics_file.exists():
        for line in metrics_file.read_text().splitlines():
            metric = json.loads(line)
            if metric.get("epoch") == 1 and metric.get("eval_scope") == "subset":
                n = metric["eval_samples"]
                rows.append({"experiment": "c4_epoch1", "mode": "rgb_ir_depth",
                             "samples": n, "hits": round(metric["acc_0.5"] * n),
                             **{k: metric[k] for k in ("acc_0.5", "mean_iou", "parse_rate")}})
    fields = ["experiment", "mode", "samples", "hits", "acc_0.5", "mean_iou", "parse_rate"]
    with (results / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r["experiment"], r["mode"]): r for r in rows}
    native = lookup.get(("native_rgb", "rgb_baseline"))
    control = lookup.get(("c4_fixed4", "rgb_ir_depth"))
    treatment = lookup.get(("t4_fixed4", "rgb_ir_depth"))
    decision = screening_decision(native, control, treatment)
    start = run_dir / "experiment_started_unix.txt"
    decision["elapsed_hours"] = (time.time() - int(start.read_text())) / 3600 if start.exists() else None
    comparisons = {}
    for left, right in (("c4", "t4"), ("c4", "w4"),
                        ("c4_seed2027", "t4_seed2027"), ("c4_seed2027", "w4_seed2027")):
        paths = [results / f"{name}_fixed4_rows.jsonl" for name in (left, right)]
        if all(p.exists() for p in paths):
            paired = compare(*(load_rows(p, "rgb_ir_depth") for p in paths), 2000, 2026)
            key = f"{left}_vs_{right}"
            (results / f"{key}_paired.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
            comparisons[key] = {k: paired[k] for k in ("delta", "transitions", "scene_bootstrap")}
    decision["paired_results"] = comparisons
    decision["best_candidates"] = sorted(
        [r for r in rows if r["experiment"].endswith("_best") and r["mode"] == "rgb_ir_depth"],
        key=lambda r: (r["hits"], r["mean_iou"], r["parse_rate"]), reverse=True,
    )
    warm = lookup.get(("w4_fixed4", "rgb_ir_depth"))
    if warm and control and native:
        if (treatment is None or (warm["hits"], warm["mean_iou"]) > (treatment["hits"], treatment["mean_iou"])) and warm["hits"] > control["hits"] and warm["hits"] >= native["hits"]:
            decision["next"] = "paired_c4_w4_seed2027"
    (results / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    text = ["# 8B 三模态实验汇总", "", "固定第 4 轮用于主对比；best 仅用于候选选模。缺失结果表示尚未完成。", "",
            "| 实验 | 模态 | 命中/总数 | ACC@0.5 | mIoU | 解析率 |",
            "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f'| {row["experiment"]} | {row["mode"]} | {row["hits"]}/{row["samples"]} | {row["acc_0.5"]:.4f} | {row["mean_iou"]:.4f} | {row["parse_rate"]:.4f} |')
    text += ["", f'后续建议：`{decision["next"]}`；实际启动还要满足剩余预算。',
             "净增 3/119 仅是优先复验门槛，不是统计显著性证明。第二个 seed 仅复验目标域阶段。",
             "各场景重采样区间和成对错变对/对变错见 paired.json；小分组仅作诊断。",
             "历史 284 条应在方案固定后单独复查一次；本工具不会读取该集合。"]
    (results / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir), ensure_ascii=False, indent=2))
