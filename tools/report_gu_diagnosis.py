"""Raw-GT diagnostic summaries for the bounded B/G*/U* experiment."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from tools.report_rematch_experiment import (
    box_iou,
    build_report,
    compare_runs,
    load_manifest,
    load_run,
    score_run as base_score_run,
    write_outputs,
)


SHARED_SEQUENCES = (
    "000002_025", "000002_041", "000002_043", "000002_074",
    "000013_016", "000014_025", "000014_052",
)
SHARED_PATTERN = re.compile(r"(?<!\d)(?:" + "|".join(SHARED_SEQUENCES) + r")(?:_suppl)?_")
MODES = ("rgb", "rgb_ir", "rgb_depth", "trimodal")


def score_run(item: dict, gt: dict) -> dict:
    metrics = base_score_run(item, gt)
    n = metrics["samples"]
    return {
        **metrics,
        "hits_0.5": round(metrics["acc_0.5"] * n) if n else 0,
        "hits_0.7": round(metrics["acc_0.7"] * n) if n else 0,
    }


def run(output: Path, name: str, gt: dict) -> dict:
    return load_run(name, output / name / "predictions.jsonl", set(gt), allow_partial=False)


def paired(left: dict, right: dict, gt: dict) -> dict:
    result = compare_runs(left, right, gt)
    return {
        "left": left["name"],
        "right": right["name"],
        "samples": result["samples"],
        "delta": result["delta"],
        "wrong_to_right": result["wrong_to_right"],
        "right_to_wrong": result["right_to_wrong"],
        "image_group_bootstrap": result["image_group_bootstrap"],
    }


def fit_summary(output: Path, models: tuple[str, ...], suffix: str) -> dict:
    m = output / "manifests"
    gt = load_manifest(m / f"fit_{suffix}_gt.json")
    metadata = {
        str(row["id"]): row
        for row in (json.loads(line) for line in (m / f"fit_{suffix}_metadata.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert set(metadata) == set(gt)
    query_keys = {sample_id: str(metadata[sample_id]["query_key"]) for sample_id in gt}
    modalities = {sample_id: str(metadata[sample_id]["target_modality"]) for sample_id in gt}
    result = {
        "outputs": len(gt),
        "unique_queries": len(set(query_keys.values())),
        "by_target_modality": {
            mode: {
                "outputs": sum(value == mode for value in modalities.values()),
                "unique_queries": len({query_keys[sample_id] for sample_id in gt if modalities[sample_id] == mode}),
            }
            for mode in sorted(set(modalities.values()))
        },
        "models": {},
        "pairs": [],
    }
    if suffix == "legacy181":
        assert result["outputs"] == 181 and result["unique_queries"] == 98
        assert {mode: entry["outputs"] for mode, entry in result["by_target_modality"].items()} == {
            "rgb": 98, "infrared": 58, "depth": 25,
        }
    runs = {model: run(output, f"{model}_fit_{suffix}", gt) for model in models}
    for model, item in runs.items():
        by_mode = {}
        for mode in sorted(set(modalities.values())):
            subset = {sample_id: gt[sample_id] for sample_id in gt if modalities[sample_id] == mode}
            by_mode[mode] = score_run(item, subset)
        query_hits = defaultdict(list)
        for sample_id, record in gt.items():
            box = item["rows"][sample_id]["primary"]
            iou = box_iou(box, record["bbox"]) if box is not None else 0.0
            query_hits[query_keys[sample_id]].append(float(iou >= 0.5))
        result["models"][model] = {
            "all": score_run(item, gt),
            "by_target_modality": by_mode,
            "query_macro_acc_0.5": sum(sum(values) / len(values) for values in query_hits.values()) / len(query_hits),
        }
    for left, right in (("m2", "g_old"), ("m2", "u_old"), ("m2", "gstar"), ("m2", "ustar"), ("g_old", "gstar"), ("u_old", "ustar"), ("gstar", "ustar")):
        if left in runs and right in runs:
            result["pairs"].append(paired(runs[left], runs[right], gt))
    return result


def prompt_sensitivity(output: Path, models: tuple[str, ...]) -> dict:
    # The canonical 83 are exactly the auxiliary-output IDs within legacy 181.
    # Recompute both prompts against the same original GT and the same IDs.
    gt = load_manifest(output / "manifests/fit_canonical_aux83_gt.json")
    metadata = {
        str(row["id"]): row
        for row in (json.loads(line) for line in (output / "manifests/fit_canonical_aux83_metadata.jsonl").read_text(encoding="utf-8").splitlines())
    }
    assert set(metadata) == set(gt)
    result = {}
    for model in models:
        legacy_gt = load_manifest(output / "manifests/fit_legacy181_gt.json")
        assert all(legacy_gt[sample_id]["bbox"] == record["bbox"] for sample_id, record in gt.items())
        legacy = run(output, f"{model}_fit_legacy181", legacy_gt)
        canonical = run(output, f"{model}_fit_canonical_aux83", gt)
        assert set(gt) <= set(legacy["rows"])
        result[model] = {
            **paired(legacy, canonical, gt),
            "by_target_modality": {
                mode: paired(
                    legacy, canonical,
                    {sample_id: gt[sample_id] for sample_id in gt if metadata[sample_id]["target_modality"] == mode},
                )
                for mode in sorted({row["target_modality"] for row in metadata.values()})
            },
        }
    return result


def city96_summary(output: Path, models: tuple[str, ...]) -> dict:
    gt = load_manifest(output / "manifests/city96_gt.json")
    assert len(gt) == 96
    runs = {
        model: {mode: run(output, f"{model}_city96_{mode}", gt) for mode in MODES}
        for model in models
    }
    result = {"samples": 96, "models": {}, "same_input_model_pairs": {}}
    for model, by_mode in runs.items():
        result["models"][model] = {
            "metrics": {mode: score_run(item, gt) for mode, item in by_mode.items()},
            "input_gain_vs_rgb": {
                mode: paired(by_mode["rgb"], by_mode[mode], gt)
                for mode in MODES if mode != "rgb"
            },
            "incremental_auxiliary_gain": {
                "depth_given_ir": paired(by_mode["rgb_ir"], by_mode["trimodal"], gt),
                "ir_given_depth": paired(by_mode["rgb_depth"], by_mode["trimodal"], gt),
            },
        }
    for mode in MODES:
        result["same_input_model_pairs"][mode] = [
            paired(runs[left][mode], runs[right][mode], gt)
            for left, right in (("m2", "c"), ("m2", "g_old"), ("m2", "u_old"), ("c", "gstar"), ("c", "ustar"), ("gstar", "ustar"))
            if left in runs and right in runs
        ]
    return result


def city412_report(output: Path, root: Path) -> dict:
    gt = load_manifest(root / "data/city/train/target_v2/qwen_generation_val.json")
    assert len(gt) == 412
    sources = {
        "M2": root / "results/next_stage_20260925/baseline_m2/predictions.jsonl",
        "C": root / "results/next_stage_20260925/c_step1500/predictions.jsonl",
        "G_old": root / "results/gu_pilot_20260926/g_2026_city200/predictions.jsonl",
        "U_old": root / "results/gu_pilot_20260926/u_2026_city200/predictions.jsonl",
        "B": output / "b_city412/predictions.jsonl",
        "Gstar": output / "gstar_city412/predictions.jsonl",
        "Ustar": output / "ustar_city412/predictions.jsonl",
    }
    runs = [load_run(name, path, set(gt), allow_partial=False) for name, path in sources.items()]
    report = build_report(gt, runs)
    write_outputs(report, output / "report_city412")
    shared = {sample_id: record for sample_id, record in gt.items() if SHARED_PATTERN.search(str(record["visible"]))}
    other = {sample_id: record for sample_id, record in gt.items() if sample_id not in shared}
    assert len(shared) == 47 and len(other) == 365, (len(shared), len(other))
    return {
        "source": "seven visually confirmed shared-location sequence patterns; different frames, not pixel duplicates",
        "shared47_ids": list(shared),
        "other365_ids": list(other),
        "models": {
            item["name"]: {"all412": score_run(item, gt), "shared47": score_run(item, shared), "other365": score_run(item, other)}
            for item in runs
        },
    }


def write(output: Path, name: str, data: dict) -> None:
    dest = output / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# G/U 诊断结果", "", "原始 GT 重算；City 96 四输入共享同一 ID/GT，fit canonical83 与 legacy181 只在共同 83 ID 上成对比较。输入增益和提示敏感性是描述性差异，不代表已识别因果机制。", ""]
    if "fit_legacy181" in data:
        fit = data["fit_legacy181"]
        lines += [f"fit：{fit['outputs']} 输出任务，{fit['unique_queries']} 个唯一 Query（按 query_key 去重）。", ""]
        for model, metrics in fit["models"].items():
            all_metrics = metrics["all"]
            lines.append(f"- {model}: ACC@0.5={all_metrics['hits_0.5']}/{all_metrics['samples']} ({all_metrics['acc_0.5']:.4f})，mIoU={all_metrics['mean_iou']:.4f}，ACC@0.7={all_metrics['hits_0.7']}/{all_metrics['samples']} ({all_metrics['acc_0.7']:.4f})；Query 宏平均 ACC@0.5={metrics['query_macro_acc_0.5']:.4f}。")
    if "city96" in data:
        lines += ["", "City96 输入模式（每模态同 96 ID）", ""]
        for model, item in data["city96"]["models"].items():
            lines.append(f"- {model}: " + "; ".join(f"{mode} ACC@0.5={item['metrics'][mode]['hits_0.5']}/96 ({item['metrics'][mode]['acc_0.5']:.4f})" for mode in MODES))
    if "city412" in data:
        lines += ["", "City412 已确认共享地点序列 47 / 其余 365（只按已目视确认的 7 种文件名模式分项）", ""]
        for model, item in data["city412"]["models"].items():
            lines.append(f"- {model}: 全412 ACC@0.5={item['all412']['hits_0.5']}/412 ({item['all412']['acc_0.5']:.4f})；47条={item['shared47']['hits_0.5']}/47；365条={item['other365']['hits_0.5']}/365；ACC@0.7={item['all412']['hits_0.7']}/412。")
    lines += ["", "成对图像组 10,000 次 bootstrap 区间、ACC@0.7、逐模态和逐 Query 统计见 summary.json；完整 City412 比较见 report_city412/report.md。", ""]
    (dest / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Rematch root; defaults to the parent of results/")
    parser.add_argument("--phase", choices=("pre", "city412", "final"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "city412":
        root = args.root.resolve() if args.root else output.parent.parent
        write(output, "report_city412_diagnostic", {"city412": city412_report(output, root)})
        return
    models = ("m2", "g_old", "u_old") if args.phase == "pre" else ("m2", "g_old", "u_old", "gstar", "ustar")
    city_models = ("m2", "c", "g_old", "u_old") if args.phase == "pre" else ("m2", "c", "g_old", "u_old", "gstar", "ustar")
    data = {
        "fit_legacy181": fit_summary(output, models, "legacy181"),
        "fit_canonical_aux83": fit_summary(output, models, "canonical_aux83"),
        "prompt_sensitivity_on_same_83_ids": prompt_sensitivity(output, models),
        "city96": city96_summary(output, city_models),
    }
    write(output, f"report_{args.phase}", data)


if __name__ == "__main__":
    main()
