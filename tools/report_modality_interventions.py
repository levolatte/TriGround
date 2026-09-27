"""Re-score fixed-checkpoint modality interventions against the raw GT manifest.

The report compares only complete, same-ID runs.  It intentionally does not
merge intervention predictions into a candidate pool: corrupted modalities
are diagnostics, not candidate sources.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from tools import report_rematch_experiment as rematch


EXPECTED_SAMPLES = 412
EXPECTED_IMAGE_GROUPS = 78
DEFAULT_BOOTSTRAP_REPLICATES = rematch.BOOTSTRAP_REPLICATES
DEFAULT_BOOTSTRAP_SEED = rematch.BOOTSTRAP_SEED

INTERVENTION_LABELS = {
    "normal": "5090 正常三模态",
    "ir_black": "红外置黑",
    "depth_black": "深度置黑",
    "both_black": "红外与深度置黑",
    "ir_shuffle": "红外异场景置换",
    "depth_shuffle": "深度异场景置换",
    "both_shuffle": "红外与深度异场景置换",
}

DISTANCE_WORDS = re.compile(r"\b(?:near|nearer|nearest|far|farther|farthest|close|closer|closest)\b", re.I)
ORDINAL_WORDS = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b", re.I
)


def _empty_run(name: str, path: Path, manifest: dict[str, dict[str, Any]], status: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": {},
        "missing_ids": list(manifest),
        "partial": True,
        "metadata": None,
        "status": status,
    }


def load_optional_run(
    name: str, path: Path, manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Load a possibly not-yet-finished arm, retaining its progress status."""
    if not path.exists():
        return _empty_run(name, path, manifest, "not_started")
    if not any(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()):
        return _empty_run(name, path, manifest, "pending")
    run = rematch.load_run(name, path, set(manifest), allow_partial=True)
    run["status"] = "partial" if run["partial"] else "complete"
    return run


def _run_summary(
    run: dict[str, Any], manifest: dict[str, dict[str, Any]], *, label: str, source: str
) -> dict[str, Any]:
    complete = run["status"] == "complete"
    metrics = rematch.score_run(run, manifest) if run["rows"] else None
    return {
        "name": run["name"],
        "label": label,
        "source": source,
        "path": run["path"],
        "status": run["status"],
        "samples": len(run["rows"]),
        "manifest_samples": len(manifest),
        "missing_samples": len(run["missing_ids"]),
        "missing_ids": run["missing_ids"],
        "comparison_eligible": complete,
        ("metrics" if complete else "observed_metrics"): metrics,
        "metadata": run["metadata"],
    }


def _raw_text(row: dict[str, Any]) -> str | None:
    for key in ("raw_text", "answer"):
        value = row["raw"].get(key)
        if isinstance(value, str):
            return value
    return None


def _query_strata(
    sample_ids: list[str], manifest: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    distance: list[str] = []
    ordinal: list[str] = []
    for sample_id in sample_ids:
        query = str(manifest[sample_id].get("query", ""))
        if DISTANCE_WORDS.search(query):
            distance.append(sample_id)
        if ORDINAL_WORDS.search(query):
            ordinal.append(sample_id)
    return {"distance_keyword": distance, "ordinal_keyword": ordinal}


def _metrics_for_ids(
    run: dict[str, Any], sample_ids: list[str], manifest: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ious = []
    parsed = 0
    for sample_id in sample_ids:
        prediction = run["rows"][sample_id]["primary"]
        if prediction is None:
            ious.append(0.0)
        else:
            parsed += 1
            ious.append(rematch.box_iou(prediction, manifest[sample_id]["bbox"]))
    return rematch._metrics(ious, parsed)


def compare_complete_runs(
    left: dict[str, Any],
    right: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Add box/text stability and query-only exploratory slices to a paired score."""
    if left["status"] != "complete" or right["status"] != "complete":
        raise ValueError("paired comparisons require two complete 412-ID runs")

    comparison = rematch.compare_runs(
        left,
        right,
        manifest,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    sample_ids = list(manifest)
    both_valid = 0
    same_box_count = 0
    coordinate_l1_values: list[float] = []
    raw_text_compared = 0
    raw_text_changed = 0
    for sample_id in sample_ids:
        left_row = left["rows"][sample_id]
        right_row = right["rows"][sample_id]
        left_box, right_box = left_row["primary"], right_row["primary"]
        if left_box is not None and right_box is not None:
            both_valid += 1
            same_box_count += int(left_box == right_box)
            coordinate_l1_values.append(
                sum(abs(first - second) for first, second in zip(left_box, right_box)) / 4
            )
        left_text, right_text = _raw_text(left_row), _raw_text(right_row)
        if left_text is not None and right_text is not None:
            raw_text_compared += 1
            raw_text_changed += int(left_text != right_text)

    left_metrics = comparison["left_metrics"]
    right_metrics = comparison["right_metrics"]
    strata = _query_strata(sample_ids, manifest)
    exploratory = {}
    for name, ids in strata.items():
        if not ids:
            continue
        left_slice = _metrics_for_ids(left, ids, manifest)
        right_slice = _metrics_for_ids(right, ids, manifest)
        exploratory[name] = {
            "samples": len(ids),
            "ids": ids,
            "left_metrics": left_slice,
            "right_metrics": right_slice,
            "delta": {
                "mean_iou": right_slice["mean_iou"] - left_slice["mean_iou"],
                "acc_0.5": right_slice["acc_0.5"] - left_slice["acc_0.5"],
            },
        }

    comparison.update(
        {
            "status": "complete",
            "same_valid_box_count": same_box_count,
            "both_valid_box_count": both_valid,
            "mean_coordinate_l1_when_both_valid": (
                sum(coordinate_l1_values) / len(coordinate_l1_values)
                if coordinate_l1_values
                else None
            ),
            "raw_text_compared_count": raw_text_compared,
            "raw_text_changed_count": raw_text_changed if raw_text_compared else None,
            "exploratory_query_slices": exploratory,
            "left_metrics": left_metrics,
            "right_metrics": right_metrics,
        }
    )
    return comparison


def _excluded_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    missing = [run["name"] for run in (left, right) if run["status"] != "complete"]
    return {
        "left": left["name"],
        "right": right["name"],
        "status": "excluded_incomplete_run",
        "incomplete_runs": missing,
        "message": "完整 412 条同 ID 前不计算最终成对指标。",
    }


def build_report(
    manifest: dict[str, dict[str, Any]],
    experiment_dir: Path,
    old_m2_path: Path,
    rgb_path: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    arms = {
        name: load_optional_run(name, experiment_dir / name / "predictions.jsonl", manifest)
        for name in INTERVENTION_LABELS
    }
    old_m2 = load_optional_run("old_m2_4090", old_m2_path, manifest)
    rgb = load_optional_run("rgb_4090", rgb_path, manifest)

    intervention_pairs = []
    for name in INTERVENTION_LABELS:
        if name == "normal":
            continue
        normal, intervention = arms["normal"], arms[name]
        if normal["status"] == intervention["status"] == "complete":
            intervention_pairs.append(
                {
                    "kind": "5090_same_checkpoint_intervention",
                    **compare_complete_runs(
                        normal,
                        intervention,
                        manifest,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=bootstrap_seed,
                    ),
                }
            )
        else:
            intervention_pairs.append(_excluded_pair(normal, intervention))

    if arms["normal"]["status"] == old_m2["status"] == "complete":
        cross_hardware = {
            "kind": "5090_vs_4090_same_checkpoint_hardware_diagnostic",
            **compare_complete_runs(
                arms["normal"],
                old_m2,
                manifest,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    else:
        cross_hardware = _excluded_pair(arms["normal"], old_m2)

    if rgb["status"] == old_m2["status"] == "complete":
        historical = {
            "kind": "4090_rgb_vs_old_m2_historical_diagnostic",
            **compare_complete_runs(
                rgb,
                old_m2,
                manifest,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    else:
        historical = _excluded_pair(rgb, old_m2)

    all_runs = [*arms.values(), old_m2, rgb]
    run_summaries = []
    for run in all_runs:
        label = INTERVENTION_LABELS.get(run["name"])
        if label is None:
            label = {
                "old_m2_4090": "旧 M2（4090）",
                "rgb_4090": "R2 RGB（4090）",
            }[run["name"]]
        source = "5090 干预实验" if run["name"] in INTERVENTION_LABELS else "4090 历史基线"
        run_summaries.append(_run_summary(run, manifest, label=label, source=source))

    return {
        "title": "5090 固定 M2 检查点模态干预诊断",
        "ground_truth_source": "raw_gt_manifest",
        "manifest_samples": len(manifest),
        "acc_threshold": rematch.ACC_THRESHOLD,
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "runs": run_summaries,
        "intervention_comparisons": intervention_pairs,
        "cross_hardware_comparison": cross_hardware,
        "historical_4090_rgb_vs_old_m2": historical,
        "caveats": [
            "置黑和图像组内打乱可能形成训练分布外输入；性能下降只说明输出对干预敏感，不能单凭下降证明模型正确使用该模态。",
            "距离词与序数词只从原始 Query 文本做词面匹配，分层结果仅供探索；它们不是人工语义标签，类别之间可以重叠。",
            "缺失或部分 run 只展示进度和已观测样本预览，不进入最终比较；完整比较均以 raw GT 的完整 manifest 和同一批 ID 重算。",
            "5090 与 4090 的比较单独标注为跨硬件诊断，不作为同硬件复现实验结论。",
        ],
    }


CSV_FIELDS = (
    "类别",
    "名称",
    "状态",
    "样本数",
    "总样本数",
    "分母",
    "解析率",
    "mIoU",
    "ACC@0.5",
    "ACC@0.7",
    "相同有效框数",
    "有效框配对数",
    "平均坐标L1",
    "原文比较数",
    "原文变化数",
    "左到右纠正数",
    "纠正涉及图像组",
    "左到右退化数",
    "退化涉及图像组",
    "mIoU差值",
    "ACC@0.5差值",
    "mIoU差值95%CI下界",
    "mIoU差值95%CI上界",
    "ACC@0.5差值95%CI下界",
    "ACC@0.5差值95%CI上界",
    "未纳入原因",
)


def _csv_metric_row(kind: str, name: str, status: str, metrics: dict[str, Any] | None, denominator: int) -> dict[str, Any]:
    if metrics is None:
        return {
            "类别": kind,
            "名称": name,
            "状态": status,
            "分母": denominator,
        }
    return {
        "类别": kind,
        "名称": name,
        "状态": status,
        "样本数": metrics["samples"],
        "分母": denominator,
        "解析率": metrics["parse_rate"],
        "mIoU": metrics["mean_iou"],
        "ACC@0.5": metrics["acc_0.5"],
        "ACC@0.7": metrics["acc_0.7"],
    }


def _comparison_csv_row(kind: str, comparison: dict[str, Any]) -> dict[str, Any]:
    if comparison.get("status") != "complete":
        return {
            "类别": kind,
            "名称": f"{comparison['left']} → {comparison['right']}",
            "状态": comparison["status"],
            "未纳入原因": ", ".join(comparison.get("incomplete_runs", [])),
        }
    bootstrap = comparison["image_group_bootstrap"]
    return {
        "类别": kind,
        "名称": f"{comparison['left']} → {comparison['right']}",
        "状态": "complete",
        "样本数": comparison["samples"],
        "总样本数": comparison["samples"],
        "分母": comparison["samples"],
        "解析率": comparison["right_metrics"]["parse_rate"],
        "mIoU": comparison["right_metrics"]["mean_iou"],
        "ACC@0.5": comparison["right_metrics"]["acc_0.5"],
        "ACC@0.7": comparison["right_metrics"]["acc_0.7"],
        "相同有效框数": comparison["same_valid_box_count"],
        "有效框配对数": comparison["both_valid_box_count"],
        "平均坐标L1": comparison["mean_coordinate_l1_when_both_valid"],
        "原文比较数": comparison["raw_text_compared_count"],
        "原文变化数": comparison["raw_text_changed_count"],
        "左到右纠正数": len(comparison["wrong_to_right"]["ids"]),
        "纠正涉及图像组": comparison["wrong_to_right"]["image_count"],
        "左到右退化数": len(comparison["right_to_wrong"]["ids"]),
        "退化涉及图像组": comparison["right_to_wrong"]["image_count"],
        "mIoU差值": comparison["delta"]["mean_iou"],
        "ACC@0.5差值": comparison["delta"]["acc_0.5"],
        "mIoU差值95%CI下界": bootstrap.get("mean_iou_delta_95ci", [None, None])[0],
        "mIoU差值95%CI上界": bootstrap.get("mean_iou_delta_95ci", [None, None])[1],
        "ACC@0.5差值95%CI下界": bootstrap.get("acc_0.5_delta_95ci", [None, None])[0],
        "ACC@0.5差值95%CI上界": bootstrap.get("acc_0.5_delta_95ci", [None, None])[1],
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    csv_rows = []
    for run in report["runs"]:
        metrics = run.get("metrics") or run.get("observed_metrics")
        row = _csv_metric_row(
            "实验臂",
            run["label"],
            run["status"],
            metrics,
            run["samples"] if run["status"] != "complete" else run["manifest_samples"],
        )
        row["样本数"] = run["samples"]
        row["总样本数"] = run["manifest_samples"]
        csv_rows.append(row)
    csv_rows.extend(
        _comparison_csv_row("5090 干预配对", pair)
        for pair in report["intervention_comparisons"]
    )
    csv_rows.append(_comparison_csv_row("5090 与 4090 跨硬件", report["cross_hardware_comparison"]))
    csv_rows.append(_comparison_csv_row("4090 历史 RGB 与 M2", report["historical_4090_rgb_vs_old_m2"]))
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    def fmt(value: Any) -> str:
        return "—" if value is None else f"{value:.4f}"

    lines = [
        f"# {report['title']}",
        "",
        f"Raw GT manifest 样本数：{report['manifest_samples']}；ACC 命中阈值：{report['acc_threshold']:.1f}。",
        f"图像组配对 bootstrap：{report['bootstrap']['replicates']} 次，seed={report['bootstrap']['seed']}。",
        "",
        "## 实验臂进度与 raw GT 重算分数",
        "",
        "完整臂的指标使用全量 manifest 分母；partial 臂的预览只按已返回样本计，不参与任何配对结论。",
        "",
        "| 实验臂 | 状态 | 已完成/总数 | 计分分母 | 解析率 | mIoU | ACC@0.5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        metrics = run.get("metrics") or run.get("observed_metrics") or {}
        denominator = run["manifest_samples"] if run["status"] == "complete" else run["samples"]
        lines.append(
            f"| {run['label']} | {run['status']} | {run['samples']}/{run['manifest_samples']} | {denominator} | "
            f"{fmt(metrics.get('parse_rate'))} | {fmt(metrics.get('mean_iou'))} | "
            f"{fmt(metrics.get('acc_0.5'))} |"
        )

    def append_comparison(title: str, comparison: dict[str, Any]) -> None:
        lines.extend(["", f"### {title}", ""])
        if comparison.get("status") != "complete":
            lines.append(
                f"未纳入成对比较：{comparison['status']}；未完成实验臂："
                f"{', '.join(comparison.get('incomplete_runs', [])) or '未知'}。"
            )
            return
        bootstrap = comparison["image_group_bootstrap"]
        lines.extend(
            [
                f"完整配对数：{comparison['samples']}；mIoU 差值（右减左）={comparison['delta']['mean_iou']:.4f}，"
                f"95% CI [{bootstrap['mean_iou_delta_95ci'][0]:.4f}, {bootstrap['mean_iou_delta_95ci'][1]:.4f}]。",
                f"ACC@0.5 差值（右减左）={comparison['delta']['acc_0.5']:.4f}，"
                f"95% CI [{bootstrap['acc_0.5_delta_95ci'][0]:.4f}, {bootstrap['acc_0.5_delta_95ci'][1]:.4f}]；"
                f"bootstrap 图像组数={bootstrap['image_groups']}。",
                f"完全相同的有效框：{comparison['same_valid_box_count']}/{comparison['both_valid_box_count']}；"
                f"两边都有有效框时平均坐标 L1={fmt(comparison['mean_coordinate_l1_when_both_valid'])}。",
                f"可比较原始生成文本：{comparison['raw_text_compared_count']} 条；变化："
                f"{comparison['raw_text_changed_count'] if comparison['raw_text_changed_count'] is not None else '未提供文本'}。",
                f"左侧错误、右侧命中：{len(comparison['wrong_to_right']['ids'])} 条，"
                f"{comparison['wrong_to_right']['image_count']} 组；左侧命中、右侧错误："
                f"{len(comparison['right_to_wrong']['ids'])} 条，{comparison['right_to_wrong']['image_count']} 组。",
                f"左错右对 IDs：{', '.join(comparison['wrong_to_right']['ids']) or '无'}。",
                f"左对右错 IDs：{', '.join(comparison['right_to_wrong']['ids']) or '无'}。",
            ]
        )
        if comparison["exploratory_query_slices"]:
            lines.extend(["", "Query 词面探索分层（类别可重叠，不是语义标注）：", ""])
            for slice_name, values in comparison["exploratory_query_slices"].items():
                label = "位置/距离词面" if slice_name == "distance_keyword" else "序数词"
                lines.append(
                    f"- {label}：{values['samples']} 条，mIoU 差值={fmt(values['delta']['mean_iou'])}，"
                    f"ACC@0.5 差值={fmt(values['delta']['acc_0.5'])}。"
                )

    lines.extend(["", "## 5090 同检查点干预：相对正常三模态", ""])
    for comparison in report["intervention_comparisons"]:
        right_label = INTERVENTION_LABELS.get(comparison["right"], comparison["right"])
        append_comparison(right_label, comparison)
    lines.extend(["", "## 硬件对照与历史 RGB 对照", ""])
    append_comparison("5090 正常臂 vs 4090 旧 M2（跨硬件诊断）", report["cross_hardware_comparison"])
    append_comparison("4090 R2 RGB vs 4090 旧 M2（历史基线诊断）", report["historical_4090_rgb_vs_old_m2"])
    lines.extend(["", "## 解释边界", "", *[f"- {note}" for note in report["caveats"]], ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="raw GT ID-to-record JSON")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--old-m2", type=Path, required=True, help="4090 old M2 predictions.jsonl")
    parser.add_argument("--rgb", type=Path, required=True, help="4090 R2 RGB predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    manifest = rematch.load_manifest(args.manifest)
    if len(manifest) != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} raw GT samples, found {len(manifest)}")
    image_groups = {rematch.image_group(record) for record in manifest.values()}
    if len(image_groups) != EXPECTED_IMAGE_GROUPS:
        raise ValueError(f"expected {EXPECTED_IMAGE_GROUPS} image groups, found {len(image_groups)}")

    report = build_report(
        manifest,
        args.experiment_dir,
        args.old_m2,
        args.rgb,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["ground_truth_path"] = str(args.manifest.resolve())
    write_outputs(report, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "manifest_samples": report["manifest_samples"],
                "runs": {run["name"]: run["status"] for run in report["runs"]},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
