"""Score one C0/A/B/V prediction file per arm against raw City GT.

Incomplete files are progress previews only. No predictions from separate files
are combined, and all paired statistics require two full runs on the same IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from tools import report_rematch_experiment as rematch
from tools.report_gu_diagnosis import SHARED_PATTERN


ARMS = ("C0", "A", "B", "V")
MIN_CLASS_SAMPLES = 16


def _hits(metrics: dict[str, Any], threshold: str = "acc_0.5") -> int:
    return round(metrics[threshold] * metrics["samples"])


def _valid_prediction(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == 4
        and all(isinstance(part, (int, float)) and not isinstance(part, bool)
                and math.isfinite(part) for part in value)
        and 0 <= value[0] < value[2] <= 1
        and 0 <= value[1] < value[3] <= 1
    )


def _load_optional(name: str, path: Path, manifest: dict[str, Any],
                   *, require_run_config: bool) -> dict[str, Any]:
    if not path.is_file() or not path.read_text(encoding="utf-8-sig").strip():
        return {
            "name": name,
            "path": str(path.resolve()),
            "rows": {},
            "missing_ids": list(manifest),
            "partial": True,
            "metadata": None,
            "status": "pending",
        }
    rows = {}
    invalid = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"{path}:{line_number}: prediction row needs an ID")
        sample_id = str(row["id"])
        if sample_id not in manifest or sample_id in rows:
            raise ValueError(f"{path}:{line_number}: duplicate or unknown ID {sample_id!r}")
        prediction = row.get("prediction")
        if _valid_prediction(prediction):
            primary = [float(value) for value in prediction]
        else:
            primary = None
            invalid[sample_id] = "parse_failure" if prediction is None else "invalid_box"
        rows[sample_id] = {
            "id": sample_id, "primary": primary,
            "candidates": [primary] if primary is not None else [], "raw": row,
        }
    run = {
        "name": name, "path": str(path.resolve()), "rows": rows,
        "missing_ids": [sample_id for sample_id in manifest if sample_id not in rows],
        "partial": len(rows) != len(manifest),
        "metadata": rematch._adjacent_metadata(path),
        "invalid_predictions": invalid,
    }
    run["status"] = "partial" if run["partial"] else "complete"
    config_path = path.parent / "run_config.json"
    if require_run_config and not config_path.is_file():
        raise ValueError(f"{path}: run_config.json is required to verify prediction source")
    run_config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else None
    metadata = run["metadata"]
    if require_run_config and run["status"] == "complete" and metadata is None:
        raise ValueError(f"{path}: complete run needs adjacent summary.json")
    if metadata is not None and isinstance(metadata["data"], dict):
        summary = metadata["data"]
        if summary.get("predictions") and str(summary["predictions"]).replace("\\", "/").rsplit("/", 1)[-1] != path.name:
            raise ValueError(f"{path}: adjacent summary names a different prediction file")
        if summary.get("samples") is not None and int(summary["samples"]) != len(rows):
            raise ValueError(f"{path}: summary sample count differs from prediction IDs")
        if run_config is not None and summary.get("adapter") != run_config.get("adapter"):
            raise ValueError(f"{path}: summary and run_config adapter differ")
    run["run_config"] = run_config
    return run


def _subset_score(run: dict[str, Any], subset: dict[str, Any]) -> dict[str, Any]:
    if not subset:
        return {"samples": 0, "hits_0.5": 0, "hits_0.7": 0, "metrics": None}
    metrics = rematch.score_run(run, subset)
    return {
        "samples": metrics["samples"],
        "hits_0.5": _hits(metrics),
        "hits_0.7": _hits(metrics, "acc_0.7"),
        "metrics": metrics,
    }


def _class_slices(
    runs: dict[str, dict[str, Any]], manifest: dict[str, Any],
    class_map: dict[str, str] | None,
) -> dict[str, Any]:
    if class_map is None:
        return {}
    if set(class_map) != set(manifest):
        raise ValueError("class map IDs must exactly match the raw GT manifest")
    result = {}
    for label in sorted(set(class_map.values())):
        subset = {sample_id: row for sample_id, row in manifest.items() if class_map[sample_id] == label}
        enough = len(subset) >= MIN_CLASS_SAMPLES
        result[label] = {
            "samples": len(subset),
            "interpretation": "descriptive_only" if enough else "insufficient_n_no_class_conclusion",
            "runs": {
                name: _subset_score(run, subset) if enough and run["status"] == "complete" else None
                for name, run in runs.items()
            },
        }
    return result


def _scene_diagnostics(
    left: dict[str, Any], right: dict[str, Any], manifest: dict[str, Any],
    pair: dict[str, Any], scene_map: dict[str, str] | None,
    *, replicates: int, seed: int,
) -> None:
    if scene_map is None:
        return
    surrogate = {sample_id: {field: scene_map[sample_id]
                             for field in rematch.IMAGE_FIELDS} for sample_id in manifest}
    entries = []
    for sample_id, gt in manifest.items():
        boxes = (left["rows"][sample_id]["primary"], right["rows"][sample_id]["primary"])
        ious = [rematch.box_iou(box, gt["bbox"]) if box is not None else 0.0 for box in boxes]
        entries.append((sample_id, ious[0], ious[1]))
    bootstrap = rematch.paired_group_bootstrap(entries, surrogate, replicates=replicates, seed=seed)
    bootstrap["scene_groups"] = bootstrap.pop("image_groups")
    pair["scene_cluster_bootstrap"] = bootstrap
    pair["scene_flips"] = {
        "rescued_scenes": sorted({scene_map[sample_id] for sample_id in pair["wrong_to_right"]["ids"]}),
        "harmed_scenes": sorted({scene_map[sample_id] for sample_id in pair["right_to_wrong"]["ids"]}),
    }


def _paired(
    left: dict[str, Any], right: dict[str, Any], manifest: dict[str, Any],
    scene_map: dict[str, str] | None, *, replicates: int, seed: int,
) -> dict[str, Any]:
    pair = rematch.compare_runs(left, right, manifest, replicates=replicates, seed=seed)
    _scene_diagnostics(left, right, manifest, pair, scene_map, replicates=replicates, seed=seed)
    return pair


def _modal_class_slices(
    pair: dict[str, Any], left: dict[str, Any], right: dict[str, Any],
    manifest: dict[str, Any], class_map: dict[str, str] | None,
) -> dict[str, Any]:
    if class_map is None:
        return {}
    result = {}
    for label in sorted(set(class_map.values())):
        subset = {sample_id: row for sample_id, row in manifest.items() if class_map[sample_id] == label}
        if len(subset) < MIN_CLASS_SAMPLES:
            result[label] = {"samples": len(subset), "status": "insufficient_n_no_class_conclusion"}
            continue
        left_score = _subset_score(left, subset)
        right_score = _subset_score(right, subset)
        rescued = [sample_id for sample_id in pair["wrong_to_right"]["ids"] if sample_id in subset]
        harmed = [sample_id for sample_id in pair["right_to_wrong"]["ids"] if sample_id in subset]
        result[label] = {
            "samples": len(subset), "status": "descriptive_only",
            "missing_input_hits": left_score["hits_0.5"],
            "full_input_hits": right_score["hits_0.5"],
            "rescued_ids": rescued, "harmed_ids": harmed,
            "rescued": len(rescued), "harmed": len(harmed), "net": len(rescued) - len(harmed),
        }
    return result


def build_report(
    manifest: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    *,
    class_map: dict[str, str] | None = None,
    scene_map: dict[str, str] | None = None,
    modal_paths: dict[tuple[str, str], Path] | None = None,
    require_run_config: bool = False,
    bootstrap_replicates: int = rematch.BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = rematch.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if "C0" not in paths:
        raise ValueError("C0 baseline path is required")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if scene_map is not None and set(scene_map) != set(manifest):
        raise ValueError("scene map IDs must exactly match the raw GT manifest")
    if class_map is not None and set(class_map) != set(manifest):
        raise ValueError("class map IDs must exactly match the raw GT manifest")
    if scene_map is not None and any(not isinstance(value, str) or not value for value in scene_map.values()):
        raise ValueError("scene map values must be non-empty scene labels")
    if class_map is not None and any(not isinstance(value, str) or not value for value in class_map.values()):
        raise ValueError("class map values must be non-empty ability labels")
    runs = {name: _load_optional(name, path, manifest, require_run_config=require_run_config)
            for name, path in paths.items()}
    source_adapters = [run["run_config"].get("adapter") for name, run in runs.items()
                       if name in ARMS and run["status"] == "complete" and run.get("run_config")]
    if len(source_adapters) != len(set(source_adapters)):
        raise ValueError("complete C0/A/B/V runs must declare distinct adapter sources")
    shared = {sample_id: row for sample_id, row in manifest.items()
              if SHARED_PATTERN.search(str(row["visible"]))}
    other = {sample_id: row for sample_id, row in manifest.items() if sample_id not in shared}
    if len(manifest) == 412 and (len(shared), len(other)) != (47, 365):
        raise ValueError("City412 known-reuse split must be 47/365")

    summaries = {}
    for name, run in runs.items():
        complete = run["status"] == "complete"
        observed = {sample_id: manifest[sample_id] for sample_id in manifest if sample_id in run["rows"]}
        metrics = rematch.score_run(run, manifest if complete else observed) if observed else None
        summaries[name] = {
            "status": run["status"],
            "path": run["path"],
            "observed": len(observed),
            "denominator": len(manifest) if complete else len(observed),
            "manifest_samples": len(manifest),
            "missing_ids": run["missing_ids"],
            "invalid_predictions": run.get("invalid_predictions", {}),
            "ranking_eligible": complete and name in ARMS,
            "metrics": metrics if complete else None,
            "preview_metrics": None if complete else metrics,
            "hits_0.5": _hits(metrics) if complete else None,
            "hits_0.7": _hits(metrics, "acc_0.7") if complete else None,
            "shared47": _subset_score(run, shared) if complete else None,
            "other365": _subset_score(run, other) if complete else None,
            "metadata": run["metadata"],
            "run_config": run.get("run_config"),
        }

    pairs = {}
    gates = {}
    c0 = runs["C0"]
    for name, run in runs.items():
        if name == "C0":
            continue
        if c0["status"] != "complete" or run["status"] != "complete":
            gates[name] = {"status": "pending_or_partial", "candidate": False}
            continue
        pair = _paired(
            c0, run, manifest, scene_map,
            replicates=bootstrap_replicates, seed=bootstrap_seed,
        )
        pairs[f"C0_to_{name}"] = pair
        rescued = len(pair["wrong_to_right"]["ids"])
        harmed = len(pair["right_to_wrong"]["ids"])
        net = rescued - harmed
        groups = pair["wrong_to_right"]["image_count"]
        gates[name] = {
            "status": "complete",
            "rescued": rescued,
            "harmed": harmed,
            "net_hits": net,
            "rescued_image_groups": groups,
            "candidate": name in ARMS and net >= 4 and groups >= 3,
            "interpretation": "paired descriptive comparison; not causal modality attribution",
        }

    for left_name, right_name in (("A", "B"), ("A", "V"), ("B", "V")):
        if all(name in runs and runs[name]["status"] == "complete" for name in (left_name, right_name)):
            pairs[f"{left_name}_to_{right_name}"] = _paired(
                runs[left_name], runs[right_name], manifest, scene_map,
                replicates=bootstrap_replicates, seed=bootstrap_seed,
            )

    modal_utility: dict[str, dict[str, Any]] = {}
    for (arm, condition), path in (modal_paths or {}).items():
        if arm not in runs:
            raise ValueError(f"modal run {arm}:{condition} has no normal --run counterpart")
        if condition not in ("ir_missing", "depth_missing", "both_missing"):
            raise ValueError(f"unknown modality intervention {condition!r}")
        missing = _load_optional(f"{arm}:{condition}", path, manifest,
                                 require_run_config=require_run_config)
        normal = runs[arm]
        if (normal.get("run_config") and missing.get("run_config")
                and normal["run_config"].get("adapter") != missing["run_config"].get("adapter")):
            raise ValueError(f"{arm}:{condition} uses a different adapter than {arm} normal")
        key = f"{arm}:{condition}"
        if normal["status"] != "complete" or missing["status"] != "complete":
            modal_utility[key] = {"status": "pending_or_partial", "samples": len(missing["rows"])}
            continue
        pair = _paired(missing, normal, manifest, scene_map,
                       replicates=bootstrap_replicates, seed=bootstrap_seed)
        pairs[f"{arm}_{condition}_to_normal"] = pair
        rescued = len(pair["wrong_to_right"]["ids"])
        harmed = len(pair["right_to_wrong"]["ids"])
        modal_utility[key] = {
            "status": "complete", "arm": arm, "condition": condition,
            "samples": len(manifest),
            "rescued": rescued, "harmed": harmed, "net": rescued - harmed,
            "rescued_ids": pair["wrong_to_right"]["ids"],
            "harmed_ids": pair["right_to_wrong"]["ids"],
            "class_slices": _modal_class_slices(pair, missing, normal, manifest, class_map),
            "interpretation": "same-checkpoint full-input minus missing-modality intervention",
        }
    modal_delta_vs_A = {}
    for key, entry in modal_utility.items():
        arm, condition = key.split(":", 1)
        reference = modal_utility.get(f"A:{condition}")
        if arm in ("B", "V") and entry["status"] == "complete" and reference and reference["status"] == "complete":
            modal_delta_vs_A[key] = {
                "net_delta": entry["net"] - reference["net"],
                "rescued_delta": entry["rescued"] - reference["rescued"],
                "harmed_delta": entry["harmed"] - reference["harmed"],
                "interpretation": "descriptive difference in same-checkpoint modality input effect",
            }

    ranked = [name for name in ARMS if name in runs and runs[name]["status"] == "complete"]
    ranked.sort(
        key=lambda name: (
            -summaries[name]["hits_0.5"],
            -summaries[name]["metrics"]["mean_iou"],
            -summaries[name]["hits_0.7"],
            ARMS.index(name),
        )
    )
    return {
        "manifest_samples": len(manifest),
        "image_groups": len({rematch.image_group(row) for row in manifest.values()}),
        "known_reuse_samples": len(shared),
        "other_samples": len(other),
        "bootstrap": {"replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "runs": summaries,
        "ranking": ranked,
        "ranking_order": ["ACC@0.5", "mIoU", "ACC@0.7"],
        "gates_vs_C0": gates,
        "pairs": pairs,
        "modal_utility": modal_utility,
        "modal_utility_delta_vs_A": modal_delta_vs_A,
        "class_slices": _class_slices(runs, manifest, class_map),
        "scene_clusters": len(set(scene_map.values())) if scene_map else None,
        "limitations": [
            "Only complete single-file runs enter ranking, pairs, or gates.",
            "Partial metrics use observed rows as preview denominator and are not comparable scores.",
            "Known reuse 47/365 is a location-sequence partition, not certified scene independence.",
            "Ability class slices below 16 samples receive no class-level conclusion; individual scene clusters are never screened by this threshold.",
            "Prediction source is fully checked only when each run has a matching run_config.json and summary.json.",
        ],
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("name", "status", "observed", "denominator", "hits_0.5", "acc_0.5",
                  "mean_iou", "hits_0.7", "acc_0.7", "parse_rate", "rank", "candidate")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, run in report["runs"].items():
            metrics = run["metrics"] or run["preview_metrics"] or {}
            writer.writerow({
                "name": name, "status": run["status"], "observed": run["observed"],
                "denominator": run["denominator"], "hits_0.5": run["hits_0.5"],
                "acc_0.5": metrics.get("acc_0.5"), "mean_iou": metrics.get("mean_iou"),
                "hits_0.7": run["hits_0.7"], "acc_0.7": metrics.get("acc_0.7"),
                "parse_rate": metrics.get("parse_rate"),
                "rank": report["ranking"].index(name) + 1 if name in report["ranking"] else None,
                "candidate": report["gates_vs_C0"].get(name, {}).get("candidate"),
            })
    lines = [
        "# C0/A/B/V 原始 GT 重算", "",
        f"总分母 {report['manifest_samples']}，图像组 {report['image_groups']}；已知地点序列 "
        f"{report['known_reuse_samples']}，其余 {report['other_samples']}。",
        "完整单文件预测才参加排序、成对比较与入围判断；partial/pending 仅为进度预览。", "",
        "| 臂 | 状态 | 样本/分母 | ACC@0.5 | mIoU | ACC@0.7 | 已知复用命中 | 其余命中 | 排名 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["runs"].items():
        m = item["metrics"] or item["preview_metrics"]
        score = "—" if m is None else f"{m['acc_0.5']:.4f}"
        miou = "—" if m is None else f"{m['mean_iou']:.4f}"
        acc7 = "—" if m is None else f"{m['acc_0.7']:.4f}"
        shared_hits = "—" if item["shared47"] is None else str(item["shared47"]["hits_0.5"])
        other_hits = "—" if item["other365"] is None else str(item["other365"]["hits_0.5"])
        rank = str(report["ranking"].index(name) + 1) if name in report["ranking"] else "—"
        lines.append(f"| {name} | {item['status']} | {item['observed']}/{item['denominator']} | "
                     f"{score} | {miou} | {acc7} | {shared_hits} | {other_hits} | {rank} |")
    lines += ["", "排序：ACC@0.5 → mIoU → ACC@0.7；相对 C0 净增至少 4 命中且救回覆盖至少 3 图像组才入围。", ""]
    for name, gate in report["gates_vs_C0"].items():
        if gate["status"] != "complete":
            lines.append(f"- {name}：待完整预测；不入围。")
        else:
            lines.append(f"- {name}：救回 {gate['rescued']}、伤害 {gate['harmed']}、净增 "
                         f"{gate['net_hits']:+d}，救回 {gate['rescued_image_groups']} 图像组；"
                         f"{'入围候选' if gate['candidate'] else '未入围'}。")
            pair = report["pairs"][f"C0_to_{name}"]
            ci = pair["image_group_bootstrap"]["acc_0.5_delta_95ci"]
            lines.append(f"  救回 IDs：{', '.join(pair['wrong_to_right']['ids']) or '无'}；"
                         f"伤害 IDs：{', '.join(pair['right_to_wrong']['ids']) or '无'}；"
                         f"图像组 bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]。")
    if report["modal_utility"]:
        lines += ["", "## 同一检查点的输入模态干预", "",
                  "每条比较固定模型权重、Query 和 GT；左侧为缺失输入，右侧为正常全输入。"]
        for key, modal in report["modal_utility"].items():
            if modal["status"] != "complete":
                lines.append(f"- {key}：pending/partial，暂不比较。")
                continue
            lines.append(f"- {key}：正常全输入救回 {modal['rescued']}、伤害 {modal['harmed']}，"
                         f"净效用 {modal['net']:+d}。")
            pair = report["pairs"][f"{modal['arm']}_{modal['condition']}_to_normal"]
            ci = (pair.get("scene_cluster_bootstrap") or pair["image_group_bootstrap"])["acc_0.5_delta_95ci"]
            lines.append(f"  图组/场景簇 bootstrap 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]；"
                         f"救回 IDs：{', '.join(modal['rescued_ids']) or '无'}；"
                         f"伤害 IDs：{', '.join(modal['harmed_ids']) or '无'}。")
            for label, ability in modal["class_slices"].items():
                if ability["status"] == "descriptive_only":
                    lines.append(f"  {label}（N={ability['samples']}）：救回 {ability['rescued']}、"
                                 f"伤害 {ability['harmed']}、净 {ability['net']:+d}。")
                else:
                    lines.append(f"  {label}（N={ability['samples']}）：不足 16，不作类级结论。")
        if report["modal_utility_delta_vs_A"]:
            lines += ["", "B/V 相对 A 的模态输入效用差（描述性）："]
            for key, delta in report["modal_utility_delta_vs_A"].items():
                lines.append(f"- {key}：救回差 {delta['rescued_delta']:+d}、伤害差 "
                             f"{delta['harmed_delta']:+d}、净差 {delta['net_delta']:+d}。")
    lines += ["", "所有 IoU 均从原始 manifest 的浮点框重新计算。翻转 ID、图像组、图像组 bootstrap 区间、47/365 和类别切片详见 summary.json。", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=predictions.jsonl")
    parser.add_argument("--modal-run", action="append", default=[],
                        metavar="ARM:CONDITION=predictions.jsonl",
                        help="same-checkpoint missing-modality intervention on the same GT IDs")
    parser.add_argument("--class-map", type=Path, help="optional JSON sample-ID to class-label map")
    parser.add_argument("--scene-map", type=Path, help="optional JSON sample-ID to scene-label map")
    parser.add_argument("--require-run-config", action="store_true",
                        help="require each present prediction file to carry evaluator run_config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=rematch.BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=rematch.BOOTSTRAP_SEED)
    args = parser.parse_args()
    specs = [rematch.parse_run_spec(spec) for spec in args.run]
    if len({name for name, _ in specs}) != len(specs):
        raise ValueError("run names must be unique; each run uses exactly one prediction file")
    paths = {name: path for name, path in specs}
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise ValueError("each run must use a distinct prediction file")
    manifest = rematch.load_manifest(args.manifest)
    modal_specs = [rematch.parse_run_spec(spec) for spec in args.modal_run]
    if len({name for name, _ in modal_specs}) != len(modal_specs):
        raise ValueError("modal run names must be unique")
    modal_paths = {}
    for name, path in modal_specs:
        arm, separator, condition = name.partition(":")
        if not separator or not arm or not condition:
            raise ValueError("--modal-run must use ARM:CONDITION=predictions.jsonl")
        modal_paths[(arm, condition)] = path
    if len({path.resolve() for path in (*paths.values(), *modal_paths.values())}) != len(paths) + len(modal_paths):
        raise ValueError("normal and modal runs must use distinct prediction files")
    if args.output_dir.resolve() in {path.parent.resolve() for path in (*paths.values(), *modal_paths.values())}:
        raise ValueError("report output directory must differ from evaluator prediction directories")
    class_map = json.loads(args.class_map.read_text(encoding="utf-8-sig")) if args.class_map else None
    scene_map = json.loads(args.scene_map.read_text(encoding="utf-8-sig")) if args.scene_map else None
    report = build_report(
        manifest, paths, class_map=class_map, scene_map=scene_map,
        modal_paths=modal_paths,
        require_run_config=args.require_run_config,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report["ground_truth_path"] = str(args.manifest.resolve())
    write_outputs(report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "ranking": report["ranking"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
