"""Optional A/M IR-reading gate followed by gated City412 two-pass evaluation.

Print the plan by default. GPU work requires --execute on the cloud host.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.run_triground_abv import BUDGET_SECONDS, execute_stage, load_budget, read_json, write_json
from tools.run_triground_tm import grounding_stage, validate_config, validate_release


ARMS = ("A", "M")
IR_CLASS = "ir"
RGB_CLASS = "rgb_sufficient"


def prediction_hits(path: Path, ids: set[str]) -> dict[str, bool]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    hits = {str(row["id"]): bool(row["acc_0.5"]) for row in rows}
    if len(hits) != len(rows) or set(hits) != ids:
        raise ValueError(f"prediction IDs differ from the frozen diagnostic set: {path}")
    return hits


def gate_result(direct: dict[str, bool], two_pass: dict[str, bool],
                classes: dict[str, str], scenes: dict[str, str],
                *, ir_present_ids: set[str]) -> dict:
    ids = set(classes)
    if set(direct) != ids or set(two_pass) != ids or set(scenes) != ids:
        raise ValueError("direct, two-pass, class and scene IDs must match exactly")
    if not ir_present_ids <= ids:
        raise ValueError("IR presence IDs must belong to the frozen diagnostic set")
    ir_ids = {sample_id for sample_id in ir_present_ids if classes[sample_id] == IR_CLASS}
    rgb_ids = {sample_id for sample_id in ir_present_ids if classes[sample_id] == RGB_CLASS}
    rescued = sorted(sample_id for sample_id in ir_ids if not direct[sample_id] and two_pass[sample_id])
    harmed = sorted(sample_id for sample_id in ir_ids if direct[sample_id] and not two_pass[sample_id])
    rgb_harmed = sorted(sample_id for sample_id in rgb_ids if direct[sample_id] and not two_pass[sample_id])
    rescued_scenes = sorted({scenes[sample_id] for sample_id in rescued})
    net = len(rescued) - len(harmed)
    # No-IR cases never execute the reader, so their stability is not a control
    # for damage caused by IR advice. Preserve the plan's 32-case promotion gate;
    # the descriptive 16-case minimum does not authorize the full City run.
    coverage = len(rgb_ids) >= 32
    passed = coverage and net >= 2 and len(rescued_scenes) >= 2 and len(rgb_harmed) <= 1
    return {
        "passed": passed,
        "ir_samples": len(ir_ids), "ir_rescued": len(rescued), "ir_harmed": len(harmed),
        "ir_net": net, "ir_rescued_ids": rescued, "ir_harmed_ids": harmed,
        "ir_rescued_scene_groups": rescued_scenes,
        "rgb_sufficient_samples": len(rgb_ids), "rgb_new_harm": len(rgb_harmed),
        "rgb_ir_control_coverage": coverage,
        "not_tested_no_ir": sorted(ids - ir_present_ids),
        "blocked_reason": None if coverage else "fewer_than_32_rgb_sufficient_cases_with_real_ir",
        "rgb_new_harm_ids": rgb_harmed,
        "criteria": {"ir_net_at_least": 2, "rescued_scene_groups_at_least": 2,
                     "rgb_sufficient_new_harm_at_most": 1,
                     "rgb_sufficient_with_ir_at_least": 32},
    }


def city_inference_manifest(source: Path, destination: Path) -> int:
    """Add explicit image order only after checking all three actual path roles."""
    records = read_json(source)
    if not isinstance(records, list):
        raise ValueError("City412 source must be a JSON array")
    for row in records:
        images = row["image"]
        if row.get("modalities") or row.get("image_order"):
            raise ValueError("City412 already has modality metadata; inspect before rewriting")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(f"City412 record is not an RGB/IR/Depth triple: {row['id']}")
        parts = [Path(str(value).replace("\\", "/")).parts for value in images]
        if not ("visible" in parts[0] and "infrared" in parts[1]
                and "depth_rgb" in parts[2]):
            raise ValueError(f"City412 image paths do not prove RGB/IR/Depth order: {row['id']}")
        row["image_order"] = ["rgb", "ir", "depth"]
    if len(records) != 412:
        raise ValueError(f"expected City412, found {len(records)} records")
    if destination.exists():
        if read_json(destination) != records:
            raise ValueError(f"existing City412 inference manifest differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, records)
    return len(records)


def two_pass_stage(config: dict, name: str, adapter: str | Path, manifest: str,
                   gt: str, output: Path) -> dict:
    stage = grounding_stage(config, name, adapter, manifest, gt, output, resume=True)
    command = stage["command"]
    command[command.index("direct")] = "ir_then_ground"
    return stage


def assert_evaluation(stage: dict) -> None:
    result = Path(stage["expected_artifact"])
    if not (result / "summary.json").is_file() or not (result / "run_config.json").is_file():
        raise RuntimeError(f"evaluation output incomplete: {result}")
    if read_json(result / "run_config.json")["inference_mode"] != "ir_then_ground":
        raise ValueError(f"wrong inference mode in {result}")
    rows = [line for line in (result / "predictions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != stage["queries"]:
        raise RuntimeError(f"evaluation has {len(rows)} of {stage['queries']} predictions: {result}")


def city_latency_output(out: Path, arm: str,
                        baseline_a_city_output: Path | None) -> Path:
    if arm == "A":
        if baseline_a_city_output is None:
            raise ValueError("A passed the IR gate: provide --baseline-a-city-output for the completed historical A City412 evaluation")
        return baseline_a_city_output
    return out / "evaluation" / arm / "city412"


def execute_once(stage: dict, out: Path, budget_dir: Path, state: dict,
                 *, resume: bool, retry_reason: str, direct_output: Path) -> None:
    records = [row for row in state["stages"] if row["name"] == stage["name"]
               and row.get("run_dir") == str(out)]
    if any(row["status"] == "complete" for row in records):
        if not resume:
            raise FileExistsError(f"{stage['name']} is complete; use --resume")
        assert_evaluation(stage)
        return
    failures = len(records)
    if failures and not retry_reason:
        raise RuntimeError(f"{stage['name']} failed; diagnose it and provide --retry-reason")
    if failures >= 3:
        raise RuntimeError(f"{stage['name']} has exhausted the original attempt plus two diagnosed retries")
    direct_rows = [json.loads(line) for line in (direct_output / "predictions.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(direct_rows) != stage["queries"]:
        raise ValueError(f"direct timing reference has {len(direct_rows)} of {stage['queries']} predictions: {direct_output}")
    observed = sum(float(row["latency_seconds"]) for row in direct_rows) / len(direct_rows)
    estimate = 100.0 + stage["queries"] * max(4.0, 2.5 * observed)
    available = state["limit_seconds"] - state["spent_seconds"]
    if estimate > available * 0.9:
        raise RuntimeError(f"{stage['name']} estimated {estimate:.0f}s exceeds 90% of remaining {available:.0f}s")
    execute_stage(stage, out, state, budget_dir)
    assert_evaluation(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-reason", default="")
    parser.add_argument("--baseline-a-city-output", type=Path,
                        help="Completed historical A City412 evaluation directory, required only if A passes the IR gate")
    args = parser.parse_args()
    config, release = read_json(args.config), read_json(args.release)
    validate_config(config)
    validate_release(release)
    out = args.output_dir.resolve()
    classes, scenes = read_json(release["class_map"]), read_json(release["scene_map"])
    normal_records = read_json(release["diagnostics"]["normal"])
    ir_present_ids = {
        str(row["id"]) for row in normal_records
        if any(m in {"ir", "infrared"} for m in row.get("modalities", row.get("image_order", [])))
        and not any(m in {"ir", "infrared"} for m in row.get("missing_modalities_actual", []))
    }
    control_ids = {i for i in ir_present_ids if classes[i] == RGB_CLASS}
    if len(classes) != 103 or sum(value == IR_CLASS for value in classes.values()) != 48 or sum(value == RGB_CLASS for value in classes.values()) != 32:
        raise ValueError("frozen diagnostic must contain 103 tasks, including IR48 and RGB-sufficient32")
    if set(classes) != set(scenes):
        raise ValueError("diagnostic class/scene IDs differ")
    city_copy = out / "inputs/city412_ir_two_pass.json"
    adapters = {"A": config["baseline_adapters"]["A"],
                "M": out / "M/main" / f"checkpoint-{release['steps']}"}
    diagnostics = {
        arm: two_pass_stage(config, f"{arm}_ir_two_pass_diagnostic_normal", adapters[arm],
                            release["diagnostics"]["normal"], release["gt_manifest"],
                            out / "evaluation" / arm / "normal_ir_then_ground")
        for arm in ARMS
    }
    city = {
        arm: two_pass_stage(config, f"{arm}_ir_two_pass_city412", adapters[arm],
                            config["city_manifest"], config["city_gt"],
                            out / "evaluation" / arm / "city412_ir_then_ground")
        for arm in ARMS
    }
    for stage in city.values():
        stage["command"][stage["command"].index("--manifest") + 1] = str(city_copy)
    if not args.execute:
        print(json.dumps({"diagnostics": list(diagnostics.values()),
                          "gate": "IR net >=2, rescued scenes >=2, RGB-sufficient new harm <=1",
                          "rgb_sufficient_with_ir": len(control_ids),
                          "gate_testable": len(control_ids) >= 32,
                          "conditional_city": list(city.values()),
                          "baseline_a_city_output": str(args.baseline_a_city_output) if args.baseline_a_city_output else None,
                          "city_inference_manifest": str(city_copy)}, ensure_ascii=False, indent=2))
        return
    if os.name != "posix":
        raise RuntimeError("GPU execution requires the Linux cloud host")
    if len(control_ids) < 32:
        raise ValueError("IR harm control is absent/insufficient: require >=32 RGB-sufficient tasks with actual IR before starting this experiment")
    budget_dir = Path(config["budget_dir"])
    state = load_budget(budget_dir)
    if state["limit_seconds"] != BUDGET_SECONDS:
        raise ValueError("IR two-pass must use the existing independent T/M 12-hour budget")
    if not (out / "task_manifest.json").is_file():
        raise FileNotFoundError("T/M run manifest is missing")
    frozen = read_json(out / "task_manifest.json")
    if frozen["config"] != config or frozen["release"] != release:
        raise ValueError("IR two-pass inputs differ from the frozen T/M task")
    gates = {}
    for arm in ARMS:
        direct_output = out / "evaluation" / arm / "normal"
        direct = prediction_hits(direct_output / "predictions.jsonl", set(classes))
        execute_once(diagnostics[arm], out, budget_dir, state, resume=args.resume,
                     retry_reason=args.retry_reason, direct_output=direct_output)
        two_pass = prediction_hits(Path(diagnostics[arm]["expected_artifact"]) / "predictions.jsonl", set(classes))
        result = gate_result(direct, two_pass, classes, scenes, ir_present_ids=ir_present_ids)
        result.update({"arm": arm, "direct": str(direct_output),
                       "two_pass": diagnostics[arm]["expected_artifact"]})
        write_json(out / "evaluation" / arm / "ir_two_pass_gate.json", result)
        gates[arm] = result
    for arm in ARMS:
        if gates[arm]["passed"]:
            direct_output = city_latency_output(out, arm, args.baseline_a_city_output)
            city_inference_manifest(Path(config["city_manifest"]), city_copy)
            execute_once(city[arm], out, budget_dir, state, resume=args.resume,
                         retry_reason=args.retry_reason,
                         direct_output=direct_output)


if __name__ == "__main__":
    main()
