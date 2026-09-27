"""Recompute paired City gates and the reviewed external gate for the G/U pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.report_rematch_experiment import compare_runs, load_manifest, load_run, score_run


def load_native_ids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else json.loads(text)
    if not isinstance(rows, list):
        raise ValueError("external native manifest must be a JSON array or JSONL")
    return [str(row["id"]) for row in rows]


def city_runs(gt: Path, paths: dict[str, Path]):
    manifest = load_manifest(gt)
    if len(manifest) != 412:
        raise ValueError(f"City gate requires the complete 412 cases, got {len(manifest)}")
    runs = {name: load_run(name, path, set(manifest), allow_partial=False) for name, path in paths.items()}
    metrics = {name: score_run(run, manifest) for name, run in runs.items()}
    return manifest, runs, metrics


def branch_pass(manifest, runs, metrics, branch: str) -> dict:
    repaired = compare_runs(runs["C"], runs[branch], manifest, replicates=0)["wrong_to_right"]["image_count"]
    branch_metrics = metrics[branch]
    passed = (
        branch_metrics["acc_0.5"] >= 300 / 412
        and repaired >= 6
        and branch_metrics["mean_iou"] >= metrics["C"]["mean_iou"]
        and branch_metrics["parse_rate"] >= metrics["C"]["parse_rate"]
    )
    return {"passed": passed, "repaired_image_groups_vs_C": repaired, "metrics": branch_metrics}


def seed_gate(gt: Path, c: Path, g: Path, u: Path) -> dict:
    manifest, runs, metrics = city_runs(gt, {"C": c, "G": g, "U": u})
    branches = {branch: branch_pass(manifest, runs, metrics, branch) for branch in ("G", "U")}
    return {
        "status": "continue" if any(item["passed"] for item in branches.values()) else "stop",
        "control": metrics["C"],
        "branches": branches,
    }


def winner_gate(seed2026: dict, seed2027: dict) -> dict:
    if seed2026["status"] != "continue":
        return {"status": "stop", "reason": "2026 gate failed"}
    g = [seed2026["branches"]["G"], seed2027["branches"]["G"]]
    u = [seed2026["branches"]["U"], seed2027["branches"]["U"]]
    differences = [u[i]["metrics"]["acc_0.5"] - g[i]["metrics"]["acc_0.5"] for i in range(2)]
    cases = [round(delta * 412) for delta in differences]
    if all(item["passed"] for item in u) and all(delta > 0 for delta in cases) and sum(cases) >= 8:
        return {"status": "select_U", "winner": "U", "acc_deltas_U_minus_G": differences}
    same_direction = all(delta >= 0 for delta in cases) or all(delta <= 0 for delta in cases)
    if all(item["passed"] for item in g) and same_direction:
        return {"status": "select_G", "winner": "G", "acc_deltas_U_minus_G": differences}
    return {"status": "stop", "reason": "two-seed direction or quality gates failed", "acc_deltas_U_minus_G": differences}


def validate_external_review(gt_path: Path, native_path: Path, review_path: Path, expected_source: str) -> dict:
    manifest = load_manifest(gt_path)
    native_ids = load_native_ids(native_path)
    reviews = [json.loads(line) for line in review_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = set(manifest)
    for name, record_ids in (("native", native_ids), ("review", [str(row["id"]) for row in reviews])):
        if len(record_ids) != 100 or len(set(record_ids)) != 100 or set(record_ids) != ids:
            raise ValueError(f"{name} must have exactly the same 100 unique IDs as external GT")
    if len(ids) != 100:
        raise ValueError("external GT must have exactly 100 unique IDs")
    if any(row.get("source") != expected_source or row.get("split") != "external_review" or row.get("review_status") != "human_accepted" for row in reviews):
        raise ValueError("external review requires explicit source, external_review split, and human_accepted status for all 100")
    for row in reviews:
        target = manifest[str(row["id"])]
        if target.get("source") != expected_source or target.get("split") != "external_review":
            raise ValueError(f"external GT source/split disagrees for {row['id']}")
    return manifest


def external_gate(gt_path: Path, native_path: Path, review_path: Path, m2_path: Path, winner_path: Path, expected_source: str) -> dict:
    manifest = validate_external_review(gt_path, native_path, review_path, expected_source)
    ids = set(manifest)
    m2 = load_run("M2", m2_path, ids, allow_partial=False)
    winner = load_run("winner", winner_path, ids, allow_partial=False)
    m2_summary = json.loads((m2_path.parent / "summary.json").read_text(encoding="utf-8"))
    winner_summary = json.loads((winner_path.parent / "summary.json").read_text(encoding="utf-8"))
    m2_host = json.loads((m2_path.parent / "evaluation_host.json").read_text(encoding="utf-8"))["hostname"]
    winner_host = json.loads((winner_path.parent / "evaluation_host.json").read_text(encoding="utf-8"))["hostname"]
    if m2_host != winner_host or m2_summary["model"] != winner_summary["model"]:
        raise ValueError("M2 and winner external evaluations require the same machine and base model")
    if Path(m2_summary["target_manifest"]).resolve() != gt_path.resolve() or Path(winner_summary["target_manifest"]).resolve() != gt_path.resolve():
        raise ValueError("external evaluations must use this original GT manifest")
    for summary in (m2_summary, winner_summary):
        if Path(summary["manifest"]).resolve() != native_path.resolve():
            raise ValueError("external evaluations must use the same reviewed native manifest")
    for field in ("min_pixels", "max_pixels", "max_new_tokens", "prompt_style", "tf32"):
        if m2_summary[field] != winner_summary[field]:
            raise ValueError(f"external evaluation setting differs: {field}")
    m2_metrics = score_run(m2, manifest)
    winner_metrics = score_run(winner, manifest)
    delta = winner_metrics["acc_0.5"] - m2_metrics["acc_0.5"]
    return {"status": "pass" if delta >= -0.03 else "stop", "acc_delta": delta, "m2": m2_metrics, "winner": winner_metrics, "hostname": m2_host}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("seed")
    first.add_argument("--gt", type=Path, required=True)
    first.add_argument("--c", type=Path, required=True)
    first.add_argument("--g", type=Path, required=True)
    first.add_argument("--u", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    choose = sub.add_parser("winner")
    choose.add_argument("--seed2026", type=Path, required=True)
    choose.add_argument("--seed2027", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    external = sub.add_parser("external")
    for option in ("gt", "native", "review", "m2", "winner", "output"):
        external.add_argument(f"--{option}", type=Path, required=True)
    external.add_argument("--expected-source", required=True)
    review = sub.add_parser("review")
    for option in ("gt", "native", "review", "output"):
        review.add_argument(f"--{option}", type=Path, required=True)
    review.add_argument("--expected-source", required=True)
    args = parser.parse_args()
    if args.command == "seed":
        result = seed_gate(args.gt, args.c, args.g, args.u)
    elif args.command == "winner":
        result = winner_gate(json.loads(args.seed2026.read_text()), json.loads(args.seed2027.read_text()))
    elif args.command == "review":
        preliminary_gt = load_manifest(args.gt)
        preliminary_native = load_native_ids(args.native)
        preliminary_reviews = [json.loads(line) for line in args.review.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(preliminary_gt) < 100 or len(preliminary_native) < 100 or len(preliminary_reviews) < 100 or any(
            row.get("review_status") != "human_accepted" for row in preliminary_reviews
        ):
            result = {"status": "wait", "reason": "external 100 human-accepted cases are incomplete",
                      "gt_rows": len(preliminary_gt), "native_rows": len(preliminary_native),
                      "review_rows": len(preliminary_reviews)}
        else:
            manifest = validate_external_review(args.gt, args.native, args.review, args.expected_source)
            result = {"status": "ready", "unique_human_reviewed_ids": len(manifest), "source": args.expected_source}
    else:
        result = external_gate(args.gt, args.native, args.review, args.m2, args.winner, args.expected_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
