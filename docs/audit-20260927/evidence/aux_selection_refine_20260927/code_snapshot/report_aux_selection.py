"""Score the complete auxiliary-selection method; GT is read only here."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from tools.prepare_aux_selection import read_rows, dump, dump_rows
from tools.report_rematch_experiment import box_iou, image_group, load_manifest, load_run, score_run, compare_runs
from tools.report_gu_diagnosis import SHARED_PATTERN


def metrics(run, gt):
    result = score_run(run, gt)
    n = result["samples"]
    return {**result, "hits_0.5": round(result["acc_0.5"] * n), "hits_0.7": round(result["acc_0.7"] * n)}


def hit(box, target):
    return box is not None and box_iou(box, target) >= .5


def candidate_summary(evidence, gt, runs):
    covered, added, missed_selection, rgb_covered, ir_covered = [], [], [], [], []
    reliability, scopes = Counter(), Counter()
    for key, row in evidence.items():
        target = gt[key]["bbox"]
        candidates = [c for c in row["candidates"] if c["role"] == "target"]
        scopes[row["query_info"]["scope"]] += 1
        for c in candidates:
            reliability[c.get("depth", {}).get("status", "unavailable")] += 1
        if any(hit(c["bbox"], target) for c in candidates):
            covered.append(key)
            if not hit(runs["C"]["rows"][key]["primary"], target):
                added.append(key)
            if not hit(runs["selection"]["rows"][key]["primary"], target):
                missed_selection.append(key)
        for modality, result in (("rgb", rgb_covered), ("ir", ir_covered)):
            if any(s["modality"] == modality and hit(s["bbox"], target)
                   for c in candidates for s in c["sources"]):
                result.append(key)
    return {"queries": len(gt), "target_pool_covered": len(covered), "target_pool_covered_ids": covered,
            "newly_covered_c_errors": len(added), "newly_covered_c_error_ids": added,
            "covered_but_selection_missed": len(missed_selection), "covered_but_selection_missed_ids": missed_selection,
            "rgb_detector_covered": len(rgb_covered), "ir_detector_covered": len(ir_covered),
            "ir_added_over_rgb_detector": sorted(set(ir_covered)-set(rgb_covered)),
            "target_depth_status_counts": dict(reliability), "scope_counts": dict(scopes),
            "interpretation": "Proposal coverage uses GT offline; it is not end-to-end accuracy. IR/RGB counts describe retained candidates."}


def run_report(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    gt = load_manifest(args.gt)
    assert len(gt) == 412
    runs = {name: load_run(name, path, set(gt), allow_partial=False) for name, path in
            (("C", args.c), ("selection", args.selected), ("full", args.final))}
    evidence = {r["id"]: r for r in read_rows(args.evidence)}
    assert set(evidence) == set(gt)
    shared = {key: row for key, row in gt.items() if SHARED_PATTERN.search(row["visible"])}
    other = {key: row for key, row in gt.items() if key not in shared}
    assert len(shared) == 47 and len(other) == 365
    scores = {name: {"all412": metrics(run, gt), "shared47": metrics(run, shared), "other365": metrics(run, other)}
              for name, run in runs.items()}
    pairs = [compare_runs(runs[left], runs[right], gt) for left, right in
             (("C", "selection"), ("C", "full"), ("selection", "full"))]
    c, full = scores["C"]["all412"], scores["full"]["all412"]
    assert c["hits_0.5"] == 292, "C baseline must reproduce the approved score"
    full_pair = pairs[1]
    delta = full["hits_0.5"] - c["hits_0.5"]
    gate = {"positive_gain": delta > 0, "net_hits_vs_c": delta,
            "replication_candidate": full["hits_0.5"] >= 300 and full_pair["wrong_to_right"]["image_count"] >= 6
                and full["mean_iou"] >= c["mean_iou"] and full["parse_rate"] >= c["parse_rate"],
            "automatic_training": False, "automatic_submission": False}
    rng = random.Random(2026)
    review_ids = []
    for transition in ("wrong_to_right", "right_to_wrong"):
        ids = sorted(full_pair[transition]["ids"])
        review_ids += [{"id": key, "reason": transition} for key in rng.sample(ids, min(8, len(ids)))]
    summary = {"scores": scores, "pairs": pairs, "gate": gate, "candidate_diagnostics": candidate_summary(evidence, gt, runs),
               "groups": len({image_group(row) for row in gt.values()}), "manual_review_ids": review_ids,
               "weights_frozen": True, "learned_new_weight_ability": False,
               "limitations": ["City412 is a repeatedly used development set", "other365 is not certified scene-independent",
                               "Depth statistics and masks are model-derived estimates", "Official-test scores unavailable"]}
    selected_rows, final_rows = read_rows(args.selected), read_rows(args.final)
    summaries = list((args.output_dir.parent/'completed_stages').glob('city412_*.json'))
    stages = [json.loads(path.read_text()) for path in summaries]
    gpu_memory = []
    for path in (args.output_dir.parent/'logs').glob('city412_*.gpu.csv'):
        for line in path.read_text().splitlines():
            fields = line.split(',')
            if len(fields) == 3:
                gpu_memory.append(float(fields[1]))
    elapsed = sum(s['elapsed_seconds'] for s in stages)
    summary['operations'] = {
        'selection_kinds': dict(Counter(r['selection_kind'] for r in selected_rows)),
        'refine_statuses': dict(Counter(r['refine_status'] for r in final_rows)),
        'final_exactly_equal_c': sum(r['prediction'] == runs['C']['rows'][r['id']]['primary'] for r in final_rows),
        'model_explicit_keep': sum(r['selection_kind'] == 'keep' for r in selected_rows),
        'scope_policy_keep': sum(r['selection_kind'] == 'scope_policy_keep' for r in selected_rows),
        'full412_stage_seconds_including_loads': elapsed,
        'seconds_per_query_amortized': elapsed/412 if stages else None,
        'sampled_gpu_peak_mib': max(gpu_memory) if gpu_memory else None,
        'gpu_sampling_interval_seconds': 1,
        'gpu_sampling_is_not_allocator_peak': True,
    }
    dump(out / "summary.json", summary)
    dump(out / "gate.json", gate)
    dump_rows(out / "manual_review_queue.jsonl", review_ids)
    lines = ["# IR/Depth选择与局部细化：完整412结果", "", "模型权重冻结；以下是推理方法成绩，不是新训练能力。", "",
             "| 条件 | ACC@0.5命中 | ACC@0.7命中 | mIoU | 解析率 | 已知重用47 | 其余365 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in scores.items():
        m = item["all412"]
        lines.append(f"|{name}|{m['hits_0.5']}/412|{m['hits_0.7']}/412|{m['mean_iou']:.6f}|{m['parse_rate']:.2%}|{item['shared47']['hits_0.5']}/47|{item['other365']['hits_0.5']}/365|")
    for pair in pairs:
        ci = pair["image_group_bootstrap"]["acc_0.5_delta_95ci"]
        lines += ["", f"- {pair['left']}→{pair['right']}：纠正{len(pair['wrong_to_right']['ids'])}，退化{len(pair['right_to_wrong']['ids'])}；"
                  f"图组10000次重采样ACC差95%区间[{ci[0]:.4f}, {ci[1]:.4f}]。"]
    pool = summary["candidate_diagnostics"]
    lines += ["", f"真实候选覆盖{pool['target_pool_covered']}/412；额外覆盖C错误{pool['newly_covered_c_errors']}题；候选中存在正确框但选择未命中{pool['covered_but_selection_missed']}题。覆盖率不是系统成绩。",
              "", f"完整方法相对C净{delta:+d}题；{'值得复验的候选，非已证突破' if gate['replication_candidate'] else '未达到预设复验候选门槛'}。",
              "", "只有净增才运行固定96题证据条件对照；固定候选含IR来源，不把RGB证据条件称为纯RGB系统。未做对照前，不将选择/细化收益归功于辅助模态。",
              "", "47条为已知地点重用，365条不等于已证明场景完全独立；没有官方测试GT。"]
    (out / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


def report_diagnostics(args):
    gt = load_manifest(args.gt)
    assert len(gt) == 96
    paths = {"rgb_evidence": args.rgb, "rgb_ir": args.rgb_ir, "trimodal": args.trimodal}
    runs = {}
    for name, path in paths.items():
        # Full trimodal412 is deliberately subset by the pre-fixed 96 IDs after inference.
        rows = [row for row in read_rows(path) if row["id"] in gt]
        subset_path = args.output_dir / f"{name}_fixed96.jsonl"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        dump_rows(subset_path, rows)
        runs[name] = load_run(name, subset_path, set(gt), allow_partial=False)
    pairs = [compare_runs(runs[a], runs[b], gt) for a,b in
             (("rgb_evidence", "rgb_ir"), ("rgb_ir", "trimodal"), ("rgb_evidence", "trimodal"))]
    dump(args.output_dir / "diagnostics96.json", {"metrics": {name: metrics(run, gt) for name, run in runs.items()},
         "pairs": pairs, "warning": "Fixed candidates retain IR proposal information; RGB evidence is not an RGB-only system."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("main")
    for name in ("gt", "c", "selected", "final", "evidence", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    p = sub.add_parser("diagnostics")
    for name in ("gt", "rgb", "rgb-ir", "trimodal", "output-dir"):
        p.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "main": run_report(args)
    else: report_diagnostics(args)


if __name__ == "__main__":
    main()
