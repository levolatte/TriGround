"""Read-only, paired 412-query audit for the G/U pilot.

Writes only docs/research/2026-09-26-gu-outcome-root-audit.json.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "M2": ROOT / "results/next_stage_20260925/baseline_m2/predictions.jsonl",
    "C": ROOT / "results/next_stage_20260925/c_step1500/predictions.jsonl",
    "G100": ROOT / "results/gu_pilot_20260926/final_cloud_snapshot/g_2026_city100/predictions.jsonl",
    "G200": ROOT / "results/gu_pilot_20260926/final_cloud_snapshot/g_2026_city200/predictions.jsonl",
    "U100": ROOT / "results/gu_pilot_20260926/final_cloud_snapshot/u_2026_city100/predictions.jsonl",
    "U200": ROOT / "results/gu_pilot_20260926/final_cloud_snapshot/u_2026_city200/predictions.jsonl",
}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def qstats(values):
    x = np.array(values, dtype=float)
    return {
        "mean": float(x.mean()), "median": float(np.median(x)),
        "q10": float(np.quantile(x, .1)), "q25": float(np.quantile(x, .25)),
        "q75": float(np.quantile(x, .75)), "q90": float(np.quantile(x, .9)),
        "min": float(x.min()), "max": float(x.max()),
    }


def load():
    runs = {}
    for name, path in SOURCES.items():
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        assert len(rows) == len({row["id"] for row in rows}) == 412, name
        runs[name] = {row["id"]: row for row in rows}
    ids = set(runs["M2"])
    assert all(set(rows) == ids for rows in runs.values())
    for id_ in ids:
        targets = [tuple(runs[name][id_]["target"]) for name in runs]
        prompts = [runs[name][id_]["prompt"] for name in runs]
        assert all(t == targets[0] for t in targets)
        assert all(p == prompts[0] for p in prompts)
        for name in runs:
            row = runs[name][id_]
            box = row["prediction"]
            assert row["parsed"] and all(math.isfinite(x) for x in box)
            assert 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
            assert abs(iou(box, targets[0]) - row["iou"]) < 1e-4, (name, id_)
    return runs, sorted(ids)


def linguistic_flags(prompt):
    query = prompt.split("Locate the object described by this query: ", 1)[1].split("\n", 1)[0].lower()
    return {
        "ordinal": bool(re.search(r"\b(first|second|third|fourth|last|leftmost|rightmost|middle|central|nearest|farthest)\b", query)),
        "spatial": bool(re.search(r"\b(left|right|top|bottom|upper|lower|front|back|behind|beside|next to|between|near|under|above|below|in front|edge|center|centre)\b", query)),
        "relation": bool(re.search(r"\b(beside|next to|between|near|behind|under|above|below|in front of|holding|wearing|with|following|facing|standing by)\b", query)),
    }


def main():
    runs, ids = load()
    groups = {id_: Path(runs["M2"][id_]["image"][0]).name for id_ in ids}
    assert len(set(groups.values())) == 78
    gt_area = {id_: area(runs["M2"][id_]["target"]) for id_ in ids}
    area_edges = np.quantile(list(gt_area.values()), [0, .25, .5, .75, 1]).tolist()
    area_bins = {id_: min(3, int(np.searchsorted(area_edges[1:-1], gt_area[id_], side="right"))) for id_ in ids}
    flags = {id_: linguistic_flags(runs["M2"][id_]["prompt"]) for id_ in ids}
    result = {
        "population": {"queries": len(ids), "image_groups": len(set(groups.values())), "gt_area_quartile_edges": area_edges,
                       "linguistic_flag_counts": {k: sum(v[k] for v in flags.values()) for k in ("ordinal", "spatial", "relation")}},
        "runs": {}, "pairs": {},
    }
    quant_rows = []
    for id_ in ids:
        target = runs["M2"][id_]["target"]
        snapped = [round(x * 1000) / 1000 for x in target]
        quant_rows.append({"id": id_, "area_quartile": area_bins[id_], "target_area": gt_area[id_],
                           "target_width": target[2]-target[0], "target_height": target[3]-target[1],
                           "snapped_iou": iou(snapped, target)})
    result["target_coordinate_quantization"] = {
        "method": "independently round each GT coordinate to nearest 0.001; GT-aware geometry diagnostic, not achievable inference",
        "snapped_gt_iou": qstats([r["snapped_iou"] for r in quant_rows]),
        "hits_05": sum(r["snapped_iou"] >= .5 for r in quant_rows),
        "below_095": sum(r["snapped_iou"] < .95 for r in quant_rows),
        "below_090": sum(r["snapped_iou"] < .9 for r in quant_rows),
        "per_area_quartile": [{"quartile":i+1,"n":len(rr),"mean_iou":mean(r["snapped_iou"] for r in rr),
                               "min_iou":min(r["snapped_iou"] for r in rr),
                               "below_095":sum(r["snapped_iou"]<.95 for r in rr)}
                              for i in range(4) for rr in [[r for r in quant_rows if r["area_quartile"]==i]]],
        "worst_10": sorted(quant_rows,key=lambda r:r["snapped_iou"])[:10],
    }
    for name, data in runs.items():
        x = [data[id_]["iou"] for id_ in ids]
        box_ratios = [area(data[id_]["prediction"]) / gt_area[id_] for id_ in ids]
        result["runs"][name] = {
            "hits_05": sum(v >= .5 for v in x), "hits_07": sum(v >= .7 for v in x), "iou": qstats(x),
            "iou_bands": {f"{lo:.1f}-{hi:.1f}": sum(lo <= v < hi for v in x) for lo, hi in zip([0,.1,.3,.5,.7,.9],[.1,.3,.5,.7,.9,1.000001])},
            "pred_to_gt_area_ratio": qstats(box_ratios),
        }
    for a, b in [("M2","G200"),("M2","U200"),("C","G200"),("C","U200"),("G200","U200"),("G100","G200"),("U100","U200")]:
        per = []
        for id_ in ids:
            ar, br = runs[a][id_], runs[b][id_]
            pa, pb, gt = ar["prediction"], br["prediction"], ar["target"]
            ca, cb = center(pa), center(pb)
            d = br["iou"] - ar["iou"]
            per.append({
                "id": id_, "group": groups[id_], "iou_a": ar["iou"], "iou_b": br["iou"], "delta": d,
                "box_agreement": iou(pa, pb), "center_shift": math.dist(ca, cb),
                "center_dx": cb[0] - ca[0], "center_dy": cb[1] - ca[1],
                "width_ratio_b_over_a": (pb[2]-pb[0])/(pa[2]-pa[0]),
                "height_ratio_b_over_a": (pb[3]-pb[1])/(pa[3]-pa[1]),
                "area_ratio_b_over_a": area(pb) / area(pa),
                "area_ratio_a_over_gt": area(pa) / gt_area[id_], "area_ratio_b_over_gt": area(pb) / gt_area[id_],
                "target_area": gt_area[id_], "area_quartile": area_bins[id_], "linguistic": flags[id_],
                "a_box": pa, "b_box": pb,
            })
        group_rows = defaultdict(list)
        for r in per:
            group_rows[r["group"]].append(r)
        d = [r["delta"] for r in per]
        high_agree = [r for r in per if r["box_agreement"] >= .7]
        low_agree = [r for r in per if r["box_agreement"] < .5]
        flips = [r for r in per if (r["iou_a"] >= .5) != (r["iou_b"] >= .5)]
        near_flips = [r for r in flips if abs(r["iou_a"] - .5) <= .1 and abs(r["iou_b"] - .5) <= .1]
        pair = {
            "hits_delta": sum(r["iou_b"] >= .5 for r in per) - sum(r["iou_a"] >= .5 for r in per),
            "iou_delta": qstats(d), "positive_negative_tie": [sum(v > 1e-6 for v in d), sum(v < -1e-6 for v in d), sum(abs(v) <= 1e-6 for v in d)],
            "delta_bands": {"loss_ge_0.3": sum(v <= -.3 for v in d), "loss_0.1_to_0.3": sum(-.3 < v <= -.1 for v in d),
                            "gain_0.1_to_0.3": sum(.1 <= v < .3 for v in d), "gain_ge_0.3": sum(v >= .3 for v in d)},
            "box_agreement": qstats([r["box_agreement"] for r in per]),
            "center_shift": qstats([r["center_shift"] for r in per]),
            "center_dx": qstats([r["center_dx"] for r in per]),
            "center_dy": qstats([r["center_dy"] for r in per]),
            "width_ratio_b_over_a": qstats([r["width_ratio_b_over_a"] for r in per]),
            "height_ratio_b_over_a": qstats([r["height_ratio_b_over_a"] for r in per]),
            "area_ratio_b_over_a": qstats([r["area_ratio_b_over_a"] for r in per]),
            "area_ratio_b_over_a_above_below_1": [sum(r["area_ratio_b_over_a"] > 1.05 for r in per), sum(r["area_ratio_b_over_a"] < .95 for r in per)],
            "high_agreement_n": len(high_agree), "high_agreement_mean_delta": mean(r["delta"] for r in high_agree),
            "low_agreement_n": len(low_agree), "low_agreement_mean_delta": mean(r["delta"] for r in low_agree) if low_agree else None,
            "low_agreement_delta_sum": sum(r["delta"] for r in low_agree),
            "top_negative_10": [{k:r[k] for k in ("id","delta","iou_a","iou_b","box_agreement","area_ratio_b_over_a")} for r in sorted(per,key=lambda x:x["delta"])[:10]],
            "top_positive_10": [{k:r[k] for k in ("id","delta","iou_a","iou_b","box_agreement","area_ratio_b_over_a")} for r in sorted(per,key=lambda x:-x["delta"])[:10]],
            "flips": [{k:r[k] for k in ("id","iou_a","iou_b","delta","box_agreement","center_shift","area_ratio_b_over_a")} for r in flips],
            "near_flip_count": len(near_flips),
            "group": {"sample_weighted_delta": mean(d), "equal_group_mean_delta": mean(mean(r["delta"] for r in rr) for rr in group_rows.values()),
                      "groups_positive_negative_tie": [sum(mean(r["delta"] for r in rr)>1e-6 for rr in group_rows.values()),
                                                       sum(mean(r["delta"] for r in rr)<-1e-6 for rr in group_rows.values()),
                                                       sum(abs(mean(r["delta"] for r in rr))<=1e-6 for rr in group_rows.values())],
                      "worst_10": [{"group":g,"n":len(rr),"mean_delta":mean(r["delta"] for r in rr),"sum_delta":sum(r["delta"] for r in rr)} for g,rr in sorted(group_rows.items(),key=lambda t:sum(r["delta"] for r in t[1]))[:10]],
                      "best_10": [{"group":g,"n":len(rr),"mean_delta":mean(r["delta"] for r in rr),"sum_delta":sum(r["delta"] for r in rr)} for g,rr in sorted(group_rows.items(),key=lambda t:-sum(r["delta"] for r in t[1]))[:10]]},
            "area_quartiles": [{"quartile":i+1,"n":len(rr),"mean_delta":mean(r["delta"] for r in rr),
                                "hit_delta":sum(r["iou_b"]>=.5 for r in rr)-sum(r["iou_a"]>=.5 for r in rr),
                                "low_agreement_n":sum(r["box_agreement"]<.5 for r in rr)}
                               for i in range(4) for rr in [[r for r in per if r["area_quartile"]==i]]],
            "linguistic": {key:{"n":len(rr),"mean_delta":mean(r["delta"] for r in rr),
                                "hit_delta":sum(r["iou_b"]>=.5 for r in rr)-sum(r["iou_a"]>=.5 for r in rr),
                                "low_agreement_n":sum(r["box_agreement"]<.5 for r in rr)}
                           for key in ("ordinal","spatial","relation") for rr in [[r for r in per if r["linguistic"][key]]]},
        }
        # Geometry-only diagnostic: retain B's center and replace its width/height
        # with A's for the same query. This uses A at evaluation time; it is not a
        # deployable correction or an unbiased validation of a new postprocessor.
        swapped = []
        for r in per:
            pa, pb = r["a_box"], r["b_box"]
            cx, cy = center(pb)
            w, h = pa[2]-pa[0], pa[3]-pa[1]
            box = [max(0, cx-w/2), max(0, cy-h/2), min(1, cx+w/2), min(1, cy+h/2)]
            target = runs[a][r["id"]]["target"]
            swapped.append((iou(box, target), r["iou_b"], r["box_agreement"]))
        pair["reference_size_same_new_center_diagnostic"] = {
            "all_mean_iou_change_vs_b": mean(x-y for x,y,_ in swapped),
            "all_hit_change_vs_b": sum(x>=.5 for x,_,_ in swapped)-sum(y>=.5 for _,y,_ in swapped),
            "high_agreement_mean_iou_change_vs_b": mean(x-y for x,y,z in swapped if z>=.7),
            "high_agreement_hit_change_vs_b": sum(x>=.5 for x,_,z in swapped if z>=.7)-sum(y>=.5 for _,y,z in swapped if z>=.7),
        }
        result["pairs"][f"{a}_to_{b}"] = pair
    out = ROOT / "docs/research/2026-09-26-gu-outcome-root-audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps({k:{"hits_delta":v["hits_delta"],"iou_delta":v["iou_delta"]["mean"],"low_agreement_n":v["low_agreement_n"],"near_flip_count":v["near_flip_count"]} for k,v in result["pairs"].items()}, indent=2))


if __name__ == "__main__":
    main()
