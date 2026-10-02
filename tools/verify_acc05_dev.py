"""独立复算比赛指标 ACC@0.5，并校验提交文件格式。

官方定义（contest/基于大模型的多模态视觉理解与推理.pdf 第6页）：
    ACC@0.5 = IoU>=0.5 的查询数 / 测试集全部查询数
    反向坐标(x1>=x2 或 y1>=y2)、越界、NaN、无效空框 -> 该预测无效（仍计入分母）

本脚本不信任已保存的 iou/hit/acc_0.5 字段，只用原始浮点框重算；并核对
预测文件内嵌 target 与独立 GT 文件是否一致。

用法：
    python tools/verify_acc05_dev.py \
        --gt <city412_gt.json> \
        --arm C0=<C0/predictions.jsonl> \
        --arm A=<A/predictions.jsonl> \
        --taxonomy A --compare A:C0 --bootstrap 5000 \
        --out-json <dir>/metric_verification.json

    python tools/verify_acc05_dev.py --submission-json <predictions.json> \
        --submission-zip <submission.zip> --out-json <dir>/submission_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def iou_xyxy(a, b) -> float:
    """标准 IoU；假定两框均为合法 xyxy（调用方先做合法性判定）。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def box_is_valid(box) -> bool:
    """官方异常框判定：非四元、非数值、NaN、反向、越界、零面积。"""
    if box is None or len(box) != 4:
        return False
    for v in box:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        if math.isnan(v) or math.isinf(v):
            return False
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 >= x2 or y1 >= y2:
        return False
    if not (0.0 <= x1 and 0.0 <= y1 and x2 <= 1.0 and y2 <= 1.0):
        return False
    return True


def group_of(sample_id: str) -> str:
    """图像组 = 去掉最后一段 Query 序号。例：city_000004_012_00000001_001 -> city_000004_012_00000001"""
    return sample_id.rsplit("_", 1)[0]


def load_gt(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    gt = {}
    for key, rec in raw.items():
        box = rec["bbox"] if isinstance(rec, dict) else rec
        gt[key] = [float(v) for v in box]
    return gt


def load_arm(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_arm(rows: list, gt: dict) -> dict:
    """只用原始浮点框重算；返回逐题结果与汇总。"""
    per_item = {}
    invalid = []
    missing = []
    target_mismatch = []
    for row in rows:
        sid = row["id"]
        if sid not in gt:
            missing.append(sid)
            continue
        pred = row.get("prediction")
        ref = gt[sid]
        embedded = row.get("target")
        if embedded is not None and any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(embedded, ref)):
            target_mismatch.append(sid)
        valid = box_is_valid(pred)
        if not valid:
            invalid.append(sid)
            value = 0.0
        else:
            value = iou_xyxy([float(v) for v in pred], ref)
        per_item[sid] = {
            "iou": value,
            "hit": bool(valid and value >= 0.5),
            "valid": bool(valid),
            "prediction": None if pred is None else [float(v) for v in pred],
            "target": ref,
            "group": group_of(sid),
        }
    denominator = len(gt)
    hits = sum(1 for v in per_item.values() if v["hit"])
    ious = [v["iou"] for v in per_item.values()]
    return {
        "observed": len(rows),
        "denominator": denominator,
        "missing_ids": missing,
        "invalid_predictions": invalid,
        "embedded_target_mismatch": target_mismatch,
        "hits_0.5": hits,
        "acc_0.5": hits / denominator,
        "mean_iou": sum(ious) / denominator,
        "items": per_item,
    }


def bootstrap_groups(arms: dict, names: tuple, rounds: int, seed: int = 2030) -> dict:
    """按图像组重采样的配对差值区间（只描述本开发集，不含训练随机性）。"""
    a, b = names
    items_a, items_b = arms[a]["items"], arms[b]["items"]
    groups = defaultdict(list)
    for sid, rec in items_a.items():
        groups[rec["group"]].append(sid)
    keys = sorted(groups)
    rng = random.Random(seed)
    deltas = []
    for _ in range(rounds):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        ids = [sid for g in pick for sid in groups[g]]
        hits_a = sum(items_a[s]["hit"] for s in ids)
        hits_b = sum(items_b[s]["hit"] for s in ids)
        deltas.append((hits_a - hits_b) / len(ids))
    deltas.sort()

    def pct(q):
        return deltas[min(len(deltas) - 1, int(q * len(deltas)))]

    point = (arms[a]["hits_0.5"] - arms[b]["hits_0.5"]) / arms[a]["denominator"]
    return {
        "comparison": f"{a}-{b}",
        "groups": len(keys),
        "rounds": rounds,
        "point_delta_pp": 100 * point,
        "ci95_pp": [100 * pct(0.025), 100 * pct(0.975)],
        "crosses_zero": pct(0.025) <= 0 <= pct(0.975),
    }


def binomial_floor(denominator: int, acc: float) -> dict:
    """单臂 95% 正态区间与最小可检测差值（同分母配对差值的粗略分辨率）。"""
    se = math.sqrt(acc * (1 - acc) / denominator)
    return {
        "denominator": denominator,
        "acc_0.5": acc,
        "se": se,
        "ci95_pp": [100 * (acc - 1.96 * se), 100 * (acc + 1.96 * se)],
        "hits_se": se * denominator,
        "one_hit_pp": 100 / denominator,
    }


def taxonomy(items: dict, gt: dict) -> dict:
    """对未命中题做几何分类；同图其它 Query 的 GT 作为竞争实例。"""
    by_group = defaultdict(list)
    for sid, box in gt.items():
        by_group[group_of(sid)].append(sid)
    buckets = Counter()
    per_case = []
    for sid, rec in items.items():
        if rec["hit"]:
            continue
        peers = [p for p in by_group[rec["group"]] if p != sid]
        best_peer, best_peer_iou = None, 0.0
        for p in peers:
            value = iou_xyxy(rec["prediction"], gt[p]) if rec["valid"] else 0.0
            if value > best_peer_iou:
                best_peer, best_peer_iou = p, value
        own = rec["iou"]
        if not rec["valid"]:
            label = "invalid_box"
        elif best_peer_iou >= 0.5 and best_peer_iou > own:
            label = "other_instance"
        elif own >= 0.4:
            label = "near_miss"
        elif own >= 0.25:
            label = "drift"
        else:
            label = "gross_miss"
        buckets[label] += 1
        per_case.append(
            {
                "id": sid,
                "group": rec["group"],
                "own_iou": round(own, 4),
                "best_peer_id": best_peer,
                "best_peer_iou": round(best_peer_iou, 4),
                "peers_in_group": len(peers),
                "label": label,
            }
        )
    multi = sum(1 for r in items.values() if len(by_group[r["group"]]) > 1)
    return {
        "failures": len(per_case),
        "buckets": dict(buckets),
        "groups": len(by_group),
        "items_in_multi_query_groups": multi,
        "cases": sorted(per_case, key=lambda r: r["own_iou"]),
    }


def check_submission(json_path: Path, zip_path: Path | None, reference_query_ids: list | None = None) -> dict:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    invalid, unnormalized, extra_fields = [], [], []
    required = {"visible", "infrared", "depth", "query", "bbox"}
    for key, rec in raw.items():
        if not isinstance(rec, dict):
            invalid.append(key)
            continue
        if not required.issubset(rec.keys()):
            extra_fields.append({"id": key, "missing": sorted(required - set(rec.keys()))})
        box = rec.get("bbox")
        if not box_is_valid(box):
            invalid.append(key)
            continue
        if any(float(v) < 0 or float(v) > 1 for v in box):
            unnormalized.append(key)
    report = {
        "file": str(json_path),
        "records": len(raw),
        "invalid_boxes": len(invalid),
        "invalid_ids_sample": invalid[:10],
        "missing_required_fields": len(extra_fields),
        "missing_fields_sample": extra_fields[:10],
        "out_of_range_boxes": len(unnormalized),
    }
    if reference_query_ids is not None:
        ref = list(reference_query_ids)
        report["reference_count"] = len(ref)
        report["id_order_identical"] = list(raw.keys()) == ref
        report["missing_from_submission"] = sorted(set(ref) - set(raw))
        report["unexpected_ids"] = sorted(set(raw) - set(ref))
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            report["zip_entries"] = names
            report["zip_single_root_json"] = names == ["predictions.json"]
            if "predictions.json" in names:
                inner = json.loads(zf.read("predictions.json").decode("utf-8"))
                report["zip_matches_json"] = inner == raw
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, help="独立 GT json（id -> bbox 或 {..., bbox}）")
    ap.add_argument("--arm", action="append", default=[], help="NAME=predictions.jsonl")
    ap.add_argument("--taxonomy", help="对哪个臂做未命中几何分类")
    ap.add_argument("--compare", help="配对比较，形如 A:C0")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--submission-json", type=Path)
    ap.add_argument("--submission-zip", type=Path)
    ap.add_argument("--reference-ids", type=Path, help="原始输入清单，用于核对提交 ID/顺序")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    out = {}
    if args.arm:
        assert args.gt, "--arm 需要 --gt"
        gt = load_gt(args.gt)
        arms = {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            result = score_arm(load_arm(Path(path)), gt)
            arms[name] = result
            out.setdefault("arms", {})[name] = {
                k: v for k, v in result.items() if k != "items"
            }
            out["arms"][name]["hits_pct"] = 100 * result["acc_0.5"]
        out["gt_file"] = str(args.gt)
        out["gt_records"] = len(gt)
        out["noise_floor"] = {
            name: binomial_floor(r["denominator"], r["acc_0.5"]) for name, r in arms.items()
        }
        if args.compare:
            a, b = args.compare.split(":")
            if args.bootstrap > 0:
                out["paired_bootstrap"] = bootstrap_groups(arms, (a, b), args.bootstrap)
        if args.taxonomy:
            out["taxonomy"] = taxonomy(arms[args.taxonomy]["items"], gt)
            out["taxonomy"]["arm"] = args.taxonomy

    if args.submission_json:
        ref = None
        if args.reference_ids and args.reference_ids.exists():
            text = args.reference_ids.read_text(encoding="utf-8-sig")
            if text.lstrip().startswith("{"):
                ref = list(json.loads(text).keys())
            else:
                ref = [json.loads(l)["id"] for l in text.splitlines() if l.strip()]
        out["submission"] = check_submission(args.submission_json, args.submission_zip, ref)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {k: v for k, v in out.items() if k not in ("taxonomy",)}
    if "taxonomy" in out:
        summary["taxonomy_buckets"] = out["taxonomy"]["buckets"]
        summary["taxonomy_failures"] = out["taxonomy"]["failures"]
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
