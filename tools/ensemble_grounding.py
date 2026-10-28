"""单目标定位集成：预测聚合不读取 GT，评分在聚合后独立执行。

配置示例（path 相对配置文件；同 family 的 weight 必须相同）：
  {"members": [{"name": "qwen_1x", "path": "qwen_1x.jsonl",
                "family": "qwen", "weight": 1},
               {"name": "qwen_2x", "path": "qwen_2x.jsonl",
                "family": "qwen", "weight": 1},
               {"name": "egm", "path": "egm.jsonl", "family": "egm", "weight": 1}]}

family 权重在配置成员间均分，每成员再在其有效候选间均分；失败成员不投票。
上述两份 Qwen 合计一票，EGM 一票。候选 score 不当作校准置信度使用。
输入支持 prediction/bbox JSONL、官方 QueryID→记录 JSON，以及
{id,candidates:[{bbox,raw_text,...}]} JSONL。坐标必须是 RGB 归一化 xyxy。

python tools/ensemble_grounding.py --config members.json --method wbf --out-dir out
python tools/ensemble_grounding.py --config members.json --compare-all --gt gt.json --out-dir out

WBF 使用 ensemble-boxes；其余算法仅依赖 NumPy。安装到当前环境：
python -m pip install ensemble-boxes==1.0.9
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_acc05_dev import box_is_valid, iou_xyxy, load_gt  # noqa: E402


METHODS = ("mean", "median", "medoid", "consensus", "cluster_medoid", "wbf")


@dataclass(frozen=True)
class Member:
    name: str
    path: Path
    family: str
    weight: float


@dataclass(frozen=True)
class Candidate:
    bbox: tuple[float, float, float, float]
    member: str
    family: str
    mass: float
    index: int


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """只在输入边界验证 ID；坏框保留为失败，不能从评分分母删除。"""
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(raw, dict):
            if "id" in raw and ("candidates" in raw or "bbox" in raw or "prediction" in raw):
                raw = [raw]
            else:
                raw = [{**record, "id": sid} for sid, record in raw.items()]
    result = {}
    for row in raw:
        sid = row["id"]
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"{path}: id 必须是非空字符串")
        if sid in result:
            raise ValueError(f"{path}: id 重复: {sid}")
        if not any(key in row for key in ("candidates", "bbox", "prediction")):
            raise ValueError(f"{path}: {sid} 缺少 candidates/prediction/bbox")
        result[sid] = row
    if not result:
        raise ValueError(f"{path}: 预测文件为空")
    return result


def load_members(config_path: Path) -> tuple[list[Member], dict[str, dict[str, dict]]]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    members = [Member(
        name=row["name"], path=(config_path.parent / row["path"]).resolve(),
        family=row.get("family", row["name"]), weight=float(row.get("weight", 1.0)),
    ) for row in config["members"]]
    validate_members(members)
    tables = {member.name: load_predictions(member.path) for member in members}
    ids = set(tables[members[0].name])
    for member in members[1:]:
        observed = set(tables[member.name])
        if observed != ids:
            raise ValueError(f"{member.name}: ID集合不一致，缺{len(ids-observed)}题、多{len(observed-ids)}题")
    return members, tables


def validate_members(members: list[Member]) -> None:
    if not members or len({m.name for m in members}) != len(members):
        raise ValueError("members 必须非空，name 不能重复")
    weights = {}
    for member in members:
        if not member.name or not member.family:
            raise ValueError("member name/family 不能为空")
        if not math.isfinite(member.weight) or member.weight <= 0:
            raise ValueError("member weight 必须是有限正数")
        if member.family in weights and member.weight != weights[member.family]:
            raise ValueError(f"family {member.family} 的成员必须使用同一 family weight")
        weights[member.family] = member.weight


def valid_boxes(row: dict) -> list[tuple[float, float, float, float]]:
    """不消费 target、iou、hit、自报 confidence；显式失败/兜底不参与投票。"""
    if row.get("parsed") is False or row.get("fallback") is True:
        return []
    items = row["candidates"] if "candidates" in row else [row]
    result = []
    for item in items:
        if item.get("parsed") is False or item.get("fallback") is True:
            continue
        box = item["prediction"] if "prediction" in item else item.get("bbox")
        if isinstance(box, (list, tuple)) and box_is_valid(box):
            result.append(tuple(float(v) for v in box))
    return result


def candidates_for_id(sid: str, members: list[Member], tables: dict) -> list[Candidate]:
    family_counts = Counter(m.family for m in members)
    candidates = []
    for member in members:
        boxes = valid_boxes(tables[member.name][sid])
        if boxes:
            mass = member.weight / family_counts[member.family] / len(boxes)
            for box in boxes:
                candidates.append(Candidate(box, member.name, member.family, mass, len(candidates)))
    return candidates


def overlap_matrix(candidates: list[Candidate]) -> np.ndarray:
    return np.array([[iou_xyxy(a.bbox, b.bbox) for b in candidates] for a in candidates])


def weighted_median(boxes: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """半票恰落在相邻坐标之间时取中点，与等权 NumPy median 一致。"""
    result = []
    half = masses.sum() / 2
    for dim in range(4):
        order = np.argsort(boxes[:, dim], kind="stable")
        cumulative = np.cumsum(masses[order])
        position = int(np.searchsorted(cumulative, half, side="left"))
        value = boxes[order[position], dim]
        if position + 1 < len(order) and math.isclose(cumulative[position], half, abs_tol=1e-12):
            value = (value + boxes[order[position + 1], dim]) / 2
        result.append(value)
    return np.array(result)


def choose_medoid(candidates: list[Candidate]) -> Candidate:
    # 包含自重：这是加权 medoid 的标准目标；等权时对所有候选同加一个常数。
    scores = overlap_matrix(candidates) @ np.array([c.mass for c in candidates])
    best = max(range(len(candidates)), key=lambda i: (round(float(scores[i]), 12), -candidates[i].index))
    return candidates[best]


def greedy_clusters(candidates: list[Candidate], threshold: float) -> list[list[Candidate]]:
    clusters: list[list[Candidate]] = []
    centers = []
    for candidate in sorted(candidates, key=lambda c: (-c.mass, c.index)):
        overlaps = [iou_xyxy(candidate.bbox, center) for center in centers]
        best = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
        if best is None or overlaps[best] < threshold:
            clusters.append([candidate])
            centers.append(np.array(candidate.bbox))
        else:
            clusters[best].append(candidate)
            group = clusters[best]
            centers[best] = np.average([c.bbox for c in group], axis=0, weights=[c.mass for c in group])
    return clusters


def aggregate(candidates: list[Candidate], method: str, threshold: float = 0.55) -> dict:
    """纯预测聚合，无 GT 参数。全失败明确返回空框；正式打包会拒绝。"""
    if method not in METHODS:
        raise ValueError(f"未知方法: {method}")
    if not 0 < threshold <= 1:
        raise ValueError("iou threshold 必须在 (0,1] 内")
    result = {"bbox": None, "parsed": False, "method": method,
              "selected_member": None, "candidate_count": len(candidates)}
    if not candidates:
        result["failure"] = "all_members_invalid"
        return result
    boxes = np.array([c.bbox for c in candidates])
    masses = np.array([c.mass for c in candidates])
    chosen = None
    if method == "mean":
        box = np.average(boxes, axis=0, weights=masses)
    elif method == "median":
        box = weighted_median(boxes, masses)
    elif method == "medoid":
        chosen = choose_medoid(candidates)
        box = chosen.bbox
    elif method == "consensus":
        overlaps = overlap_matrix(candidates)
        votes = (overlaps >= threshold) @ masses
        smooth = overlaps @ masses
        i = max(range(len(candidates)), key=lambda k: (
            round(float(votes[k]), 12), round(float(smooth[k]), 12), -candidates[k].index))
        chosen = candidates[i]
        box = chosen.bbox
    elif method == "cluster_medoid":
        clusters = greedy_clusters(candidates, threshold)
        winning = max(clusters, key=lambda group: (
            round(sum(c.mass for c in group), 12), -min(c.index for c in group)))
        chosen = choose_medoid(winning)
        box = chosen.bbox
        result["cluster_count"] = len(clusters)
        result["winning_cluster_mass"] = sum(c.mass for c in winning)
    else:
        from ensemble_boxes import weighted_boxes_fusion

        # 每个有效候选作为一个 WBF 输入流，权重已按 family→member→candidate 分摊。
        # score 全1；avg 的输出排序等价于簇票质量排序，不冒充正确概率。
        fused, scores, _ = weighted_boxes_fusion(
            [[c.bbox] for c in candidates], [[1.0] for _ in candidates],
            [[0] for _ in candidates], weights=masses, iou_thr=threshold,
            skip_box_thr=0.0, conf_type="avg", allows_overflow=False,
        )
        # WBF 原库仅分数排序；同分时按与首个配置候选的重叠打破平局。
        i = max(range(len(fused)), key=lambda k: (
            round(float(scores[k]), 12), iou_xyxy(fused[k], candidates[0].bbox), -k))
        box = fused[i]
        result["cluster_count"] = len(fused)
        result["fusion_support"] = float(scores[i])
    box = [float(v) for v in box]
    if not box_is_valid(box):
        result["failure"] = "invalid_fused_box"
        return result
    result.update(bbox=box, parsed=True)
    if chosen is not None:
        result["selected_member"] = chosen.member
        result["selected_candidate_index"] = chosen.index
    return result


def aggregate_table(members: list[Member], tables: dict, method: str, threshold: float = 0.55) -> dict:
    return {sid: {"id": sid, **aggregate(candidates_for_id(sid, members, tables), method, threshold)}
            for sid in tables[members[0].name]}


def score_predictions(rows: dict, gt: dict) -> dict:
    if set(rows) != set(gt):
        raise ValueError("预测与 GT 的 ID 集合不一致")
    hits, ious, invalid = {}, [], []
    for sid, row in rows.items():
        box = row["bbox"]
        valid = isinstance(box, (list, tuple)) and box_is_valid(box)
        value = iou_xyxy(box, gt[sid]) if valid else 0.0
        hits[sid] = bool(valid and value >= 0.5)
        ious.append(value)
        if not valid:
            invalid.append(sid)
    return {"denominator": len(gt), "hits": sum(hits.values()),
            "acc_0.5": sum(hits.values()) / len(gt), "mean_iou": float(np.mean(ious)),
            "invalid_ids": invalid, "hit_by_id": hits}


def evaluate(members: list[Member], tables: dict, outputs: dict, gt: dict, baseline: str | None = None) -> dict:
    """评分阶段单独接收 GT。个体按首个有效候选评分，多候选另报 oracle。"""
    if not all(box_is_valid(box) for box in gt.values()):
        raise ValueError("GT 含非法归一化框")
    individual = {}
    for member in members:
        rows = {}
        for sid, row in tables[member.name].items():
            boxes = valid_boxes(row)
            rows[sid] = {"bbox": list(boxes[0]) if boxes else None}
        individual[member.name] = score_predictions(rows, gt)
    baseline = baseline or members[0].name
    reference = individual[baseline]["hit_by_id"]
    methods = {}
    for name, rows in outputs.items():
        score = score_predictions(rows, gt)
        observed = score["hit_by_id"]
        score["rescued"] = sum(observed[sid] and not reference[sid] for sid in gt)
        score["damaged"] = sum(reference[sid] and not observed[sid] for sid in gt)
        score["net_hits"] = score["rescued"] - score["damaged"]
        methods[name] = score
    oracle_ids = [sid for sid in gt if any(
        iou_xyxy(c.bbox, gt[sid]) >= 0.5 for c in candidates_for_id(sid, members, tables))]
    return {"baseline": baseline, "individual_first_valid_candidate": individual,
            "methods": methods, "original_candidate_oracle_hits": len(oracle_ids),
            "original_candidate_oracle_acc": len(oracle_ids) / len(gt),
            "oracle_note": "仅为选择原框的上限，几何融合生成的新框可能超过它"}


def package_predictions(rows: dict, queries: dict) -> dict:
    if set(rows) != set(queries):
        raise ValueError("预测与原 queries 的 ID 集合不一致")
    invalid = [sid for sid in queries if not rows[sid]["parsed"] or not box_is_valid(rows[sid]["bbox"])]
    if invalid:
        raise ValueError(f"不能生成正式提交：{len(invalid)}题无合法预测，示例 {invalid[:3]}")
    return {sid: {**record, "bbox": rows[sid]["bbox"]} for sid, record in queries.items()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="medoid")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--baseline", help="评分净变化参考成员；默认配置第一项")
    parser.add_argument("--queries", type=Path, help="提供时只更新 bbox 生成正式 JSON")
    args = parser.parse_args()
    members, tables = load_members(args.config)
    methods = METHODS if args.compare_all else (args.method,)
    outputs = {method: aggregate_table(members, tables, method, args.iou_threshold) for method in methods}
    queries = json.loads(args.queries.read_text(encoding="utf-8-sig")) if args.queries else None
    for method, rows in outputs.items():
        directory = args.out_dir / method if args.compare_all else args.out_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "predictions.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows.values()), encoding="utf-8")
        if queries is not None:
            write_json(directory / "predictions.json", package_predictions(rows, queries))
    report = {
        "members": [{"name": m.name, "path": str(m.path), "family": m.family, "weight": m.weight} for m in members],
        "methods": list(methods), "iou_threshold": args.iou_threshold,
        "family_weighting": "同family总weight均分给配置成员；再均分其有效候选；失败成员不投票",
        "failures": {name: sum(not row["parsed"] for row in rows.values()) for name, rows in outputs.items()},
    }
    if args.gt:
        report["evaluation"] = evaluate(members, tables, outputs, load_gt(args.gt), args.baseline)
    write_json(args.out_dir / "ensemble_report.json", report)
    print(json.dumps({"output": str(args.out_dir), "methods": list(methods), "failures": report["failures"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
