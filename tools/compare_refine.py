"""对比「第一遍」与「裁剪-复核」两轮预测，按失分类型分别报净变化。

## 为什么必须分类报，而不是只看总分

裁剪会丢掉目标周围的上下文。对"框太小、位置对"的题，放大是帮忙；
对"彻底找错物体"的题，那些被丢掉的多实例上下文恰恰是消歧依据，放大是帮倒忙。
两类题目的净变化可能互相抵消，总分看起来"没变化"——那会掩盖真正的机制。

所以本脚本按第一遍的失分形态把题分组，逐组报 `rescued` / `damaged`：

* **紧框组**：中心基本对（偏移 < 0.3 倍对角）但框明显偏小（尺寸比 < 0.85）
* **错例组**：中心偏得远（偏移 ≥ 0.5 倍对角），基本是找错了物体
* **其余**：介于两者之间

## 判据

**不只看总分。** 采纳条件建议为：总分净增 ≥ +8 题，且错例组净损失 ≤ 3 题。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.verify_acc05_dev import box_is_valid, iou_xyxy, load_gt  # noqa: E402


def load_rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in
            (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip())}


def classify(box: list[float] | None, gt: list[float]) -> dict:
    """按第一遍的形态给题打标签。box 非法时归为「无效」。"""
    if box is None or not box_is_valid(box):
        return {"iou": 0.0, "kind": "invalid"}
    iou = iou_xyxy(box, gt)
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    size_ratio = (box_area / gt_area) ** 0.5 if gt_area > 0 and box_area > 0 else 0.0
    gt_c = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    box_c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    diag = ((gt[2] - gt[0]) ** 2 + (gt[3] - gt[1]) ** 2) ** 0.5
    offset = (((gt_c[0] - box_c[0]) ** 2 + (gt_c[1] - box_c[1]) ** 2) ** 0.5) / diag if diag else 0.0
    if iou >= 0.5:
        kind = "hit"
    elif offset < 0.3 and size_ratio < 0.85:
        kind = "tight"          # 位置对、框太小 —— 剪裁应该帮到这一类
    elif offset >= 0.5:
        kind = "wrong_object"   # 彻底找错 —— 剪裁可能帮倒忙
    else:
        kind = "other"
    return {"iou": iou, "kind": kind, "size_ratio": size_ratio, "offset": offset}


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--min-gain", type=int, default=8)
    parser.add_argument("--max-wrong-object-loss", type=int, default=3)
    args = parser.parse_args()

    gt = load_gt(args.gt)
    base = load_rows(args.baseline)
    cand = load_rows(args.candidate)
    ids = [key for key in base if key in cand and key in gt]
    if not ids:
        raise ValueError("三轮的 ID 没有交集")

    groups: dict[str, dict[str, int]] = {}
    base_hits = cand_hits = 0
    details = []
    for sample_id in ids:
        b = base[sample_id].get("bbox")
        c = cand[sample_id].get("bbox")
        info = classify(b, gt[sample_id])
        b_hit = info["iou"] >= 0.5
        c_hit = bool(c is not None and box_is_valid(c) and iou_xyxy(c, gt[sample_id]) >= 0.5)
        base_hits += b_hit
        cand_hits += c_hit
        group = groups.setdefault(info["kind"], {"n": 0, "rescued": 0, "damaged": 0, "kept": 0})
        group["n"] += 1
        if c_hit and not b_hit:
            group["rescued"] += 1
        elif b_hit and not c_hit:
            group["damaged"] += 1
        elif b_hit:
            group["kept"] += 1
        details.append({"id": sample_id, "kind": info["kind"], "base_iou": round(info["iou"], 4),
                        "cand_iou": round(iou_xyxy(c, gt[sample_id]) if c and box_is_valid(c) else 0.0, 4),
                        "base_hit": b_hit, "cand_hit": c_hit})

    net = cand_hits - base_hits
    wrong_loss = groups.get("wrong_object", {}).get("damaged", 0) - groups.get("wrong_object", {}).get("rescued", 0)
    verdict = "adopt" if (net >= args.min_gain and wrong_loss <= args.max_wrong_object_loss) else "reject"
    reason = []
    if net < args.min_gain:
        reason.append(f"净增 {net} < {args.min_gain}")
    if wrong_loss > args.max_wrong_object_loss:
        reason.append(f"错例组净损失 {wrong_loss} > {args.max_wrong_object_loss}")

    print(f"  基线命中 {base_hits}/{len(ids)}   裁剪复核 {cand_hits}/{len(ids)}   净 {net:+d}")
    print()
    print(f"  {'组':<14} {'题数':>4} {'救回':>4} {'损害':>4} {'保住':>4} {'净':>4}")
    for kind in ("hit", "tight", "other", "wrong_object", "invalid"):
        if kind not in groups:
            continue
        g = groups[kind]
        print(f"  {kind:<14} {g['n']:>4} {g['rescued']:>4} {g['damaged']:>4} "
              f"{g['kept']:>4} {g['rescued']-g['damaged']:>+4d}")
    print()
    print(f"  判定: {verdict}" + (f"（{'; '.join(reason)}）" if reason else ""))

    report = {"baseline_hits": base_hits, "candidate_hits": cand_hits, "net": net,
              "queries": len(ids), "groups": groups, "verdict": verdict,
              "verdict_reason": "; ".join(reason), "details": details}
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
