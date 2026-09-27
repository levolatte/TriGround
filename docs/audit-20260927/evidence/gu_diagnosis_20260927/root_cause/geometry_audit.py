"""CPU-only, source-checked geometry audit for the 83 accepted auxiliary tasks."""

import collections
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/gu_diagnosis_20260927"
SNAP = BASE / "cloud_snapshot"
OUT = BASE / "root_cause"
MODELS = {"m2": "m2_fit_legacy181", "gstar": "gstar_fit_legacy181", "ustar": "ustar_fit_legacy181"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def indexed(rows, key):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), f"duplicate {key}"
    return result


def box_equal(a, b):
    return len(a) == len(b) == 4 and all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(a, b))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    assert 0 <= ax1 < ax2 <= 1 and 0 <= ay1 < ay2 <= 1, a
    assert 0 <= bx1 < bx2 <= 1 and 0 <= by1 < by2 <= 1, b
    w = max(0, min(ax2, bx2) - max(ax1, bx1))
    h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = w * h
    return intersection / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection)


def metrics(rows, field):
    values = [row[field] for row in rows]
    return {
        "n": len(values),
        "mean_iou": statistics.mean(values),
        "median_iou": statistics.median(values),
        "min_iou": min(values),
        "hits_0_5": sum(x >= 0.5 for x in values),
        "acc_0_5": statistics.mean(x >= 0.5 for x in values),
        "hits_0_7": sum(x >= 0.7 for x in values),
        "acc_0_7": statistics.mean(x >= 0.7 for x in values),
    }


def paired(rows, transferred, actual):
    return {
        "transferred_minus_actual_mean_iou": statistics.mean(row[transferred] - row[actual] for row in rows),
        "transferred_only_hits_0_5": sum(row[transferred] >= 0.5 and row[actual] < 0.5 for row in rows),
        "actual_only_hits_0_5": sum(row[actual] >= 0.5 and row[transferred] < 0.5 for row in rows),
        "both_hits_0_5": sum(row[transferred] >= 0.5 and row[actual] >= 0.5 for row in rows),
        "both_misses_0_5": sum(row[transferred] < 0.5 and row[actual] < 0.5 for row in rows),
        "transferred_only_hits_0_7": sum(row[transferred] >= 0.7 and row[actual] < 0.7 for row in rows),
        "actual_only_hits_0_7": sum(row[actual] >= 0.7 and row[transferred] < 0.7 for row in rows),
    }


def main():
    gt = read_json(SNAP / "manifests/fit_legacy181_gt.json")
    manifest = indexed(read_json(SNAP / "manifests/fit_legacy181.json"), "id")
    meta = indexed(read_jsonl(SNAP / "manifests/fit_legacy181_metadata.jsonl"), "id")
    source = indexed(read_jsonl(SNAP / "source/pending_candidates.jsonl"), "task_id")
    assert len(gt) == len(manifest) == len(meta) == 181
    assert set(gt) == set(manifest) == set(meta)
    assert collections.Counter(row["target_modality"] for row in gt.values()) == {"rgb": 98, "infrared": 58, "depth": 25}

    rgb_by_source = {}
    for task_id, row in gt.items():
        original = source[task_id]
        assert row["source_id"] == meta[task_id]["source_id"] == f'{original["bundle_id"]}::{original["origin_query_id"]}', task_id
        assert row["candidate_query_id"] == original["query_id"] and row["candidate_origin_query_id"] == original["origin_query_id"], task_id
        assert row["query"] == meta[task_id]["query"] == original["query"]
        assert row["target_modality"] == meta[task_id]["target_modality"] == original["output_modality"]
        assert box_equal(row["bbox"], original["bbox_xyxy_normalized"]), task_id
        assert box_equal(row["rgb_bbox"], original["rgb_gt_bbox_xyxy_normalized"]), task_id
        assert manifest[task_id]["image"] == [row["visible"], row["infrared"], row["depth"]], task_id
        if row["target_modality"] == "rgb":
            assert box_equal(row["bbox"], row["rgb_bbox"]), task_id
            assert row["source_id"] not in rgb_by_source
            rgb_by_source[row["source_id"]] = task_id
    assert len(rgb_by_source) == 98

    predictions = {}
    for model, directory in MODELS.items():
        pred = indexed(read_jsonl(SNAP / directory / "predictions.jsonl"), "id")
        assert len(pred) == 181 and set(pred) == set(gt), model
        for task_id, row in pred.items():
            assert row["parsed"] and row["prediction"] is not None, (model, task_id)
            assert box_equal(row["target"], gt[task_id]["bbox"]), (model, task_id)
            assert row["image"] == manifest[task_id]["image"], (model, task_id)
            # Cloud report uses float32 box arithmetic; double precision differs by <2e-6.
            assert math.isclose(iou(row["prediction"], gt[task_id]["bbox"]), row["iou"], abs_tol=2e-6), (model, task_id)
        predictions[model] = pred

    rows = []
    for task_id, target in gt.items():
        modality = target["target_modality"]
        if modality == "rgb":
            continue
        source_id = target["source_id"]
        rgb_id = rgb_by_source[source_id]
        rgb = gt[rgb_id]
        assert target["query_key"] == rgb["query_key"] and target["query"] == rgb["query"]
        assert box_equal(target["rgb_bbox"], rgb["bbox"]), task_id
        assert [target[k] for k in ("visible", "infrared", "depth")] == [rgb[k] for k in ("visible", "infrared", "depth")]
        row = {
            "aux_id": task_id,
            "rgb_id": rgb_id,
            "source_id": source_id,
            "query": target["query"],
            "modality": modality,
            "rgb_gt_box": rgb["bbox"],
            "aux_gt_box": target["bbox"],
            "rgb_gt_to_aux_iou": iou(rgb["bbox"], target["bbox"]),
            "rgb_gt_to_aux_max_abs_coord_shift": max(abs(a - b) for a, b in zip(rgb["bbox"], target["bbox"])),
        }
        for model, pred in predictions.items():
            rgb_pred = pred[rgb_id]["prediction"]
            aux_pred = pred[task_id]["prediction"]
            row[f"{model}_rgb_prediction"] = rgb_pred
            row[f"{model}_aux_prediction"] = aux_pred
            row[f"{model}_rgb_prediction_to_aux_iou"] = iou(rgb_pred, target["bbox"])
            row[f"{model}_actual_aux_iou"] = iou(aux_pred, target["bbox"])
            row[f"{model}_transfer_minus_actual_iou"] = row[f"{model}_rgb_prediction_to_aux_iou"] - row[f"{model}_actual_aux_iou"]
            row[f"{model}_rgb_prediction_vs_aux_prediction_iou"] = iou(rgb_pred, aux_pred)
            row[f"{model}_rgb_prediction_vs_aux_prediction_max_abs_coord_shift"] = max(abs(a - b) for a, b in zip(rgb_pred, aux_pred))
        rows.append(row)
    rows.sort(key=lambda r: r["aux_id"])
    assert len(rows) == 83 and collections.Counter(r["modality"] for r in rows) == {"infrared": 58, "depth": 25}

    train = {name: read_jsonl(SNAP / f"manifests/train_{name}_metadata.jsonl") for name in ("b", "gstar", "ustar")}
    for name, entries in train.items():
        assert len(entries) == 1600 and sum(row["is_new"] for row in entries) == (0 if name == "b" else 400), name
    new = [row for row in train["ustar"] if row["is_new"]]
    assert collections.Counter(row["target_modality"] for row in new) == {"rgb": 217, "infrared": 128, "depth": 55}
    assert len({row["source_id"] for row in new}) == 98
    assert {row["source_task_id"] for row in new} == set(gt)
    assert [row["source_task_id"] if row["is_new"] else None for row in train["gstar"]] == [row["source_task_id"] if row["is_new"] else None for row in train["ustar"]]
    assert [row.get("mapped_from_source_task_id") for row in train["b"]] == [row["source_task_id"] if row["is_new"] else None for row in train["ustar"]]

    groups = {"all": rows, "infrared": [row for row in rows if row["modality"] == "infrared"], "depth": [row for row in rows if row["modality"] == "depth"]}
    summary = {
        "definition": "All boxes are original normalized floating-point xyxy. IoU is intersection/union. RGB GT copy is a geometric reference, not a model result; RGB prediction transfer reuses each model's separately prompted same-query RGB output, not an actual auxiliary output.",
        "sources": {
            "raw_candidates": "cloud_snapshot/source/pending_candidates.jsonl",
            "accepted_gt": "cloud_snapshot/manifests/fit_legacy181_gt.json",
            "manifest": "cloud_snapshot/manifests/fit_legacy181.json",
            "metadata": "cloud_snapshot/manifests/fit_legacy181_metadata.jsonl",
            "predictions": {m: f"cloud_snapshot/{d}/predictions.jsonl" for m, d in MODELS.items()},
            "train_metadata": [f"cloud_snapshot/manifests/train_{m}_metadata.jsonl" for m in ("b", "gstar", "ustar")],
        },
        "counts": {
            "accepted_tasks": 181,
            "unique_sources_and_queries": 98,
            "accepted_by_modality": {"rgb": 98, "infrared": 58, "depth": 25},
            "new_presentations": 400,
            "new_presentations_by_modality": {"rgb": 217, "infrared": 128, "depth": 55},
        },
        "groups": {},
    }
    for group, subset in groups.items():
        data = {
            "rgb_gt_copy_geometric_reference": metrics(subset, "rgb_gt_to_aux_iou"),
            "rgb_gt_max_abs_coord_shift_median": statistics.median(row["rgb_gt_to_aux_max_abs_coord_shift"] for row in subset),
            "models": {},
        }
        for model in MODELS:
            transfer = f"{model}_rgb_prediction_to_aux_iou"
            actual = f"{model}_actual_aux_iou"
            data["models"][model] = {
                "rgb_prediction_transfer": metrics(subset, transfer),
                "actual_aux_output": metrics(subset, actual),
                "paired_transfer_vs_actual": paired(subset, transfer, actual),
                "rgb_prediction_vs_aux_prediction_median_iou": statistics.median(row[f"{model}_rgb_prediction_vs_aux_prediction_iou"] for row in subset),
                "rgb_prediction_vs_aux_prediction_median_max_abs_coord_shift": statistics.median(row[f"{model}_rgb_prediction_vs_aux_prediction_max_abs_coord_shift"] for row in subset),
            }
        summary["groups"][group] = data

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "geometry.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "geometry_rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value for key, value in row.items()})
    print(json.dumps(summary["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
