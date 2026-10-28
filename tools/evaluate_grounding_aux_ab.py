"""Prepare the corrected City evaluation set and compare grounding/aux runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import build_user_content, load_model_images
from tools.predict_native_submission import MAX_NEW_TOKENS, MIN_PIXELS
from tools.predict_submission_vllm import check_upscale_consistency, upscale_images
from tools.verify_acc05_dev import box_is_valid, iou_xyxy


BOOTSTRAP_ROUNDS = 5000
BOOTSTRAP_SEED = 2026
CHOICES = {"A", "B", "both", "neither", "unknown"}
MEDIA_FIELDS = ("visible", "infrared", "depth")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _media_path(value: str, image_root: Path, field: str) -> str:
    parts = list(PurePosixPath(value.replace("\\", "/")).parts)
    while parts and parts[0] == "..":
        parts.pop(0)
    if not parts or parts[0] != field or ".." in parts:
        raise ValueError(f"expected {field} media path, got {value!r}")
    relative = Path(*parts)
    root = image_root.resolve()
    resolved = (root / relative).resolve()
    return resolved.relative_to(root).as_posix()


def prepare(gt_path: Path, corrections_path: Path, output_dir: Path,
            image_root: Path) -> dict[str, Any]:
    source = json.loads(gt_path.read_text(encoding="utf-8-sig"))
    corrections = json.loads(corrections_path.read_text(encoding="utf-8-sig"))
    cases = corrections["cases"]
    by_id = {str(case["id"]): case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("corrections contain duplicate IDs")
    if set(by_id) - set(source):
        raise ValueError("corrections contain IDs absent from City GT")

    status_counts = Counter(case["status"] for case in cases)
    expected_status = {"keep": 28, "corrected": 51, "deferred": 11, "partial": 1}
    if dict(status_counts) != expected_status:
        raise ValueError(f"unexpected correction statuses: {dict(status_counts)}")
    query_changed = {sid for sid, case in by_id.items() if case["query_changed"]}
    needs_inference = {sid for sid, case in by_id.items() if case["requires_new_inference"]}
    declared_inference = set(corrections["summary"]["requires_new_inference_ids"])
    if query_changed != needs_inference or query_changed != declared_inference or len(query_changed) != 27:
        raise ValueError("changed-query IDs and new-inference IDs must match the 27 reviewed changes")

    queries: dict[str, dict[str, str]] = {}
    gt: dict[str, list[float]] = {}
    groups: dict[str, str] = {}
    regression: list[str] = []
    reviewed: list[str] = []
    unresolved: list[str] = []
    for sample_id, record in source.items():
        if not isinstance(record, dict) or not record.get("query"):
            raise ValueError(f"{sample_id}: invalid City query record")
        queries[sample_id] = {
            **{field: _media_path(record[field], image_root, field) for field in MEDIA_FIELDS},
            "query": record["query"],
        }
        bbox = record["bbox"]
        if not box_is_valid(bbox):
            raise ValueError(f"{sample_id}: invalid GT bbox")
        gt[sample_id] = [float(value) for value in bbox]
        groups[sample_id] = PurePosixPath(record["visible"].replace("\\", "/")).stem
        case = by_id.get(sample_id)
        if case is None:
            regression.append(sample_id)
        elif case["status"] in {"keep", "corrected"}:
            reviewed.append(sample_id)
        else:
            unresolved.append(sample_id)
        if case is not None and case["query_changed"] and case["new_query"] != record["query"]:
            raise ValueError(f"{sample_id}: corrected query does not match GT")

    if len(source) != 412 or len(regression) != 321 or len(reviewed) != 79 or len(unresolved) != 12:
        raise ValueError("City partitions must contain 321 regression, 79 reviewed, and 12 unresolved IDs")
    city400 = regression + reviewed
    all412 = list(source)
    subsets = {
        "_metadata": {
            "primary": "city400",
            "prediction_only": ["unresolved12"],
            "descriptive_only": ["all412_descriptive_only"],
            "note": "ACC@0.5 and mIoU use city400; all412 is descriptive only.",
        },
        "city400": city400,
        "regression321": regression,
        "reviewed79": reviewed,
        "unresolved12": unresolved,
        "all412_descriptive_only": all412,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "queries.json", queries)
    write_json(output_dir / "gt.json", gt)
    write_json(output_dir / "groups.json", groups)
    write_json(output_dir / "subsets.json", subsets)
    return {
        "query_count": len(queries), "gt_count": len(gt),
        "query_changed_count": len(query_changed), "city400": len(city400),
        "regression321": len(regression), "reviewed79": len(reviewed),
        "unresolved12": len(unresolved), "image_root": str(image_root.resolve()),
        "all412_role": "descriptive_only",
    }


def load_gt(path: Path) -> dict[str, list[float]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("GT must be a non-empty object keyed by ID")
    gt = {}
    for sample_id, record in raw.items():
        box = record["bbox"] if isinstance(record, dict) else record
        if not box_is_valid(box):
            raise ValueError(f"{sample_id}: invalid GT bbox")
        gt[str(sample_id)] = [float(value) for value in box]
    return gt


def load_predictions(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [dict(record, id=sample_id) for sample_id, record in raw.items()]
    raise ValueError(f"unsupported prediction file: {path}")


def index_predictions(rows: list[dict[str, Any]], gt: dict[str, list[float]],
                      queries: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    indexed = {}
    for number, row in enumerate(rows, 1):
        sample_id = str(row["id"])
        if sample_id not in gt:
            raise ValueError(f"prediction row {number} has extra ID: {sample_id}")
        if sample_id in indexed:
            raise ValueError(f"duplicate prediction ID: {sample_id}")
        if queries is not None:
            if "query" not in row:
                raise ValueError(f"prediction has no query for {sample_id}; use raw predictions.jsonl")
            if row["query"] != queries[sample_id]["query"]:
                raise ValueError(f"query mismatch for {sample_id}")
        indexed[sample_id] = row
    return indexed


def load_groups(path: Path | None, gt: dict[str, list[float]]) -> dict[str, str]:
    if path is None:
        return {sample_id: sample_id.rsplit("_", 1)[0] for sample_id in gt}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("groups must map IDs to image-group names")
    groups = {str(sample_id): str(group) for sample_id, group in raw.items()}
    missing = set(gt) - set(groups)
    extra = set(groups) - set(gt)
    if missing or extra:
        raise ValueError(f"group IDs differ from GT: missing={len(missing)}, extra={len(extra)}")
    return groups


def load_subsets(path: Path | None, gt: dict[str, list[float]]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if path is None:
        return {"all": list(gt)}, {"primary": "all", "prediction_only": [], "descriptive_only": []}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    metadata = raw.pop("_metadata", {})
    subsets = {name: [str(sample_id) for sample_id in ids]
               for name, ids in raw.items() if not name.startswith("_")}
    for name, ids in subsets.items():
        if not ids:
            raise ValueError(f"subset {name} is empty")
        if len(ids) != len(set(ids)):
            raise ValueError(f"subset {name} has duplicate IDs")
        if set(ids) - set(gt):
            raise ValueError(f"subset {name} contains IDs absent from GT")
    if not subsets:
        raise ValueError("subsets file has no subsets")
    metadata.setdefault("primary", next(iter(subsets)))
    metadata.setdefault("prediction_only", [])
    metadata.setdefault("descriptive_only", [])
    if metadata["primary"] not in subsets:
        raise ValueError("primary subset is absent")
    marked = set(metadata["prediction_only"]) | set(metadata["descriptive_only"])
    if marked - set(subsets):
        raise ValueError("subset role metadata names an absent subset")
    if metadata["primary"] in marked:
        raise ValueError("primary subset cannot be prediction-only or descriptive-only")
    return subsets, metadata


def _parse_state(row: dict[str, Any] | None) -> tuple[bool | None, bool | None]:
    if row is None:
        return None, None
    parsed = row.get("parsed")
    fallback = row.get("fallback")
    if parsed is not None and not isinstance(parsed, bool):
        raise ValueError(f"parsed must be boolean for {row.get('id')}")
    if fallback is not None and not isinstance(fallback, bool):
        raise ValueError(f"fallback must be boolean for {row.get('id')}")
    if parsed is not None and fallback is not None and parsed == fallback:
        raise ValueError(f"parsed/fallback flags conflict for {row.get('id')}")
    if parsed is None and fallback is not None:
        parsed = not fallback
    if fallback is None and parsed is not None:
        fallback = not parsed
    return parsed, fallback


def score_items(ids: list[str], gt: dict[str, list[float]],
                predictions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    items = {}
    for sample_id in ids:
        row = predictions.get(sample_id)
        pred = None if row is None else (row["prediction"] if "prediction" in row else row.get("bbox"))
        valid = box_is_valid(pred)
        iou = iou_xyxy([float(value) for value in pred], gt[sample_id]) if valid else 0.0
        parsed, fallback = _parse_state(row)
        items[sample_id] = {
            "prediction": [float(value) for value in pred] if valid else None,
            "valid": valid, "iou": iou, "hit": bool(valid and iou >= 0.5),
            "parsed": parsed, "fallback": fallback, "missing": row is None,
        }
    return items


def _arm_metrics(ids: list[str], items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = [items[sample_id] for sample_id in ids]
    denominator = len(ids)
    known = [item for item in values if item["parsed"] is not None]
    parsed = sum(item["parsed"] is True for item in known)
    fallback = sum(item["fallback"] is True for item in known)
    hits = sum(item["hit"] for item in values)
    missing_ids = [sample_id for sample_id in ids if items[sample_id]["missing"]]
    invalid_ids = [sample_id for sample_id in ids if not items[sample_id]["missing"] and not items[sample_id]["valid"]]
    return {
        "denominator": denominator, "observed": denominator - len(missing_ids),
        "missing_count": len(missing_ids), "missing_ids": missing_ids,
        "invalid_count": len(invalid_ids), "invalid_ids": invalid_ids,
        "hits_0.5": hits, "acc_0.5": hits / denominator if denominator else None,
        "mean_iou": sum(item["iou"] for item in values) / denominator if denominator else None,
        "parse_status_known": len(known), "parsed_count": parsed,
        "parsed_rate": parsed / len(known) if known else None,
        "fallback_count": fallback,
        "fallback_rate": fallback / len(known) if known else None,
    }


def _prediction_diagnostics(ids: list[str], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [predictions.get(sample_id) for sample_id in ids]
    states = [_parse_state(row) for row in rows if row is not None]
    known = [state for state in states if state[0] is not None]
    parsed = sum(state[0] is True for state in known)
    fallback = sum(state[1] is True for state in known)
    missing_ids = [sample_id for sample_id, row in zip(ids, rows) if row is None]
    return {
        "denominator": len(ids), "observed": len(ids) - len(missing_ids),
        "missing_count": len(missing_ids), "missing_ids": missing_ids,
        "parse_status_known": len(known),
        "parsed_count": parsed,
        "parsed_rate": parsed / len(known) if known else None,
        "fallback_count": fallback,
        "fallback_rate": fallback / len(known) if known else None,
    }


def paired_bootstrap(ids: list[str], groups: dict[str, str],
                     items_a: dict[str, dict[str, Any]], items_b: dict[str, dict[str, Any]],
                     rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_group[groups[sample_id]].append(sample_id)
    names = sorted(by_group)
    rng = random.Random(seed)
    deltas = []
    for _ in range(rounds):
        chosen = [names[rng.randrange(len(names))] for _ in names]
        selected_ids = [sample_id for group in chosen for sample_id in by_group[group]]
        delta = sum(items_b[sample_id]["hit"] - items_a[sample_id]["hit"]
                    for sample_id in selected_ids) / len(selected_ids)
        deltas.append(delta)
    deltas.sort()

    def percentile(q: float) -> float:
        return deltas[min(len(deltas) - 1, int(q * len(deltas)))]

    point = (sum(items_b[sample_id]["hit"] - items_a[sample_id]["hit"] for sample_id in ids)
             / len(ids))
    return {
        "comparison": "B-A", "unit": "image_group", "groups": len(names),
        "rounds": rounds, "seed": seed, "point_delta_pp": 100 * point,
        "ci95_pp": [100 * percentile(0.025), 100 * percentile(0.975)],
    }


def compare(gt_path: Path, groups_path: Path | None, subsets_path: Path | None,
            arms: list[tuple[str, Path]], output_dir: Path,
            queries_path: Path | None = None) -> dict[str, Any]:
    if len(arms) != 2 or {name for name, _ in arms} != {"A", "B"}:
        raise ValueError("compare requires exactly --arm A=... and --arm B=...")
    gt = load_gt(gt_path)
    queries = json.loads(queries_path.read_text(encoding="utf-8-sig")) if queries_path else None
    if queries is not None and set(queries) != set(gt):
        raise ValueError("query manifest IDs differ from GT")
    groups = load_groups(groups_path, gt)
    subsets, metadata = load_subsets(subsets_path, gt)
    prediction_indexes = {
        name: index_predictions(load_predictions(path), gt, queries)
        for name, path in arms
    }
    all_ids = list(gt)
    prediction_only = set(metadata["prediction_only"])
    descriptive_only = set(metadata["descriptive_only"])
    scored_subsets = {name: ids for name, ids in subsets.items()
                      if name not in prediction_only and name not in descriptive_only}
    score_ids = list(dict.fromkeys(sample_id for ids in scored_subsets.values() for sample_id in ids))
    scored = {name: score_items(score_ids, gt, prediction_indexes[name])
              for name, _ in arms}

    results: dict[str, Any] = {}
    for name, ids in subsets.items():
        role = ("primary" if name == metadata["primary"] else
                "prediction_only" if name in prediction_only else
                "descriptive_only" if name in descriptive_only else "scored")
        if role in {"prediction_only", "descriptive_only"}:
            arm_metrics = {arm_name: _prediction_diagnostics(ids, prediction_indexes[arm_name])
                           for arm_name, _ in arms}
        else:
            arm_metrics = {arm_name: _arm_metrics(ids, scored[arm_name])
                           for arm_name, _ in arms}
        a_metrics, b_metrics = arm_metrics["A"], arm_metrics["B"]
        entry: dict[str, Any] = {"role": role, "count": len(ids), "arms": arm_metrics}
        if role not in {"prediction_only", "descriptive_only"}:
            a_items, b_items = scored["A"], scored["B"]
            rescued = [sample_id for sample_id in ids if not a_items[sample_id]["hit"] and b_items[sample_id]["hit"]]
            harmed = [sample_id for sample_id in ids if a_items[sample_id]["hit"] and not b_items[sample_id]["hit"]]
            rescued_groups = sorted({groups[sample_id] for sample_id in rescued})
            harmed_groups = sorted({groups[sample_id] for sample_id in harmed})
            entry.update({
                "delta_acc_0.5_pp": 100 * (b_metrics["acc_0.5"] - a_metrics["acc_0.5"]),
                "paired_bootstrap": paired_bootstrap(ids, groups, a_items, b_items),
                "rescued_ids": rescued, "rescued_groups": rescued_groups,
                "harmed_ids": harmed, "harmed_groups": harmed_groups,
            })
        results[name] = entry

    cases = []
    scored_ids = set(score_ids)
    for sample_id in all_ids:
        membership = [name for name, ids in subsets.items() if sample_id in ids]
        case = {
            "id": sample_id, "group": groups[sample_id], "subsets": membership,
            "scored": sample_id in scored_ids,
        }
        if sample_id in scored_ids:
            case["target"] = gt[sample_id]
        case["arms"] = {}
        for name, _ in arms:
            item = scored[name].get(sample_id)
            if item is None:
                item = score_items([sample_id], gt, prediction_indexes[name])[sample_id]
            prediction = {key: item[key] for key in ("prediction", "valid", "parsed", "fallback", "missing")}
            if sample_id in scored_ids:
                prediction.update({"iou": item["iou"], "hit": item["hit"]})
            case["arms"][name] = prediction
        cases.append(case)

    summary = {
        "gt_file": str(gt_path), "gt_count": len(gt),
        "query_manifest": str(queries_path) if queries_path else None,
        "query_checks": "every observed prediction must contain the exact manifest query" if queries_path else "not requested",
        "primary_subset": metadata["primary"],
        "all412_role": "descriptive_only" if "all412_descriptive_only" in descriptive_only else None,
        "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "seed": BOOTSTRAP_SEED, "unit": "image_group"},
        "subsets": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    (output_dir / "report.md").write_text(_comparison_report(summary), encoding="utf-8")
    return summary


def _comparison_report(summary: dict[str, Any]) -> str:
    lines = ["# Grounding A/B comparison", ""]
    lines.append(f"Primary subset: **{summary['primary_subset']}**.")
    if summary["all412_role"]:
        lines.append("The 412-row view is descriptive only; the primary ACC@0.5 and mIoU use city400.")
    lines += ["", "| Subset | Role | N | A ACC@0.5 | B ACC@0.5 | B−A pp | A mIoU | B mIoU |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in summary["subsets"].items():
        if result["role"] in {"prediction_only", "descriptive_only"}:
            lines.append(f"| {name} | {result['role']} | {result['count']} | — | — | — | — | — |")
            continue
        a, b = result["arms"]["A"], result["arms"]["B"]
        lines.append(f"| {name} | {result['role']} | {result['count']} | {a['acc_0.5']:.4f} | {b['acc_0.5']:.4f} | {result['delta_acc_0.5_pp']:+.2f} | {a['mean_iou']:.4f} | {b['mean_iou']:.4f} |")
    lines += ["", "## Prediction diagnostics", "",
              "| Subset | Arm | Observed / N | Missing | Parsed rate | Fallback rate |",
              "|---|---|---:|---:|---:|---:|"]
    for name, result in summary["subsets"].items():
        for arm in ("A", "B"):
            metrics = result["arms"][arm]
            parsed = "—" if metrics["parsed_rate"] is None else f"{metrics['parsed_rate']:.3f}"
            fallback = "—" if metrics["fallback_rate"] is None else f"{metrics['fallback_rate']:.3f}"
            lines.append(f"| {name} | {arm} | {metrics['observed']} / {metrics['denominator']} | "
                         f"{metrics['missing_count']} | {parsed} | {fallback} |")
        if result["role"] in {"primary", "scored"}:
            lines.append("")
            lines.append(f"{name}: rescued {len(result['rescued_ids'])} IDs in "
                         f"{len(result['rescued_groups'])} groups; harmed {len(result['harmed_ids'])} IDs in "
                         f"{len(result['harmed_groups'])} groups.")
            ci = result["paired_bootstrap"]["ci95_pp"]
            lines.append(f"Paired image-group bootstrap B−A 95% CI: [{ci[0]:+.2f}, {ci[1]:+.2f}] pp.")
    lines += ["", "Rescued and harmed IDs are listed in `summary.json`; bootstrap uses 5000 rounds and seed 2026.", ""]
    return "\n".join(lines)


def _target_choice(record: dict[str, Any]) -> tuple[str, str | None]:
    target = record.get("expected", record.get("target"))
    if target is None:
        assistants = [message for message in record.get("messages", [])
                      if message.get("role") == "assistant"]
        if assistants:
            target = assistants[-1].get("content")
    if isinstance(target, str):
        target = json.loads(target)
    if not isinstance(target, dict) or target.get("choice") not in CHOICES:
        raise ValueError(f"{record.get('id')}: aux manifest has no valid target choice")
    return target["choice"], target.get("reason")


def _response_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _response_text(value.get("content", value.get("text", value.get("response", ""))))
    if isinstance(value, list):
        return "\n".join(_response_text(item) for item in value if item is not None).strip()
    return ""


def parse_choice_response(value: Any) -> tuple[str | None, str | None, bool]:
    if isinstance(value, dict) and "choice" in value:
        payload = value
    else:
        text = _response_text(value)
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None, None, False
    if not isinstance(payload, dict):
        return None, None, False
    choice = payload.get("choice")
    reason = payload.get("reason")
    if choice not in CHOICES:
        return None, reason if isinstance(reason, str) else None, False
    return choice, reason if isinstance(reason, str) else None, True


def score_aux(manifest_path: Path, predictions_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("aux evaluation manifest is empty")
    targets: dict[str, dict[str, Any]] = {}
    categories = {}
    for record in manifest:
        sample_id = str(record["id"])
        if sample_id in targets:
            raise ValueError(f"duplicate aux manifest ID: {sample_id}")
        choice, reason = _target_choice(record)
        category = record.get("task_category", record.get("category", "unspecified"))
        targets[sample_id] = {"choice": choice, "reason": reason}
        categories[sample_id] = str(category)

    predicted = {}
    for row in read_jsonl(predictions_path):
        sample_id = str(row["id"])
        if sample_id not in targets:
            raise ValueError(f"aux prediction has extra ID: {sample_id}")
        if sample_id in predicted:
            raise ValueError(f"duplicate aux prediction ID: {sample_id}")
        value = row.get("response", row.get("prediction", row.get("raw_text")))
        choice, reason, parsed = parse_choice_response(value)
        predicted[sample_id] = {"choice": choice, "reason": reason, "parsed": parsed}

    cases = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for sample_id, target in targets.items():
        output = predicted.get(sample_id, {"choice": None, "reason": None, "parsed": False})
        correct = output["choice"] == target["choice"]
        case = {
            "id": sample_id, "category": categories[sample_id],
            "expected": target["choice"], "prediction": output["choice"],
            "reason": output["reason"], "parsed": output["parsed"],
            "correct": correct, "missing": sample_id not in predicted,
        }
        cases.append(case)
        by_category[categories[sample_id]].append(case)
        confusion[target["choice"]][output["choice"] or "parse_failure"] += 1

    denominator = len(targets)
    parsed_count = sum(case["parsed"] for case in cases)
    correct_count = sum(case["correct"] for case in cases)
    summary = {
        "manifest": str(manifest_path), "predictions": str(predictions_path),
        "samples": denominator, "observed": len(predicted),
        "missing_count": denominator - len(predicted), "parsed_count": parsed_count,
        "parse_rate": parsed_count / denominator if denominator else None,
        "correct": correct_count, "accuracy": correct_count / denominator if denominator else None,
        "by_category": {
            name: {"samples": len(rows), "correct": sum(row["correct"] for row in rows),
                  "accuracy": sum(row["correct"] for row in rows) / len(rows)}
            for name, rows in sorted(by_category.items())
        },
        "confusion": {answer: dict(counts) for answer, counts in sorted(confusion.items())},
        "interpretation": "Aux choice accuracy is separate from grounding ACC@0.5 and is not a bbox gain claim.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "aux_summary.json", summary)
    (output_dir / "aux_cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    _write_aux_report(output_dir / "aux_report.md", summary)
    return summary


def _write_aux_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Auxiliary choice evaluation", "",
             f"Accuracy: {summary['correct']}/{summary['samples']} = {summary['accuracy']:.4f}; "
             f"parse rate: {summary['parse_rate']:.4f}; missing: {summary['missing_count']}.",
             "", "| Category | N | Accuracy |", "|---|---:|---:|"]
    for category, row in summary["by_category"].items():
        lines.append(f"| {category} | {row['samples']} | {row['accuracy']:.4f} |")
    lines += ["", summary["interpretation"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_aux_requests(tokenizer, records: list[dict[str, Any]], upscale: float
                       ) -> list[dict[str, Any]]:
    requests = []
    for record in records:
        sample_id = str(record["id"])
        manifest_messages = record["messages"]
        if any(message.get("role") == "assistant" for message in manifest_messages):
            raise ValueError(f"{sample_id}: aux inference manifest must not contain assistant messages")
        user_messages = [message for message in manifest_messages if message.get("role") == "user"]
        if len(user_messages) != 1 or len(manifest_messages) != 1:
            raise ValueError(f"{sample_id}: aux inference requires one user-only message")
        prompt = user_messages[0]["content"]
        if not isinstance(prompt, str):
            raise ValueError(f"{sample_id}: prepared aux prompt must be a string")
        image_paths = [Path(path) for path in record["images"]]
        images = upscale_images(load_model_images(image_paths), upscale)
        messages = [{"role": "user", "content": build_user_content(
            prompt, images, prompt_has_image_placeholders=True)}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not rendered.rstrip().endswith("</think>"):
            raise ValueError(
                f"{sample_id}: rendered auxiliary prompt did not close the disabled thinking block")
        requests.append({
            "id": sample_id,
            "request": {"prompt": rendered, "multi_modal_data": {"image": images}},
        })
    return requests


def infer_aux(manifest_path: Path, output_path: Path, model: str,
              adapter: Path | None = None, *, min_pixels: int = MIN_PIXELS,
              max_pixels: int = 8294400, model_max_length: int = 32768,
              max_new_tokens: int = MAX_NEW_TOKENS, upscale: float = 2.0,
              gpu_memory_utilization: float = 0.90, max_num_seqs: int = 32,
              max_lora_rank: int = 32, chunk_size: int = 8,
              mamba_cache_mode: str = "none", no_prefix_caching: bool = False
              ) -> dict[str, Any]:
    check_upscale_consistency(upscale, max_pixels)
    manifest = read_jsonl(manifest_path)
    if not manifest:
        raise ValueError("aux inference manifest is empty")
    ids = [str(record["id"]) for record in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("aux inference manifest contains duplicate IDs")
    max_images = max(len(record["images"]) for record in manifest)
    adapter_path = adapter.resolve() if adapter else None
    if adapter_path is not None and not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found: {adapter_path}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=model, dtype="bfloat16", max_model_len=model_max_length,
        gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
        enable_prefix_caching=not no_prefix_caching,
        mamba_cache_mode=mamba_cache_mode,
        enable_lora=adapter_path is not None, max_lora_rank=max_lora_rank,
        limit_mm_per_prompt={"image": max_images},
        mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )
    tokenizer = llm.get_tokenizer()
    lora_request = (LoRARequest("trained", 1, str(adapter_path))
                    if adapter_path is not None else None)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                              skip_special_tokens=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(manifest), chunk_size):
            records = manifest[start:start + chunk_size]
            requests = build_aux_requests(tokenizer, records, upscale)
            outputs = llm.generate([item["request"] for item in requests], sampling,
                                   use_tqdm=False, lora_request=lora_request)
            for item, output in zip(requests, outputs):
                handle.write(json.dumps({
                    "id": item["id"], "response": output.outputs[0].text.strip(),
                }, ensure_ascii=False) + "\n")
            handle.flush()
    return {
        "model": model, "adapter": str(adapter_path) if adapter_path else None,
        "manifest": str(manifest_path), "output": str(output_path),
        "samples": len(manifest), "min_pixels": min_pixels, "max_pixels": max_pixels,
        "model_max_length": model_max_length, "max_new_tokens": max_new_tokens,
        "upscale": upscale, "max_num_seqs": max_num_seqs, "chunk_size": chunk_size,
        "enable_prefix_caching": not no_prefix_caching,
        "mamba_cache_mode": mamba_cache_mode,
    }


def _parse_arms(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--arm must be NAME=predictions.jsonl")
        parsed.append((name, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--gt", type=Path, required=True)
    prepare_parser.add_argument("--corrections", type=Path, required=True)
    prepare_parser.add_argument("--image-root", type=Path, required=True,
                                help="Data root containing visible/, infrared/, and depth/; paths are not checked")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--gt", type=Path, required=True)
    compare_parser.add_argument("--groups", type=Path)
    compare_parser.add_argument("--subsets", type=Path)
    compare_parser.add_argument("--arm", action="append", required=True)
    compare_parser.add_argument("--queries", type=Path)
    compare_parser.add_argument("--output-dir", type=Path, required=True)

    aux_parser = subparsers.add_parser("aux")
    aux_parser.add_argument("--manifest", type=Path, required=True)
    aux_parser.add_argument("--predictions", type=Path, required=True)
    aux_parser.add_argument("--output-dir", type=Path, required=True)

    infer_parser = subparsers.add_parser("infer-aux")
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--adapter", type=Path)
    infer_parser.add_argument("--manifest", type=Path, required=True)
    infer_parser.add_argument("--output", type=Path, required=True,
                              help="输出仅含 id 和 response 的 JSONL 文件")
    infer_parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS)
    infer_parser.add_argument("--max-pixels", type=int, default=8294400)
    infer_parser.add_argument("--model-max-length", type=int, default=32768)
    infer_parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    infer_parser.add_argument("--upscale", type=float, default=2.0)
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    infer_parser.add_argument("--max-num-seqs", type=int, default=32)
    infer_parser.add_argument("--max-lora-rank", type=int, default=32)
    infer_parser.add_argument("--chunk-size", type=int, default=8)
    infer_parser.add_argument("--mamba-cache-mode", default="none")
    infer_parser.add_argument("--no-prefix-caching", action="store_true")

    args = parser.parse_args()
    if args.command == "infer-aux":
        result = infer_aux(
            args.manifest, args.output, args.model, args.adapter,
            min_pixels=args.min_pixels, max_pixels=args.max_pixels,
            model_max_length=args.model_max_length, max_new_tokens=args.max_new_tokens,
            upscale=args.upscale, gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs, max_lora_rank=args.max_lora_rank,
            chunk_size=args.chunk_size, mamba_cache_mode=args.mamba_cache_mode,
            no_prefix_caching=args.no_prefix_caching,
        )
    elif args.command == "prepare":
        result = prepare(args.gt, args.corrections, args.output_dir, args.image_root)
    elif args.command == "compare":
        result = compare(args.gt, args.groups, args.subsets, _parse_arms(args.arm),
                         args.output_dir, args.queries)
    elif args.command == "aux":
        result = score_aux(args.manifest, args.predictions, args.output_dir)
    else:
        raise ValueError(f"unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
