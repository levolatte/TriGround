"""Build verifier supervision from real TRAIN-only baseline predictions.

No synthetic perfect proposal, diagnostic question, or GT region is placed in
the prompt. The retained grounder is an inference dependency, not a trainable
component. This exports unique examples, not an invented 600-step schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from tools.grounding_revision import revision_prompt
from tools.prepare_triground_tm_data import _bbox_task
from tools.prepare_qwen3vl_native_sft import bbox_to_qwen1000


def box_iou(a, b):
    if a is None:
        return 0.0
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = x * y
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def make_revision_example(source, baseline, original, grounder_adapter):
    if source["split"] != "train" or source.get("selected_for_diagnostic"):
        raise ValueError(f"held-out source cannot become correction training: {source['task_id']}")
    if baseline["target"] != source["bbox"]:
        raise ValueError(f"baseline GT differs from reviewed floating GT: {source['task_id']}")
    if baseline["prompt"] != original["conversations"][0]["value"]:
        raise ValueError(f"baseline used a different full query/prompt: {source['task_id']}")
    proposal = baseline["prediction"]
    # Keep the existing contest-success decision; do not regress a new rounded
    # box for an already correct proposal. KEEP is a literal copy at inference.
    keep = box_iou(proposal, source["bbox"]) >= .5
    answer = ({"action": "keep"} if keep else
              {"action": "replace", "bbox_2d": bbox_to_qwen1000(source["bbox"])})
    row = dict(original)
    row.update({
        "id": "revision:" + source["task_id"], "task_type": "rgb_revision",
        "coordinate_system": "rgb_qwen_0_1000", "expected_answer": answer,
        "grounder_adapter": grounder_adapter, "baseline_prediction": proposal,
        "baseline_raw_text": baseline["raw_text"], "rgb_gt_bbox": source["bbox"],
        "baseline_iou": box_iou(proposal, source["bbox"]),
        "conversations": [
            {"from": "human", "value": revision_prompt(baseline["prompt"], proposal)},
            {"from": "gpt", "value": json.dumps(answer, separators=(",", ":"))},
        ],
    })
    return row


def build(release_dir: Path, baseline_dir: Path, output: Path):
    def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
    def rows(p):return [json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines() if x]
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    release = read(release_dir / "release.json")
    source = {r["task_id"]:r for r in rows(release_dir / "source_reviews.jsonl")}
    selected = {r["source_task_id"] for r in read(Path(release["manifests"]["M"]))
                if r["task_category"] != "old_city"}
    diagnostic_sources = {r["source_task_id"] for r in read(Path(release["diagnostics"]["normal"]))}
    if selected & diagnostic_sources:
        raise ValueError("training/diagnostic source overlap")
    config = read(baseline_dir / "run_config.json")
    prediction_rows = rows(baseline_dir / "predictions.jsonl")
    predictions = {}
    for row in prediction_rows:
        if not row["id"].startswith("tm:fit:"):
            raise ValueError("expected original training-fit predictions, not City validation or diagnostics")
        key = row["id"].removeprefix("tm:fit:")
        if key in predictions:
            raise ValueError(f"duplicate baseline prediction: {key}")
        predictions[key] = row
    data = []
    for task in sorted(selected):
        original = _bbox_task(source[task], "revision:"+task, output/"assets")
        data.append(make_revision_example(source[task], predictions[task], original, config["adapter"]))
    output.mkdir(parents=True)
    (output/"train.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    summary = {
        "role": "auxiliary_verifier_only", "retained_grounder": config["adapter"],
        "examples": len(data), "actions": dict(Counter(r["expected_answer"]["action"] for r in data)),
        "scene_groups": len({r["scene_id"] for r in data}),
        "training_only": True, "baseline_source": str(baseline_dir),
        "new_training_executed": False, "gt_used_only_for_supervision_and_scoring": True,
        "limitations": ["normal-input proposals only; no invented baseline predictions for degraded images",
                        "labels optimize ACC@0.5 retention; KEEP does not improve an already passing box",
                        "this pack does not certify modality necessity or independent scene generalization"],
    }
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir",type=Path,required=True)
    parser.add_argument("--baseline-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.release_dir,args.baseline_dir,args.output_dir),ensure_ascii=False,indent=2))
