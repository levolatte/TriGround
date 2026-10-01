"""Cache real retained-A predictions and causal context; never fabricate proposals."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch

from tools.train_grounding_structure import read, write, load_model
from tools.structured_grounder import QwenStructureRuntime, generate
from tools.evaluate_pretrained_grounder import parse_generated_bbox
from tools.prepare_grounding_revision import box_iou


def input_identity(row):
    # Exact input metadata, no GT, no file hashes. Resume rejects changed conditions.
    keys = ("id", "prompt", "image", "modalities", "depth_encoding", "depth_raw",
            "missing_modalities_actual", "degraded_modality")
    return {key: row[key] for key in keys if key in row}


def load_cache(path, model, adapter, rows):
    metadata = {"model":str(model), "adapter":str(adapter), "inputs":[input_identity(r) for r in rows],
                "max_new_tokens":128, "do_sample":False,
                "retained_adapter_dtype":"bfloat16", "tf32":False, "matmul_precision":"highest"}
    if Path(path).exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["metadata"] != metadata:
            raise ValueError("proposal cache uses a different model, input, or decode setting")
    else:
        cache = {"metadata":metadata, "adapter":str(adapter), "rows":{}}
    return cache


def baseline_summary(rows, predictions):
    groups={'all_conditions':rows,
            'city412':[r for r in rows if r.get('task_category')=='old_city' and r.get('condition')=='normal'],
            'diagnostic119':[r for r in rows if r.get('task_category')!='old_city' and r.get('condition')=='normal']}
    result={}
    for name,group in groups.items():
        if not group:continue
        ious=[box_iou(predictions[r['id']]['prediction'],r['rgb_gt_bbox']) for r in group]
        result[name]={'denominator':len(group),'correct':sum(x>=.5 for x in ious),
                      'miou':sum(ious)/len(ious),'parse_failures':sum(predictions[r['id']]['prediction'] is None for r in group)}
    return result


def cache_predictions(args):
    if not torch.cuda.is_available():
        raise RuntimeError("real 8B proposal generation requires the authorized GPU stage")
    rows = read(args.inputs)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("cache input IDs must be unique")
    cache = load_cache(args.output, args.model, args.adapter, rows)
    model, processor = load_model(args.model, args.adapter, trainable=False)
    model.eval().requires_grad_(False)
    runtime = QwenStructureRuntime(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for index, row in enumerate(rows):
        if row["id"] in cache["rows"]:
            continue
        query_started=time.monotonic()
        result = generate(model, processor, runtime, row)
        result["prediction"] = parse_generated_bbox(result["raw_text"])
        result['wall_seconds']=time.monotonic()-query_started
        cache["rows"][row["id"]] = result
        if (index+1)%10 == 0 or index+1 == len(rows):
            torch.save(cache, args.output)
            print(f"A cache {len(cache['rows'])}/{len(rows)}", flush=True)
    torch.save(cache, args.output)
    write(args.output.with_suffix(".summary.json"), {"rows":len(rows), "cached":len(cache['rows']),
        "parse_failures":sum(x['prediction'] is None for x in cache['rows'].values()),
        "elapsed_seconds":time.monotonic()-started, "adapter":str(args.adapter),
        "baseline_metrics":baseline_summary(rows,cache['rows'])})
    runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model", "adapter", "inputs", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    cache_predictions(parser.parse_args())
