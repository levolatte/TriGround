"""Cache Qwen multimodal / EGM RGB candidates for offline ensemble experiments.

Each invocation loads one model. Run members sequentially on the 96GB host.
Training targets in input manifests are never included in model requests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_pretrained_grounder import (
    EGM_PROMPT, build_user_content, extract_query, parse_generated_bbox,
    resolve_image_path,
)
from tools.prepare_qwen3vl_native_sft import depth_mm_to_grayscale_rgb, native_prompt


def load_queries(path: Path, data_root: Path, modality: str) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        records = [dict(row, id=key) for key, row in payload.items()] if isinstance(payload, dict) else payload
    rows, seen = [], set()
    aliases = {"rgb": "rgb", "visible": "rgb", "ir": "infrared", "infrared": "infrared", "depth": "depth"}
    for row in records:
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"duplicate query ID: {sample_id}")
        seen.add(sample_id)
        query = row.get("query")
        if not query:
            messages = row.get("messages", row.get("conversations", []))
            user = [m for m in messages if m.get("role", m.get("from")) in {"user", "human"}]
            query = extract_query(user[-1].get("content", user[-1].get("value", "")))
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{sample_id}: empty query")
        if "images" in row or "image" in row:
            values = row.get("images", row.get("image"))
            values = [values] if isinstance(values, str) else values
            views = row.get("modalities", row.get("image_order"))
            if views is None:
                views = {1: ["rgb"], 3: ["rgb", "infrared", "depth"]}.get(len(values))
            if views is None or len(views) != len(values):
                raise ValueError(f"{sample_id}: provide modalities matching image paths")
            if len({aliases[view.lower()] for view in views}) != len(views):
                raise ValueError(f"{sample_id}: duplicate modalities")
            paths = {aliases[view.lower()]: str(resolve_image_path(value, data_root, path, from_data_root=True))
                     for view, value in zip(views, values)}
            depth_policy = row.get("depth_policy", "visual")
        else:
            paths = {view: str(resolve_image_path(row[key], data_root, path, from_data_root=True))
                     for view, key in (("rgb", "visible" if "visible" in row else "rgb"),
                                       ("infrared", "infrared"), ("depth", "depth")) if row.get(key)}
            if row.get("aux"):
                paths[aliases[row["aux_type"].lower()]] = str(resolve_image_path(
                    row["aux"], data_root, path, from_data_root=True))
            official_views = all(row.get(key) for key in ("visible", "infrared", "depth"))
            depth_policy = row.get("depth_policy", "millimeter" if official_views else "visual")
        views = ("rgb",) if modality == "rgb" else tuple(v for v in ("rgb", "infrared", "depth") if v in paths)
        if not views or views[0] != "rgb" or (modality == "trimodal" and len(views) != 3):
            raise ValueError(f"{sample_id}: requested {modality}, available views: {list(paths)}")
        rows.append({"id": sample_id, "query": query.strip(),
                     "image_paths": {view: paths[view] for view in views},
                     "modalities": list(views), "depth_policy": depth_policy})
    if not rows:
        raise ValueError("no queries")
    return rows


def prompt_for(row: dict, profile: str) -> str:
    if profile == "egm":
        if row["modalities"] != ["rgb"]:
            raise ValueError("EGM uses its trained RGB interface; choose --modality rgb")
        return EGM_PROMPT.replace("{query}", row["query"])
    return native_prompt(row["query"], tuple(row["modalities"]), row["depth_policy"])


def load_images(row: dict, upscale: float, max_pixels: int | None = None) -> list[Image.Image]:
    images = []
    for view, path in row["image_paths"].items():
        with Image.open(path) as source:
            array = np.asarray(source)
            if view == "depth" and array.ndim == 2 and array.dtype == np.uint16:
                if row["depth_policy"] not in {"millimeter", "sensor_linear_20000"}:
                    raise ValueError("raw uint16 depth needs an explicit sensor policy or prepared depth visualization")
                image = Image.fromarray(depth_mm_to_grayscale_rgb(array))
            else:
                image = source.convert("RGB").copy()
        if upscale != 1:
            image = image.resize((round(image.width * upscale), round(image.height * upscale)), Image.Resampling.LANCZOS)
            if upscale > 1 and max_pixels is not None and max_pixels < image.width * image.height * 0.95:
                raise ValueError("raise --max-pixels with --upscale; otherwise the processor undoes the enlargement")
        images.append(image)
    return images


def build_request(row: dict, tokenizer, profile: str, upscale: float, max_pixels: int | None = None) -> dict:
    images = load_images(row, upscale, max_pixels)
    prompt = prompt_for(row, profile)
    messages = [{"role": "user", "content": build_user_content(
        prompt, images, prompt_has_image_placeholders=profile == "qwen")}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=profile == "egm")
    return {"prompt": text, "multi_modal_data": {"image": images}}


def candidate_from_output(output, profile: str = "qwen") -> dict:
    # The last answer is parsed after generation. EGM reasoning can mention earlier boxes.
    answer = output.text.rsplit("</think>", 1)[-1]
    bbox = parse_generated_bbox(answer)
    if profile == "egm" and output.finish_reason == "length" and "</think>" not in output.text:
        bbox = None
    return {"sample_index": output.index, "bbox": bbox, "parsed": bbox is not None,
            "raw_text": output.text, "generated_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
            "cumulative_logprob": output.cumulative_logprob}


def load_cache(path: Path, rows: list[dict], profile: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    expected = {row["id"]: row for row in rows}
    cached = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sample_id = row["id"]
        if sample_id not in expected or sample_id in cached:
            raise ValueError(f"cache contains duplicate or unknown ID: {sample_id}")
        for key in ("query", "image_paths", "modalities", "depth_policy"):
            if row[key] != expected[sample_id][key]:
                raise ValueError(f"cache {key} changed: {sample_id}")
        if row["prompt"] != prompt_for(expected[sample_id], profile):
            raise ValueError(f"cache prompt changed: {sample_id}")
        cached[sample_id] = row
    return cached


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--profile", choices=("qwen", "egm"), required=True)
    parser.add_argument("--modality", choices=("rgb", "trimodal", "all"), required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-new-tokens", type=int, help="default: Qwen 128, EGM 4096")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=2073600)
    parser.add_argument("--upscale", type=float, default=1)
    parser.add_argument("--model-max-length", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--mamba-cache-mode", default="none")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true", help="write query-only inputs without loading a model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples < 1 or (args.num_samples > 1 and args.temperature <= 0):
        raise ValueError("multiple samples require positive temperature")
    if args.upscale <= 0 or args.chunk_size < 1 or args.limit < 0:
        raise ValueError("upscale/chunk-size must be positive; limit must be nonnegative")
    args.max_new_tokens = args.max_new_tokens or (4096 if args.profile == "egm" else 128)
    if args.profile == "egm" and args.modality != "rgb":
        raise ValueError("EGM requires --modality rgb")
    rows = load_queries(args.queries.resolve(), args.data_root.resolve(), args.modality)
    if args.limit:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in {"prepare_only", "output_dir"}}
    config["engine"] = "vllm"
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("run configuration changed; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows_path = args.output_dir / "candidates.jsonl"
    cached = load_cache(rows_path, rows, args.profile)
    pending = [(index, row) for index, row in enumerate(rows) if row["id"] not in cached]
    if args.prepare_only:
        path = args.output_dir / "inputs.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                for image_path in row["image_paths"].values():
                    if not Path(image_path).is_file():
                        raise FileNotFoundError(image_path)
                handle.write(json.dumps({**row, "prompt": prompt_for(row, args.profile)}, ensure_ascii=False) + "\n")
        print(json.dumps({"queries": len(rows), "inputs": str(path), "model_loaded": False}))
        return
    started = time.perf_counter()
    if pending:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                  max_model_len=args.model_max_length, max_num_seqs=args.max_num_seqs,
                  gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                  mamba_cache_mode=args.mamba_cache_mode, enable_lora=args.adapter is not None,
                  max_lora_rank=args.max_lora_rank,
                  limit_mm_per_prompt={"image": max(len(row["modalities"]) for row in rows)},
                  mm_processor_kwargs={"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})
        tokenizer = llm.get_tokenizer()
        adapter = LoRARequest("trained", 1, str(args.adapter.resolve())) if args.adapter else None
        with rows_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.chunk_size):
                batch = pending[start:start + args.chunk_size]
                requests = [build_request(row, tokenizer, args.profile, args.upscale, args.max_pixels) for _, row in batch]
                # Stable per-query seeds make resumed sampling independent of chunk boundaries.
                sampling = [SamplingParams(n=args.num_samples, temperature=args.temperature, top_p=args.top_p,
                                           seed=args.seed + index, max_tokens=args.max_new_tokens,
                                           skip_special_tokens=True) for index, _ in batch]
                outputs = llm.generate(requests, sampling, lora_request=adapter, use_tqdm=True)
                for (_, row), output in zip(batch, outputs, strict=True):
                    result = {**row, "prompt": prompt_for(row, args.profile),
                              "candidates": [candidate_from_output(item, args.profile) for item in output.outputs]}
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    cached[row["id"]] = result
                handle.flush()
                print(json.dumps({"completed": len(cached), "total": len(rows),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    candidates = [candidate for row in cached.values() for candidate in row["candidates"]]
    summary = {"queries": len(rows), "candidates": len(candidates),
               "valid_candidates": sum(item["parsed"] for item in candidates),
               "queries_without_valid_candidate": sum(not any(c["parsed"] for c in row["candidates"]) for row in cached.values()),
               "length_stopped_candidates": sum(item["finish_reason"] == "length" for item in candidates),
               "generated_tokens": sum(item["generated_tokens"] for item in candidates),
               "this_invocation_seconds": round(time.perf_counter() - started, 1)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
