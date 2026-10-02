"""闸门 1：确认 vLLM 环境与我们 HF 环境对同一批查询编码出**完全相同的 token 序列**。

为什么必须先做且不能省：
  * vLLM 装在独立 venv 里，自带一套 transformers 依赖，版本与我们训练/验证用的 5.16.1
    不同；chat template 与图像处理器的行为都可能随之变化；
  * 提示词一旦漂移，模型看到的输入就变了，输出会静默变差——本地不报任何错，
    只表现为"分数莫名其妙掉了"。

做法：让**两个解释器各自加载 processor**，对同一批查询导出 `input_ids` 与
`image_grid_thw`，再逐条比对。这直接检验"vLLM 会用到的那套 processor"是否与 HF 侧一致，
比比对渲染文本更接近真正会出问题的地方。

用法（云端）：
    # 主环境
    /root/miniconda3/bin/python tools/verify_vllm_token_parity.py --export \
        --model <model> --queries <queries.json> --data-root <root> \
        --out /root/parity_hf.json --limit 24
    # vLLM 环境（同一份脚本、不同的解释器）
    /root/vllm_env/bin/python tools/verify_vllm_token_parity.py --export \
        --model <model> --queries <queries.json> --data-root <root> \
        --out /root/parity_vllm.json --limit 24
    # 比对（任一环境均可）
    /root/miniconda3/bin/python tools/verify_vllm_token_parity.py \
        --compare /root/parity_hf.json /root/parity_vllm.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_pretrained_grounder import build_user_content  # noqa: E402
from tools.predict_native_submission import (  # noqa: E402
    PROMPTS,
    image_paths,
    load_modal_images,
    load_queries,
)

MIN_PIXELS = 200704
MAX_PIXELS = 2073600


def selected_items(queries: Path, limit: int) -> list[tuple[str, dict[str, Any]]]:
    source = load_queries(queries, None)
    items = list(source.items())
    # 均匀取样而不是取前 N 条：官方集按图组排列，取前 N 条会集中在少数场景，
    # 覆盖不到不同的查询文本长度与图像尺寸组合。
    step = max(1, len(items) // limit) if limit else 1
    return items[::step][:limit]


def export(model: str, queries: Path, data_root: Path, limit: int, output: Path) -> None:
    import transformers
    from transformers import AutoProcessor

    version = transformers.__version__
    processor = AutoProcessor.from_pretrained(model, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
                                              local_files_only=True)
    records = []
    for sample_id, record in selected_items(queries, limit):
        images = load_modal_images(image_paths(record, data_root, queries), "trimodal")
        prompt = PROMPTS["trimodal"](record["query"])
        messages = [{"role": "user", "content": build_user_content(
            prompt, images, prompt_has_image_placeholders=True)}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False)
        records.append({
            "id": sample_id,
            "query": record["query"],
            "input_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
        })
    output.write_text(json.dumps({"transformers": version, "records": records}),
                      encoding="utf-8")
    print(f"导出 {len(records)} 条（transformers {version}）-> {output}")


def compare(hf_path: Path, other_path: Path) -> int:
    hf = json.loads(hf_path.read_text(encoding="utf-8"))
    other = json.loads(other_path.read_text(encoding="utf-8"))
    print(f"transformers: HF 侧 {hf['transformers']}  vs  对侧 {other['transformers']}")
    by_id = {item["id"]: item for item in other["records"]}
    shared = [item for item in hf["records"] if item["id"] in by_id]
    print(f"可比较条数: {len(shared)}")

    token_mismatch, grid_mismatch = [], []
    for item in shared:
        peer = by_id[item["id"]]
        if item["image_grid_thw"] != peer["image_grid_thw"]:
            grid_mismatch.append(item["id"])
        if item["input_ids"] != peer["input_ids"]:
            token_mismatch.append(item["id"])

    print(f"image_grid_thw 不一致: {len(grid_mismatch)}")
    for key in grid_mismatch[:3]:
        print(f"  {key}")
    print(f"input_ids 不一致: {len(token_mismatch)}")
    for key in token_mismatch[:3]:
        item = next(entry for entry in shared if entry["id"] == key)
        peer = by_id[key]
        left, right = item["input_ids"], peer["input_ids"]
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        print(f"  {key}: 长度 {len(left)} vs {len(right)}，首个差异位置 {first}")

    if token_mismatch or grid_mismatch:
        print("\nPARITY_FAILED —— 不允许用 vLLM 跑正式提交")
        return 1
    print("\nPARITY_OK —— 两侧 token 完全一致")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))
    if not args.export:
        raise SystemExit("需要 --export 或 --compare")
    export(args.model, args.queries, args.data_root, args.limit, args.out)


if __name__ == "__main__":
    main()
