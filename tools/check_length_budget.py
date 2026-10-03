"""核实混合集里最长样本的 token 数是否会被 max_length 截断。

为什么必须查：若某个样本超过 `max_length`，ms-swift 会**截断**它。截断可能切掉
assistant 的答案，于是这条样本变成"只有输入没有监督"——**不报错、不计数**，
白白浪费一个 batch 位置，还会让 loss 曲线看起来正常。

判据：把每条样本按训练时的真实处理（图像按 max_pixels 缩放 + chat template）算 token 数，
报告最大值与超限比例。City 图像是 1920x1080，在 max_pixels=2073600 下每图约 1980 token，
所以"三张 City 图 + 长查询"是最坏情况。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", type=pathlib.Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-pixels", type=int, default=2073600)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--sample", type=int, default=400,
                        help="抽样条数；全量算太慢，且只需要看分布尾部")
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    rows = [json.loads(line) for line in
            args.mixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    # 按"查询最长 + 图像最多"排序，优先检查最坏情况
    rows.sort(key=lambda r: -len(r["messages"][0]["content"]))
    candidates = rows[:args.sample // 2] + rows[::max(1, len(rows) // (args.sample // 2))]
    candidates = candidates[:args.sample]

    counts: list[tuple[int, str, int]] = []
    for row in candidates:
        images = []
        for path in row["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        # 训练时的三模态输入始终是 3 个槽位；缺失的模态用全黑占位
        while len(images) < 3:
            size = images[0].size if images else (640, 512)
            images.append(Image.new("RGB", size, (0, 0, 0)))
        messages = [{"role": "user",
                     "content": [{"type": "image"} for _ in images]
                     + [{"type": "text", "text": row["messages"][0]["content"]
                         .replace("<image>\n", "").replace("<image>", "")}]},
                    {"role": "assistant", "content": row["messages"][1]["content"]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)
        inputs = processor(text=[text], images=images, return_tensors="pt",
                           min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        total = int(inputs["input_ids"].shape[1])
        counts.append((total, row.get("capability", "?"), len(row["messages"][0]["content"])))

    counts.sort(reverse=True)
    over = [c for c in counts if c[0] > args.max_length]
    print(f"抽查 {len(counts)} 条（含查询最长的样本）")
    print(f"  token 数 最大 {counts[0][0]}  中位 {counts[len(counts)//2][0]}  "
          f"最小 {counts[-1][0]}")
    print(f"  超过 max_length({args.max_length}) 的: {len(over)} 条 "
          f"({len(over)/len(counts)*100:.1f}%)")
    if over:
        print("\n  超限样本（这些会被截断，可能切掉答案）:")
        for total, capability, chars in over[:10]:
            print(f"    {total:6d} token  capability={capability}  提示词 {chars} 字符")
        print("\n  结论：需要提高 max_length，或限制查询长度/图像分辨率")
        raise SystemExit(1)
    print("\n  结论：全部在预算内，不会被截断")


if __name__ == "__main__":
    main()
