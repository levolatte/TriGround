"""解析 ms-swift 训练日志，给出 B3（训练前预检）与 B4（正式训练）的客观判读。

存在意义：训练日志长达数千行，`s/step`、峰值显存、可训练参数量、监督 token 数散落在不同位置，
靠肉眼读容易漏掉"其实没在训 LoRA"或"已经接近显存上限"这类致命问题。
本脚本把这些数字抽出来，并直接给出通过/不通过，避免主观判断。

用法：
    python scripts/check_training.py <log> [--steps-budget 600] [--vram-limit-gib 83.6]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ms-swift 的 trainable params 行形如：
#   trainable params: 30,670,848 || all params: 8,800,000,000 || trainable%: 0.3485
TRAINABLE_RE = re.compile(
    r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)\s*\|\|\s*trainable%:\s*([\d.]+)"
)
# tqdm 的进度行里带速率，例如： 12%|█▏ | 72/600 [05:21<39:16, 4.47s/it]
PROGRESS_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)(s/it|it/s)\]")
VRAM_RE = re.compile(r"(?:max_memory_allocated|peak(?:_memory)?_allocated)[^\d]*([\d.]+)\s*(GiB|GB|MiB|MB)")
OOM_MARKERS = ("torch.OutOfMemoryError", "CUDA out of memory", "OutOfMemoryError")
LOSS_RE = re.compile(r"'loss':\s*([\d.]+)")


def to_seconds(stamp: str) -> float:
    parts = [int(part) for part in stamp.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--steps-budget", type=int, default=600,
                        help="正式训练的总步数，用于外推总时长")
    parser.add_argument("--vram-limit-gib", type=float, default=83.6,
                        help="显卡可用显存上限（本机 RTX 6000D 为 83.6 GiB）")
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    problems: list[str] = []

    print(f"日志: {args.log}  ({len(text.splitlines())} 行)")

    # 1) 是否在训 LoRA：可训练参数必须 > 0 且远小于全量
    trainable = TRAINABLE_RE.findall(text)
    if not trainable:
        problems.append("日志里找不到 trainable params 行，无法确认 LoRA 是否生效")
    else:
        count, total, percent = trainable[-1]
        count_value = int(count.replace(",", ""))
        total_value = int(total.replace(",", ""))
        print(f"可训练参数     : {count} / {total}  ({percent}%)")
        if count_value == 0:
            problems.append("可训练参数为 0 —— LoRA 没有挂上，等于没训练")
        if count_value >= total_value:
            problems.append("可训练参数等于全部参数 —— 退化成全量微调，显存与预期完全不符")

    # 2) 显存
    vram = VRAM_RE.findall(text)
    if vram:
        value, unit = vram[-1]
        gib = float(value) / 1024 if unit in ("MiB", "MB") else float(value)
        if unit in ("GB",):
            gib = float(value) / 1.073741824
        print(f"峰值显存       : {gib:.1f} GiB (上限 {args.vram_limit_gib} GiB)")
        if gib > args.vram_limit_gib * 0.95:
            problems.append(f"峰值显存 {gib:.1f} GiB 已用掉上限的 95% 以上，正式训练很可能 OOM")
    else:
        print("峰值显存       : 日志未记录（ms-swift 不一定打印，不影响通过）")

    # 3) 速度与总时长外推
    progress = PROGRESS_RE.findall(text)
    if progress:
        done, total, elapsed, remaining, rate, unit = progress[-1]
        per_step = float(rate) if unit == "s/it" else 1.0 / float(rate)
        print(f"进度           : {done}/{total}  已用 {elapsed}  剩余 {remaining}")
        print(f"每步耗时       : {per_step:.2f} 秒")
        if args.steps_budget:
            hours = per_step * args.steps_budget / 3600
            print(f"按此速度，{args.steps_budget} 个优化步预计 {hours:.1f} 小时"
                  f"（每步会消费 gradient_accumulation_steps 个样本）")
    else:
        print("进度           : 日志里还没有 tqdm 进度行（可能刚开始或已结束）")

    losses = LOSS_RE.findall(text)
    if losses:
        tail = [float(value) for value in losses[-5:]]
        print(f"最近 loss      : {['%.4f' % v for v in tail]}")

    for marker in OOM_MARKERS:
        if marker in text:
            problems.append(f"日志里出现显存溢出标记：{marker}")

    print()
    if problems:
        print("预检不通过：")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("TRAINING_CHECK_OK")


if __name__ == "__main__":
    main()
