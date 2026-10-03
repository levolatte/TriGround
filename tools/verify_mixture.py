"""混合集出厂验收：训练启动前必须全过的闸门。

## 为什么要有这个工具

上一轮（Arm 2）踩了两个只能靠事后审计才发现的坑：

1. **55.6% 的样本深度图是未渲染的原始 `I;16`**，直转 RGB 后 64~80% 像素饱和为白，
   深度模态等于噪声。这个错误**不会让训练报错**——loss 照降、分数照出，
   只是深度白喂了。
2. **训练集混入了官方测试集帧**（City 114 帧、RGBDT500 161 帧逐字节相同）。

两个坑的共同点是：**错了也不报错**。因此必须在训练前用独立脚本正面验证，
而不是相信构建器的自述。

## 闸门

* `depth`：每一张深度图必须是 3 通道 8 位，且白色像素占比 < 5%
* `contamination`：混合集引用的全部图像与官方测试集内容哈希零交集
* `placeholder`：`<image>` 占位符数 == 图像数 == 模态数
* `existence`：每条引用的图像都真实存在
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np
from PIL import Image

# 深度图白色像素占比上限，**按深度策略分别设**。阈值按实测分布标定：
#
#   * `millimeter`（City / RGBDT500）：值域 0~20000 映射到 0~255。
#     正确渲染实测均值 **0.01%**、最大 **6.06%**；误喂原始 uint16 则 **64~80%**。
#     上限取 10%——既容得下"原始深度里有少量 ≈1 的像素被映成最亮"这种真实情况，
#     又与误喂原始图的 64% 留出 6 倍余量。
#   * `sensor_linear_20000`（RoboRefIt）：P01-P99 百分位拉伸，
#     **饱和是设计的一部分**。实测均值 0.91%、最大 36.59%，故上限取 45%。
#   * `visual`：同前者（当前构建器不用这个值，保留条目以免将来静默套用毫米阈值）。
WHITE_SHARE_LIMIT = {
    "millimeter": 0.10,
    "sensor_linear_20000": 0.45,
    "visual": 0.45,
}
DEFAULT_WHITE_SHARE_LIMIT = 0.10
# 退化判据：**常数图**才是要拦的——那种图深度模态没有任何信息。
# 阈值按实测标定：RoboRefIt 有一批传感器量化很粗的帧（唯一值 16~19、标准差 4.7~17.7），
# 它们是**色阶少但有真实梯度**，不是退化；真退化是标准差 ≈0、唯一值 1~2。
# 故取 MIN_STD=3 / MIN_UNIQUE=8：既能放过量化帧，又能拦住常数图。
#
# 注意：**只靠梯度拦不住"误喂原始 uint16"**——那种图直转 RGB 后白色占比 80%
# 但标准差仍很高（截断后还有对比度）。所以白色占比是主判据，梯度是补充。
MIN_STD = 3.0
MIN_UNIQUE = 8


def md5_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_mixture(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def gate_existence(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        for image in row["images"]:
            if not pathlib.Path(image).is_file():
                problems.append(f"{row['id']}: 图像不存在 {image}")
    return problems


def gate_placeholder(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        prompt = row["messages"][0]["content"]
        count = prompt.count("<image>")
        if count != len(row["images"]):
            problems.append(f"{row['id']}: 占位符 {count} 个，图像 {len(row['images'])} 张")
        if len(row["modalities"]) != len(row["images"]):
            problems.append(f"{row['id']}: 模态 {len(row['modalities'])} 个，"
                            f"图像 {len(row['images'])} 张")
    return problems


def gate_depth(rows: list[dict], sample: int | None) -> list[str]:
    """按来源的深度策略分别判定：白色占比（主）+ 梯度（区分常数图）。

    这是唯一能同时拦住两种失效的判据组合：
      * **误喂原始 uint16** → 白色占比 64~80%（梯度仍高，只靠梯度拦不住）
      * **常数图** → 梯度接近 0（白色占比可能正常，只靠白色占比拦不住）
    """
    problems: list[str] = []
    # 采集每条样本的深度图，并记下它所属的策略
    path_to_policy: dict[str, str] = {}
    for row in rows:
        if "depth" not in row["modalities"]:
            continue
        index = row["modalities"].index("depth")
        path_to_policy.setdefault(row["images"][index],
                                  row.get("depth_policy", "millimeter"))
    paths = sorted(path_to_policy)
    if sample is not None and len(paths) > sample:
        step = len(paths) / sample
        paths = [paths[int(i * step)] for i in range(sample)]

    by_policy: dict[str, list[float]] = collections.defaultdict(list)
    for path_str in paths:
        policy = path_to_policy[path_str]
        limit = WHITE_SHARE_LIMIT.get(policy, DEFAULT_WHITE_SHARE_LIMIT)
        path = pathlib.Path(path_str)
        with Image.open(path) as image:
            array = np.asarray(image)
            mode = image.mode
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            problems.append(f"深度图位深/通道不对: {path} mode={mode} "
                            f"shape={array.shape} dtype={array.dtype}")
            continue
        gray = array[:, :, 0]
        share = float((array >= 250).mean())
        by_policy[policy].append(share)
        if share > limit:
            problems.append(f"[{policy}] 深度图白色占比 {share:.1%} 超上限 {limit:.0%}: {path}")
        if float(gray.std()) < MIN_STD or len(np.unique(gray)) < MIN_UNIQUE:
            problems.append(f"[{policy}] 深度图退化（标准差 {gray.std():.1f} / "
                            f"唯一值 {len(np.unique(gray))}）: {path}")
    for policy, shares in sorted(by_policy.items()):
        print(f"  [{policy}] 检查 {len(shares)} 张：白色占比 均值 {np.mean(shares):.2%} / "
              f"最大 {np.max(shares):.2%}（上限 {WHITE_SHARE_LIMIT.get(policy, DEFAULT_WHITE_SHARE_LIMIT):.0%}）")
    return problems


def gate_contamination(rows: list[dict], test_root: pathlib.Path) -> list[str]:
    test_hashes = {md5_file(p) for p in test_root.rglob("*.png")}
    print(f"  测试集哈希 {len(test_hashes)} 个；逐图比对混合集引用…")
    problems = []
    checked = 0
    for path_str in sorted({image for row in rows for image in row["images"]}):
        path = pathlib.Path(path_str)
        if not path.is_file():
            continue
        checked += 1
        if md5_file(path) in test_hashes:
            problems.append(f"与测试集逐字节相同: {path}")
    print(f"  比对 {checked} 张唯一图像")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mixture", type=pathlib.Path, required=True)
    parser.add_argument("--test-root", type=pathlib.Path, required=True)
    parser.add_argument("--depth-sample", type=int, default=None,
                        help="深度图抽查张数，默认全查")
    parser.add_argument("--skip-contamination", action="store_true",
                        help="跳过污染比对（该步最慢，仅在已单独验证过时使用）")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = load_mixture(args.mixture)
    print(f"混合集 {args.mixture}: {len(rows)} 条")

    failures: dict[str, list[str]] = {}
    print("\n[1/4] 图像存在性")
    failures["existence"] = gate_existence(rows)
    print(f"  {'通过' if not failures['existence'] else str(len(failures['existence'])) + ' 个问题'}")

    print("\n[2/4] 占位符与模态一致性")
    failures["placeholder"] = gate_placeholder(rows)
    print(f"  {'通过' if not failures['placeholder'] else str(len(failures['placeholder'])) + ' 个问题'}")

    print("\n[3/4] 深度图健康（必须是渲染后的 8 位 RGB）")
    failures["depth"] = gate_depth(rows, args.depth_sample)
    print(f"  {'通过' if not failures['depth'] else str(len(failures['depth'])) + ' 个问题'}")

    if args.skip_contamination:
        print("\n[4/4] 污染比对 —— 已跳过")
    else:
        print("\n[4/4] 与官方测试集的污染比对")
        failures["contamination"] = gate_contamination(rows, args.test_root)
        print(f"  {'通过' if not failures['contamination'] else str(len(failures['contamination'])) + ' 个问题'}")

    total = sum(len(v) for v in failures.values())
    print("\n" + "=" * 60)
    if total:
        print(f"验收失败：{total} 个问题")
        for gate, problems in failures.items():
            for problem in problems[:5]:
                print(f"  [{gate}] {problem}")
            if len(problems) > 5:
                print(f"  [{gate}] …另有 {len(problems) - 5} 个")
        raise SystemExit(1)
    print("MIXTURE_VERIFY_OK")


if __name__ == "__main__":
    main()
