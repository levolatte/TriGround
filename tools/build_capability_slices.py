"""按**失败模式**构建能力切片，并预混合成单一训练清单。

## 为什么按失败模式而不是按数据集

City412 上的失分分解（B2 零样本 27B 原生，115 个未命中）显示：

    尺寸桶          题数   未命中   指错目标   定位不精
    极小 <0.2%      143     75       45         26
    小 0.2-1%       147     32       10         22
    中 1-5%          80      8        0          8
    大 >5%           42      0        0          0

即 **65% 的失分集中在最小的那一桶**，且失败原因分成两类：
**55 题是"指错目标"（判别）**、**56 题是"框不准"（精度）**。
按数据集堆数据解决不了这个问题，必须按失败模式配比。

## 为什么预混合成单文件，而不用 `--interleave_prob`

ms-swift 的 `stopping_strategy` **默认是 `first_exhausted`**——最短的数据集耗尽就停止。
若直接把 181 条 RGBDT500 混进去而不改这个默认值，训练会在最短集耗尽后**静默停止**，
不报任何错。预混合成单文件既避开这个陷阱，又让配比可审计、可复现。

## 已核查的事实（不是假设）

* RGBT 的 `尺寸档 == 'small'` 是极小目标的**超集**：三个源里 `normal` 标签中
  面积 <0.2% 的占比**均为 0.0%**。所以先用标签粗筛、再按真实 bbox 面积精筛。
* RGBT 的 bbox 是 `[x, y, w, h]` 绝对像素，**不是 xyxy**（写成 xyxy 不报错但框全错）。
* RGBT 的 `converted/*.jsonl` **不带**条件标签，标签只在 raw `.pth` 里，因此必须回读 raw。

## 用法

    python tools/build_capability_slices.py \
        --rgbt-root <RGBT-GroundBench> --roborefit-root <RoboRefIt> \
        --rgbdt-sft <native_sft.jsonl> --rgbdt-assets <解包目录> \
        --city-manifest <qwen_generation_train_100.json> --city-image-root <city/train> \
        --out-dir /root/sprint_qwen36/slices
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import pickle
import random
import sys
import zipfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.prepare_qwen3vl_native_sft import native_prompt  # noqa: E402

NORMALIZE = 1000.0
DEPTH_POLICY_MM = "millimeter"
DEPTH_POLICY_SENSOR = "sensor_linear_20000"
# 极小目标阈值（归一化面积）。与 City412 分析口径一致。
TINY_AREA = 0.002
# 官方测试集有 7.5% 的查询超过 100 字符，而 City 训练集一条都没有
LONG_QUERY_CHARS = 100
# 官方查询的长度上限。实测官方分布：min 5 / 中位 49 / p75 73 / **max 298**。
# 超过这个长度的样本基本是 RGBT-MFAD 的冗长生成式描述（实测最长 1766 字符，
# 形如「...likely a compact car given its size compared to other vehicles visible
# further ahead along...」），与官方的简洁指代风格相差很远。放进训练等于教模型
# 输出啰嗦描述——**在 ACC@0.5 这种要求精确框的任务上是有害的**。
MAX_QUERY_CHARS = 300
SOURCES = ("flir", "m3fd", "mfad")
CLASS_NAMES = {
    "flir": {1: "truck", 2: "car", 3: "person", 4: "bicycle"},
    "m3fd": {1: "people", 2: "car", 3: "bus", 4: "motorcycle", 5: "lamp", 6: "truck"},
    "mfad": {2: "car", 3: "truck"},
}
DEGENERATE = ("No ", "is present", "There is no")


class SliceError(RuntimeError):
    """切片构建中的硬错误。宁可失败，也不要静默产出坏样本。"""


def read_pth(path: pathlib.Path) -> list:
    """读 torch ZIP 归档里的标注（标准库即可，不需要 torch）。"""
    with zipfile.ZipFile(path) as archive:
        member = next((n for n in archive.namelist() if n.endswith("data.pkl")), None)
        if member is None:
            raise SliceError(f"{path}: 没有 data.pkl")
        return pickle.loads(archive.read(member))


def to_answer(box_xyxy_norm: tuple[float, float, float, float]) -> str:
    """归一化 xyxy -> 训练用的答案 JSON。"""
    x1, y1, x2, y2 = box_xyxy_norm
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise SliceError(f"bbox 越界或退化: {box_xyxy_norm}")
    quantized = [round(v * NORMALIZE) for v in (x1, y1, x2, y2)]
    return json.dumps({"bbox_2d": quantized}, separators=(",", ":"))


def make_record(sample_id: str, modalities: tuple[str, ...], images: list[str], query: str,
                box: tuple, *, depth_policy: str, capability: str,
                source: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """组装一条 swift SFT 样本，并带上用于事后审计的元数据。

    **模板必须与真实模态数一致**：`native_prompt` 对 2 视图走 generic 分支
    （`<image>\\n<image>` + "These are views..."），对三模态走 `_trimodal_prompt`
    （三个占位符 + "These are **aligned** views..."）。
    这里曾用"三模态模板 + 缺失说明"来表示只有 2 个视图的样本，结果提示词里有 3 个
    `<image>` 却只给 2 张图——处理器会错位。收到队内 `structure_noncity` 数据后确认：
    **既有约定就是"几个视图用几个占位符"**，因此按实际模态构造。
    """
    for path in images:
        if not pathlib.Path(path).is_file():
            raise SliceError(f"{sample_id}: 图像不存在 {path}")
    if len(modalities) != len(images):
        raise SliceError(f"{sample_id}: 模态 {modalities} 与图像数 {len(images)} 不一致")
    prompt = native_prompt(query, modalities, depth_policy)
    if prompt.count("<image>") != len(images):
        raise SliceError(f"{sample_id}: 占位符 {prompt.count('<image>')} 个，"
                         f"但图像 {len(images)} 张")
    record = {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": to_answer(box)},
        ],
        "images": images,
        "capability": capability,
        "source": source,
        "modalities": list(modalities),
        "missing_modalities": [m for m in ("infrared", "depth") if m not in modalities],
    }
    if extra:
        record.update(extra)
    return record


def assign_primary_capability(area: float, query_chars: int, occlusion: int) -> str | None:
    """按优先级给一条样本分配**唯一**的能力归属。

    为什么必须唯一：一条 RGBT 样本可能同时是小目标、被遮挡、又是长查询。若三个切片
    各自独立抽样，它会在同一次训练里出现最多三次，把预定的配比悄悄放大成 3 倍。
    这里按优先级只归一类，保证池与池不相交、配比可复现。

    优先级：极小目标 > 判别（遮挡） > 长查询。理由是极小目标是我们**最主要的失分桶**
    （占全部失分的 65%），它的样本供应也最紧张，优先保住。
    """
    if area < TINY_AREA:
        return "small"
    if occlusion > 0:
        return "discriminate"
    if query_chars > LONG_QUERY_CHARS:
        return "longquery"
    return None


def build_rgbt(root: pathlib.Path, rng: random.Random,
               sources: tuple[str, ...] = SOURCES) -> dict[str, list[dict[str, Any]]]:
    """RGBT-GroundBench：极小目标 / 遮挡判别 / 长查询。两模态（RGB+IR），深度置空。

    `sources` 可收窄，便于测试；生产上默认三个源都要，缺任何一个都硬失败——
    否则会以为数据用上了，实际少了一个源。
    """
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    image_root = root / "raw" / "image_data"
    for source in sources:
        pth = root / "raw" / f"rgbtvg_{source}" / f"rgbtvg_{source}_train.pth"
        if not pth.is_file():
            raise SliceError(f"缺少 RGBT 训练标注: {pth}")
        rows = read_pth(pth)
        mapping = CLASS_NAMES[source]
        for index, row in enumerate(rows):
            filename, size, box, query = row[0], row[1], row[2], str(row[3])
            class_id, size_class, occlusion = row[4], str(row[5]), int(row[6])
            if any(marker in query for marker in DEGENERATE):
                continue
            # 过长的生成式描述与官方简洁风格相差太远，剔除而不是训练进去
            if len(query) > MAX_QUERY_CHARS:
                continue
            if class_id not in mapping:
                continue
            rgb = image_root / source / "rgb" / filename
            ir = image_root / source / "ir" / filename
            # 缺图必须硬失败：静默跳过会让人以为这批数据都用上了
            if not rgb.is_file() or not ir.is_file():
                raise SliceError(f"RGBT 图像缺失: {rgb} / {ir}")
            width, height = float(size["width"]), float(size["height"])
            if width <= 0 or height <= 0:
                continue
            x, y, w, h = (float(v) for v in box)
            # ⚠️ 源标注是 xywh 绝对像素
            xyxy = (x / width, y / height, (x + w) / width, (y + h) / height)
            if not (0.0 <= xyxy[0] < xyxy[2] <= 1.0 and 0.0 <= xyxy[1] < xyxy[3] <= 1.0):
                continue
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            capability = assign_primary_capability(area, len(query), occlusion)
            if capability is None:
                continue
            sample_id = f"rgbt:{source}:{pathlib.Path(filename).stem}:{index}"
            out[capability].append(make_record(
                sample_id, ("rgb", "infrared"), [str(rgb), str(ir)], query, xyxy,
                depth_policy=DEPTH_POLICY_MM,
                capability=capability, source=f"rgbt_{source}",
                extra={"area": round(area, 6), "query_chars": len(query),
                       "size_class": size_class, "occlusion": occlusion}))
    rng.shuffle(out["small"])
    return out


def build_roborefit(root: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    """RoboRefIt：真实 RGB-D、66 类、同类多实例歧义。两模态（RGB+Depth），红外置空。"""
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    manifest = root / "subset" / "converted" / "train.jsonl"
    if not manifest.is_file():
        raise SliceError(f"缺少 RoboRefIt 转换产物: {manifest}")
    base = manifest.parent
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rgb = (base / row["rgb"]).resolve()
        depth = (base / row["depth"]).resolve()
        if not rgb.is_file() or not depth.is_file():
            raise SliceError(f"{row['id']}: RoboRefIt 图像缺失 {rgb} / {depth}")
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise SliceError(f"{row['id']}: bbox 越界 {row['bbox']}")
        # RoboRefIt 的 depth 是可视化图（P01-P99 拉伸），物理单位未知
        record = make_record(
            f"robo:{row['id']}", ("rgb", "depth"), [str(rgb), str(depth)],
            row["query"], (x1, y1, x2, y2),
            depth_policy=DEPTH_POLICY_SENSOR,
            capability="discriminate", source="roborefit",
            extra={"area": round((x2 - x1) * (y2 - y1), 6),
                   "query_chars": len(row["query"]), "class_name": row.get("class_name")})
        out["discriminate"].append(record)
    return out


def build_rgbdt(sft: pathlib.Path, assets: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    """RGBDT500 人工验收子集：唯一真三模态来源。"""
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    if not sft.is_file():
        raise SliceError(f"缺少 RGBDT500 SFT 清单: {sft}")
    for line in sft.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        images = [(assets / p).resolve() for p in row["image"]]
        for path in images:
            if not path.is_file():
                raise SliceError(f"{row['id']}: RGBDT500 图像缺失 {path}")
        human = row["conversations"][0]["value"]
        answer = row["conversations"][1]["value"]
        # RGBDT500 自带渲染好的 prompt（含 sensor_linear_20000 的深度说明），
        # 直接复用，保证与其人工验收时的输入一致
        if human.count("<image>") != len(images):
            raise SliceError(f"{row['id']}: 占位符 {human.count('<image>')} 个，"
                             f"图像 {len(images)} 张")
        record = {
            "id": f"rgbdt:{row['id']}",
            "messages": [{"role": "user", "content": human},
                         {"role": "assistant", "content": answer.strip()}],
            "images": [str(p) for p in images],
            "capability": "trimodal",
            "source": "rgbdt500",
            "modalities": ["rgb", "infrared", "depth"],
            "missing_modalities": [],
        }
        out["trimodal"].append(record)
    return out


def build_city(manifest: pathlib.Path, image_root: pathlib.Path, scene_cap: int,
               rng: random.Random) -> list[dict[str, Any]]:
    """City 锚点：按每场景上限抽样，保持目标域主导。

    计划里的取舍：把 `scene_cap` 从 6 提到 `scene_cap`，而不是取消上限——
    取消会改变训练分布（少数场景权重上升），提高上限则既多用数据又保留防过采样。

    **路径解析**：清单里的图像路径形如 `../visible/x.png`，是相对**清单自身所在目录**
    （`target_v2/`）的。若拿 `image_root / "../visible/x.png"` 去拼会得到
    `data/city/visible/...`，那是错的——所以先按清单目录解析，解析不到再退回 image_root。
    """
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = list(records.values())
    base = manifest.parent
    by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        scene = pathlib.Path(str(row["visible"])).stem.rsplit("_", 1)[0]
        by_scene[scene].append(row)

    def resolve(value: str) -> pathlib.Path:
        candidate = (base / value).resolve()
        return candidate if candidate.is_file() else (image_root / value).resolve()

    out: list[dict[str, Any]] = []
    for scene, rows in by_scene.items():
        rng.shuffle(rows)
        for row in rows[:scene_cap]:
            rgb, ir, depth = resolve(row["visible"]), resolve(row["infrared"]), resolve(row["depth"])
            if not rgb.is_file() or not ir.is_file() or not depth.is_file():
                raise SliceError(
                    f"{row['query'][:40]}: City 图像缺失 "
                    f"(rgb={rgb} ir={ir} depth={depth})")
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            out.append(make_record(
                f"city:{scene}:{len(out)}", ("rgb", "infrared", "depth"),
                [str(rgb), str(ir), str(depth)], row["query"], (x1, y1, x2, y2),
                depth_policy=DEPTH_POLICY_MM,
                capability="city_anchor", source="city",
                extra={"area": round((x2 - x1) * (y2 - y1), 6),
                       "query_chars": len(row["query"]),
                       "class_name": row.get("class_name")}))
    rng.shuffle(out)
    return out


def mix(slices: dict[str, list[dict[str, Any]]], quotas: dict[str, int],
        seed: int) -> list[dict[str, Any]]:
    """按配额预混合。配额是"使用多少条"，超出则无放回抽样，不足则全用并告警。"""
    rng = random.Random(seed)
    mixed: list[dict[str, Any]] = []
    for name, quota in quotas.items():
        pool = list(slices.get(name, []))
        if not pool:
            raise SliceError(f"配额要求 {name}={quota}，但该切片为空")
        rng.shuffle(pool)
        if len(pool) < quota:
            print(f"  ⚠️ {name} 只有 {len(pool)} 条，少于配额 {quota}，按实际数量使用")
            quota = len(pool)
        mixed.extend(pool[:quota])
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbt-root", type=pathlib.Path, required=True)
    parser.add_argument("--roborefit-root", type=pathlib.Path, required=True)
    parser.add_argument("--rgbdt-sft", type=pathlib.Path, required=True)
    parser.add_argument("--rgbdt-assets", type=pathlib.Path, required=True)
    parser.add_argument("--city-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--city-image-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--scene-cap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    # 配额。City 占 50%，相对 Arm 1 的 100% 已大幅降低——这正是 Arm 2 相对 Arm 1 的意义：
    # 在 City 质量存疑的前提下，把一半的训练信号让给真实采集的能力数据。
    #
    # 配额是算出来的，不是拍的：City 清单在 scene_cap=10 下最多只能给出 3464 条，
    # 因此总混合量上限 = 3464 / 0.5 = 6928（City 占 50%，同时保证三模态过半，
    # 降低"深度常黑 -> 忽略深度"的模态捷径风险）。
    # 扣掉 city 与 trimodal 后，两模态能力切片总预算 = 6928 - 3464 - 181 = 3283，
    # 按 small:discriminate:longquery ≈ 1500:1200:583 分配。
    parser.add_argument("--quota-city", type=int, default=3464)
    parser.add_argument("--quota-small", type=int, default=1500)
    parser.add_argument("--quota-discriminate", type=int, default=1200)
    parser.add_argument("--quota-trimodal", type=int, default=181)
    parser.add_argument("--quota-longquery", type=int, default=583)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print("构建各切片…")
    slices: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for partial in (build_rgbt(args.rgbt_root, rng),
                    build_roborefit(args.roborefit_root),
                    build_rgbdt(args.rgbdt_sft, args.rgbdt_assets)):
        for key, value in partial.items():
            slices[key].extend(value)
    slices["city_anchor"] = build_city(args.city_manifest, args.city_image_root,
                                       args.scene_cap, rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in slices.items():
        path = args.out_dir / f"slice_{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  slice_{name}.jsonl  {len(rows)} 条")

    quotas = {"city_anchor": args.quota_city, "small": args.quota_small,
              "discriminate": args.quota_discriminate, "trimodal": args.quota_trimodal,
              "longquery": args.quota_longquery}
    print("\n预混合（配额）:", quotas)
    mixed = mix(slices, quotas, args.seed)
    mixture_path = args.out_dir / "mixture.jsonl"
    with mixture_path.open("w", encoding="utf-8") as handle:
        for row in mixed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_capability = collections.Counter(row["capability"] for row in mixed)
    by_source = collections.Counter(row["source"] for row in mixed)
    three_modal = sum(1 for row in mixed if len(row["modalities"]) == 3)
    summary = {
        "seed": args.seed, "scene_cap": args.scene_cap,
        "mixture_size": len(mixed), "mixture_path": str(mixture_path),
        "by_capability": dict(by_capability), "by_source": dict(by_source),
        "three_modal_share": round(three_modal / len(mixed), 4),
        "tiny_area_share": round(
            sum(1 for row in mixed if row.get("area", 1) < TINY_AREA) / len(mixed), 4),
        "slice_sizes": {name: len(rows) for name, rows in slices.items()},
        "quotas": quotas,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
