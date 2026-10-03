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

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tools.prepare_qwen3vl_native_sft import (  # noqa: E402
    depth_mm_to_grayscale_rgb,
    native_prompt,
)

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
        # 把深度策略落进记录：出厂验收闸门要靠它选阈值。
        # 不同策略的合法白色占比差别很大（毫米映射 0%，百分位拉伸可到 37%），
        # 一刀切的阈值要么放过坏图，要么误杀好图。
        "depth_policy": depth_policy,
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
    """RoboRefIt：真实 RGB-D、66 类、同类多实例歧义。两模态（RGB+Depth），红外置空。

    ## 深度图必须用可视化版，不能用原始版

    本函数原先读 `converted/train.jsonl`，其 `depth` 指向 `final_dataset/train/depth/`，
    那是**原始 `I;16`**（conversion_report 实测 200 张全是 uint16，max 65535）。
    把这种图直接交给图像处理器转 RGB，**值 >255 会被截断**——实测白色像素占比 71~80%，
    深度模态等于噪声。（原注释写的是"depth 是可视化图（P01-P99 拉伸）"，
    说明意图如此，只是路径取错了。）

    正解是 `converted/train_depth_visual_fixed_p01_p99.jsonl`，其 `depth` 指向
    `depth_visual_fixed_p01_p99/`，2000 张全是 RGB 可视化图。

    ## 提示词策略不能照抄清单里的 `depth_policy`

    清单里 `depth_policy: "visual"` 说的是**文件格式**（"这个文件已经是可视化图"），
    而 `native_prompt` 的 `depth_policy` 参数说的是**该给模型哪条亮度说明**——
    两者同名但语义不同。实测把 `"visual"` 透传过去，`native_prompt` 会**一条深度说明都不加**
    （其分支是 `elif ... and depth_policy != "visual"` 才报错，等于 `"visual"` 静默跳过），
    模型无从知道那个亮度代表什么。

    清单里的 `depth_visualization: "fixed_raw_p01_p99_v1"` 与 `depth_unit: "unknown"`
    正好对应 `DEPTH_POLICY_SENSOR` 那条说明（"fixed inverse mapping of the raw sensor
    values… physical distance unit is not established"），因此**用该策略**，
    并**断言来源格式**——将来清单换成别的渲染方式时会明确失败，而不是悄悄用错说明。
    """
    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    manifest = root / "subset" / "converted" / "train_depth_visual_fixed_p01_p99.jsonl"
    if not manifest.is_file():
        raise SliceError(
            f"缺少 RoboRefIt 的可视化深度清单: {manifest}\n"
            f"（不能用 converted/train.jsonl —— 它指向原始 I;16 深度，会被截断成白图）")
    base = manifest.parent
    expected_visualization = "fixed_raw_p01_p99_v1"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rgb = (base / row["rgb"]).resolve()
        depth = (base / row["depth"]).resolve()
        if not rgb.is_file() or not depth.is_file():
            raise SliceError(f"{row['id']}: RoboRefIt 图像缺失 {rgb} / {depth}")
        if row.get("depth_visualization") != expected_visualization:
            raise SliceError(
                f"{row['id']}: 深度可视化方式变了（{row.get('depth_visualization')!r} "
                f"≠ {expected_visualization!r}），"
                f"提示词里的亮度说明需要重新核对，不能沿用 DEPTH_POLICY_SENSOR")
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise SliceError(f"{row['id']}: bbox 越界 {row['bbox']}")
        record = make_record(
            f"robo:{row['id']}", ("rgb", "depth"), [str(rgb), str(depth)],
            row["query"], (x1, y1, x2, y2),
            depth_policy=DEPTH_POLICY_SENSOR,
            capability="discriminate", source="roborefit",
            extra={"area": round((x2 - x1) * (y2 - y1), 6),
                   "query_chars": len(row["query"]), "class_name": row.get("class_name")})
        out["discriminate"].append(record)
    print(f"  RoboRefIt: {len(out['discriminate'])} 条，"
          f"深度可视化={expected_visualization}，提示词策略={DEPTH_POLICY_SENSOR}")
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


def build_rgbdt_prepared(prepared_root: pathlib.Path, rendered_dir: pathlib.Path,
                         rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    """队友新传的 RGBDT500_prepared：992 条真三模态，121 类，唯一同源于测试集的来源。

    ## 为什么值得用

    * **每条都是完整三模态**（visible/infrared/depth 三路径齐全，已逐条校验 992/992）。
    * **深度是 `I;16` uint16 max 19999、1920×1080——与官方测试集格式完全一致**
      （实测官方测试集 depth 也是 `I;16` max 19999 1920×1080）。
    * 赛题 PDF 参考资源 [1] 即 RGBDT500 论文，正当性最强。
    * 121 类，远多于当前混合集的 72 类。

    ## 两处必须处理的缺陷

    1. **深度是原始位深**，直接交给处理器转 RGB 会截断成白图（同 City/RoboRefIt 的坑）。
       这里用**推理时的同一个函数** `depth_mm_to_grayscale_rgb` 现场渲染，
       保证 train/inference 逐像素一致。
    2. **`weak_label: true`**——查询由 Qwen API 生成 + **同模型自审**（非独立验证）。
       这靠配比稀释，不在本函数过滤；但查询长度仍按官方上限截断。

    ## 关于与测试集重叠

    实测 161/992 样本与官方测试集逐字节相同。官方已确认外部数据集合规，
    剔除与否是**测量决策**（让官方分数可反映泛化），由 `--exclude-frame-keys` 统一处理，
    本函数不内嵌这份清单。
    """
    manifest = prepared_root / "grounding_final_train.json"
    if not manifest.is_file():
        raise SliceError(f"缺少 RGBDT500_prepared 标注: {manifest}")
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, dict):
        raise SliceError(f"预期 {manifest} 是 id -> 记录 的字典")

    out: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for sample_id, row in records.items():
        query = str(row["query"]).strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            continue
        rgb = (prepared_root / row["visible"]).resolve()
        ir = (prepared_root / row["infrared"]).resolve()
        raw_depth = (prepared_root / row["depth"]).resolve()
        for path in (rgb, ir, raw_depth):
            if not path.is_file():
                raise SliceError(f"{sample_id}: RGBDT500_prepared 图像缺失 {path}")
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise SliceError(f"{sample_id}: bbox 越界 {row['bbox']}")

        # 深度现场渲染。产物按样本 id 命名，重跑时直接复用，避免每次重建都重渲 7G。
        depth_path = rendered_dir / f"{sample_id}.png"
        if not depth_path.is_file():
            with Image.open(raw_depth) as image:
                raw = np.asarray(image)
            visual = depth_mm_to_grayscale_rgb(raw)
            Image.fromarray(visual).save(depth_path)
            rendered += 1

        area = (x2 - x1) * (y2 - y1)
        capability = assign_primary_capability(area, len(query), 0)
        # 归入 trimodal 池：它同时是"三模态协同"的能力信号，也是本轮的分布锚点
        target = capability if capability in ("small",) else "trimodal"
        out[target].append(make_record(
            f"rgbdt500p:{sample_id}", ("rgb", "infrared", "depth"),
            [str(rgb), str(ir), str(depth_path)], query, (x1, y1, x2, y2),
            depth_policy=DEPTH_POLICY_MM,
            capability=target, source="rgbdt500_prepared",
            extra={"area": round(area, 6), "query_chars": len(query),
                   "class_name": row.get("class_name"),
                   "sequence_id": row.get("sequence_id"),
                   "weak_label": bool(row.get("weak_label"))}))
    print(f"  RGBDT500_prepared: 三模态 {len(out['trimodal'])} 条 / "
          f"极小目标 {len(out['small'])} 条（本次新渲染 {rendered} 张深度）")
    return out


def build_city(manifest: pathlib.Path, image_root: pathlib.Path, scene_cap: int,
               rng: random.Random) -> list[dict[str, Any]]:
    """City 锚点：按每场景上限抽样，保持目标域主导。

    计划里的取舍：把 `scene_cap` 从 6 提到 `scene_cap`，而不是取消上限——
    取消会改变训练分布（少数场景权重上升），提高上限则既多用数据又保留防过采样。

    **路径解析**：清单里的图像路径形如 `../visible/x.png`，是相对**清单自身所在目录**
    （`target_v2/`）的。若拿 `image_root / "../visible/x.png"` 去拼会得到
    `data/city/visible/...`，那是错的——所以先按清单目录解析，解析不到再退回 image_root。

    ## 深度图必须用预渲染版

    清单的 `depth` 指向 `../depth/x.png` → `city/train/depth/`，那是**原始 `I;16`**
    （max 19999）。实测直接交给图像处理器转 RGB 后**白色像素占 64~72%**，
    深度模态等于噪声——而推理侧 `predict_native_submission.py` 是用
    `depth_mm_to_grayscale_rgb()` 正确渲染的，两边不一致。

    预渲染版就在 `target_v2/qwen3vl_native_sft/depth_rgb/`（759 张 RGB），
    实测与现场渲染 `depth_mm_to_grayscale_rgb()` **逐像素完全一致**，因此直接指向它。
    """
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(records, dict):
        records = list(records.values())
    base = manifest.parent
    rendered_root = base / "qwen3vl_native_sft" / "depth_rgb"
    if not rendered_root.is_dir():
        raise SliceError(f"缺少 City 预渲染深度目录: {rendered_root}")
    by_scene: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        scene = pathlib.Path(str(row["visible"])).stem.rsplit("_", 1)[0]
        by_scene[scene].append(row)

    def resolve(value: str) -> pathlib.Path:
        candidate = (base / value).resolve()
        return candidate if candidate.is_file() else (image_root / value).resolve()

    def resolve_depth(value: str) -> pathlib.Path:
        """原始深度路径 -> 预渲染路径（同名）。查不到就硬失败，不退回原始图。"""
        rendered = rendered_root / pathlib.Path(value).name
        if not rendered.is_file():
            raise SliceError(f"City 预渲染深度缺失: {rendered}（原始路径 {value}）")
        return rendered

    out: list[dict[str, Any]] = []
    for scene, rows in by_scene.items():
        rng.shuffle(rows)
        for row in rows[:scene_cap]:
            rgb, ir = resolve(row["visible"]), resolve(row["infrared"])
            depth = resolve_depth(row["depth"])
            if not rgb.is_file() or not ir.is_file():
                raise SliceError(
                    f"{row['query'][:40]}: City 图像缺失 (rgb={rgb} ir={ir})")
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
    """按配额预混合。配额是"使用多少条"，超出则无放回抽样，不足则全用并告警。

    配额为 0 表示**该切片整体不用**（例如 Arm 3 弃用 City）。此时必须跳过而不是
    照样构建——否则会白读一遍 City 清单，还会因空池触发下面的硬失败。
    """
    rng = random.Random(seed)
    mixed: list[dict[str, Any]] = []
    for name, quota in quotas.items():
        if quota <= 0:
            print(f"  {name}: 配额 0，跳过")
            continue
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


def frame_key_of(path: str) -> str:
    """把一张图的路径映射到帧键，与 `audit_test_overlap.frame_key` 保持同一规则。

    同一帧的原始图与渲染图 stem 相同（`depth/003353.png` vs `depth_rgb/003353.png`），
    因此天然归到同一键——这正是剔除要的粒度：任一模态被污染，整帧剔除。
    """
    stem = pathlib.Path(path).stem
    for suffix in ("_color", "_depth", "_infrared", "_nir", "_ir", "_rgb", "_visible"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def drop_excluded(rows: list[dict[str, Any]], excluded: set[str]) -> tuple[list, int]:
    """剔除引用了污染帧的样本。

    按**帧**而非按文件剔除：只删被查出的那个文件，同帧的渲染副本会漏网。
    """
    if not excluded:
        return rows, 0
    kept = [row for row in rows
            if not any(frame_key_of(image) in excluded for image in row["images"])]
    return kept, len(rows) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgbt-root", type=pathlib.Path, required=True)
    parser.add_argument("--roborefit-root", type=pathlib.Path, required=True)
    parser.add_argument("--rgbdt-sft", type=pathlib.Path, required=True)
    parser.add_argument("--rgbdt-assets", type=pathlib.Path, required=True)
    # 队友新传的 RGBDT500_prepared。可选：不传则完全不参与，便于复现 Arm 2。
    parser.add_argument("--rgbdt-prepared", type=pathlib.Path, default=None,
                        help="RGBDT500_prepared/train 目录（含 grounding_final_train.json）")
    parser.add_argument("--rgbdt-rendered-dir", type=pathlib.Path, default=None,
                        help="渲染后深度的缓存目录（默认 out-dir/rgbdt_depth_rgb）")
    parser.add_argument("--city-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--city-image-root", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    # 污染剔除清单，由 tools/audit_test_overlap.py 产出
    parser.add_argument("--exclude-json", type=pathlib.Path, default=None,
                        help="audit_test_overlap.py 的输出，按帧剔除与测试集重叠的样本")
    parser.add_argument("--scene-cap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    # Arm 3 配额：**用满全部可用数据**，而不是按固定数字抽样。
    #
    # 相对 Arm 2 的三点变化：
    #   1. **City 配额 0**（弃用）——RGBDT500_prepared 是更好的锚点：同源于测试集、
    #      格式一致（uint16 max 19999 / 1920x1080）、121 类、深度已按推理口径渲染。
    #   2. **外部数据用满**——Arm 2 只用了 RGBT 的 11%、RoboRefIt 的 6.5%。
    #      实测可用总量 16,442 条（small 6,609 / discriminate 6,149 / longquery 2,767
    #      / trimodal 917），是 Arm 2 的 2.4 倍。
    #   3. **真三模态从 181 提到 917**（RGBDT500_prepared 全部干净样本 + 旧 RGBDT500）。
    #
    # 配额写成略高于实测可用量：`mix()` 会在池不足时告警并按实际数量使用，
    # 这样"有多少用多少"，且可用量变化时会显式告警而不是静默缩水。
    parser.add_argument("--quota-city", type=int, default=0)
    parser.add_argument("--quota-small", type=int, default=7000)
    parser.add_argument("--quota-discriminate", type=int, default=6500)
    parser.add_argument("--quota-trimodal", type=int, default=1000)
    parser.add_argument("--quota-longquery", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    excluded: set[str] = set()
    if args.exclude_json is not None:
        excluded = set(json.loads(
            args.exclude_json.read_text(encoding="utf-8"))["excluded_frame_keys"])
        print(f"污染剔除清单: {len(excluded)} 个帧键（来自 {args.exclude_json}）")

    print("构建各切片…")
    slices: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    builders = [build_rgbt(args.rgbt_root, rng), build_roborefit(args.roborefit_root),
                build_rgbdt(args.rgbdt_sft, args.rgbdt_assets)]
    if args.rgbdt_prepared is not None:
        builders.append(build_rgbdt_prepared(
            args.rgbdt_prepared,
            args.rgbdt_rendered_dir or (args.out_dir / "rgbdt_depth_rgb"),
            rng))
    for partial in builders:
        for key, value in partial.items():
            slices[key].extend(value)
    if args.quota_city > 0:
        slices["city_anchor"] = build_city(args.city_manifest, args.city_image_root,
                                           args.scene_cap, rng)
    else:
        print("  City: 配额 0，不构建（Arm 3 弃用 City）")

    total_dropped = 0
    if excluded:
        print("\n按帧剔除污染样本：")
        for name in list(slices):
            slices[name], dropped = drop_excluded(slices[name], excluded)
            total_dropped += dropped
            if dropped:
                print(f"  {name}: 剔除 {dropped} 条")
        print(f"  合计剔除 {total_dropped} 条")

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
        "excluded_frame_keys": len(excluded),
        "dropped_for_contamination": total_dropped,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
