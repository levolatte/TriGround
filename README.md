# TriGround：RGB–红外–深度三模态视觉指代定位

TriGround 基于 `Qwen3-VL-2B-Instruct`，面向 RGB（可见光）、IR（红外）和 Depth（深度）三模态视觉指代定位任务。模型接收对齐的三模态图像和文本查询，输出查询目标在可见光图像中的归一化边界框 `[x1, y1, x2, y2]`。


本仓库实现并比较了两条主要技术路线：

- **方案一：弱监督早期融合。** 使用大量目标域弱监督数据，直接训练早期融合模块，使 RGB、IR 和 Depth 特征在较早阶段交互。该路线实现简单、能充分利用低成本数据，但容易受到错误方位描述、噪声框和数据偏差影响。
- **方案二：独立 adaptor 预训练后融合。** 先在其他具有明确模态监督的数据集上分别训练 IR adaptor 和 Depth adaptor，使两个分支先获得独立的模态表征能力；随后在目标任务上进行联合校准，并通过查询条件融合模块整合 RGB、IR 和 Depth 信息。

## 比赛测试结果

| 方案 | 核心训练方式 | 比赛正确率 |
| --- | --- | ---: |
| 方案一：弱监督早期融合 | 大量弱监督数据直接训练早期融合模块 | 0.6404 |
| **方案二：独立 adaptor 后融合** | IR/Depth adaptor 分别在其他数据集训练，再进行联合融合 | **0.6785** |

方案二比方案一高 **0.0381**，即 **3.81 个百分点**。比赛结果表明，先让辅助模态分支学习较明确的模态能力，再进行融合，比依赖大量噪声弱监督数据直接学习早期融合更加有效。因此，当前以**方案二**作为推荐的比赛路线。

这项结论来自比赛测试结果；仓库内 `combined284` 的本地验证结果用于训练诊断和模型选择，不能与比赛正确率直接横向比较。

## 模型结构

方案二的主要数据流如下：

```text
RGB ────────────────────────────────┐
                                    │
IR ─── IR adaptor ──────────────────┼── 查询条件联合融合 ── Qwen bbox 输出
                                    │
Depth ─ Depth adaptor ──────────────┘
```

RGB 保留为主要视觉路径；IR 和 Depth 分别通过轻量 adaptor 提取增量模态信息，融合器结合文本 query 决定如何注入这些信息。发布配置冻结 Qwen 语言模型和视觉主干，主要训练项目自定义的 adaptor 与 fusion 参数。

## 已发布模型

模型参数可从 [models-v1.0.0 Release](https://github.com/levolatte/TriGround/releases/tag/models-v1.0.0) 下载。

| 模型文件 | 对应路线 | 用途 |
| --- | --- | --- |
| `triground-parallel-a-v1.pt` | 方案二：独立 adaptor 后融合 | 当前推荐的比赛路线 |
| `triground-rdt-ws-v1-manual-ft1.pt` | 方案一：弱监督早期融合 | 早期融合对照与目标域微调实验 |

发布文件只包含 TriGround 的 adaptor/fusion 等项目参数，不包含 Qwen3-VL-2B-Instruct 主干。模型来源、配置和 SHA-256 校验值见 [MODEL_REGISTRY.md](MODEL_REGISTRY.md) 与 [模型说明](release_models/MODEL_CARD.md)。

## 安装

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows 下使用以下命令激活环境：

```powershell
.venv\Scripts\activate
```

请另外下载 `Qwen/Qwen3-VL-2B-Instruct`，或者在配置文件的 `model.backbone` 中填写已有的本地模型路径。

## 数据格式

数据清单为以 Query ID 为键的 JSON 对象。每条记录包含对齐的三模态图像路径、文本查询和归一化 `xyxy` 边界框：

```json
{
  "query-id": {
    "visible": "Images/visible/000001.png",
    "infrared": "Images/infrared/000001.png",
    "depth": "Images/depth/000001.png",
    "query": "车辆旁边的行人",
    "bbox": [0.12, 0.21, 0.38, 0.84]
  }
}
```

图像路径默认相对于清单文件解析。训练边界框必须满足：

```text
0 <= x1 < x2 <= 1
0 <= y1 < y2 <= 1
```

## 训练

公开配置使用 Hugging Face 模型 ID 和相对数据路径，运行前需要根据本地数据位置修改清单路径。

### 方案一：弱监督早期融合

历史实验先在弱监督数据上训练 RDT-deep 早期融合模块，再进行低学习率延长训练：

```bash
python tools/preflight.py --config configs/multimodal_rdt_deep_reviewed.yaml
python train.py --config configs/multimodal_rdt_deep_reviewed.yaml
python train.py --config configs/multimodal_rdt_deep_reviewed_extend_e5.yaml
```

随后可在人工复核的目标域数据上微调：

```bash
python train.py --config configs/triground_rdt_ws_v1_manual_ft1.yaml
```

### 方案二：独立 adaptor 预训练后融合

该路线先分别训练 IR 和 Depth adaptor，再进行联合校准与查询条件融合：

```bash
python train.py --config configs/stage1a_ir.yaml
python train.py --config configs/stage1b_depth.yaml
python train.py --config configs/stage2_joint_calibration.yaml
python train.py --config configs/stage2_weak1024_raw.yaml
python train.py --config configs/stage2_clean_after_weak1024.yaml
python train.py --config configs/stage2_joint_fusion_v2.yaml
```

其中弱监督步骤用于补充目标域覆盖，不替代 IR/Depth adaptor 的独立模态训练。实际复现时应根据数据质量决定是否启用弱监督步骤，并保留独立验证集选择最佳 checkpoint。

## 评估

不同模型需要使用与其结构对应的配置。评估方案二：

```bash
python evaluate.py \
  --config configs/stage2_joint_fusion_v2.yaml \
  --checkpoint triground-parallel-a-v1.pt \
  --manifest path/to/validation.json \
  --output evaluation.json
```

评估方案一时，将配置替换为 `configs/triground_rdt_ws_v1_manual_ft1.yaml`，权重替换为 `triground-rdt-ws-v1-manual-ft1.pt`。

## 生成比赛提交文件

提交工具会保留输入 JSON 的其他字段，只替换 `bbox`；同时检查 Query 顺序、坐标归一化和边界框有效性，支持断点续推，并生成包含预测 JSON 的 ZIP 文件。

方案二示例：

```bash
python tools/predict_competition_submission.py \
  --config configs/stage2_joint_fusion_v2.yaml \
  --checkpoint triground-parallel-a-v1.pt \
  --queries path/to/queries.json \
  --data-root path/to/dataset \
  --progress outputs/progress.jsonl \
  --output-json outputs/predictions.json \
  --output-zip outputs/predictions.zip
```

## 模型文件内容

发布的 checkpoint 包含：

- 项目自定义模型参数；
- 架构与训练配置；
- epoch、step 和模型选择分数等元数据；
- 稳定的模型名称和标签。

不包含：

- Qwen 主干参数；
- 优化器、学习率调度器和混合精度 scaler；
- 训练数据、比赛输入或图片。

请使用 `mm_grounding.checkpoint.load_model_checkpoint` 加载模型。

## 复现说明与局限

- CPU 环境可以运行代码测试，实际训练需要 CUDA GPU。
- 三模态图像需要正确对齐，深度单位及预处理必须与配置一致。
- `combined284` 中的 `new154` 部分与方案一使用过的原始弱监督数据存在重合，因此其绝对指标不能作为完全独立的泛化结论。
- 比赛测试结果证明方案二在当前评测上的综合表现更好，但仍建议通过 RGB、RGB+IR、RGB+Depth、三模态、错配模态和零模态消融进一步判断各分支贡献。
- 发布权重必须配合相应配置和 Qwen3-VL-2B-Instruct 主干使用。

## 测试

```bash
pytest -q
ruff check .
```

## 许可

目前尚未选择明确的再分发许可证，版权归项目所有者。接受第三方复用或贡献前，应补充代码与模型许可证。Qwen 主干及各训练数据集继续遵循其各自许可证。
