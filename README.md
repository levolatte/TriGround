# TriGround：RGB–红外–深度三模态视觉指代定位

本仓库同时保留已发布的 2B 基线与新的 8B/DeepStack 实验实现；当前推荐复现实验是 Qwen3-VL-8B 五阶段流程。8B 配置见 [升级说明](QWEN3_VL_8B_UPGRADE.md)，集群冒烟测试与完整训练评估见 [Slurm 操作说明](SLURM.md)。下面的比赛分数仍是历史 2B 路线结果，不代表新 8B 实验成绩。

TriGround 面向 RGB（可见光）、IR（红外）和 Depth（深度）三模态视觉指代定位任务，同时支持历史 `Qwen3-VL-2B-Instruct` 路线和当前 `Qwen3-VL-8B-Instruct` 实验路线。模型接收对齐的三模态图像和文本查询，输出查询目标在可见光图像中的归一化边界框 `[x1, y1, x2, y2]`。


本仓库实现并比较了两条主要技术路线：

- **方案一：弱监督早期融合。** 使用大量目标域弱监督数据，直接训练早期融合模块，使 RGB、IR 和 Depth 特征在较早阶段交互。该路线实现简单、能充分利用低成本数据，但容易受到错误方位描述、噪声框和数据偏差影响。
- **方案二：独立 adaptor 预训练后融合。** 先在其他具有明确模态监督的数据集上分别训练 IR adaptor 和 Depth adaptor，使两个分支先获得独立的模态表征能力；随后在目标任务上进行联合校准，并通过查询条件融合模块整合 RGB、IR 和 Depth 信息。

## 比赛测试结果

| 方案 | 核心训练方式 | 比赛正确率 |
| --- | --- | ---: |
| 方案一：弱监督早期融合 | 大量弱监督数据直接训练早期融合模块 | 0.6404 |
| **方案二：独立 adaptor 后融合** | IR/Depth adaptor 分别在其他数据集训练，再进行联合融合 | **0.6785** |

方案二比方案一高 **0.0381**，即 **3.81 个百分点**。比赛结果表明，先让辅助模态分支学习较明确的模态能力，再进行融合，比依赖大量噪声弱监督数据直接学习早期融合更加有效。因此，历史比赛流程以**方案二**作为表现更好的路线。

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

## 已发布模型（历史 2B）

模型参数可从 [models-v1.0.0 Release](https://github.com/levolatte/TriGround/releases/tag/models-v1.0.0) 下载。

| 模型文件 | 对应路线 | 用途 |
| --- | --- | --- |
| `triground-rdt-ws-v1-manual-ft1.pt` | 方案一：弱监督早期融合 | 模型注册表中的推荐发布 checkpoint |
| `triground-parallel-a-v1.pt` | 方案二：独立 adaptor 后融合 | Parallel-A 替代融合基线 |

这些发布文件只包含 TriGround 的 adaptor/fusion 等项目参数，不包含 Qwen3-VL-2B-Instruct 主干。模型来源、配置和 SHA-256 校验值见 [MODEL_REGISTRY.md](MODEL_REGISTRY.md) 与 [模型说明](release_models/MODEL_CARD.md)。当前 8B 五阶段流程尚未在此 Release 中提供训练完成的 checkpoint，历史 2B 分数也不能当作 8B 结果。

## 安装

需要 Python 3.10 或更高版本。下面的方式适合开发和 CPU 测试；正式 GPU 实验推荐按 [Slurm 操作说明](SLURM.md) 使用 `scripts/setup_gpu.sh` 安装与 CUDA 匹配的固定依赖。

```bash
git clone --branch qwen3-vl-8b https://github.com/levolatte/TriGround.git
cd TriGround
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows 下使用以下命令激活环境：

```powershell
.venv\Scripts\activate
```

当前 8B 流程需要另外下载 `Qwen/Qwen3-VL-8B-Instruct`；历史 2B 配置才使用 `Qwen/Qwen3-VL-2B-Instruct`。模型权重不包含在本仓库中。

## 复现实验前提与数据放置

**只克隆本仓库不能直接开始训练。** 仓库已经包含模型实现、五阶段配置、训练/评估工具、数据重叠审计和 Slurm 脚本，不依赖其他未公开的项目代码；但以下大文件和运行环境必须由实验者另行准备：

- `Qwen/Qwen3-VL-8B-Instruct` 的完整权重和配置；
- RGBT-Ground-Dataset、RoboRefIt 和项目目标域 `city_detection_prepared` 数据；
- 与集群驱动匹配的 Python、PyTorch/CUDA 依赖；
- 支持 BF16 且显存不少于 23,000 MiB 的 CUDA GPU（推荐 L40 48 GB）。

仓库的 `.gitignore` 会排除数据、Qwen 权重、训练 checkpoint 和 `runs/`，所以它们不会随 `git clone` 获得。使用默认 8B 配置和 Slurm 脚本时，推荐把外部资源放在仓库同级：

```text
experiment-root/
├── TriGround/                         # 本仓库；名称可不同
├── models/
│   └── Qwen3-VL-8B-Instruct/          # 必须含 config.json 和完整权重
├── datasets/
│   ├── RGBT-Ground-Dataset/
│   │   └── extracted/
│   │       ├── subsets/train_50.jsonl
│   │       └── manifests/val.jsonl
│   └── RoboRefIt/
│       └── manifests/
│           ├── formal_subsets/train_50.jsonl
│           └── testA.jsonl
└── city_detection_prepared/
    └── train/target_v2/
        ├── manual_split/train_100.json
        ├── manual_split/val.json
        └── train_weak_scene_safe.json
```

每份 manifest 中还必须包含可读取的 RGB/IR/Depth 图片路径。相对图片路径以 **manifest 所在目录** 为基准；若 manifest 保存的是绝对路径，则该路径必须在计算节点上同样有效。

上述五份训练/验证 manifest 必须在提交作业前已经生成并检查完毕；Slurm 脚本只验证和使用它们，不会从原始数据自动下载或制作这些划分。仓库中的 `tools/prepare_rgbt_groundbench.py`、`tools/prepare_roborefit.py` 和 `tools/prepare_target_v2_data.py` 可用于相应的数据转换，但原始数据、人工复核结果及其许可仍由实验者负责准备。

不采用上述默认布局时，无需修改仓库中的 YAML，可在提交 Slurm 作业前覆盖位置：

```bash
export BACKBONE=/shared/models/Qwen3-VL-8B-Instruct
export RGBT_ROOT=/shared/datasets/RGBT-Ground-Dataset
export ROBOREFIT_ROOT=/shared/datasets/RoboRefIt
export CITY_ROOT=/shared/datasets/city_detection_prepared
```

完成上述准备后，先运行 `sbatch scripts/qwen3_vl_8b_smoke.slurm`；冒烟任务成功后再运行 `sbatch scripts/qwen3_vl_8b_formal.slurm`。完整环境变量、集群参数和产物说明见 [SLURM.md](SLURM.md)。

## 数据格式

数据清单支持两种形式：以 Query ID 为键的 JSON 对象，或每行一个记录的 JSONL。每条记录包含对齐的三模态图像路径、文本查询和归一化 `xyxy` 边界框；JSONL 记录还应直接包含唯一 `id`。以下是 JSON 对象示例：

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

### 当前推荐：Qwen3-VL-8B 五阶段流程

准备好前述外部资源后，先提交冒烟任务，再提交正式训练；两份脚本会生成本次运行专用配置，不修改仓库中的源 YAML：

```bash
sbatch scripts/qwen3_vl_8b_smoke.slurm
sbatch scripts/qwen3_vl_8b_formal.slurm
```

正式任务按 `stage1a_ir → stage1b_depth → stage2_joint → stage2_weak → stage2_clean` 顺序运行。推荐使用 [SLURM.md](SLURM.md) 中的 `afterok` 提交方式，确保冒烟成功后才启动正式任务。

以下两节记录的是历史 2B 路线，不是当前 8B 五阶段流程。

### 历史 2B 方案一：弱监督早期融合

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

### 历史 2B 方案二：独立 adaptor 预训练后融合

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

## 历史 2B 发布模型评估

不同模型需要使用与其结构对应的配置。以下示例评估历史 2B 方案二；8B 正式脚本会在五阶段训练后自动执行原生 RGB 和最终多模态全量评估：

```bash
python evaluate.py \
  --config configs/stage2_joint_fusion_v2.yaml \
  --checkpoint triground-parallel-a-v1.pt \
  --manifest path/to/validation.json \
  --output evaluation.json
```

评估方案一时，将配置替换为 `configs/triground_rdt_ws_v1_manual_ft1.yaml`，权重替换为 `triground-rdt-ws-v1-manual-ft1.pt`。

## 使用历史 2B 模型生成比赛提交文件

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
- 历史比赛测试结果表明方案二在当时评测上的综合表现更好，但仍建议通过 RGB、RGB+IR、RGB+Depth、三模态、错配模态和零模态消融进一步判断各分支贡献。
- 发布权重必须配合相应配置和 Qwen3-VL-2B-Instruct 主干使用。

## 测试

```bash
pytest -q
ruff check .
```

## 许可

本仓库当前采用保留所有权利的专有声明；未经版权所有者事先书面许可，不授予使用、复制、修改、发布或分发代码及模型产物的权利，详见 [LICENSE](LICENSE)。Qwen 主干、第三方依赖及各训练数据集继续遵循其各自许可证。
