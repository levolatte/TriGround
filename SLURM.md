# Qwen3-VL-8B 的 Slurm 冒烟与正式实验

本流程针对 `QWEN3_VL_8B_UPGRADE.md` 中的 E 配置：冻结 8B 主干、BF16、四个 DeepStack/末层融合边界、单卡执行。集群须提供可运行该配置的 GPU，目标是 L40 48 GB；实际峰值以冒烟结果为准。

仓库本身包含完整的训练、评估和作业编排代码，但不包含训练数据、Qwen3-VL-8B 权重或 CUDA 运行环境。仅执行 `git clone` 不足以开始实验；必须按下文放置外部资源。

## 两个入口

| 文件 | 工作内容 | 默认资源 |
| --- | --- | --- |
| `scripts/qwen3_vl_8b_smoke.slurm` | 单测、数据划分审计、IR/Depth/Joint 三种结构各两步 AdamW、两条样本四模式生成与解析检查 | 1 GPU、8 CPU、64 GB 主存、2 小时 |
| `scripts/qwen3_vl_8b_formal.slurm` | 五阶段顺序训练，每阶段先预检，最后全量评估原生 RGB 与训练后模型 | 1 GPU、8 CPU、64 GB 主存、72 小时 |

分区、账户、GPU 类型名与时限依集群配置由 `sbatch` 参数指定，脚本没有编造固定集群名称。上面的主存不是显存。脚本不安装环境、不下载模型、不自动提交比赛结果。

## 1. 环境与数据准备

在登录/下载节点进入 **TriGround 仓库根目录**。激活已有 CUDA 环境，或在允许安装的节点运行 `bash scripts/setup_gpu.sh`。后者创建 `.venv`，安装 CUDA 12.8 的 PyTorch 2.8.0 和固定实验依赖；集群 CUDA/驱动要求不同时，应使用对应的 PyTorch 环境。

默认配置约定仓库、模型和数据按以下方式存放。`TriGround/` 可以换成其他仓库目录名，但各外部目录与仓库的相对关系需要保持不变：

```text
experiment-root/
├── TriGround/
├── models/Qwen3-VL-8B-Instruct/
├── datasets/RGBT-Ground-Dataset/
├── datasets/RoboRefIt/
└── city_detection_prepared/
```

```bash
cd /your/path/experiment-root/TriGround
source .venv/bin/activate
export PYTHON="$PWD/.venv/bin/python"

# 方案 A：提前下载到共享 Hugging Face 缓存，计算节点离线读取。
export HF_HOME=/your/shared/cache/huggingface
python tools/download_model.py --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --cache-dir "$HF_HOME/hub"
export BACKBONE=Qwen/Qwen3-VL-8B-Instruct

# 方案 B：若已整理成完整本地模型目录，直接指定该目录。
# export BACKBONE=/your/shared/models/Qwen3-VL-8B-Instruct

# 数据不在默认同级目录时，按集群实际位置覆盖；结构见下表。
export RGBT_ROOT=/your/shared/datasets/RGBT-Ground-Dataset
export ROBOREFIT_ROOT=/your/shared/datasets/RoboRefIt
export CITY_ROOT=/your/shared/datasets/city_detection_prepared
```

| 变量 | 目录下应存在的清单 |
| --- | --- |
| `RGBT_ROOT` | `extracted/subsets/train_50.jsonl`、`extracted/manifests/val.jsonl` |
| `ROBOREFIT_ROOT` | `manifests/formal_subsets/train_50.jsonl`、`manifests/testA.jsonl` |
| `CITY_ROOT` | `train/target_v2/manual_split/train_100.json`、`manual_split/val.json`、`train_weak_scene_safe.json`，后三者均相对 `train/target_v2/` |

清单内的图片路径也必须在计算节点有效。相对图片路径以清单文件所在目录为基准；仅更改清单根目录不能修复其中旧机器的绝对图片路径。未设置数据变量时使用源 YAML 的默认相对路径：`../datasets/RGBT-Ground-Dataset`、`../datasets/RoboRefIt` 和 `../city_detection_prepared`。

可选变量：

- `BACKBONE`：完整本地模型目录，或已经下载进 `HF_HOME` 的 Hugging Face ID。Slurm 默认值为 `../models/Qwen3-VL-8B-Instruct`，并在启动 GPU 工作前检查本地目录中的 `config.json`。
- `CONFIG_DIR`：自定义的五份 8B YAML 所在目录，默认 `configs`。文件名需保持 `qwen3_vl_8b_<stage>.yaml`。
- `EVAL_MANIFEST`：训练结束后评估的**带标签独立保留集**。默认使用最终配置的 119 条复核验证集，此时报告只是验证成绩，不称作独立测试成绩。若使用独立 284 条清单，须显式设置此变量，脚本不猜测文件路径。
- `RUN_ROOT`：本次新实验目录，必须尚不存在；默认 `runs/slurm-smoke-<jobid>` 或 `runs/slurm-formal-<jobid>`。使用依赖作业时不要给两个作业设置相同目录。
- `IR_CHECKPOINT`：可选的已完成 Stage1A `best_phase_a.pt`。正式脚本验证文件存在后跳过 IR 训练，并让 Joint 阶段直接加载该权重。
- `DEPTH_CHECKPOINT`：可选的已完成 Stage1B `best_phase_a.pt`。正式脚本验证文件存在后跳过 Depth 训练，并让 Joint 阶段直接加载该权重。
- `REPO_DIR`：计算节点上的仓库根目录；默认 Slurm 提交工作目录。跨目录提交时必须使用 `sbatch --chdir=/your/path/experiment-root/TriGround` 或设置 `REPO_DIR`。
- `SMOKE_SCAN_SAMPLES`：冒烟扫描的训练样本数，默认 64。

## 2. 提交冒烟及正式任务

从仓库根目录提交，替换集群分区；若需要账户，在两个命令中都加 `--account=账户名`。若集群规定 GPU 类型资源，例如 `gpu:l40:1`，用管理员公布的类型覆盖默认 `--gres=gpu:1`。

```bash
export GPU_PARTITION=YOUR_GPU_PARTITION
smoke_job=$(sbatch --parsable --partition="$GPU_PARTITION" \
  scripts/qwen3_vl_8b_smoke.slurm)

# 只有冒烟作业成功退出，正式任务才会运行。
sbatch --partition="$GPU_PARTITION" \
  --dependency="afterok:${smoke_job%%;*}" \
  scripts/qwen3_vl_8b_formal.slurm
```

复用已经完成的 IR 第一轮时，为新任务保留新的 `RUN_ROOT`，只需把现有权重作为输入：

```bash
export IR_CHECKPOINT=/path/to/completed-run/stage1a_ir/best_phase_a.pt
sbatch --partition="$GPU_PARTITION" scripts/qwen3_vl_8b_formal.slurm
```

此时任务从 Depth 开始；Joint 阶段加载该 IR 权重和本次新训练的 Depth 权重。若同时提供 `DEPTH_CHECKPOINT`，任务将从 Joint 开始。

默认 72 小时只是资源申请值，不是已测得的耗时保证；需按集群时限和冒烟表现调整 `--time`。若暂时只想检查环境，只提交第一条。脚本不设置 `CUDA_VISIBLE_DEVICES`，使用 Slurm 分配的设备。

Slurm 的工作目录、命令行覆盖和 `afterok` 语义见 [官方 sbatch 文档](https://slurm.schedmd.com/sbatch.html)，GPU 资源名称见 [官方 GRES 文档](https://slurm.schedmd.com/gres.html)。批作业脚本会被 Slurm 复制到运行目录，因此不使用脚本自身路径反推仓库。

## 3. 冒烟检查什么

`tools/prepare_slurm_run.py` 会从正式源配置生成本次运行副本，原 YAML 不变。冒烟副本：

- 清除所有前序检查点与恢复状态，因此 Joint 冒烟不要求已经完成 Stage1 训练。
- 暂停模态 dropout，避免随机丢模态影响梯度检查。
- 令 `min_pixels=max_pixels=802816`，按最大配置预算处理输入；扫描候选后记录实际最大的视觉词元数。图像尺寸取整可能使实际值不恰好等于预算，不能把扫描样本称为全数据集最大样本。
- 分别对红外、深度、联合模型运行两次完整优化器更新，记录实际 GPU、损失、梯度、DeepStack 索引、峰值 allocated/reserved 显存和耗时。
- 联合模型第二步必须有 Query 编码器与联合融合的有限非零梯度，冻结主干和冻结适配器不得出现参数梯度。
- 用两条样本调用四种模式的生成和框解析，每种模式至少应有一个可解析输出。这里不设置准确率门槛，不把未训练模型的输出当模型成绩。

冒烟优化发生在内存中，不保存供正式训练使用的检查点。正式任务从源配置规定的起点开始。

## 4. 正式实验与产物

```text
Stage1A IR（3轮）
  → Stage1B Depth（3轮）
  → Stage2 Joint（1轮，合并两个Stage1的best）
  → Stage2 Weak（1轮）
  → Stage2 Clean（1轮）
  → 原生RGB及最终模型的全量评估
```

单卡任务中两个 Stage1 顺序运行。正式副本保留源配置的像素范围、梯度累积、学习率与随机种子，只调整数据/主干位置、运行输出和前序权重路径。Stage2 预检要求验证集恰为 119 条。

所有真实模型预检均在独立进程中临时关闭模态 dropout（随机丢弃），以稳定检查梯度通路；日志记录源配置和预检使用的值。正式训练仍使用源配置的 dropout。

每一步只读取本次运行目录中的前序 `best_phase_a.pt`。合并后的训练链按 `ACC@0.5 → mIoU → ACC@0.7 → parse_rate` 选 best，并额外保留 mIoU-best 与 last。最终 `evaluate.py` 不传 `--subset-size`，遍历全部指定样本。

产物位于本次 `RUN_ROOT`：

```text
configs/                       实际运行配置
plan.json                      阶段及最终评估范围
audit_*.json                   ID/图像/场景重叠报告
logs/                          每条预检、训练、评估命令日志
stage1a_ir/ ... stage2_clean/   各阶段权重、配置与训练指标
native_rgb.json/.jsonl         未加载任务权重的RGB基线（正式任务）
final_evaluation.json/.jsonl   最终模型四模式指标和逐样本证据
smoke_generation.json/.jsonl   两条样本的生成检查（仅冒烟任务）
```

Slurm 标准输出/错误另保存在提交工作目录的 `slurm-<jobname>-<jobid>.out/.err`，即使 Python 或数据预检很早失败也有日志。脚本启用 `set -euo pipefail`，Python 失败不会被 `tee` 掩盖。

准备阶段用已有 `audit_manifest_overlap.py` 核对各单模态 train/val，并将五阶段全部训练来源与 119 条验证集、可选保留集比较；出现重叠即退出，不自动删样本。该工具按清单 ID/场景字段与路径比较，跨数据集同名 ID 可能产生误报，图片复制改名也不能仅凭路径排除；出现异常应核对数据来源与报告。8B 文档只明确弱集排除了 284 集场景，是否也排除 119 条验证集仍须这一步实测。

**EVAL_MANIFEST 不能填官方无标签 Query JSON。** 正式比赛预测是另一个任务；这两个脚本不把官方测试数据用于训练或选模。

## 5. 失败与重跑

任一步失败会中止，后续阶段不会继续。已完成的权重和日志保留，脚本不会自动选旧运行的模型或重试。重新提交默认产生新 jobid 和新目录；本脚本没有承诺五阶段自动断点续训。若要接着已有阶段训练，应依据保存配置和 checkpoint 明确安排，而不是覆盖原实验目录。

## 本地验证记录

脚本的 Bash 语法、配置重定位/初始化依赖测试和合并后 CPU 单测在本地验证；真实 L40 显存、CUDA/BF16 优化器及集群调度必须由冒烟作业验证。具体本轮测试结果见仓库 HANDOFF 与工作区阶段记录。
