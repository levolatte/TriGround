# 2026-09-09 分支同步与 Slurm 脚本交付

> 后续更新：用户另行授权“推送”后，已成功执行 `git push origin main`，远端 main 更新至 `2571a4c`。正文“未推送”描述的是初次交付时的状态；集群作业仍未运行。

## 请求与范围

用户要求拉取队友最新分支、实现冒烟测试与正式测试的 `.slurm` 脚本，并明确正式测试为“完整五阶段训练后评估”。随后根据队友的分支不同步说明，要求顺便合并两个分支。本轮完成本地合并与脚本实现，不启动集群作业、不推送远端、不提交比赛结果。

实际 Git 仓库是 `F:/AIC/code`。根 `AGENTS.md` 已读取并遵循，未修改；赛题材料与工程接手记录见前一日的 `2026-09-08-contest-and-current-code-review.md`。本次任务属于既有工程的合并与作业编排，没有引入新的训练框架。

## 分支事实及合并选择

- 拉取后的 `origin/main` 为 `770928e`，包含冻结 Qwen3-VL-8B、面向 L40 的预检、DeepStack 覆盖和五阶段训练配置。
- 拉取后的 `origin/public-v1` 为 `e3e5ede`，包含 Query 位置编码与因果诊断、ACC@0.5 选模、逐样本评估及发布清理。
- 两分支共同祖先为 `abcd311013096a103d2201b138ceea52b3ab6478`，不是简单快进关系。
- 在本地 `main` 合并 `origin/public-v1`，保留两个分支的提交历史，处理 11 个冲突文件。
- 合并提交为 `2571a4c`（`Merge public-v1 into main and add 8B Slurm workflows`），未推送远端。

核心保留项：

1. 8B 主干动态维度，独立的 adaptor（适配器）、fusion（融合）、Query 编码器维度；显式融合层 `[8,16,24,26]`；在 DeepStack 提取前注入融合结果。
2. Query 可选正弦位置编码，以及 IR/Depth/Query 尺度干预；联合模态尺度为零时的 RGB 路径；完整残差的模态随机丢弃语义。
3. 初始化权重的碰撞检测、checkpoint（权重检查点）格式版本及运行环境信息。
4. `ACC@0.5 → mIoU → ACC@0.7 → parse_rate` 选模，保留 ACC-best、mIoU-best、last，并重载最优权重后评估。
5. 合并两个分支的真实预检功能：两步 AdamW 更新、Query/融合非零有限梯度、冻结参数无梯度、DeepStack 索引与显存记录。

沿用 public-v1 的旧日志、旧脚本、旧配置和 egg-info 清理；恢复仍有用途的 `scripts/setup_gpu.sh`，将其结尾提示更新为 8B 预检入口。六份 main 的旧 2B 配置原来使用个人机器绝对模型路径，改为公开模型 ID，同时通过 `backbone_revision` 保留原版本。未改变相应训练超参数。

## 脚本与运行约定

完整集群操作说明位于 `code/SLURM.md`。

| 文件 | 作用 |
| --- | --- |
| `code/scripts/qwen3_vl_8b_smoke.slurm` | 单测、数据重叠审计、IR/Depth/Joint 各两步真实优化、两条样本四模式生成解析 |
| `code/scripts/qwen3_vl_8b_formal.slurm` | Stage1A IR → Stage1B Depth → Stage2 Joint → Weak → Clean，随后原生 RGB 与最终模型全量评估 |
| `code/scripts/slurm_env.sh` | 已有 Python 环境、共享离线模型缓存、CUDA/BF16 检查及环境记录 |
| `code/tools/prepare_slurm_run.py` | 生成独立配置副本、数据根目录覆盖、本次运行的前序权重连接与划分审计 |
| `code/tests/test_prepare_slurm_run.py` | 配置隔离、训练超参数保留、缺失清单、祖先训练与保留集审计 |
| `code/tests/test_slurm_scripts.py` | 使用真实 Bash 和模型命令替身验证流程顺序及失败传播 |

默认单节点、单任务、单 GPU、8 CPU、64 GB 主存；冒烟申请 2 小时，正式申请 72 小时。它们是可覆盖的资源申请值，并非已测耗时或显存保证。分区、账户和 GPU 类型名称依真实集群通过 sbatch 参数提供。

冒烟清除前序权重依赖并提高输入至最大配置像素预算，检查从零初始化出发的梯度通路；不会保存供正式训练使用的权重。所有真实预检临时关闭随机模态丢弃，训练仍遵循源配置。正式五阶段每阶段读取本次运行目录中的前序最佳权重，任何失败立即停止。每次运行需要一个新输出目录，避免混用历史检查点。

默认最终评估使用 119 条复核验证集，此结果应称为验证成绩。通过 `EVAL_MANIFEST` 可以指定带标签的独立保留集，并对全部祖先训练清单检查重叠。不能用无 bbox 的官方测试 Query 做离线 ACC 评估。本地缺少队友外部训练清单，因此不能证明弱监督清单与 119 条验证集实际无重叠；脚本会在运行前执行审计并遇重叠停止。跨数据源的相同 ID 若引发误报，应核实数据标识含义，不能直接跳过检查。

## 验证结果

新建本地 `.venv`，使用 Python 3.11、PyTorch 2.8.0+cpu、Transformers 4.57.3 和仓库固定实验依赖。没有下载 Qwen 权重。

- 完整 `python -m pytest -q`：**99 passed in 19.46s**。
- 真实 Bash 编排测试：冒烟三种结构各两步；正式五阶段后才评估、评估不截断样本；第三阶段故意失败时经 tee 仍返回错误并停止；生成全部解析失败时冒烟失败。
- 数据审计测试：弱监督阶段与验证场景重叠时拒绝继续；显式保留集审计包含全部五阶段训练来源。
- 新增 Python 脚本、测试及预检的 Ruff 检查通过；全工程 Python compileall 通过；四份相关 Shell/Slurm 文件的 `bash -n` 通过；`pip check` 无依赖冲突。
- 合并冲突标记已清除，最终 `git diff --cached --check` 通过；顺带清理了上游六个文件的末尾多余空行。`.gitattributes` 固定 Shell/Slurm 使用 LF 行尾。

未验证：真实集群提交、L40 最大输入显存、Qwen3-VL-8B 前反向和生成、五阶段训练耗时及最终准确率。以上应通过交付的冒烟与正式作业执行获得，不能将本地替身测试当成 GPU 实验成绩。

## 下一步

1. 将本地合并结果按团队流程同步到远端/集群；本轮未自动推送。
2. 准备 CUDA 环境、共享模型缓存和 RGBT/RoboRefIt/City 清单，核对清单内图像路径。
3. 按 `SLURM.md` 提交冒烟；确认数据审计、两步梯度、四模式生成和显存日志。
4. 冒烟成功后，通过 afterok 依赖启动完整五阶段训练；保留本次配置、日志、最佳权重和逐样本评估文件。
