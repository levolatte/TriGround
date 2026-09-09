# TriGround Handoff

## 2026-09-09 分支合并与 Slurm 流程

- 将 `main@770928e` 的 8B/DeepStack 与 `public-v1@e3e5ede` 的 Query 因果诊断、ACC 选模和逐样本评估合并。下方 Query A/B 状态是历史记录，尚无新实验成绩。
- 保留显式融合层 `[8,16,24,26]`、分离的 adaptor/fusion/query 维度、零初始化、稀疏初始化碰撞检测、checkpoint 格式与运行环境信息；同时保留位置编码与三种尺度干预。
- 新入口：`scripts/qwen3_vl_8b_smoke.slurm`、`scripts/qwen3_vl_8b_formal.slurm`。正式任务是五阶段训练后全量评估，操作说明见 `SLURM.md`。
- 冒烟使用独立临时配置，无需前序权重，覆盖 IR/Depth/Joint 两步优化与四模式生成解析。正式配置保留训练参数，前序权重全部指向本次运行目录。
- `tools/prepare_slurm_run.py` 负责路径配置、运行隔离和已有数据重叠审计；`scripts/slurm_env.sh` 使用已经激活的 Python 与共享模型缓存。
- 集群分区、账户、真实路径通过命令参数/环境变量提供；本地没有 Slurm/L40，未提交 GPU 作业。真实 GPU 预检与模型成绩仍须集群运行。
- 本地验证：Windows / Python 3.11 / PyTorch 2.8.0+cpu / Transformers 4.57.3，完整 `pytest -q` 为 **99 passed**。包含四项真实 Bash 编排测试（模型命令替身）、配置隔离/权重链和数据审计测试；新增脚本及预检 Ruff、Python compileall、Bash 语法检查、依赖一致性检查均通过。GPU 显存和训练效果未验证。
- 合并时将 main 中六份 2B 旧配置的本机模型绝对路径替换为模型 ID，并以 `backbone_revision` 保留原版本。所有真实预检临时关闭模态 dropout，以免随机丢弃导致梯度误报；训练配置不变。
- 本轮只完成本地合并，未推送远端。使用前按 `SLURM.md` 配置模型缓存、训练清单路径及集群分区；先冒烟，成功后再执行完整五阶段流程。

## 以下为历史 Query A/B 记录

## 当前阶段

Query 位置编码 A/B 的本地实现已完成；正式云端预检、训练、离线 119 条评估和两次官方提交尚未执行。完整命令和判定规则见 `EXPERIMENT_QUERY_POSITION_AB.md`。

## 已完成

- 新增无参数动态正弦 Query 位置编码，默认 `none` 保持旧模型兼容。
- 新增 v3 Control/Treatment 配置，只有输出目录与位置编码开关不同。
- 全训练链改为 `ACC@0.5 → mIoU → ACC@0.7 → parse_rate` 选模，并单独保留 mIoU 最优和 last checkpoint。
- 修复最佳 checkpoint 重载后误复用最后一轮指标的问题。
- 修复 Joint Fusion 诊断没有作用于真实 `joint_stage_fusions` 的问题；scale=0 直接跳过对应模态路径。
- 增加逐样本评估证据、A/B 成对比较、scene bootstrap、尺寸/类别/已有 Query 类型分层。
- 增加初始化点解析、配置等价检查、整条已知初始化训练链重合审计和梯度预检。
- `combined284` 已降为纯诊断，不允许输出主选 checkpoint。
- 提交工具支持 9555 条强制检查，并记录解析失败/fallback。

## 下一执行者必须先做

1. 在云电脑按 `EXPERIMENT_QUERY_POSITION_AB.md` 安装固定环境并运行完整测试。
2. 运行 `prepare_query_position_ab.py`，确认两组共同初始化来源。
3. 对所有已知祖先训练清单运行 overlap audit；任何 ID、图像或 scene/sequence 重合都必须停止训练。
4. 两份配置分别跑真实 Qwen forward/backward 预检，确认 Query Encoder、Joint Fusion 有梯度且冻结 adaptor 无梯度。
5. Treatment → Control 分别启动训练进程。
6. 每组三类 checkpoint 跑四模式评估；best-ACC 模型再跑 scale/mismatch 诊断。
7. 固定提交 Control best-ACC 与 Treatment best-ACC，返回两个官方分数。

## 当前验证状态

- 本机 pytest 环境缺少 PyTorch 与 `huggingface_hub`，完整测试在收集阶段因依赖缺失而停止，尚未进入测试断言。
- 已完成全仓 Python AST 语法解析、`git diff --check` 和不依赖 PyTorch 的成对比较测试。
- 云端 GPU 预检与完整 pytest 是正式训练的硬门槛，不得跳过。

## 暂不处理

Direct BBox Head 等九项后续方向只记录在实验文档的“待研究计划”中。本轮结果回来前不要实现，也不要提前修改 README、模型注册表或模型卡里的主模型定义。
