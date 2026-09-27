# GPT Pro 审计入口：TriGround 全仓库快照

本分支 `audit/gpt-pro-20260927` 基于本地原 `main`，提交截至2026-09-27的当前源码、脚本、测试及最近实验的文本证据。GitHub默认分支仍是 `public-v1`；审计时必须明确选择本分支，不能只读默认分支。

## 从这里开始

1. [DATASETS.md](DATASETS.md)：各来源、划分、深度编码、人工验收、独立Query与训练呈现的区别。
2. [最新实验结论](evidence/aux_selection_refine_20260927/CONCLUSIONS.md)：C292→选择273→完整实测v1 221，以及预检漏检和裁剪边界实现偏差。
3. [此前G/U根因审计](research/2026-09-27-gu-deep-root-cause.md)：对应监督、几何捷径、训练输入和评测的已证事实/不确定性。
4. [B/G*/U*完整报告](evidence/gu_diagnosis_20260927/RESULTS.md)、[第一轮G/U报告](evidence/gu_pilot_20260926/RESULTS.md)。
5. 下列实际源码与逐条证据，而不是只相信上述文字结论。

## 最新源码版本尤其需要分清

- 仓库根 `tools/predict_aux_selection.py`：**事后修正为居中扩大再截边的版本，仅CPU测试通过，没有GPU成绩。** 它不能对应221这个已测结果。
- [最新实测v1源码](evidence/aux_selection_refine_20260927/code_final/)：真正产生选择273、完整221的版本，包含评分前空检测/数字ID接口修复。
- [首次冻结源码](evidence/aux_selection_refine_20260927/code_snapshot/) 和 [评分前补丁](evidence/aux_selection_refine_20260927/code_patches/) 均保留。
- [边界裁剪修正版](evidence/aux_selection_refine_20260927/code_corrections_not_evaluated/) 明确未GPU评测。
- `tools/aux_selection_evidence.py`：真实RGB/IR检测、预测框提示SAM、主体Depth可靠性与前后关系。
- `tools/run_aux_selection_queue.py`：有限8小时、GPU串行、冻结输入、按阶段落盘、有净收益才96对照。
- `tools/report_aux_selection.py`：原浮点GT、完整412、候选覆盖、翻转、78图组配对重采样。
- `tools/prepare_gu_diagnosis.py`、`prepare_gu_same_day_pilot.py`、`prepare_same_day_pilot.py`、`scripts/run_gu_diagnosis.sh`、`run_gu_pilot.sh`、`run_qwen3vl_native_lora.sh`：此前严格配对训练的生成与执行。
- `annotation_builders/` 保留此前散落在本地 `.work` 的一次性标注构建脚本，硬编码路径和人工对象描述属于历史产物，不应当作自动可迁移标注器运行。

## 当前数据与模型事实

| 项目 | 核实状态 |
|---|---|
| 主任务 | RGB/IR/Depth与英文Query输入，最终预测RGB框 |
| City训练 | 3707条记录；重复图像与同场景Query须单列统计 |
| City验证 | 412题、78图组；已知47题地点与训练重用，其余365未被证明完全场景独立 |
| 新人审放行 | 181输出任务，98Query/98图组；RGB98、IR58、Depth25，是对象对应材料 |
| 人审来源 | 两份100检查项的真实CSV，5拒绝、66尚不具备放行决策不混入181 |
| 本地参考 | M2 285、C1500 292、旧G 288、旧U 283、B 287、G* 284、U* 286，分母均412 |
| 最新冻结方法 | 选择273、完整v1 221；没有权重更新、没有官方推理 |
| 官方测试 | 5690题无GT；M2/C均0.7144是用户提供的比赛反馈，不是本地计算结果 |

不要把“对齐模态中同物体的框对应”解释成“辅助证据帮助区分不同对象”。对181条的拟合检查属于训练内能力诊断，不是新场景泛化成绩。旧样本、复用Query、重复呈现和增强不得重复计数为新增数据。

## 随分支提供的证据

- `datasets/released181/`：真实训练输出记录、任务元信息、放行统计。
- `datasets/human_decisions/`：用户导出的两包人工决定原CSV。
- `datasets/rgbdt_trial200/`：RGBDT首200图组对象证据及审核汇总。
- `datasets/inventories/`：历史去重库存快照，按日期选择，不相加。
- `evidence/gu_pilot_20260926/`：第一轮G/U配对呈现、200结果与报告。
- `evidence/gu_diagnosis_20260927/`：B/G*/U*实际呈现和消费记录、96真实输入组合、181/83拟合预测、训练配置及报告。
- `evidence/aux_selection_refine_20260927/`：GT-free推理清单与独立评分GT、完整412候选/Depth统计/预测、模型原文、裁剪变换、配置、全部翻转ID及12例独立目视文字记录。
- `research/`：70份原研究Markdown快照，包括旧计划和旧结果。最新事实优先，历史计划不等于真实执行。
- [FILE_INVENTORY.json](FILE_INVENTORY.json)：逐文件的来源位置、仓库位置和字节数；不是模型或数据哈希。

原始图像、掩码PNG、图册图片、模型权重、tokenizer大文件、压缩归档、缓存、凭据及AGENTS.md不公开上传。记录中的 `F:/AIC/...` 和 `/root/...` 是原运行路径，克隆后并不存在；不能据此声称已经成功复现。图册文字保留，但其中本地图像链接在GitHub中不可用，需要实际视觉复核时请明确索取相应原图或图册，不凭文字假称看过图。

## 推荐给GPT Pro的审计任务

可复制 [AUDIT_REQUEST.md](AUDIT_REQUEST.md)。重点要求逐条引用源码行、真实数据记录和结果，不接受只提出更大模型或更多标注的泛化建议。

1. 核对实现是否满足计划，尤其候选来源、Depth单位/主体污染、候选编号绑定、同一对象细化和坐标所属图像。
2. 核对181放行、人审CSV到训练输出的映射，以及G/U和B/G*/U*逐条配对、消费顺序、实际提示差异。
3. 分开核对模型环境、候选覆盖、选错对象、框边界、提示/输出协议和统计波动；不能根据分数臆测模型内部注意力。
4. 核对评分有无GT泄漏、整数化评分、分母变化、失败回退、事后择优、版本混淆及未证实的泛化结论。
5. 评估预检为何没有拦住训练16例细化14→13→9的异常，建议最少且有效的行为验收。
6. 最后只建议一个最有证据的下一实验，给出代表输入、明确比较对象、停止条件和最小算力；不要把尚未评测的修正代码当成功方案。

## 环境和验证边界

最新实测云端Python来自 `/root/miniconda3/bin/python`，PyTorch2.8.0+cu128、Transformers5.14.1、RTX4090。GroundingDINO/SAM使用隔离venv并复用现有系统包；实际加载版本和命令以结果配置为准。仓库早期 `pyproject.toml` 仍约束Transformers<5，不能把旧安装说明误当本轮精确环境；本次提交是审计快照，没有悄悄升级依赖或重跑GPU。

最新开跑前27项CPU测试、C固定8题精确复现通过；边界修正后选择器11项CPU测试通过。这些接口检查不能代替行为质量检查。源码历史、报告里的失败和修正前输出全部保留，供审计追溯。
