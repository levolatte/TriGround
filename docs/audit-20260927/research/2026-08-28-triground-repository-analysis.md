# TriGround 队友仓库阅读与实验审计报告

> 日期：2026-08-28  
> 队友：沈茜琳（sql）  
> 上游仓库：`https://github.com/levolatte/TriGround.git`  
> 本地位置：`F:\AIC\code`（仓库内容直接以 `code` 为根目录）  
> 分支：`public-v1`  
> 审计提交：`ad45af93a61339bd96f7c9a809f0858b0c8cc7e5`  
> 审计性质：只读分析；未修改 TriGround 源码、配置或 Git 状态

## 1. 审计范围与结论口径

本轮严格按“先 README、后源码”的顺序完成了以下工作：

1. 阅读当前 `README.md`、模型注册表、模型卡、发布说明与发布清单；
2. 遍历当前分支全部 97 个受跟踪文件，包括 75 个 Python 文件、9 个 YAML 配置、4 个 Markdown 文档、2 个审核页面和其余元数据；
3. 重点逐段阅读 `src/mm_grounding/`、`train.py`、`evaluate.py`、全部当前配置、提交脚本、数据转换/审核工具与 21 个测试文件；
4. 阅读 14 个提交的 Git 历史，查看公开发布前被删除的配置、实验日志和数据审计报告，以复原实验演进；
5. 对 75 个 Python 文件做无依赖的语法树解析，全部通过；检查当前工作树保持干净；
6. 未下载 Qwen 主干、发布权重或外部数据，也未运行训练/推理。

证据强度需要明确区分：

- **比赛分数**来自最新 README 的记录，当前仓库没有排行榜截图、提交回执、原始预测或对应评测日志，故可视为队友报告的结果，但本轮无法独立复核；
- **`combined284` 指标**来自模型注册表和模型卡，当前分支没有原始评测 JSON，而且文档主动声明其中一部分与弱监督训练数据重叠；
- **早期 1,831 条验证实验**保存在 Git 历史的原始日志中，能直接复核日志数值，但数据划分与后两类结果不同，不能横向比较。

## 2. 一页结论

TriGround 是一个以 `Qwen3-VL-2B-Instruct` 为冻结主干、让模型直接生成归一化边界框 JSON 的多模态视觉指代定位项目。它不是 GroundingDINO 检测器，也没有候选框生成与重排序链路；仓库内未出现 GroundingDINO、候选生成或 reranking（重排序）实现。

仓库真正形成了两条可发布路线：

- **方案一：RDT-deep 弱监督路线。** IR/Depth 先形成视觉提示，并在 Qwen 每个视觉块前反复更新和注入。README 将其概括为“弱监督早期融合”，但按代码看，它并不是只在输入端融合一次，而是深层循环提示注入。比赛正确率记录为 `0.6404`。
- **方案二：并行三流＋查询条件联合融合。** RGB、IR、Depth 共享同一套冻结 Qwen 视觉块权重，但分别前向；IR/Depth 每层接轻量残差 adaptor（适配器），最后在查询条件下做联合可靠性融合。比赛正确率记录为 `0.6785`，比方案一高 `0.0381`，即 3.81 个百分点，是最新 README 推荐路线。

我对当前成果的总体判断是：

1. **方案二的比赛结果是仓库里最重要的新证据。** 它推翻了旧发布文档基于 `combined284` 得出的“RDT 推荐”结论。
2. **代码思路完整，实验迭代痕迹真实。** 仓库不是只有结果说明；数据转换、人工审核、场景防泄漏划分、稀疏 checkpoint（检查点）、训练、消融和提交工具都已实现。
3. **公开分支更像“科研快照”，还不是即拉即跑的复现包。** 数据、权重、运行日志不在 Git 中；若干旧工具仍指向已删除配置；依赖声明有缺口；最新比赛结论尚未同步到模型卡与发布清单。
4. **模型选择目标与比赛指标不一致。** 训练和组合结果排序都以 mIoU（平均交并比）选优，而比赛目标是 ACC@0.5；这可能直接错过比赛指标最优 checkpoint。
5. **最终联合融合模型的模态缩放诊断存在实现盲区。** 当前诊断脚本只修改旧的独立融合模块，不能缩放最终启用的联合融合模块；RGB/缺模态/错配模态实验仍可用，但 IR/Depth scale sweep（尺度扫描）对最终模型无效。

## 3. 仓库结构与职责

| 区域 | 作用 | 审计结论 |
| --- | --- | --- |
| `README.md` | 项目入口、路线与比赛结果 | 最新、唯一写明 0.6404/0.6785 的地方 |
| `train.py` | 构建处理器、模型、数据加载器并启动训练 | 单机 PyTorch 训练入口，没有 Accelerate 分布式封装 |
| `evaluate.py` | RGB、RGB+IR、RGB+Depth、三模态评估 | 并行路线支持单辅助模态消融，RDT 只支持完整三模态 |
| `src/mm_grounding/` | 数据、融合器、模型挂接、训练引擎、指标、checkpoint | 核心实现集中，抽象层不多，符合竞赛代码风格 |
| `configs/` | 两条发布路线的 9 份配置 | 配置能复原主要训练阶段，但依赖本地未发布的数据路径和前序 checkpoint |
| `tools/` | 数据转换、人工审核、诊断、评估、提交 | 覆盖面很广；部分为旧实验遗留，当前不可直接运行 |
| `tests/` | 21 个测试文件、63 个测试函数 | 重点覆盖融合恒等初始化、梯度、三流执行、数据转换和稀疏 checkpoint |
| `release_models/` | 模型卡、发布说明、清单与哈希 | 实际 `.pt` 不在 Git；角色标注滞后于最新比赛结果 |

核心模块分工如下：

- `data.py`：清单读取、三模态预处理、Qwen 对话监督构造；
- `adapters.py`：RDT、并行 adaptor、查询编码器、独立融合和联合融合；
- `model.py`：把自定义融合上下文接入 Qwen 视觉前向，并管理可训练参数；
- `engine.py`：自回归 bbox 解析、评估、优化器、训练阶段、早停与 checkpoint 选择；
- `checkpoint.py`：保存项目参数并组合加载多个阶段 checkpoint；
- `metrics.py` / `boxes.py`：mIoU、ACC@0.5、ACC@0.7、L1 和 GIoU。

## 4. 任务形式与数据流

### 4.1 输入与监督

清单支持 JSON 对象或 JSONL。每条数据包含 RGB、IR、Depth、自然语言 Query 和归一化 `xyxy` 框。

- RGB：按 RGB 图像读入；
- IR：按灰度读入，再转成三通道供 Qwen 图像处理器使用；
- Depth：若原图是多通道只取第一通道；按 `depth_scale=1000` 转成米；保留 `(0, 20m]`；第一通道为 `log1p(distance)` 归一化值，第二通道为有效性掩码，第三通道为零；深度图缩放使用最近邻；
- Query：一份进入 Qwen 文本提示，另一份单独分词后供并行路线的查询编码器使用。

训练提示要求 Qwen 精确生成：

```json
{"bbox_2d":[x1,y1,x2,y2]}
```

坐标被量化到 0～1000 的整数。训练损失主体是 Qwen 原生 causal LM（因果语言模型）token loss；虽然代码保留可选辅助 bbox head（边界框头）和 Vision LoRA，但所有发布配置都关闭了它们。

### 4.2 推理与指标

评估时从生成文本中用正则解析四个坐标，除以 1000 还原归一化框。解析失败在评估中被记作零面积框，因此 IoU 为零；比赛提交工具则默认回退为整图框 `[0,0,1,1]`，并记录回退数量。

报告指标包括：

- mIoU：平均 IoU；
- ACC@0.5：IoU 不低于 0.5 的比例，也是本赛题主指标；
- ACC@0.7：更严格的定位正确率；
- L1 coordinate error：坐标平均绝对误差；
- parse rate、generation cap hit rate：输出可解析率和达到生成长度上限的比例。

## 5. 方案一：RDT-deep 弱监督路线

### 5.1 实现方式

RDT-deep 先从 IR 和 Depth 原始 patch（图块）构造初始提示。Qwen 视觉前向被局部替换：在每个冻结视觉 block（块）之前，当前 RGB hidden state（隐藏状态）与上一层提示共同更新新提示，再将提示注入 RGB 流。因而“早期融合”只是 README 的高层概括，代码本质是**每层都有一次提示更新和注入**。

初始化使用零恢复投影与很小的残差尺度，目标是初始时严格保持 RGB 行为；测试覆盖了初始恒等性和每层提示参数能够收到梯度。

### 5.2 配置链路

1. `multimodal_rdt_deep_reviewed.yaml`：弱监督数据，2 个 epoch，学习率 `4e-5`；
2. `multimodal_rdt_deep_reviewed_extend_e5.yaml`：从历史 best checkpoint 续训到 epoch 5，学习率降为 `2e-5`；
3. `triground_rdt_ws_v1_manual_ft1.yaml`：在 923 条人工复核目标域样本上训练 1 个 epoch，学习率 `3e-6`，用独立 119 条验证集选择。

发布配置始终冻结 Qwen 语言和视觉主干，不启用 Vision LoRA，也不启用辅助 bbox head。

### 5.3 结果

- README 记录的比赛正确率：`0.6404`；
- `combined284`：mIoU `0.5952`，ACC@0.5 `69.01%`，ACC@0.7 `54.93%`；
- 历史 1,152 条人工复核验证：RDT ACC@0.5 为 `62.41%`（719/1152），RGB 为 `62.07%`（715/1152），仅多 4 条；文档明确说明 RDT 的 mIoU 和 ACC@0.7 仍低于 RGB。

`combined284` 的 `new154` 与该路线原始弱监督训练数据存在重叠，因此 69.01% 不能作为干净泛化成绩。

## 6. 方案二：并行三流与查询条件联合融合

### 6.1 它并非三套独立 Qwen

并行路线只保留一套冻结 Qwen 视觉权重，但同一视觉 block 会分别处理 RGB、IR 和 Depth token。每层执行顺序是：

1. RGB 经共享视觉 block；
2. IR 经同一 block，再经该层 IR adaptor；
3. Depth 经同一 block，再经该层 Depth adaptor；
4. 到配置的融合层边界时，将辅助信息以残差形式注入 RGB。

因此它是**共享权重、三次前向、分模态轻量适配**，不是三份主干参数。显存中的主干参数没有复制三份，但视觉 block 计算量大致接近 RGB 单流的三倍，外加 adaptor 和融合器开销。

当前全部并行配置都设置 `parallel_fusion_stages: 1`，按代码只会选中最后一个视觉 block 边界。这意味着 IR/Depth adaptor 在每层工作，但真正的跨模态注入只发生在视觉主干末端，并不是多层反复融合。

### 6.2 查询编码与最终联合融合

Query 使用冻结 Qwen 的 token embedding（词元嵌入），并 `detach`（切断主干梯度），再分别经过小型 IR/Depth Transformer 查询编码器。

最终 `JointQueryAwareStageFusion` 的核心步骤是：

- 分别投影 RGB、IR、Depth；
- 让 RGB 图像 token 对 Query 做 cross-attention（交叉注意力）；
- 在每个对齐空间 token 上，对 RGB、IR、Depth、language 四项做模态注意力；
- 显式计算 RGB–IR、RGB–Depth、IR–Depth 的余弦一致性；
- 用 token 级和 sample（样本）级两个可靠性门控控制残差；
- 以 RGB 为主路径，仅把有门控的联合残差加回 RGB。

最终配置采用 `modality_dropout=0.2` 和零初始化恢复投影，以便从严格 RGB 恒等起点训练，并学习在辅助模态不可靠时关闭注入。

### 6.3 分阶段实验链

| 阶段 | 数据与训练 | 加载关系 | 训练重点 |
| --- | --- | --- | --- |
| Stage 1A IR | RGBT-GroundBench 的 `train_50` 分组子集，3 epoch | 从冻结 Qwen 开始 | 只训练 IR adaptor、IR 查询编码及对应融合参数 |
| Stage 1B Depth | RoboRefIt 的 `train_50` 分组子集，3 epoch | 从冻结 Qwen 开始 | 只训练 Depth adaptor、Depth 查询编码及对应融合参数 |
| Joint calibration | 人工复核目标域 `train_100`，2 epoch | 顺序加载 IR、Depth 两个 checkpoint | 冻结两侧 adaptor，校准联合任务参数 |
| Weak 1024 | 场景去重、类别/尺度分层的 1,024 条弱监督目标域数据，1 epoch | 加载 calibration | 低学习率补目标域覆盖 |
| Clean after weak | 人工复核 `train_100`，1 epoch | 加载 Weak 1024 | 用干净数据收尾 |
| Joint fusion v2 | 人工复核 `train_100`，2 epoch | 加载 clean checkpoint | 冻结 adaptor，训练新的联合可靠性融合 |

这里的 `train_50` 是由分组子集工具生成的 50% 子集名称，不是 50 条样本。弱监督步骤在 README 中被描述为可按数据质量决定是否启用。

一个容易忽略的实现细节是：最终 `stage2_joint_fusion_v2.yaml` 没有开启 `warm_start_joint_fusion_from_legacy`。所以它会加载前序 adaptor、查询编码器和旧独立融合参数，但实际启用的新 `joint_stage_fusions` 通过零恢复投影从 RGB-safe（保持 RGB）起点训练；旧 `stage_fusions` 随 checkpoint 保留但被冻结且不参与最终融合。

### 6.4 结果

- README 记录的比赛正确率：`0.6785`；
- 相对 RDT：`+0.0381`，即 `+3.81` 个百分点；
- `combined284`：mIoU `0.5882`，ACC@0.5 `67.96%`，ACC@0.7 `55.28%`；
- 当前模型注册表写明发布文件来自 `runs/stage2_joint_fusion_v2/last_phase_a.pt`，不是 `best_phase_a.pt`。

比赛结果与本地 `combined284` 排名方向相反：本地 mIoU/ACC@0.5 偏向 RDT，比赛测试明显偏向 Parallel。这说明 `combined284` 既存在泄漏，也可能与正式测试分布不一致，不能再承担路线裁决的主要角色。

## 7. 指标证据总表

| 数据/来源 | RGB | RDT-deep | Parallel | 能否直接比较 | 证据限制 |
| --- | ---: | ---: | ---: | --- | --- |
| 比赛正确率（README） | 未提供 | 0.6404 | **0.6785** | 两路线可比 | 只有 README 结果文字，无原始评测材料 |
| `combined284` mIoU | 0.5848 | **0.5952** | 0.5882 | 同表可比 | `new154` 与 RDT 弱监督数据重叠 |
| `combined284` ACC@0.5 | 67.25% | **69.01%** | 67.96% | 同表可比 | 不是比赛集；原始结果文件未发布 |
| `combined284` ACC@0.7 | 53.87% | 54.93% | **55.28%** | 同表可比 | Parallel 只比 RDT 高 1 个预测 |
| 历史 reviewed-1152 ACC@0.5 | 62.07% | **62.41%** | 未提供 | RGB/RDT 可比 | RDT 仅多 4 条；其他指标低于 RGB |
| 早期 full-val-1831 ACC@0.5 | **46.31%** | 早期 legacy Phase A 39.54%；Phase B 43.64% | 尚未形成 | 仅早期实验可比 | 不是当前发布路线，也不是当前验证集 |

## 8. 从 Git 历史复原的实验演进

公开分支共 14 个提交，主线很清楚：

1. 2026-08-24 导入已有多模态 grounding（定位）项目；
2. 同日加入 query-aware（查询感知）分阶段并行路线、RGBT 和 RoboRefIt 转换器；
3. 2026-08-25 串联轻量阶段、补充下载和阶段检查工具；
4. 2026-08-27 清理成公开发布分支，删除大量旧配置、日志和本地运行脚本，加入模型卡和发布元数据；
5. 2026-08-28 最新提交只改 README，补入中文说明和比赛分数，将 Parallel 改为推荐路线。

发布前历史日志显示，早期路线并不成功：

- 原生 Qwen RGB 在旧 1,831 条验证集：mIoU `0.4051`，ACC@0.5 `0.4631`，ACC@0.7 `0.3687`；
- 早期 legacy 多模态 Phase A：mIoU `0.3441`，ACC@0.5 `0.3954`；
- 旧 Phase B 延长到 epoch 10：mIoU `0.3725`，ACC@0.5 `0.4364`，仍低于 RGB；
- 一次 Phase A 后进入 BF16 训练时，曾因 PyTorch GradScaler 不支持 BF16 unscale 崩溃；当前代码已通过“只在 FP16 启用 GradScaler”修复。

这组日志说明队友不是一次得到最终结构，而是先发现“直接多模态注入会伤害 RGB”，随后持续强化 RGB-safe 初始化、分模态预训练、冻结 adaptor、查询门控和联合可靠性融合。

## 9. 数据工程与人工审核

仓库的数据侧投入很重，且对最终结果可能与模型结构同等重要。

### 9.1 外部数据转换

- RGBT-GroundBench 转换器读取 FLIR、MFAD、M3FD 三个来源的官方 grounding 标注，形成 RGB+IR 清单；
- RoboRefIt 转换器查找 train/testA/testB 标注，解析 RGB+Depth 路径，规范化、裁剪并审计 bbox；
- 还保留 RefCOCO、Visual Genome 转换器，但不在当前两条发布配置中。

### 9.2 划分与弱监督

- `build_grouped_subsets.py` 按场景或指定 group key 划分嵌套子集，避免同一场景进入训练和验证；
- `select_weak_subset.py` 按类别和目标尺度分层，且每个场景最多一条，精确选择 1,024 条；
- `enrich_city_queries.py` 根据 GT bbox 的位置、同类序数和邻近关系扩充弱 Query。它能补空间/关系语言，但也会把标签几何直接写进文本，天然存在模板偏差和噪声放大风险；
- `prepare_target_v2_data.py` 合并人工审核结果，剔除不合格样本，并按图像/序列排除重叠；审核中的 `modality_need` 只作为分析字段，不直接作为训练标签。

### 9.3 审核工具

仓库有两套本地网页审核界面：

- 快速 grounding 审核：切换 RGB/IR/Depth，显示 GT/预测，标注有效、歧义、错框、漏标、不可辨认、模态错位及失败原因；
- 候选集审核：在数据质量之外，额外标注去向（测试、补充训练、暂缓）和模态需求（RGB、IR、Depth、IR+Depth、不确定）。

历史数据审计报告还记录了：训练 10,684 条/1,479 场景，验证 3,229 条/372 场景；目标明显偏小，训练集中 7,950 个框面积低于图像 1%；近重复图像哈希仍发现 41 个跨划分候选对；深度样本均为 uint16，零值比例均值约 28.3%；类别分布明显不均衡。这是早期原始/优化数据的历史状态，不代表最终人工复核划分的精确统计。

## 10. 训练、checkpoint 与提交实现

### 10.1 训练与 checkpoint

- 优化器只接收当前阶段 `requires_grad=True` 的任务参数；
- 发布路线全在 Phase A，不启用 Vision LoRA；
- 支持梯度累积、AMP、梯度 checkpointing、早期 probe（小样本探针）和早停；
- Stage 1 checkpoint 只保存可训练分支；joint 阶段会保存全部 `fusion.*`，以便组合 IR/Depth/旧融合状态；
- 多个初始化 checkpoint 以 `strict=False` 顺序组合，适合独立训练 IR 和 Depth 后合并；
- 发布工具再导出不含优化器/调度器/scaler 的推理 checkpoint，Qwen 主干始终需另行获取。

### 10.2 提交工具

提交脚本有几项实用设计：

- 保留原 Query ID、顺序和所有非 bbox 字段；
- 用 JSONL 增量记录进度，支持断点续推；
- 对 bbox 归一化、顺序、完整性和 ZIP 内容做边界检查；
- 最终 ZIP 只包含结果 JSON；
- 解析失败默认写整图框，并单独计数。

## 11. 当前最重要的问题与风险

### 11.1 最新推荐结论未同步到发布元数据

最新 README 把 `triground-parallel-a-v1.pt` 定义为当前推荐路线；但以下文件仍把 RDT 标为 recommended（推荐），把 Parallel 标为 baseline/alternative（基线/备选）：

- `MODEL_REGISTRY.md`；
- `release_models/MODEL_CARD.md`；
- `release_models/v1/release-manifest.json`。

原因可从历史确认：`models-v1.0.0` 标签指向 `bcd4804`，而比赛结果在其后的 `ad45af9` 才加入，只更新了 README。当前使用模型时应以最新比赛结论为准，但发布元数据需要队友确认后统一。

### 11.2 模型选择指标与比赛指标错位

训练引擎保存 best checkpoint、早停、早期 probe 和组合结果排序都使用 `mean_iou`。本赛题目标却是 ACC@0.5。两者相关但不等价，尤其当大量预测集中在 IoU 0.4～0.6 附近时，mIoU 最优点未必使过阈值样本最多。

这不是抽象风险：`combined284` 上 RDT 的 mIoU 更高，而 Parallel 的正式比赛 ACC 更高。后续讨论实验时，至少应把“按 mIoU 选”与“按 ACC@0.5 选”视为两个不同实验口径。

### 11.3 最终联合模型的 scale 诊断没有作用到活动模块

`diagnose_modality_interventions.py` 的 `temporary_modality_scales()` 只遍历 `model.fusion.stage_fusions`，修改旧独立 IR/Depth 残差尺度。最终 Parallel v2 实际使用 `joint_stage_fusions`，所以该脚本的 IR/Depth scale sweep 不会改变活动联合融合器。对应测试也只构造了旧 `stage_fusions`，没有覆盖最终 joint 模式。

不受此问题影响的部分包括：RGB-only、只给 IR、只给 Depth、完整三模态、替换成错配 IR/Depth 等输入干预。受影响的是通过 scale 参数声称“增减某个模态贡献”的结论。

### 11.4 公开复现链不闭合

- 当前 checkout 没有数据、训练输出、原始评测报告和 `.pt` 权重；权重只在发布文档中给出外部 Release；
- 比赛分数没有对应 checkpoint 元数据、预测文件或评测回执；
- `combined284` 没有原始 manifest 和评测 JSON，且缺少每个发布模型的精确重叠矩阵；
- 5 个旧 preflight/diagnose 工具仍引用发布时已删除的 YAML：`multimodal.yaml`、`multimodal_extension.yaml`、`multimodal_from_a3.yaml`、`multimodal_safe_v2.yaml`、`multimodal_safe_v3_bbox.yaml`；
- 两个渲染脚本硬编码 `D:\AIC\...`；`start_reviewer.cmd` 也依赖队友本地相邻数据和 `.conda-env`；
- `requirements-experiment.txt` 提到已经不存在的 `scripts/setup_gpu.sh`；
- `predict_competition_submission.py` 依赖 `tqdm`，但 `pyproject.toml` 和 `requirements-experiment.txt` 都没有声明它；`accelerate`、`safetensors`、`pyarrow` 只在实验 requirements 中，不在基础安装依赖中；
- `LICENSE` 是 All rights reserved（保留所有权利），与 README 的“尚未选择许可证”含义基本一致：未经许可不能复用或分发。

这些问题不否定实验结论，但意味着新机器复现前需要先确定“最小发布路径”和“历史工具是否继续维护”，不能假设所有 `tools/` 都属于当前路线。

## 12. 测试与本机验证

测试设计本身较有针对性，63 个测试函数覆盖：

- RGB 恒等初始化和恢复投影梯度；
- RDT 每层提示更新；
- 并行三流是否真的执行三次共享视觉 block；
- IR-only、Depth-only、joint 阶段的可训练参数；
- 冻结 adaptor、joint fusion 替换旧独立融合、warm start helper；
- 深度编码、Query token mask、稀疏 checkpoint；
- RGBT/RoboRefIt/RefCOCO/Visual Genome 转换；
- 场景安全划分、弱子集、人工审核与本地审核 API。

本机验证结果：

- 75/75 Python 文件完成 AST 语法解析，0 错误；
- `F:\AIC\code` 工作树干净；远端、分支与提交正确；
- 本机当前 Python 3.11 环境没有 torch、transformers、numpy、Pillow、PyYAML，也没有 pytest/ruff，因此 `pytest -q` 和 `ruff check .` 均因工具未安装而未执行测试主体；
- 本轮没有为只读审计临时安装训练环境，所以不能把 README 的“CPU 测试通过”当作本机已复验结论。

## 13. 建议优先讨论的问题（不涉及架构设计）

在开始下一步设计前，我建议先和 sql 对齐以下事实：

1. `0.6785` 对应的精确 checkpoint 是否就是发布的 `triground-parallel-a-v1.pt`，还是另一个本地/提交 checkpoint？
2. `0.6404` 与 `0.6785` 是否来自同一轮、同一测试集、同一提交规则，是否有提交 ID 或结果截图可归档？
3. Parallel 发布模型为何选 `last_phase_a.pt` 而不是训练引擎产生的 `best_phase_a.pt`？是否做过 ACC@0.5 口径的离线重排？
4. Parallel 的 Weak 1024 与 `combined284/new154` 是否有样本重叠？当前文档只明确写了 RDT 的重叠。
5. 最终模型实际完成过哪些有效消融：RGB、RGB+IR、RGB+Depth、完整三模态、错配模态、缺模态？scale sweep 若使用当前脚本需要重新解释。
6. `train_100`、`val`、manual 923/119 和比赛提交所用数据的最终 manifest 能否只在队内安全归档，并附样本数与场景重叠审计？
7. 发布文档究竟要以“比赛最优 Parallel”还是“本地 combined284 最优 RDT”为推荐口径，二者应避免继续混写。

这些问题会决定我们如何评价现有成果和规划后续实验，但本报告不提出新架构，也未对现有代码做修改。

## 14. 最终判断

TriGround 已经提供一个有竞争力的纯生成式三模态 baseline（基线），尤其方案二的比赛结果值得作为下一阶段讨论的起点。其真正有价值的部分不仅是联合融合器，还包括：独立模态预训练、RGB-safe 初始化、冻结主干的小参数训练、人工审核目标域数据、场景级防泄漏以及一套可恢复的比赛推理工具。

但目前最需要做的不是立刻叠加新模块，而是先把**比赛结果—checkpoint—数据划分—消融证据**四者对齐。只要这条证据链补齐，我们就能准确判断 0.6785 的提升来自独立 adaptor、目标域数据处理、最终联合门控，还是这些因素的组合，也能避免被 `combined284` 的泄漏和 mIoU 选优口径误导。
