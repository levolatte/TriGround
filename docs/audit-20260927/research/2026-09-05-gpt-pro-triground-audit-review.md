# GPT Pro 对 TriGround 审计的复核报告

> 日期：2026-09-05  
> 被复核材料：`pasted-text.txt`（网页版 GPT Pro 对 TriGround 的系统性审计）  
> 对照仓库：`F:\AIC\code`  
> 分支：`public-v1`  
> 提交：`ad45af93a61339bd96f7c9a809f0858b0c8cc7e5`  
> 复核性质：只读分析；未修改 TriGround 源码、配置或 Git 状态，也未开始架构设计

## 1. 一句话结论

这是一份**质量明显高于一般代码点评、可以作为讨论底稿，但不能直接照单实施**的审计。

它对仓库现状的主要技术事实判断大多准确，尤其抓住了：

1. 训练和 checkpoint（检查点）选择使用 `mean_iou`，与比赛唯一主指标 `ACC@0.5` 不一致；
2. 发布模型仍以坐标字符串的自回归交叉熵为主要定位监督；
3. Parallel 辅助 Query 编码器没有位置编码；
4. 发布配置只在最后一个视觉层边界做一次联合融合，并冻结 IR/Depth adaptor（适配器）；
5. 数据、验证、推理和发布文档仍有明显欠账。

但它有一个贯穿全文的问题：**经常把“代码事实”“合理研究假设”“论文在别的任务上的证据”“已经由本仓库实验验证的结论”混在一起。** 因此，它适合用于生成实验假设，不适合直接当作下一版架构规格书。

如果必须打一个分数：

| 维度 | 评价 |
| --- | --- |
| 代码事实核对 | 8.5/10，整体很准 |
| 赛题理解 | 9/10，与官方规则基本一致 |
| 论文引用 | 7/10，方向相关，但有跨任务外推和个别引文漂移 |
| 因果判断 | 6/10，缺少仓库原始预测、严格消融和误差分析支撑 |
| 实施优先级 | 7/10，前几项有价值，但跳过了验证与诊断地基 |
| 综合 | **约 7.5/10：值得采纳其问题清单，不应原样采纳其结论强度与实施顺序** |

## 2. 我复核了什么

本轮重新核对了：

- 官方 9 页赛题 PDF 的任务、指标、鲁棒性、输出合法性和数据使用要求；
- 附文全部 929 行内容及 5 个外部引用；
- 当前 `public-v1` 的 README、发布配置、模型实现、训练引擎、数据处理、checkpoint、诊断和人工审核工具；
- Git 历史中已经删除的 Direct BBox Head（直接框回归头）实验配置与预检脚本；
- 仓库公开的线上成绩、`combined284` 指标和历史 1,152 条验证记录；
- Grounding DINO、RGBT-Ground、RGBDT500/RDTTrack、Qwen3-VL 与 RoboRefIt 的一手材料。

官方规则明确：输入为空间对齐的 RGB、红外、深度和英文 Query，输出 RGB 坐标系中的归一化单框；唯一主指标为 `ACC@0.5`；反向、越界、NaN 和空框直接无效。官方还明确要求在模糊 RGB、噪声深度和低对比红外等情况下保持稳定。因此，附文围绕“指标对齐、模态质量和框合法性”展开是正确的。参见[官方赛题 PDF](../../contest/基于大模型的多模态视觉理解与推理.pdf)。

## 3. 先看仓库真实结果：附文哪些结论能被成绩支持

### 3.1 公开结果汇总

| 模型/路线 | 比赛 ACC@0.5 | `combined284` mIoU | `combined284` ACC@0.5 | 折算正确数 | `combined284` ACC@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB 基线 | 未公开 | 0.5848 | 67.25% | 191/284 | 53.87% |
| RDT 弱监督路线 | 0.6404 | **0.5952** | **69.01%** | **196/284** | 54.93% |
| Parallel adaptor 路线 | **0.6785** | 0.5882 | 67.96% | 193/284 | **55.28%** |

这些数字给出四个很重要的结论：

1. **线上只证明 Parallel 整条流水线比 RDT 整条流水线高 3.81 个百分点。** 两条路线同时改变了结构、数据、训练顺序和 checkpoint，不能把差值单独归因于“独立 adaptor 预训练”。
2. **线上结果没有证明三模态优于 RGB。** 仓库没有同一比赛测试上的 RGB-only 分数；所以附文“线上结果已经证明三模态有作用”的措辞过强。
3. **本地与线上排序反转。** `combined284` 上 RDT 的 mIoU 和 ACC@0.5 都高于 Parallel，线上却相反。这强烈支持附文关于验证集不足、域偏移和模型选择问题的担忧。
4. **本地优势很小。** Parallel 对 RGB 的 ACC@0.5 只多 2 个样本，RDT 只多 5 个样本；而 `combined284` 的 `new154` 又与弱监督训练数据重叠，不能据此作强泛化或因果结论。

历史 1,152 条验证中，RDT 的 ACC@0.5 是 719/1152，RGB 是 715/1152，仅多 4 条，同时 mIoU 和 ACC@0.7 还更低。这进一步说明“辅助模态有效”目前是**有迹象、未被干净消融证明**。

此外，比赛分数目前只见于 [README](../../code/README.md)，公开分支没有排行榜回执、原始提交预测或对应评测日志。本轮可以把它们视为队友报告结果，但无法独立复算。

## 4. 附文判断最准确的部分

### 4.1 P0-1：模型选择指标错位——正确，而且是最确定的问题

[训练引擎](../../code/src/mm_grounding/engine.py)中的恢复基线、Early Probe（早期探测）、最优保存、Early Stopping（早停）和 checkpoint 分数都使用 `mean_iou`；[选择工具](../../code/tools/select_fusion_checkpoint.py)与[组合结果排序工具](../../code/tools/rank_combined_results.py)也按 mIoU 排序。比赛则只计 `ACC@0.5`。

附文给出的两个 IoU 序列反例是成立的，`mean_iou` 最优不保证 `ACC@0.5` 最优。它提出以 `(acc_0.5, mean_iou, acc_0.7, parse_rate)` 排序，并同时保留 ACC 最优、mIoU 最优和最后一个 epoch，也比简单地把全部逻辑硬换成一个离散指标更稳妥。

需要补充的只有统计稳定性：当前独立验证集只有 119 条，单个样本就对应约 0.84 个百分点；若真实准确率约为 68%，普通二项近似的 95% 波动范围约为 ±8.4 个百分点，而且同一场景的多个 Query 还不是独立样本。因此，正确做法不是“只看 ACC”，而是：

- ACC@0.5 作为主选择目标；
- mIoU、ACC@0.7 和解析率作为并列/破平指标；
- 保存全部关键 checkpoint，使用同一组逐样本预测做配对比较；
- 按场景而不是随机 Query 做 bootstrap（自助抽样）或分组复核。

### 4.2 P0-3：Query 分支丢失词序——这是全文最漂亮的代码发现

[model.py](../../code/src/mm_grounding/model.py)直接从 Qwen 词嵌入表取 `query_input_ids` 的静态 embedding，并立即 `detach()`；[QueryTokenEncoder](../../code/src/mm_grounding/adapters.py)只有投影、`TransformerEncoder` 和归一化，没有任何位置编码。

无位置编码的自注意力对 Token 排列是置换等变的；后续把这些 Token 当作跨注意力 Key/Value 时，对同一组 Token 的排列基本不敏感。因此，“person left of car”与“car left of person”在**辅助模态融合分支**里确实可能近似等价。

附文同时正确强调：主 Qwen Prompt 仍保留词序，所以整个模型不是词袋；受损的是决定 IR/Depth 如何注入 RGB 的辅助 Query 路径。这个限定非常专业，建议保留原判断。

### 4.3 P0-4：发布配置只有一个晚期融合边界——事实正确

[发布配置](../../code/configs/stage2_joint_fusion_v2.yaml)确实是：

```yaml
parallel_fusion_stages: 1
parallel_joint_fusion: true
freeze_parallel_adapters: true
```

层索引算法保证单 Stage 落在最后一个视觉 Block。附文据此说“三流直到最后才发生跨模态交互”是对的。

但要精确表述：IR 和 Depth 并非直到最后都未经处理，它们在每个共享 Qwen 视觉层后都有自己的 adaptor；真正只发生一次的是**跨模态融合**。所以“太晚”是很合理的实验假设，不是已有结果。

### 4.4 P1/P2 的多项工程事实也准确

以下判断均能从代码直接确认：

- IR 先按灰度读取再复制成三通道；Depth 编为“对数距离、有效掩码、零通道”，二者都进入 Qwen 的 RGB 图像处理器和 Patch Embed；
- 弱 Query 主要由 bbox 几何生成左右、序数、图像区域和最近目标关系，确有明显模板捷径；
- [prepare_target_v2_data.py](../../code/tools/prepare_target_v2_data.py)会把“无 decision、无修正文案”的候选默认为 `valid`，并把 `hold` 放入训练集；
- 当前没有系统化的同步几何增强、模态退化训练、多尺度/滑窗推理或模型集成；
- 推理是 `do_sample=False` 的单次贪心生成，解析失败时提交工具默认回退整图框；
- README 推荐 Parallel，而模型注册表、模型卡和发布清单仍把 RDT 标为推荐；
- 稀疏 checkpoint 使用 `strict=False`，并对 `trainable_only` 文件忽略全部 missing keys；
- 模型通过 Qwen 视觉模块的内部属性和前向挂接实现融合，而 `pyproject.toml` 允许 `transformers>=4.57.3,<5`。

其中人工审核问题需要加限定：另一个通用工具 [apply_grounding_reviews.py](../../code/tools/apply_grounding_reviews.py)要求评审覆盖完整，未分类样本不会默认保留。因此这不是“所有人工审核脚本”的统一行为，而是 `prepare_target_v2_data.py` 这条流水线的具体风险。

## 5. 附文合理、但说得过于确定的部分

### 5.1 P0-2：Direct BBox Head 值得优先试，但尚不是被验证的修复

附文对当前状态的描述准确：

- 所有发布配置都关闭 `auxiliary_bbox_enabled`；
- [model.py](../../code/src/mm_grounding/model.py)已经实现 `cx,cy,w,h` 的 Sigmoid 直接回归、SmoothL1 与 GIoU；
- `coordinate_token_loss` 只是被记录，没有额外加入总损失；坐标 Token 仍只作为完整语言模型 CE 的一部分；
- [config.py](../../code/src/mm_grounding/config.py)把辅助框头限制在旧的 `safe_post_embed` 路线，Parallel/RDT 不能直接打开配置使用。

但 Git 历史还揭示了附文没有写出的事实：仓库曾有 `multimodal_safe_v3_bbox.yaml`，以 `L1=2.0`、`GIoU=1.0` 启用直接框头，并有 [preflight_safe_v3_bbox.py](../../code/tools/preflight_safe_v3_bbox.py)检查梯度隔离和输出有限性；公开历史里却没有对应训练日志、验证结果或 checkpoint。换言之，它是**实现过并预检过的实验设想，不是成功实验**。

因此，更严谨的结论应是：

> 连续几何监督高度合理，应作为优先 A/B 实验；但不能在没有结果时断言它必然优于生成框，更不能默认“直接框为主、生成框兜底”。

还要注意：当前直接头只读取“答案开始前一个位置”的单个最终语言隐藏状态；它是否保留足够精细的空间信息未知。同时运行生成框和直接框还可能增加一次完整前向或要求重构推理接口。RDTTrack 使用 L1/GIoU 能证明这类损失适合连续定位，但 RDTTrack 是给定首帧目标的**视频跟踪**，不是语言指代定位，不能直接充当本赛题效果保证。[RDTTrack 一手论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf)确实冻结 RGB 主干并使用 Focal、L1、GIoU，但迁移到 TriGround 仍须实验。

### 5.2 “1→2→4 个融合 Stage、解冻最后 adaptor”是消融，不是定论

增加融合边界和低学习率解冻最后 2～4 个 adaptor 有充分直觉，也基本已有配置支持。不过：

- 早期跨模态交互可能同时带来辅助模态噪声污染；
- 多个新 Stage 会增加显存、训练难度和所需步数；
- 当前最终 Joint Fusion（联合融合）是零初始化恢复层，训练只有 2 个 epoch；
- `freeze_parallel_adapters: true` 会覆盖 adaptor 的训练开关，不能只添加 `parallel_adapter_train_last_n` 而不调整冻结逻辑/配置。

[RGBT-Ground](https://arxiv.org/html/2512.24561v1)强力支持“RGB 预训练特征与热红外存在域差异、需非对称适配和语言引导融合”的方向，并在自己的 RGB-T 数据上做了消融；但它没有 Depth，也不是 TriGround 的 Qwen 三流实现。因此它支撑“值得试”，不支撑“2～4 层必然更好”。

### 5.3 模态专属 Stem、质量估计、配准修复方向正确，但应先做轻量诊断

“IR/Depth 被当作特殊 RGB”是公平批评，但附文略低估了现有 per-layer adaptor 已经承担的域适配作用，也低估了“只给辅助流加 LoRA”的实现成本：三流共享同一组 Qwen 视觉 Block 参数，普通 LoRA 会同时影响 RGB，若要仅作用于某一流，需要条件化或分流实现。

外部数据域差异确实存在：

- [RoboRefIt 官方资料](https://luyh20.github.io/RoboRefIt.github.io/)说明其数据来自杂乱日常室内/机器人抓取场景，并使用机器人指令风格 Query，与赛题多场景分布不同；
- RGBT-Ground 自己也承认数据包含 aligned/weakly aligned（对齐/弱对齐）图像对，并主动过滤严重错位样本。

因此先做配准质量分数、坏样本过滤和分层指标是扎实建议；直接上跨光谱光流、Homography（单应变换）或可变形注意力则可能过重，而且 RGB/热红外外观差异会让普通光流本身不可靠。

### 5.4 数据增强和推理增强不能全部当成免费收益

附文建议的同步裁剪、缩放和模态退化有价值，但有几项需谨慎：

- 水平翻转必须可靠改写 `left/right`、序数和关系对象，否则制造错标签；
- 单独 jitter（抖动）真值框会让框偏离真实物体，不是天然的正增强；
- 滑窗可能丢掉 Query 所需的全局参照物；
- 坐标平均可能把两个各自合理的候选平均成错误区域；
- 集成需要可信置信度，而当前生成式框没有校准好的框置信度。

所以这些是待测工具箱，不应整体列为固定主线。

## 6. 附文中证据不足或明显过度推断的地方

### 6.1 “当前最影响排名的主因就是四个结构错位”无法由现有证据推出

仓库没有：

- 完整排行榜位置和多次提交轨迹；
- 对应线上提交的逐样本预测；
- 官方测试标签；
- 按 Query 类型、目标大小、模态质量的错误分析；
- 针对四个 P0 的受控消融。

因此我们只能说四点都是高价值嫌疑，不能说它们已经被证明是当前排名的主要瓶颈。真正瓶颈也可能是目标域训练样本太少、弱标签错误、基础视觉分辨率、测试域偏移或候选目标识别失败。

### 6.2 “线上结果证明独立 adaptor 路线有效”是流水线级结论，不是组件级结论

0.6785 > 0.6404 能证明某次 Parallel 提交优于某次 RDT 提交；不能单独证明：

- IR 有贡献；
- Depth 有贡献；
- 独立预训练是增益来源；
- Joint Fusion 是增益来源；
- 三模态优于同一 Qwen RGB 基线。

这些都需要同 checkpoint、同数据、同推理协议下的 RGB、RGB+IR、RGB+Depth、RGB+IR+Depth、错配模态和随机/零模态配对结果。

### 6.3 候选生成 + 重排序路线有潜力，但附文忽略了“召回上限”

[Grounding DINO](https://arxiv.org/html/2303.05499)确实是语言条件开放集检测器，包含语言引导 Query 选择和跨模态解码器，也在 RefCOCO/+/g 上评估；所以拿它做候选生成是合理研究方向。

但其论文也明确指出：不使用 REC（指代表达理解）数据时，Grounding DINO 和 GLIP 在 REC 上表现并不好，加入 RefCOCO 类数据后才显著提升。对本赛题还有一个更直接的风险：附文建议只从 RGB 生成 Top-K，而赛题恰恰包含模糊/弱光 RGB；若真值目标不在 Top-K，后面的 Qwen 和三模态重排序永远无法恢复。

因此，在讨论整套候选架构前，必须先测：

```text
candidate recall@1 / @5 / @10 / @20
按小目标、弱光、遮挡、关系型 Query 分组的 recall@K
```

只有候选召回显著高于当前 ACC 上限，这条路线才值得成为主线。否则它最多是补充分支。

### 6.4 Qwen3-VL 引用没有证明“小 VLM 在复杂 Grounding 中容易失败”

[Qwen3-VL 技术报告](https://arxiv.org/html/2511.21631)确实证明存在 2B/4B/8B/32B 等版本，也展示更大模型在很多通用推理任务上更强，并包含 2D/3D Grounding 评测。但该报告没有得出附文所写的那条具体结论——“小型 VLM 在多个相似候选、复杂关系描述和长推理指令下更容易失败”；相反，报告多次强调小模型也具有竞争力。

所以“4B/8B 可能更强”是合理常识与待测假设，附文的引文却不足以支撑那句强表述。

### 6.5 Checkpoint 哈希建议不应照搬

附文指出稀疏 checkpoint 可能静默漏载新模块，这个风险成立；建议对预期项目参数做白名单匹配、记录实际加载和缺失键，也很实用。

但它进一步建议普遍增加 `model_config_hash`、`dataset_manifest_hash` 等哈希机制，既超出当前竞赛收益，也与本项目 `AGENTS.md` 明确的“非系统重要边界不主动增加哈希机制”相冲突。更合适的是保存可读配置、仓库提交、主干 ID/版本、模块名称和 expected/loaded keys；不扩张成生产级完整性系统。

## 7. 附文漏掉的关键问题

### 7.1 最终 Joint 模型的模态 scale sweep 实际无效

[diagnose_modality_interventions.py](../../code/tools/diagnose_modality_interventions.py)中的 `temporary_modality_scales()` 只修改 `model.fusion.stage_fusions` 里的旧双分支 `residual_scale`；最终配置启用的却是 `joint_stage_fusions`。实际前向在 `parallel_joint_fusion: true` 时选择后者。

结果是：

- RGB-only 模式仍有效；
- 换错 IR/Depth 图的 mismatch（错配）实验仍有效；
- `ir_scale_0/0.25/0.5/1.0`、`depth_scale_*` 和三模态 scale sweep 对最终模型不起预期作用；所谓 scale=0 仍可能使用完整联合融合。

测试文件只构造了旧 `stage_fusions` 假模型，因此没有发现这个盲区。附文称赞“已有模态消融和错配诊断”，却没有核对其是否真正作用于发布模型。这是本轮发现的最重要遗漏。

### 7.2 Parallel 发布权重来自 `last_phase_a.pt`，不是 `best_phase_a.pt`

[MODEL_REGISTRY.md](../../code/MODEL_REGISTRY.md)明确记录 `triground-parallel-a-v1.pt` 来自 `runs/stage2_joint_fusion_v2/last_phase_a.pt`。这意味着附文围绕“mIoU 选错最佳 checkpoint”的批评虽然对训练系统成立，却不能自动解释 0.6785 的发布权重选择；首先要查明为什么发布的是 last、last 与 best 的各项指标分别是多少。

### 7.3 最终 Joint Fusion 没有使用已经实现的 legacy warm start

代码有 `warm_start_joint_fusion_from_legacy`，可以把旧独立融合的投影权重平均迁移到联合融合；最终配置没有开启。它只加载旧 adaptor、Query encoder 和 legacy `stage_fusions`，随后冻结 legacy 分支，新的 `joint_stage_fusions` 以零恢复层开始短短 2 个 epoch 训练。

这不是必然错误，甚至可能是为了保护 RGB 的有意选择；但在判断“融合层数不够”前，应先确认现有单层联合融合是否已经得到充分训练。

### 7.4 依赖版本问题比附文说的稍轻，但仍存在

`pyproject.toml` 的确放宽到 `transformers>=4.57.3,<5`，真实 Qwen 内部挂接容易被小版本改变；但 `requirements-experiment.txt` 已精确固定 `transformers==4.57.3`。所以这是“默认安装入口与实验锁定方式不一致”，不是完全没有版本固定。

### 7.5 仓库证据链不足比架构美丑更优先

当前公开分支没有发布权重本体、数据、训练指标、原始验证预测和排行榜回执。README、模型卡和注册表又互相冲突。没有这些材料，任何新结构即使涨分，也很难知道究竟是哪一项改变生效。

## 8. 我建议如何使用这份审计

不把它当成“下一版架构设计”，而把它改造成一个按证据推进的实验清单。

### 第一层：先修测量与证据，不改模型主体

1. 固定一个真正 scene-safe（场景隔离）的干净主验证集；
2. checkpoint 以 ACC@0.5 为主、mIoU/ACC@0.7/解析率破平，并保留 ACC 最优、mIoU 最优和 last；
3. 修正最终 Joint Fusion 的 scale 干预，再跑 RGB、单模态、三模态、错配和缩放诊断；
4. 保存逐样本预测，按 Query 类型、目标大小和场景分组；
5. 对齐 README、模型注册表、模型卡和发布来源。

这是当前最高优先级。没有它们，后面的结构实验会继续被 119 条小验证集和失效诊断误导。

### 第二层：做彼此独立的小实验，不一次堆成“大主线”

建议分别验证：

1. Query 位置编码：旧版 vs 加位置；
2. Fusion Stage：1 vs 2，确认有收益后再试 4；
3. adaptor：全冻结 vs 只训练最后 1～2 层；
4. 输出监督：仅生成 CE vs 生成 CE + Direct Head；
5. 推理：generated vs direct vs 简单校准融合。

每次只改变一项，报告 ACC@0.5、mIoU、解析率、按场景的配对差异和计算成本。这样才能知道 GPT Pro 的哪条假设成立。

### 第三层：再决定是否进入较大路线

- 模态专属 Stem、质量估计和复杂配准模块；
- Grounding DINO/MDETR 候选生成 + 三模态重排序；
- 更大 Qwen 或专门 Grounding 训练。

这几项都可能有较高上限，但它们不应掩盖当前最便宜、最确定的评测和诊断问题。

## 9. 最终复核结论

我赞同附文的核心精神：**TriGround 不是“做坏了”，而是已经得到一个完整、认真、能参赛的 baseline；下一步不该盲目延长同一训练，而要增强指标对齐、语言关系表示、几何监督和模态适配。**

但我会把它的最终结论改写为：

> 当前最确定的问题是验证、模型选择和诊断链路不够可靠；Query 辅助分支缺位置编码则是最明确的结构缺陷。Direct BBox Head、多阶段融合、adaptor 解冻、专属 Stem 和候选重排序都是高价值假设，但尚未由本仓库实验验证。应先建立可信评测与逐项消融，再讨论哪一项进入正式架构。

因此，这份 GPT Pro 审计的正确用法是：

- **采纳问题清单的大部分内容；**
- **降低其中因果措辞的确定性；**
- **把“直接实施的新主线”拆成单变量实验；**
- **在架构讨论前先补验证、诊断和证据链。**

本轮仅完成复核与报告，没有修改队友仓库，也没有为下一版模型作架构定案。

