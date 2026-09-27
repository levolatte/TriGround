# 2026-09-22 当前仓库与 8B 证据审计

## 范围、方法与结论边界

本审计在 2026-09-22 对 `F:\AIC\code` 做静态、只读检查，并读取现有本地实验材料和数据压缩包；**没有启动训练、评估、推理，也没有访问任何云端服务**。赛题规则以 `F:\AIC\contest\基于大模型的多模态视觉理解与推理.pdf` 为准。代码仓库当前 `main` 的提交为 `b184384eee7d7a8f4e23efd4dc1de8827d224108`（`Add budgeted 8B trimodal experiments and training fixes`）。工作区已有未提交的 `HANDOFF.md` 与 `tools/diagnose_8b_fusion.py` 修改；本报告没有改动它们。

赛题要求 RGB、红外（IR）、深度与英文 Query（查询语句）共同输入，输出 RGB 坐标系中归一化的唯一目标框；主指标为 `ACC@0.5`（IoU 不低于 0.5 的准确率）。官方明确称三种图像在空间上对齐，但未提供相机标定参数或对象级跨模态对应标签（竞赛 PDF 第 1、3--4 页；交叉核对见 [2026-09-18-contest-task-crosscheck.md](F:\AIC\docs\research\2026-09-18-contest-task-crosscheck.md:5)）。

本次最重要的结论如下。

1. 当前 8B 路径确实保留了 Qwen3-VL 的原生视觉 `merger`（合并器）和 DeepStack（深层堆叠）输入位置，但主干权重在现有阶段完全冻结。可训练的是融合模块、两套辅助模态适配器和 Query 编码器；而当前 8B 配置又冻结了辅助模态适配器。因此，已经测到的是“冻结 Qwen 主干下，融合/Query 小模块的筛选结果”，不是视觉主干微调结果。
2. 现有 `rgb_only` 评估**不会**在额外 Query 融合支路中混入信息：模型在建立融合上下文前直接返回，且原生 Qwen 一直通过主对话提示词接收 Query。它是同一冻结 RGB 主干的推理消融，不能代替“投入同等可训练参数、专门训练过的 RGB-only（仅 RGB）控制组”。
3. 已有 8B 数据支持“多模态融合路径会改变预测，且某些单次筛选优于 RGB 基线”；不足以支持“提升稳定”“已利用正确跨模态对象对应”“正弦 Query 位置编码有效”“warm start（热启动）有效”或“冻结主干就是性能受限根因”。尤其不能由旧消融直接推断最后一项。
4. target_v2 的标注数据只可在**同一 visible 图、对应 IR/Depth 路径一致、同类且不同 bbox**的前提下，构造受限的已知目标框候选负例；它不含全图对象清单、实例 ID、掩码、相机标定或显式跨模态匹配标注，不能据此编造未查询对象的负例或像素/对象级 RGB--IR--Depth 匹配标签。
5. 本地证据目录只有弱监督及其对照的预生成配置，没有已完成的弱监督指标、结果、日志或输出目录。它只能说明当时准备/开始了任务，不能在本地证实停止后的结果。

## 当前 8B 结构与冻结状态

### 原生 Qwen 入口、并行三流和 merger

`MultiModalGrounder` 当前默认的 8B 融合类型是 `parallel_backbone`（并行主干），并构建 `ParallelBackboneFusion`；它还检查融合位置是否覆盖原生 DeepStack 位置和最终第 26 层：[model.py](F:\AIC\code\src\mm_grounding\model.py:296)--[model.py](F:\AIC\code\src\mm_grounding\model.py:405)。

在补丁后的视觉前向中，RGB、IR 和 Depth（深度）各自经过**同一个** Qwen 视觉 `patch_embed`（图像分块嵌入）和同一位置嵌入；随后每个冻结视觉块分别跑三次。IR/Depth 在每层之后经过自己的可训练 `ModalityBackboneAdapter`（模态主干适配器），到指定层时同 RGB 融合：[model.py](F:\AIC\code\src\mm_grounding\model.py:136)--[model.py](F:\AIC\code\src\mm_grounding\model.py:229)。融合后的 RGB token（词元）继续走 Qwen 原生 `deepstack_merger_list`，最后走原生 `vision.merger`；IR 和 Depth 流不单独送入 merger。这是“用融合后的 RGB 流接回原生 VLM（视觉语言模型）接口”，不是重写 merger。

`ParallelBackboneFusion` 有独立的 IR/Depth 适配器、两个 Query 编码器、旧版分阶段融合与当前联合融合模块；真正被调用的是 `joint_stage_fusions`：[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:868)--[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:1068)。联合模块在同一空间 token 位置上用 Query 注意力和模态注意力做加权，初始可靠性门偏向关闭：[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:601)--[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:865)。它不是显式的跨空间对象匹配器，因而不能从结构本身推出已经学到 RGB/IR/Depth 的实例对应。

### Query 的两条路径及 RGB-only 是否混淆

主 Qwen 对话提示词始终含原始 Query。数据整理器还把 Query 单独分词为 `query_input_ids`：[data.py](F:\AIC\code\src\mm_grounding\data.py:166)--[data.py](F:\AIC\code\src\mm_grounding\data.py:224)。在三模态模式，这个单独序列经两个小型 `QueryTokenEncoder` 编码；其输入是冻结的 Qwen 词嵌入，随后是 128 维投影和一层 Transformer，并非另一个语言模型：[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:364)--[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:427)。

关键的是，`rgb_only=True` 时 `_set_fusion_context` 在读取辅助图像或调用两个 Query 编码器之前就返回：[model.py](F:\AIC\code\src\mm_grounding\model.py:520)--[model.py](F:\AIC\code\src\mm_grounding\model.py:602)。虽然评估输入字典仍携带 `query_input_ids`（[engine.py](F:\AIC\code\src\mm_grounding\engine.py:147)--[engine.py](F:\AIC\code\src\mm_grounding\engine.py:171)），`_qwen_inputs` 不会将它传入原生 Qwen：[model.py](F:\AIC\code\src\mm_grounding\model.py:604)--[model.py](F:\AIC\code\src\mm_grounding\model.py:624)。因此：

| 问题 | 静态证据 | 可以下的结论 | 不能下的结论 |
|---|---|---|---|
| RGB-only 是否额外使用融合 Query 支路 | `rgb_only` 直接返回 | 否；只有主提示词 Query 进入原生 Qwen | 不能称它为等参数、经训练的 RGB-only 模型 |
| 多模态是否同时改变多项因素 | 三模态打开 IR、Depth、Query 编码器、联合融合及其门控 | 是，结果差异是完整融合路径的效果 | 不能把三模态与 RGB 的差异分配给任一传感器、Query 编码器或门控 |
| 是否保留原生 VLM 推理入口 | 最终调用 `self.backbone.generate` | 是 | 不代表原生主干参数参与训练 |

`evaluate.py` 的联合评估会依次跑三模态、RGB+IR、RGB+Depth 和 RGB；RGB 分支传入 `rgb_only=True`：[evaluate.py](F:\AIC\code\evaluate.py:171)--[evaluate.py](F:\AIC\code\evaluate.py:206)。脚本中的 native RGB 也是 `evaluate.py --config native_rgb.yaml --rgb-only`：[run_8b_experiment.sh](F:\AIC\code\scripts\run_8b_experiment.sh:98)--[run_8b_experiment.sh](F:\AIC\code\scripts\run_8b_experiment.sh:100)，不是另一套已训练 RGB 网络。证据中 native 行没有 checkpoint，`native_rgb.json` 为 76/119、mIoU 0.542532；各训练 checkpoint 的 RGB 消融也报出相同的 76/119 与 0.542532（[native_rgb.json](F:\AIC\docs\research\8b-run-20260918-evidence\results\native_rgb.json:1)，[summary.md](F:\AIC\docs\research\8b-run-20260918-evidence\results\summary.md:5)）。这符合上述实现，但只验证推理开关，不验证已适配 RGB 对照。

### Freeze（冻结）与 LoRA（低秩适配）

阶段 A 先将全部参数设为不可训练，再仅打开融合、Query 编码器和可选的模态适配器；当前 `freeze_parallel_adapters=true` 会再次关闭 IR/Depth 适配器：[model.py](F:\AIC\code\src\mm_grounding\model.py:434)--[model.py](F:\AIC\code\src\mm_grounding\model.py:473)。注意 `parallel_adapter_train_last_n` 只影响辅助流适配器，不会解冻 Qwen 视觉块：[model.py](F:\AIC\code\src\mm_grounding\model.py:452)--[model.py](F:\AIC\code\src\mm_grounding\model.py:459)。C4/T4 的 8B 配置明确设置了该冻结项，阶段 B 为 0，且 `vision_lora=false`：[c4.yaml](F:\AIC\docs\research\8b-run-20260918-evidence\configs\c4.yaml:23)--[c4.yaml](F:\AIC\docs\research\8b-run-20260918-evidence\configs\c4.yaml:42)。

仓库已有对 Qwen 视觉线性层插入 LoRA 的实现和单元测试：[lora.py](F:\AIC\code\src\mm_grounding\lora.py:10)--[lora.py](F:\AIC\code\src\mm_grounding\lora.py:103)，[test_lora.py](F:\AIC\code\tests\test_lora.py:32)--[test_lora.py](F:\AIC\code\tests\test_lora.py:43)。然而配置验证显式禁止在 `rdt_deep` 和 `parallel_backbone` 启用视觉 LoRA：[config.py](F:\AIC\code\src\mm_grounding\config.py:127)--[config.py](F:\AIC\code\src\mm_grounding\config.py:128)，当前实验也全部关闭。故“LoRA 已有代码”不能写成“8B 已做视觉 LoRA 对照”。

同理，目前没有以下受控实验：同一配置只解冻少量 Qwen 视觉层、只开启视觉 LoRA，或用等量训练参数训练 RGB-only 对照。旧版的“最后若干层”若仅指 `parallel_adapter_train_last_n`，并不改变 Qwen 主干。因此，**冻结主干是根因**是待检验假设，不是现有旧消融可推出的结论。

## 8B 结果：已有对照能证明什么

8B 证据的配置生成器自身注明 C4 与 T4 同时改变了 `query_position_encoding`（Query 位置编码）与 `warm_start`：[prepare_8b_experiment.py](F:\AIC\code\tools\prepare_8b_experiment.py:305)--[prepare_8b_experiment.py](F:\AIC\code\tools\prepare_8b_experiment.py:348)。热启动仅从旧模块复制 RGB/IR/Depth 投影和语言注意力的一部分，不是对联合模块的完整复制：[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:991)--[adapters.py](F:\AIC\code\src\mm_grounding\adapters.py:1012)。

| 可比较项 | 119 条验证集结果 | 可靠结论 | 不可靠结论 |
|---|---:|---|---|
| native RGB | 76/119，63.87% | 该冻结原生 RGB 推理点的本地基线 | 不是经同预算训练的 RGB 对照 |
| C4（seed 2026）三模态 | 73/119，61.34% | 该筛选点低于 RGB | 不能代表 C4 总体能力 |
| T4（seed 2026）三模态 | 79/119，66.39% | 此次完整配置高于 RGB 3 命中 | 不可归因给位置编码或热启动单项 |
| W4（seed 2026）三模态 | 79/119，66.39% | 在该 seed 与 T4 相同 | 不能证明位置编码无效，只是缺少稳定证据 |
| C4（seed 2027）三模态 | 82/119，68.91% | 结果会随 seed 大幅波动 | 不能以单 seed 排序配置 |
| T4（seed 2027）三模态 | 74/119，62.18% | C4/T4 相对关系发生反转 | 不能以 seed 2026 的正差作为最终选择依据 |
| W4（seed 2027）三模态 | 76/119，63.87% | warm-start 与位置的交互仍不清楚 | 不能断言热启动稳定有效 |

以上汇总见 [summary.md](F:\AIC\docs\research\8b-run-20260918-evidence\results\summary.md:5)--[summary.md](F:\AIC\docs\research\8b-run-20260918-evidence\results\summary.md:29)。`decision.json` 的同 seed C4→T4 配对差为 +6 命中，bootstrap（自助法）区间为正；但第二个 seed 为 -8 命中且区间为负：[decision.json](F:\AIC\docs\research\8b-run-20260918-evidence\results\decision.json:1)。这足以否定“单次正向筛选已证明稳定机制”的表述。

八样本诊断记录到非零融合增量与门控值，说明残差融合代码在执行（例如 [t4_diagnostics.json](F:\AIC\docs\research\8b-run-20260918-evidence\results\t4_diagnostics.json:1)）。但这些记录没有 Query 置零/打乱比较、模态错配比较或跨模态实例匹配真值，不能证明传感器信息被正确因果使用。仓库已有此类推理诊断入口：`query_correct/query_zero/query_shuffled` 和模态缩放/错配模式定义于 [diagnose_modality_interventions.py](F:\AIC\code\tools\diagnose_modality_interventions.py:56)--[diagnose_modality_interventions.py](F:\AIC\code\tools\diagnose_modality_interventions.py:97)，其解释边界也写在 [QUERY_FUSION_CAUSAL_DIAGNOSTIC.md](F:\AIC\code\docs\QUERY_FUSION_CAUSAL_DIAGNOSTIC.md:1)--[QUERY_FUSION_CAUSAL_DIAGNOSTIC.md](F:\AIC\code\docs\QUERY_FUSION_CAUSAL_DIAGNOSTIC.md:44)。本次没有运行它；现有证据目录也没有这些诊断的结果文件。

## IR、Depth 表示与配准假设

官方定义 IR 为 3 通道、通道视觉一致且数值越大表示温度越高；Depth 为单通道 `uint16`、单位 mm、有效范围预期为 0.3--20m（竞赛 PDF 第 3 页）。当前数据集实现把 IR 以灰度 `L` 读取、双线性 resize 后复制为 RGB 三通道；这与官方“视觉一致三通道”的语义一致，但会忽略任何压缩导致的细微通道差异：[data.py](F:\AIC\code\src\mm_grounding\data.py:81)--[data.py](F:\AIC\code\src\mm_grounding\data.py:118)。

Depth 的编码为：取首通道，按 mm 除以 1000，保留 `(0,20]`，以 `log1p` 值写入第 0 通道，以有效掩码写入第 1 通道，第 2 通道置零：[data.py](F:\AIC\code\src\mm_grounding\data.py:17)--[data.py](F:\AIC\code\src\mm_grounding\data.py:29)。三种图各自经图像处理器，整理器只断言处理后的 `image_grid_thw` 相等：[data.py](F:\AIC\code\src\mm_grounding\data.py:211)--[data.py](F:\AIC\code\src\mm_grounding\data.py:223)。这依赖官方的全局空间对齐承诺；代码没有也无法从现有标注中验证逐图几何配准。

本地 `F:\AIC\data` 的文件清点发现：RGB 共有 1902 张 1920×1080 PNG、97 张 640×360 JPEG、1 张 1920×1080 JPEG；IR 同样有 1903 张 PNG 与 97 张 JPEG；Depth 为 1903 张 1920×1080 `I;16` PNG，另有 97 张 640×360 RGB JPEG。文件名集合三模态一致，而这 97 个非 `uint16` 深度文件被 177 个 Query 引用。对于 JPEG，当前编码会将 0--255 的像素当 mm 并转为 0--0.255m，全部落在官方预期下限 0.3m 以下。这是**深度编码格式未决问题**：现有代码没有证据支持这样解释 JPEG 深度；它也不是“模态未对齐”的证据。重构前应至少按格式单独报告/诊断这 177 条，明确原始 JPEG 的含义后再选择编码，不能凭推断悄然当作毫米深度。

## 可用数据、负例和匹配标签的边界

官方无标签数据 `F:\AIC\data\queries.json` 有 9,555 条 Query，字段只有 `visible/infrared/depth/query`；没有 bbox、类别、sequence、对象清单、实例 ID、掩码或标定。它覆盖 2,000 组三模态图像，其中 1,923 张有多个 Query（最多 14 条），但同图或同文本重复并不能推出同类别或同一对象。因此不能自动为官方无标签数据生产可靠同类负例或跨模态匹配标签。

本地 `F:\Downloads\target_v2.zip` 的有标注部分不同：923 条训练、119 条验证和 284 条保留标注均具有 `bbox`、`class_name`、`sequence_id`、`visible/infrared/depth`、Query 及复核相关字段；标签说明也记录了这一分割：[2026-09-18-target-v2-labels.md](F:\AIC\docs\research\2026-09-18-target-v2-labels.md:5)--[2026-09-18-target-v2-labels.md](F:\AIC\docs\research\2026-09-18-target-v2-labels.md:30)。`sequence_id` 只是场景/序列元数据，不能当作同图候选池：训练集有 249 个 sequence、269 条 visible 路径（32 个 sequence 含多条 visible 路径），验证集有 24 个 sequence、26 条 visible 路径。每个压缩包内 visible 路径均唯一映射到一对 IR/Depth 路径；按 `(visible, infrared, depth, class_name)` 而非 sequence 聚合，训练集有 241 个含不同 bbox 的同图同类组、覆盖 693 条记录（738 个不同框对）；验证集为 31 组、89 条、95 对；284 条保留集为 38 组、109 条、114 对。弱监督 992 条中没有这类同图同类多框组。

上述计数是**候选池规模**，不是可直接写入训练的负例数。候选只能从同一 `(visible, infrared, depth, class_name)` 组的不同 bbox 中挑选；同 bbox 的不同 Query 绝不可互作负例。候选生成还应排除高度重叠框（本次按 IoU≥0.5 清点，三个有标注分割的这些不同框对均为 0 个；这不代替逐项复核）、读取修订后的 `query`，并在发现同物体/标签冲突时剔除。`review_decision=query_corrected` 表示应使用修订后的 Query；它本身不保证跨记录不存在标签冲突。它不是完整场景标注：

| 目标 | 是否可由现有标注可靠得到 | 原因 |
|---|---|---|
| 同图同类、不同已知目标框候选负例 | 可条件构造，限 target_v2 有标注分割 | 需同一 `visible/infrared/depth` 路径、同 `class_name`、不同 bbox，且通过重叠/冲突筛除 |
| 未被 Query 标注的同类对象负例 | 不可以 | 无对象清单或实例标注 |
| RGB--IR--Depth 对象级匹配标签 | 不可以 | 没有跨模态对应 ID、掩码或标定 |
| 像素级配准监督 | 不可以 | 仅有官方对齐声明，缺少几何真值 |
| 官方无标签的同类负例 | 不可以 | 缺 `class_name/bbox/sequence` |

现有 `review_grounding.py` 会按可见图像和 `class_name` 标出同类复核风险：[review_grounding.py](F:\AIC\code\tools\review_grounding.py:108)--[review_grounding.py](F:\AIC\code\tools\review_grounding.py:129)；`select_scene_coverage_candidates.py` 也能复用其类别/空间桶元数据。这两者都只是人工复核/抽样工具，没有输出训练用负例，也没有候选框排序损失。当前训练整理器只持有单一 bbox 目标，故若要做候选框排序，必须新增明确的数据清单、候选区域表征和损失/评估入口，不能把现有复核脚本误称作已实现的训练功能。

## 弱监督的本地证据状态

`F:\AIC\code\HANDOFF.md` 的未提交末段记录了 2026-09-19 11:42 启动弱监督比较运行的意图与 PID，顺序为 `weak → weak_clean → control → control_clean`。证据目录中也有四份于 11:41 生成的配置：`weak.yaml`、`weak_clean.yaml`、`control.yaml`、`control_clean.yaml`。然而 `F:\AIC\docs\research\8b-run-20260918-evidence` 及 `F:\AIC\results` 下没有对应的运行目录、metrics、预测、日志或汇总结果。

所以本地可报告的状态只有“已有方案和启动记录，未发现完成结果”。由于本次禁止云端访问，无法判断远端任务是被停止、仍存在还是结果尚未同步；也不能据本地缺失断言训练从未执行。

## 9/22 复赛的可复用入口与审计约束

复赛主路线采用“新专家 → 原生 RGB LoRA → 三图”的顺序；本审计只给出可复用入口与旧路线的边界，不把旧并行融合重构或全套消融设为首轮前置条件。

1. **新专家与原生 RGB LoRA 阶段。** 当前原生 RGB baseline 已证实不进入旧 IR/Depth/额外 Query 融合支路，且主提示词仍保留 Query（[model.py](F:\AIC\code\src\mm_grounding\model.py:537)--[model.py](F:\AIC\code\src\mm_grounding\model.py:556)，[data.py](F:\AIC\code\src\mm_grounding\data.py:132)--[data.py](F:\AIC\code\src\mm_grounding\data.py:209)）。现有 LoRA 注入实现可复用，但 `parallel_backbone` 的 LoRA 禁令只约束旧并行路径，不能被误读为原生 RGB LoRA 不可行：[lora.py](F:\AIC\code\src\mm_grounding\lora.py:10)--[lora.py](F:\AIC\code\src\mm_grounding\lora.py:103)，[config.py](F:\AIC\code\src\mm_grounding\config.py:127)--[config.py](F:\AIC\code\src\mm_grounding\config.py:128)。
2. **三图阶段。** 保留现有融合后 RGB 回接 `deepstack_merger_list`/`vision.merger` 的事实和入口；它可以作为迁移时的参考，当前没有证据要求重写 merger。对 Depth 要先保留 JPEG 与 `I;16` 的数据切片，避免沿用未经证实的 JPEG 毫米解释：[data.py](F:\AIC\code\src\mm_grounding\data.py:17)--[data.py](F:\AIC\code\src\mm_grounding\data.py:29)。
3. **旧融合干预仅作补充证据。** [diagnose_modality_interventions.py](F:\AIC\code\tools\diagnose_modality_interventions.py:56)--[diagnose_modality_interventions.py](F:\AIC\code\tools\diagnose_modality_interventions.py:97) 的 Query 置零/打乱、模态缩放和错配可用于解释仍要复用的旧 checkpoint；它不应阻塞首轮新路线，也不能单独证明新结构优越。
4. **同图同类候选监督须单独实现。** 如后续需要，从 target_v2 生成 `(query, positive_bbox, distinct_same_image_same_class_negative_bboxes)` 清单，并落实重叠/冲突筛选。当前没有候选区域排序训练入口，`auxiliary_bbox` 也不能直接开启：它只支持 `safe_post_embed`，不支持旧并行融合路径：[config.py](F:\AIC\code\src\mm_grounding\config.py:174)--[config.py](F:\AIC\code\src\mm_grounding\config.py:175)。

首轮实验可以按主路线的最小验证集与预算执行；当需要声明可复现增益或比较多模态机制时，再补充固定划分、重复 seed、RGB 与各模态组合以及配对统计。
