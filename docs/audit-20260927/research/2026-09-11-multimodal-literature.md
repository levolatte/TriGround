# 多模态视觉指代定位文献与源码审计

日期：2026-09-11。范围：RGB、热红外（IR/TIR）和深度图结合英文 Query 的视觉指代定位（Visual Grounding / Referring Expression Comprehension）。本文只提出可证伪的实验假设，不把其他数据集上的论文成绩外推为本赛题成绩。

## 1. 结论与行动顺序

当前 TriGround 的主线是合理的：冻结 Qwen3-VL 主干，分别适配 IR、深度，再在视觉层内进行 Query 条件融合。它已经覆盖了三个最必要的对照路径：RGB、单辅助模态、IR+深度联合。现阶段最有价值的工作不是替换成一套新的论文架构，而是先回答四个尚未被数据证明的问题：

1. IR 和深度在本赛题中分别帮助哪些 Query，而非只看总 ACC@0.5；
2. 训练时的随机模态丢弃能否抵御实际的模态缺失，且不会削弱三模态正常输入；
3. 对齐误差是否会让当前逐位置融合产生负迁移；
4. 深度的米制与无效像素是否真的被“近/远、前/后、最近/最远、序数”等语言约束利用。

应以小规模、分组隔离的人工复核集先完成下列最小实验矩阵，并且只在有明确增益后改模型：

| 优先级 | 假设 | 最小比较 | 通过标准 |
| --- | --- | --- | --- |
| P0 | 辅助模态有净增益 | RGB；RGB+IR；RGB+Depth；RGB+IR+Depth | 在同一不参与训练的复核验证集上，联合路径的 ACC@0.5 高于 RGB，且不只来自少量样本 |
| P0 | 增益与 Query 语义有关 | 按外观、颜色、弱光、关系/距离、序数、其他分桶报告 | 每个桶同时给样本数、ACC@0.5、平均 IoU、相对 RGB 的差值；样本很少的桶只作定性线索 |
| P0 | 当前融合没有损害正常输入 | 对联合权重将 IR 或 Depth 推理尺度设为 0，并与原生 RGB 基线比较 | “联合权重的 rgb_only”不应明显低于无训练 RGB 基线；否则先查融合残差/训练泄漏 |
| P1 | 随机缺失与错位是不同问题 | 真实三模态、缺 IR、缺深度、缺两者；IR/Depth 同向平移 8/16/32 像素 | 缺失和错位分别报告；不能用“模态 dropout 有效”替代错位鲁棒性结论 |
| P1 | 深度语义优于把深度当普通灰度图 | 现有 `log1p + 有效掩码`；原始/线性单通道编码 | 只接受在关系/距离桶和整体指标上可复现的优势 |

这里的 P0/P1 是实验排序，不是已观察到的效果。官方评测数据只能推理，所有训练、选模和阈值选择都必须留在允许使用的公开或人工复核训练来源中，并在技术报告中说明来源与用途。

## 2. 赛题边界与审计对象

赛题的事实以本地[官方题目 PDF](<F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf>)为准：每条样本有空间对齐的 RGB、IR、Depth 和英文 Query；框在 RGB 坐标系输出为归一化 `xyxy`；主指标是 ACC@0.5。PDF 将 IR 定义为三通道同值的热灰度图，将深度定义为毫米级单通道 `uint16`，零或过小值无效，可感知距离约 0.3–20 m。这意味着“将三张图按 RGB 图尺寸重采样”是输入几何的必要操作，但不等于模型真的理解距离关系。

本文审计的工程版本为 `code/` 的提交 `2571a4c8ba6b7d870971ac895d12b465627a2f2d`。审计中没有改动模型、配置、权重、数据或提交文件，也没有下载训练数据或大模型。外部公开源码仅作只读比对，固定到以下提交，避免把仓库后续变动写成论文原始实现：

| 项目 | 固定版本 | 用途 |
| --- | --- | --- |
| RGBT-GroundBench / RGBT-VGNet | `e8869905d4f74a813a2fc667d922838db79f8acb` | RGB–热红外指代定位及 Query 引导融合 |
| VL-Grasp / RoboRefIt | `dd6bd6d7b4045b8b72df7d4bebb6ff4a1344076f` | RGB-D 指代定位的数据和模型接口 |
| CMNeXt | `a9d605d4a0702dce1579910d8dc93b0e817571a4` | 多辅助模态的特征选择和互校准 |

## 3. 当前 TriGround 已实现什么，以及尚未覆盖什么

### 3.1 输入与 Query 条件路径

当前数据读取不是把深度简单复制成 RGB：`encode_depth_image` 先取单通道、按 `depth_scale=1000` 转米，保留 `0 < d <= 20` 的有效像素，对有效距离做 `log1p` 归一化，并把有效性放入第二通道；IR 以双线性插值、深度以最近邻插值对齐至 RGB 尺寸。见[data.py](<F:/AIC/code/src/mm_grounding/data.py:17>)和[data.py](<F:/AIC/code/src/mm_grounding/data.py:107>)。这比 RGB-D 文献中将读到的深度直接拼为第四通道更符合本赛题的米制定义。

Query 有两条用途不同的路径：生成提示中要求模型返回框，另外单独 tokenize 原 Query，为融合模块提供 token 表示，见[data.py](<F:/AIC/code/src/mm_grounding/data.py:132>)和[data.py](<F:/AIC/code/src/mm_grounding/data.py:184>)。模型从冻结词嵌入取得 Query token 后 `detach()`，交给独立 IR/Depth Query 编码器，见[model.py](<F:/AIC/code/src/mm_grounding/model.py:540>)。这使辅助模态的更新可以由表达式调制，同时不更新 Qwen 语言主干。

### 3.2 层内并行融合

并行视觉前向让 RGB、IR、Depth 经过同一冻结视觉块；IR/Depth 输出各自先过适配器，然后对 RGB token 做融合。融合发生在 DeepStack 特征收集之前，因此配置的层（8、16、24、26）在前 3 个 DeepStack 输出和末层输出处都能影响主干注入。关键顺序见[model.py](<F:/AIC/code/src/mm_grounding/model.py:136>)、[model.py](<F:/AIC/code/src/mm_grounding/model.py:204>)和[Qwen3-VL-8B 升级说明](<F:/AIC/code/QWEN3_VL_8B_UPGRADE.md:1>)。

联合模块将 RGB、IR、Depth、Query 上下文堆成每个位置的四个候选，用模态注意力与 MLP 混合；三种成对余弦相似度进入 token/sample 可靠性门，最后以守护残差写回 RGB token。源码见[adapters.py](<F:/AIC/code/src/mm_grounding/adapters.py:601>)。它满足两项很重要的工程约束：输出投影可零初始化，从而初始化时精确保留 RGB；推理时把 `ir_fusion_scale` 或 `depth_fusion_scale` 设为零，可作为同权重消融。

### 3.3 现有鲁棒性边界

训练期随机按样本丢弃 IR 或 Depth，见[adapters.py](<F:/AIC/code/src/mm_grounding/adapters.py:674>)。它模拟的是“整张样本的模态缺失”，不是下列任何情况：传感器相对平移、尺度/裁剪差、错误配对的辅助图、局部无效深度、热饱和或深度量纲错误。由于当前融合以对应 token 计算 RGB–IR、RGB–Depth 余弦，并以对应位置进行注意力，错位会同时破坏内容匹配与可靠性特征；有门控不表示已经对错位稳健。

还应留意已有数据审计发现的边界：官方初赛输入中有 97 组深度文件为 640×360 RGB JPEG，而大多数深度为 1920×1080 的 16 位单通道图。当前读取会取其第一通道并按毫米解释。未弄清这 97 组的编码含义前，不应将这些样本上的深度消融归因为“深度无效”或“模型失败”。这属于数据语义问题，优先级高于添加更复杂的融合层。

## 4. 可迁移的公开工作

### 4.1 RGB–热红外指代定位：最接近任务的证据

[RGBT-GroundBench](https://arxiv.org/abs/2512.24561) 是目前直接对应 RGB–TIR 指代定位的公开基准：21,535 对 RGB–TIR、38,760 个带表达式和框的目标，并按场景、光照/天气、目标大小/遮挡标注。它的价值不在于可以直接外推分数，而在于其统一 RGB、TIR、RGB+TIR 协议与条件分桶：低光、小目标、遮挡是辅助热信息最可能改变结论的条件。

其公开实现将高秩 LoRA 给 TIR 分支（默认 RGB rank 16、IR rank 48），而不是假定两模态的预训练域差相同；再以语言作为 Query 对视觉 token 做跨模态交互（LAVS），最后按光照和特征差异做可靠性融合。具体实现可复核：[异构 LoRA](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/e8869905d4f74a813a2fc667d922838db79f8acb/models/mmvg.py#L955-L980)、[LAVS](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/e8869905d4f74a813a2fc667d922838db79f8acb/models/mmvg.py#L519-L626)、[IAFv3 门控](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/e8869905d4f74a813a2fc667d922838db79f8acb/models/mmvg_fusion.py#L1181-L1357)。

对 TriGround 的结论是有限而明确的：已经具备“辅助模态独立适配 + Query 条件融合 + 可靠性门”这三种结构思想，没有证据支持在训练前照搬 RGBT-VGNet。优先借鉴它的**评测方式**：将低照、热显著、遮挡、小目标和普通光照分别报告 RGB/IR/联合差值。该项目 README 所称的默认 IAFv3/HiLoRA 是当前仓库版本的配置，不能和论文中 AMA/LAVS/TPF 名称混作同一已验证消融。

### 4.2 RGB-D 指代定位与空间关系

[Refer-it-in-RGBD](https://arxiv.org/abs/2103.07894) 将 RGB-D 指代定位拆为对象候选与语言匹配，说明深度的价值常出现在“同类对象之间的关系判别”，而不只在目标分类。[VL-Grasp](https://arxiv.org/abs/2308.00640) 的 RoboRefIt 将语言指向与操作场景结合，是室内 RGB-D 表达式数据的可用外部训练/诊断参考，但域、目标类别、视角和最终任务均与本赛题不同。

其源码也给出反例：RoboRefIt 的 `RGBD` 路径用默认 `cv2.imread` 读 depth，取第一个通道、硬编码 reshape 为 480×640，然后与 RGB 拼成 4 通道，[固定版本第 229–270 行](https://github.com/luyh20/VL-Grasp/blob/dd6bd6d7b4045b8b72df7d4bebb6ff4a1344076f/RoboRefIt/datasets/grounding_datasets/refer_dataset.py#L229-L270)。该代码没有保持毫米尺度、有效值掩码或显式几何关系模块。因此它不能成为替换当前深度预处理的理由；其更合适的作用是为“近/远、左/右、前/后、遮挡”Query 构造补充训练或对抗测试时的候选来源。

[RoboRefer](https://arxiv.org/abs/2506.04308) 和 [RoboSpatial](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_RoboSpatial_Towards_Coordinate_Aware_Spatial_Reasoning_in_Vision-Language_Models_CVPR_2025_paper.html) 的共同提醒是：空间指代表达不只依赖深度图，参考系、参照对象、关系链和序数都必须正确绑定角色。若 Query 为 “the second person from the left” 或 “the vehicle behind the bus”，仅让深度影响视觉 token 仍可能失败。现阶段应先用反事实 Query 检验模型是否响应关系词，再决定是否需要候选框重排序器或显式关系头。

建议的反事实集必须有已知正确答案，并且不用于训练或模型选择：对同一场景的 Query 仅交换 `left/right`、`nearest/farthest`、序数或参照物；人工确认两个 Query 分别指向两个不同目标。记录两问是否同时正确、预测框中心是否按预期改变，而不是只记录原 Query 的单点 IoU。

### 4.3 通用 RGB-X 融合：适合作为模块库，不是端到端答案

[CMX](https://arxiv.org/abs/2203.04838) 为 RGB 加 Depth、Thermal、Event、LiDAR 等模态的语义分割设计了特征互校准（FRM）和跨路径融合（FFM）。它证明了“先校准，再交互”比盲拼接更值得试验，但训练目标是像素分割，既没有语言 Query，也没有单目标框指标。

[CMNeXt](https://arxiv.org/abs/2303.01480) 进一步以 Self-Query Hub 对多个辅助模态打分、逐元素取最大，再经 FRM 和 FFM 在四个阶段融合。[固定实现](https://github.com/holdon1/CMNeXt/blob/a9d605d4a0702dce1579910d8dc93b0e817571a4/semseg/models/backbones/cmnext.py#L277-L361)可见其选择及四层融合；[FFM](https://github.com/holdon1/CMNeXt/blob/a9d605d4a0702dce1579910d8dc93b0e817571a4/semseg/models/modules/ffm.py#L159-L189)先走双向 cross path 再压缩通道。它最值得借鉴的是**多辅助源竞争前先评估可靠性**，而非其分割头或完整骨干。

当前联合融合已包含 IR/Depth 的可观测相似度与样本/位置门，因此应先做一个极小消融：保留全部数据、参数预算和训练轮数，只将三种一致性特征替换为无一致性特征的门，或在门前加入轻量的每模态 score。若低光或深度关系桶有稳定收益，再考虑引入 CMX/CMNeXt 式 FRM；不能因为 CMX 在语义分割有效就假定它会提升 ACC@0.5。

### 4.4 缺失模态：与错位分开评价

[Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation](https://arxiv.org/abs/2310.03986) 使用参数高效的中间特征调制处理训练/测试模态组合不一致；[MAGIC](https://arxiv.org/abs/2407.11344) 通过模态可用性和相似性来选择融合来源。两者的可靠结论是：缺失模态应在训练和测试的各组合中独立报告，不能只测试“全模态训练、全模态测试”。它们都不是语言指代定位论文，也没有证明自己的模块会适合 Qwen3-VL 或本赛题。

TriGround 的随机模态 dropout 已是符合这一结论的低成本起点。应将现有联合训练与 `modality_dropout=0` 做完全相同的对比，并测试 4 个推理组合：RGB、RGB+IR、RGB+Depth、RGB+IR+Depth。缺模态实验须传入真正缺失的流或明确零尺度；不要以“将一张与当前图无关的辅助图替代缺失”来模拟缺失，因为那测试的是错误配对。

对于错位，最小可靠测试是只对辅助图应用已知像素平移（例如 8、16、32 px），不动 RGB、Query 或真值框。保持所有图像的边界填充规则一致，单独记录每种偏移方向和幅度。若只在正负偏移平均后报告，左右不对称和截断效应会被掩盖。若该实验发现显著下降，再考虑：训练时辅助模态随机小平移、在融合前学习有限偏移、或把融合从同位置 token 扩展到局部邻域；在这之前不应预先加入可变形注意力等高风险模块。

## 5. 具体实验协议

### 5.1 固定条件

每一组比较固定：人工复核集的 group-safe 划分、随机种子、预处理、训练样本数、训练步数、解码提示、最大生成长度、框解析和提交前合法化。只改变一项。验证集只用于比较，不与弱监督或官方测试混合。输出至少包含：`n`、有效框率、ACC@0.5、平均 IoU、相对 RGB 的 delta ACC、95% bootstrap 区间或至少逐 Query 成败表。

未跑实验前不报告任何“提升 X%”。对于 n 很小的语义桶，置信区间会很宽；它们用于发现错误模式，不能作为架构结论。

### 5.2 Query 分桶应可追溯

第一版可以人工审查优先、规则辅助。每个 Query 允许多标签，保留原句和标注理由：

| 标签 | 初始触发词示例 | 需要人工确认的歧义 |
| --- | --- | --- |
| 颜色/外观 | red, black, white, wearing, striped | 颜色到底修饰目标还是参照物 |
| 光照/热线索 | bright, dark, illuminated, warm | 表达式是否真的涉及可见热差 |
| 二维位置 | left, right, top, bottom, middle | 相对图像还是相对参照物 |
| 深度/距离 | near, far, closest, furthest, front, behind | 相机距离、遮挡层次、运动方向是否混淆 |
| 序数/计数 | first, second, third, last, another | 排序方向和候选集合 |
| 参照关系 | next to, beside, between, behind | 参照物的身份及关系方向 |
| 困难视觉 | small, occluded, crowded | 由人工按图像确认 |

规则只用来生成待复核清单，不要把词频当真值。尤其 `front` 可能指车辆正面，`right` 可能是目标自身的右侧，`near` 可能是二维邻近而非米制距离。

### 5.3 可靠性门是否做了正确选择

当前模块能输出或通过轻量日志记录 token/sample gate、三项相似度、IR/Depth 的有效性和最终框。为避免诊断本身改变模型，先在 `eval()` 下记录已有张量，不训练新的探针。对每个 Query 计算：

* `ΔIoU_IR = IoU(RGB+IR) - IoU(RGB)`；
* `ΔIoU_D = IoU(RGB+Depth) - IoU(RGB)`；
* `ΔIoU_joint = IoU(RGB+IR+Depth) - IoU(RGB)`；
* 分别在 RGB+IR、RGB+Depth、Triple 三种运行中记录联合 token gate 和 sample gate，以及 RGB–IR、RGB–Depth、IR–Depth 一致性。当前 Joint 没有两套独立的 IR gate/Depth gate，不能把共同残差门解释成各模态单独权重。

然后只做相关和分位数表：高 gate 的样本是否确实更常有正 `ΔIoU`；低 gate 是否保护 RGB；错位后 gate 是否下降并限制负迁移。如果没有这种关系，门可能只是在吸收训练分布偏差，不能以“可解释性”宣传。此诊断必须在不用于反向传播的独立验证数据上完成。

## 6. 何时才值得改模型

满足下列任一证据后，再选一个最小改动进入训练：

1. 深度关系桶和反事实集均显示 RGB/IR 不足、而正确米制深度有稳定正增益：增加轻量几何关系候选重排序，而不是重新训练 Qwen 主干；
2. 低光/热显著桶中 IR 强、正常光照中常为负：用现有门的条件损失或轻量照度特征校准，先与静态缩放比较；
3. 缺失模态显著退化且 dropout 改善缺失路径但不损伤联合路径：保留 dropout，并按可用组合训练/报告；
4. 已证实的 8–16 px 小错位便会使联合路径显著低于 RGB：加入仅作用于辅助输入的小平移增强，验证后再考虑可学习对齐。

若 P0 实验未显示联合路径比 RGB 好，先核对深度文件语义、训练标签质量、框解析和验证划分；不要以更大模型、更多融合层或更长训练掩盖负结果。

## 7. 参考资料

外部资料均为公开文献/官方源码，访问于 2026-09-11：

1. Zhao et al., [RGBT-GroundBench: Visual Grounding Beyond RGB in Complex Real-World Scenarios](https://arxiv.org/abs/2512.24561), 2025/2026；[官方实现](https://github.com/crazyxiaoxi/RGBT-GroundBench)。
2. Lu et al., [VL-Grasp: a 6-Dof Interactive Grasp Policy for Language-Oriented Objects in Cluttered Indoor Scenes](https://arxiv.org/abs/2308.00640), IROS 2023；[官方实现](https://github.com/luyh20/VL-Grasp)。
3. Zhang et al., [CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers](https://arxiv.org/abs/2203.04838), T-ITS 2023；[官方实现](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)。
4. Zhang et al., [Delivering Arbitrary-Modal Semantic Segmentation](https://arxiv.org/abs/2303.01480), CVPR 2023；[CMNeXt 官方实现](https://github.com/holdon1/CMNeXt)。
5. Reza et al., [Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation](https://arxiv.org/abs/2310.03986), 2023，TPAMI 2024。
6. Zheng et al., [Centering the Value of Every Modality: Towards Efficient and Resilient Modality-agnostic Semantic Segmentation](https://arxiv.org/abs/2407.11344), ECCV 2024。
7. Wu et al., [Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images](https://arxiv.org/abs/2103.07894), CVPR 2021。
8. Zhou et al., [RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](https://arxiv.org/abs/2506.04308), NeurIPS 2025。
9. Lin et al., [RoboSpatial: Towards Coordinate-Aware Spatial Reasoning in Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_RoboSpatial_Towards_Coordinate_Aware_Spatial_Reasoning_in_Vision-Language_Models_CVPR_2025_paper.html), CVPR 2025。

## 8. 交接

下一位执行实验的成员应先读本文、第 1 节的 P0 矩阵、[当前工程审计](<F:/AIC/docs/research/2026-09-08-contest-and-current-code-review.md>)、[Qwen3-VL 升级说明](<F:/AIC/code/QWEN3_VL_8B_UPGRADE.md>)和本地官方题目 PDF。开始实验前先确认那 97 组异常深度文件的语义，并生成不触碰官方测试集的分组隔离验证清单。实验结束后把完整命令、配置提交、权重来源、样本数、指标和逐桶结果追加到本文或新的实验报告中；不要将本文的文献推断改写成实验结果。
