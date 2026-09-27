# TriGround 8B 模型架构审计

审计日期：2026-09-11  
审计对象：F:/AIC/code，提交 2571a4c8ba6b7d870971ac895d12b465627a2f2d  
范围：当前 Qwen3-VL-8B TriGround 的数据流、可训练参数、损失、checkpoint 链和可验证的风险。本文不包含训练、下载权重、GPU 推理或修改模型代码。文中“风险”和“方向”均是待证伪的实验假设，不是对当前线上分数根因的断言。

## 结论摘要

当前推荐路线不是把三张图拼接后送入新视觉网络，而是把同一个冻结的 Qwen3-VL 视觉主干顺序运行三次：RGB 为主流，IR 和深度为辅助流。每个辅助流在每个视觉块之后经过小型残差 adaptor（适配器）；RGB 在第 8、16、24、26 个块后接收联合 IR、深度和 Query 的残差。前 3 个注入点恰好位于 Qwen 原生 DeepStack 特征提取之前，最后一个位于最终 merger（视觉令牌压缩器）之前。因此，Qwen 既有的视觉令牌压缩、DeepStack 和语言模型注入路径仍在工作。

8B 的 Stage 1A/1B 分别训练 IR 和深度支路；Stage 2 改为新的 Joint Fusion（联合融合）模块，并冻结两类视觉 adaptor。Stage 2 的 Query Encoder 仍可训练，语言和视觉主干均冻结。生成监督仍是 Qwen 的 bbox_2d JSON 自回归交叉熵；辅助 Direct BBox Head 已在工程中实现，但配置明确禁止它和当前 parallel_backbone 路线同时启用，所以它不参与 8B 训练和推理。

最值得优先验证的架构问题有三类：辅助 Query 路径默认无位置编码，因而对词序近似不敏感；联合阶段把两个独立单模态训练得到的 Query 表示直接求均值，没有显式对齐；联合残差同时受零初始化、双 -2 可靠性门和 BF16 写回约束，实际残差信号是否足够大必须在真实预检中量化。三者都有明确代码证据，但都尚无真实 8B 训练结果，不能据此归因。

## 赛题目标与工程边界

官方 PDF 和网页材料要求对对齐的 RGB、热红外、深度和英文 Query 做视觉定位，输出 RGB 图中的归一化 xyxy 框。评分是 IoU 不低于 0.5 的 ACC@0.5；反向框、越界、NaN 和空框直接无效。官方明确把空间关系、计数、指代、弱光、遮挡和传感器质量退化纳入考察，允许公开外部训练数据，但禁止将测试集用于训练或人工标注。

当前工程的 checkpoint 选择顺序为 ACC@0.5、mean IoU、ACC@0.7、解析率，和主指标一致，见 code/src/mm_grounding/engine.py:18-34。评估会解析生成 JSON，解析失败用零框计分，另外记录解析率和达到生成长度上限的比例，见 code/src/mm_grounding/engine.py:93-206。

## 8B 实际规格、令牌预算和融合参数量

读取 Hugging Face 官方 Qwen/Qwen3-VL-8B-Instruct 的 main/config.json（只读取配置 JSON，未下载权重）得到：

| 项目 | 实际值 | 架构含义 |
| --- | --- | --- |
| 文本 hidden size / 层数 | 4096 / 36 | Query 裸词嵌入的输入宽度为 4096；主语言模型冻结 |
| Vision depth / hidden size / heads | 27 / 1152 / 16 | 融合配置的最终层 26 是 0-based 最后一个视觉块 |
| patch size / temporal patch size | 16 / 2 | 一个原始图像 patch 为 3 x 2 x 16 x 16 = 1536 输入维，先映射到 1152 维视觉 token |
| spatial merge size | 2 | 一个语言侧视觉 token 汇聚 2 x 2 个视觉 patch |
| DeepStack 层 | [8, 16, 24] | 与前 3 个融合层精确重合 |
| Vision out hidden size | 4096 | merger 输出与语言模型视觉接口对齐 |

8B YAML 设置 min_pixels=200704、max_pixels=802816。按 patch=16、merge=2 计算，单张图的原始视觉 patch token 预算为 784 到 3136，送到语言模型的 merged visual token 为 196 到 784。三流并行的视觉块部分在最大预算下要处理约 3 x 3136 个 token；只有 RGB 的 784 个 merged token 及其 DeepStack 特征进入语言模型。这解释了为什么显存预检必须在最大视觉预算执行。

用上述真实规格只实例化 ParallelBackboneFusion（不构造 Qwen 权重）得到 97,746,514 个融合模块参数。其中 IR/Depth adapters 各 14,245,659，IR/Depth Query Encoder 各 731,136，legacy stage fusions 为 24,190,992，joint stage fusions 为 43,601,932。Stage 1 每个单模态实际激活约 27.07M 参数；8B Stage 2 冻结两类 adapter 和 legacy stage fusions 后，仍训练两个 Query Encoder 加四层 joint fusion，共 45,064,204 个参数。参数量并不等于训练显存占用；三流冻结视觉块的激活仍须为这些参数的梯度保留。

## 审计环境与兼容性验证

本机只读验证环境为 PyTorch 2.8.0+cpu 和 Transformers 4.57.3。当前安装的 Qwen3-VL 视觉前向位于 code/.venv/Lib/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py:703-753：其标准顺序是 patch_embed、位置编码、逐块视觉计算、DeepStack merger、最终 merger。TriGround 的 parallel 前向复现了这条顺序，并在逐块循环中插入辅助流与融合，见 code/src/mm_grounding/model.py:136-229。

相关 CPU 测试已执行：

| 验证 | 结果 | 覆盖内容 |
| --- | --- | --- |
| pytest tests/test_model.py tests/test_adapters.py tests/test_config.py tests/test_auxiliary_training.py -q | 44 passed | 三流前向、DeepStack 注入顺序、单模态训练参数、Joint Fusion、零初始化梯度、配置约束、辅助几何损失 |
| Query 无位置编码置换实验 | Encoder 最大差 3.58e-7；完整 Joint Fusion 最大差 0 | 证实辅助 Query 路径对同时置换词序近似不变 |
| 本地 Transformers 源码核对 | 兼容 | 自定义视觉方法的签名和输出二元组与当前库一致 |

上述测试不包含 8B 权重、真实图像处理、CUDA、显存和训练效果。8B GPU 冒烟仍是正式训练门槛。

## 数据和监督数据流

### 1. 输入准备

GroundingDataset 读取 RGB、IR、深度、英文 Query 和归一化 xyxy 标注。它会校验 bbox 的有限性、范围和 x1 小于 x2、y1 小于 y2，见 code/src/mm_grounding/data.py:81-118。

RGB 直接以 RGB 图读取。IR 按灰度读取，双线性缩放到 RGB 尺寸后再转回三通道；这符合官方“IR 三通道内容高度一致”的说明。深度图读为原始数组后，先按毫米比例换算，仅将 `0 < d <= 20 m` 作为有效值，对其取 log1p；超范围值作为无效值置零。再构造三通道伪彩图：[归一化距离、有效深度掩码、零]，并用最近邻缩放，见 code/src/mm_grounding/data.py:17-29、103-117。

NativeGroundingCollator 的语言主提示为“根据 referring expression 定位并输出 bbox_2d JSON”。训练答案是 0 到 1000 的整数坐标 JSON。RGB 由 Qwen processor 同时处理图像和完整聊天文本；独立的裸 Query 还会被 tokenizer 编码一次，供融合模块使用。IR 和深度分别用同一个 image_processor 处理，并强制要求 image_grid_thw 与 RGB 相同，见 code/src/mm_grounding/data.py:121-224。

### 2. 三条视觉流

一次非 RGB-only 的 parallel_backbone 前向在 code/src/mm_grounding/model.py:161-229 执行：

1. RGB、IR、深度的扁平图像 patch 各自经过同一个冻结 Qwen patch_embed；三者加同一位置编码。
2. 对每个 Qwen 视觉块，先运行 RGB；再运行 IR、深度；各辅助输出再分别经过该层的 ModalityBackboneAdapter。
3. ModalityBackboneAdapter 是 LayerNorm、降维线性层、GELU、升维线性层的残差。升维层权重和偏置为零初始化，代码在 code/src/mm_grounding/adapters.py:339-361。
4. 当层号为 8、16、24、26 时，融合模块把残差写回 RGB；辅助流不会接收 RGB 融合后的反馈，仍独立向前演化。
5. 在 8、16、24 层，写回后的 RGB 特征立即经过 Qwen 原生 deepstack_merger_list；26 层写回后经过最终 merger。

当前 8B 配置的 [8,16,24,26] 与升级说明的原生 DeepStack 对齐要求一致，见 code/QWEN3_VL_8B_UPGRADE.md:7-21 和五份 configs/qwen3_vl_8b_*.yaml。构造时还会检查配置融合层覆盖主干实际的 DeepStack 层和最后一层，见 code/src/mm_grounding/model.py:393-405。

### 3. Query 进入融合的方式

在开始视觉前向前，模型调用冻结 Qwen 的 get_input_embeddings 取得裸 Query 的词嵌入，随后 detach；它不会运行 Qwen 的文本 Transformer。两套独立、可训练的 QueryTokenEncoder 分别处理同一 Query，一套给 IR，一套给深度，见 code/src/mm_grounding/model.py:540-577、code/src/mm_grounding/adapters.py:364-427、1020-1037。

每个 QueryTokenEncoder 为 LayerNorm 加线性投影、一个 TransformerEncoderLayer、LayerNorm。默认 8B 配置为输入语言宽度到 128 维、一层、4 头、无 dropout、无位置编码。它接收的是预训练且冻结的 Qwen 词表嵌入，所以输入不是随机向量；但自定义 Query Encoder 本身从随机初始化开始训练，且没有使用 Qwen 的上下文化文字隐藏状态。

JointQueryAwareStageFusion 在 code/src/mm_grounding/adapters.py:601-865 完成每个融合层的计算：

1. RGB、IR、深度 token 分别投影到 512 维。
2. 对每个样本，将可用 IR 和深度 Query token 逐 token 求均值；以 RGB 空间 token 为 Query、该平均 Query 为 Key/Value 做语言交叉注意力，见 696-735。
3. 对每一个空间位置，把 RGB、IR、深度、语言上下文当作 4 个小序列 token 做模态注意力。
4. 用 RGB 与两类传感器的余弦相似度、语言上下文和模态注意力输出驱动 token reliability 与 sample reliability 两个门；二者相乘且必须有辅助模态。
5. 联合 MLP 的 restore 输出乘门和残差尺度后写回 RGB。

### 4. Qwen 语言模型和生成

融合后的视觉输出仍由 Qwen3VLForConditionalGeneration 处理。Qwen 原生代码会把最终图像特征和 DeepStack 特征同时交给语言侧，见本地 Transformers 源码 1050-1064、以及 784-804 的 DeepStack 参数说明。TriGround 训练使用完整提示加真实 bbox JSON 的 labels，并关闭 cache。推理调用原生 generate，贪心生成，见 code/src/mm_grounding/model.py:668-730、767-790。

## 8B 五阶段训练链和可训练参数

所有 8B 配置冻结 Qwen 语言和视觉主干、关闭 Vision LoRA；config.py 还显式禁止 parallel_backbone 同时启用 Vision LoRA，见 code/src/mm_grounding/config.py:122-123。set_phase_a_trainable 先冻结整个模型，再按阶段开启融合模块，见 code/src/mm_grounding/model.py:434-473。

| 阶段 | 初始化 | 可训练部分 | 冻结部分 | 数据与训练预算 |
| --- | --- | --- | --- | --- |
| Stage 1A IR | 无 | ir_adapters、ir_query_encoder、每层 legacy stage_fusions 的 IR 支路 | Qwen、深度支路 | RGBT train_50，3 epoch，lr 3e-5，accum 16 |
| Stage 1B Depth | 无 | depth_adapters、depth_query_encoder、每层 legacy stage_fusions 的深度支路 | Qwen、IR 支路 | RoboRefIt train_50，3 epoch，lr 3e-5，accum 16 |
| Stage 2 Joint | 合并 1A 与 1B 稀疏 checkpoint | joint_stage_fusions、两个 Query Encoder | Qwen、两类 adapters、legacy stage_fusions | 923 条人工复核训练集，1 epoch，lr 1e-5，accum 16 |
| Stage 2 Weak | Stage 2 Joint | joint_stage_fusions、两个 Query Encoder | 同上 | 992 条场景安全弱监督，1 epoch，lr 1e-5 |
| Stage 2 Clean | Stage 2 Weak | joint_stage_fusions、两个 Query Encoder | 同上 | 回到 923 条人工复核训练集，1 epoch，lr 3e-6 |

其中“冻结 adaptor”不等于冻结 Query Encoder：freeze_parallel_adapters 只匹配 ir_adapters 和 depth_adapters，两个 Query Encoder 会继续可训练。Joint 模式还会冻结 legacy stage_fusions，只使用 joint_stage_fusions，见 code/src/mm_grounding/model.py:460-473。

Stage 1 checkpoint 是仅保存可训练参数的稀疏 checkpoint；Joint 阶段为便于后续链式加载会保存全部 fusion 参数。多个初始化 checkpoint 的键冲突、未知键和形状不一致都会 fail fast，见 code/src/mm_grounding/checkpoint.py:20-113。这条链避免了 IR 与深度 checkpoint 静默互相覆盖。

默认 8B Stage 2 没有开启 warm_start_joint_fusion_from_legacy。代码存在一个受控 warm-start：复制 IR/深度辅助投影、平均两套 RGB 投影与 language attention，但故意不复制 legacy restore，见 code/src/mm_grounding/adapters.py:977-1012。它适合做单变量消融，而不是直接视为默认修复。

## 损失和 bbox 路线

8B 当前有效训练目标只有 Qwen 的生成交叉熵。评估时生成结果经正则解析为 0 到 1 的 xyxy 框，解析失败为零框，见 code/src/mm_grounding/engine.py:93-206。

工程还保留了只适用于 safe_post_embed 的辅助 Direct BBox Head。它从首个答案 token 前一个 prompt token 的最终隐藏状态预测 sigmoid(cx,cy,w,h)，优化 Smooth L1 和 GIoU；coordinate_token_loss 仅作监控而不加入 total loss，见 code/src/mm_grounding/model.py:625-730、code/src/mm_grounding/boxes.py:6-54。config.py:169-170 禁止在 parallel_backbone 下开启该头，所以 8B 方案没有使用框回归监督，也不会以 Direct BBox Head 作为提交输出。

## 代码可确认的风险与限制

以下条目按“代码已确认的机制”记录，不能外推出“已经导致低分”。

### 1. 辅助 Query 路径默认丢失词序信息

证据：8B 配置没有设置 query_position_encoding，因此使用默认 none；QueryTokenEncoder 只在 sinusoidal 选项开启时加入位置编码，见 code/src/mm_grounding/adapters.py:397-427。无位置的 self-attention 对同时置换输入 token 是等变的；随后视觉交叉注意力把这些 token 当 Key/Value，对其排列不敏感。

CPU 的最小实测中，置换同一 Query 的词序后，Encoder 重排回原序的最大绝对差为 3.58e-7，完整 Joint Fusion 输出差为 0。主 Qwen 仍会读到原始完整提示并具有位置信息，因此这只限制辅助融合条件，不代表整个模型听不懂词序；但它会削弱融合模块对“左侧”“右侧”“站在……前”“第二个”等关系语义的利用。

### 2. 两个单模态 Query latent 直接平均，缺少显式跨模态对齐

证据：Stage 1 训练了独立的 ir_query_encoder 和 depth_query_encoder；Joint 阶段在 code/src/mm_grounding/adapters.py:712-720 直接平均二者的 token 表示，再送一个共同 language_attention。工程没有共享权重、对齐损失或 modality type embedding。

这不一定错误，因为 Stage 2 两个 encoder 仍可共同训练；但两个表示在不同 Stage 1 数据和不同目标下学得的坐标系没有保证可相加。当前 Joint 阶段又只在零初始化新 restore 的条件下训练三段短阶段，因此“平均是否是有效融合算子”必须测量，不能依靠直觉。

### 3. 联合残差初值可能被双门和 BF16 写回削弱

证据：token_reliability 和 sample_reliability 的最终 bias 都初始化为 -2，见 code/src/mm_grounding/adapters.py:652-668。无训练时两门约为 sigmoid(-2) 的平方，即约 0.014；同时 restore 权重为零。最终写回时 gate 和 residual 被转换为 rgb_tokens.dtype 后相加，见 853-865。8B RGB 视觉 token 为 BF16，因此幅度低于 BF16 有效间隔的小残差可能在写回后没有改变 token。

这是一条需要真实模型激活验证的数值风险。restore 的梯度虽不会为零，但门会缩放它；一轮联合训练是否已把门和残差推到有效范围未知。应记录四个融合层的 gate 分位数、写回前 residual/RGB RMS、BF16 写回后的 changed-token 比例，而不是从初始化值推断最终效果。

### 4. Stage 1 学到的 legacy fusion 在 Stage 2 默认被旁路

证据：8B Stage 1 用 parallel_joint_fusion: false，Stage 2 用 true；ParallelBackboneFusion 同时持有 legacy stage_fusions 和 joint_stage_fusions，但真正 fuse 时只选择后者，见 code/src/mm_grounding/adapters.py:1039-1068。Joint 阶段又冻结 legacy stage_fusions，见 code/src/mm_grounding/model.py:465-470。

这意味着 Stage 1 确实给 adapters 和 Query Encoder 提供了初始化，但其已学 IR/Depth 融合投影、语言注意力和 restore 不会直接执行。它是“新 Joint 结构优先安全 RGB 保持”的设计选择，不是 checkpoint 加载错误；不过在小数据、短训练时可能牺牲样本效率。

### 5. IR 和尤其深度首先经过冻结的 RGB patch_embed，缺少模态输入 stem

证据：IR、深度均直接调用 vision.patch_embed，见 code/src/mm_grounding/model.py:161-184；各模态 adaptor 只在已经运行一次冻结视觉块之后才生效，见 204-210。IR 的灰度复制造成的域差异较小；深度则是 log 距离、有效掩码和零通道构成的伪 RGB，见 code/src/mm_grounding/data.py:17-29。

这条设计保留三流形状一致与 Qwen 重用，但第一层深度视觉表示只能依赖未针对深度训练的 RGB patch_embed。后续 adapter 可能足够补偿，也可能不够；必须与一个轻量前置 stem 的对照实验比较。

### 6. 梯度检查点由视觉块基类处理，显存仍需真实测量

不能因为视觉循环没有显式调用 checkpoint，就认定梯度检查点未实现。本地 Transformers 4.57.3 的 `Qwen3VLVisionBlock` 继承 `GradientCheckpointingLayer`（modeling_qwen3_vl.py:251），后者在 `__call__` 中根据 `gradient_checkpointing` 和 `training` 自动包装重计算（modeling_layers.py:35-59）；`PreTrainedModel._set_gradient_checkpointing` 会向各子模块设置开关（modeling_utils.py:3716-3732）。TriGround 按正常 `block(...)` 方式调用这些块，正式训练也会 `model.train()`，所以不能把此机制列为已缺失功能。

仍需在真实预检和训练中核对各块的训练状态、开关与峰值显存。三流冻结参数不等于无需为 adapter/fusion 保留反向路径；检查点重计算也不保证最大 802816 像素预算一定能在目标 GPU 上运行。应测量后再调整分辨率或显存策略。

## 最值得试的六个方向

| 优先级 | 方向与假设 | 最小证伪实验 | 代价 |
| --- | --- | --- | --- |
| 1 | Query 位置编码：正弦位置编码能让辅助融合区分词序关系，从而提升关系、序数和空间类 Query。 | 用同一初始化、同一 seed、同一训练链比较 none 与 sinusoidal；119 条逐样本成对 ACC@0.5、mIoU、Query 类型分层，并先做 query_correct/zero/shuffled 干预。已有 configs/stage2_joint_fusion_v3_control.yaml 和 positional.yaml 可作为 2B 参考，8B 需复制为严格同变量配置。若融合 Query 干预无收益或 sinusoidal 不改善则否证。 | 低到中，两个完整 Joint 训练。 |
| 2 | 替换 IR/depth Query 的裸均值：两套单模态表示不能默认落在相同坐标系，显式保留模态身份会更稳定。 | 保持 encoder 宽度和训练预算，比较当前逐 token mean、把两路 token concat 后用 modality type embedding 送共同注意力、共享一个 Query Encoder 加两个小投影。记录同句两路 token cosine、gate、query 干预和 119 条指标。mean 不劣则否证。 | 中，需要小范围模型改动和三臂消融。 |
| 3 | 激活并校准 Joint 残差：双 -2 门和 BF16 写回可能使有效残差过小；更开放的初始门或更长的联合适配可提高实际融合强度。 | 先在 preflight 加四层 gate 分位数、残差/RGB RMS、写回前后 changed-token 比例。固定全部其他设置，对比 gate bias -2、-1、0；必要时只延长 Joint 阶段。若当前已有足够 delta、开门不提高 ACC 或提升鲁棒性下降，则否证。 | 低到中，需要指标钩子和少量训练。 |
| 4 | 受控 warm-start Joint Fusion：复用 Stage 1 对辅助特征和 Query 的已学投影，可在小样本 Stage 2 提高起点。 | 在 8B Stage 2 Joint 配置只开启 warm_start_joint_fusion_from_legacy: true；保持 restore 零初始化、数据、seed、步数完全一致。比较第 2、10、末步的 delta/gate 和 119 条最终指标。若没有更快收敛或最终下降则否证。 | 低，已有实现，主要是新配置与一次对照。 |
| 5 | 模态专属输入 stem：让深度和 IR 在进入冻结 RGB patch_embed 前先做轻量、可训练的传感器映射，减少首层域偏移。 | 先只加深度 branch 的小型 patch-level 或 1x1 stem，RGB 完全不变；与当前 log-depth 编码按相同 Stage 1B 初始化、步数和分辨率对比，再进入固定 Joint 阶段。若 RGB+Depth、Triple 和低质深度分层均无提升则否证。 | 中，涉及输入路径和 Stage 1/2 重训。 |
| 6 | RGB 指令微调 LoRA 作为独立对照：当前 8B 完全冻结语言模型，只靠生成交叉熵适应坐标 JSON；Qwen 官方 qkv/o LoRA 微调入口可测试语言定位与 JSON 生成是否是主要短板。 | 先做 RGB-only Qwen 8B SFT LoRA，使用同一人工复核训练集与独立验证，比较原生 RGB、冻结 TriGround 的 rgb_only、RGB SFT LoRA。仅当 RGB 对照明确提升，才评估如何与冻结三流融合兼容。若 RGB SFT 不提高解析率或 ACC，则不把 LoRA 混入多模态路线。 | 中到高，需要单独训练入口、显存评估和严控可比性。 |

Direct BBox Head 适合列为后续第七方向：官方只关心合法 bbox，直接几何损失可能改善阈值附近框；但它当前被 config 限制为 safe_post_embed，不能作为“只改 8B YAML”的实验。应先在 parallel_backbone 上设计不会泄漏答案 token 的读取位置、生成与直接框的选择/融合规则，再与纯生成路线做同初始化对照。

## 建议的实验顺序

1. 在真实 8B GPU 冒烟完成前，不推断任何分数原因。先验证最大视觉 token 预算、两步后上游梯度、冻结参数无梯度、四层融合门和 BF16 写回有效比例。
2. 将 Query 的现有因果干预扩展到 8B，然后先做 none 与 sinusoidal 的严格成对 A/B。该项改动最小，并且直接针对官方的空间关系和指代要求。
3. 从同一个 Stage 1 checkpoint 做 current、warm-start 和两路 Query 不求均值三组。每组保留 119 条逐样本结果，按场景分组 bootstrap，避免把相关帧当独立样本解释。
4. 若融合残差确实有效但深度独立模式仍弱，再投入模态输入 stem；若 RGB-only 本身已是瓶颈，则先完成 RGB SFT LoRA 对照。

## 证据路径

- 官方规则：F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf，第 1-9 页；F:/AIC/contest/全球校园人工智能算法精英大赛 多模态赛道.html。
- 8B 设计与数据链：F:/AIC/code/QWEN3_VL_8B_UPGRADE.md；F:/AIC/code/configs/qwen3_vl_8b_stage1a_ir.yaml、qwen3_vl_8b_stage1b_depth.yaml、qwen3_vl_8b_stage2_joint.yaml、qwen3_vl_8b_stage2_weak.yaml、qwen3_vl_8b_stage2_clean.yaml。
- 模型与损失：F:/AIC/code/src/mm_grounding/model.py、adapters.py、data.py、boxes.py、engine.py、checkpoint.py、config.py。
- 当前本地依赖实现：F:/AIC/code/.venv/Lib/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py。
- 既有 Query 实验设计：F:/AIC/code/EXPERIMENT_QUERY_POSITION_AB.md、QUERY_FUSION_CAUSAL_DIAGNOSTIC.md。
