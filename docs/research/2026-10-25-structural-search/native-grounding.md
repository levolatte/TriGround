# 原生多模态大模型的空间读出：论文、源码与本项目可证伪方案

2026-10-25。研究对象是现役 Qwen3.6-27B（本地接口实际为 `qwen3_5`）直接看 RGB、红外、深度及完整 Query，一次预测 RGB 坐标系中的归一化 BBox。本文是论文与源码审读、结构假设和实验设计；**未实现新模型、未运行 GPU 实验，不把分割数据集的 cIoU 写成比赛 ACC@0.5**。当前链路和旧实验的数据边界见[仓库取证](history-and-current.md)、[实际 Qwen 接口](../2026-10-25-architecture-rethink/qwen-interface.md)。比赛规则以本地 `contest/基于大模型的多模态视觉理解与推理.pdf` 为准：三图空间对齐，唯一文本目标，预测可见光图中的 `[x1,y1,x2,y2]∈[0,1]`，IoU≥0.5 算命中；深度原始值 0/过小可无效，外部训练数据可用，测试集不可训练。

## 一、先界定“原生”和项目历史

这里的“原生”指语义决策来自当前多模态大模型，空间监督的梯度能回到它的语言 LoRA 或内部视觉表示，输出不用再把冻结模型生成的框交给另一个检测器作选择。**它并不等于没有小读出头**：bbox 最终总须由参数或解码规则转换为 4 个数。若给模型一个先验建议框 A，后续区域头只学改 A，就仍是旧项目二次修正范式。

必须对照的旧事实如下。

- 旧 8B 的 R/S/S+G 已有四出口模态残差、数值 Depth、目标条件读出、连续几何头，且以冻结模型建议框 A 为锚二次处理。各臂约 296/296/298/296，Depth 正常/置空在 55 题上同为 45；仅把 head 改连续、把深度再拼一遍，不能构成新的路线或有效证据。[历史记录](history-and-current.md)
- 旧 DINO+SAM 候选池覆盖 339/412，但 292 个 C 直接框经编号选择到 273、再局部细化到 221。主要失败在目标身份与坐标系协议，不说明检测器本身无用，更不能把 339 个“有正确候选”当最终成绩。[历史记录](history-and-current.md)
- 现役 27B 的正式最好已知为 Arm2 ck500 二倍尺寸 .8012；新 226 对/157 图的辅助 A/B 训练尚未跑。City 的新版 400 主评分也未推理。这 157 张不同三模态图是现有专项池，可用于诊断与适配；结构训练还可使用合规外部公开数据。[历史记录](history-and-current.md)

## 二、六条近邻方法的真实结构

以下优先看原论文、附录和公开实现。不同论文的分割 cIoU、gIoU、RefCOCO 系列与本赛 BBox ACC 不可直接比较；模型底座和训练集也完全不同。

| 方法 | 图像/语言/空间输出的连接 | 梯度、数据、代价及直接启示 |
| --- | --- | --- |
| **LISA**（2023） | LLaVA 生成 `<SEG>`；该 token 的隐藏态经投影成为 SAM mask decoder 提示。SAM ViT-H 图像编码另行提供高分辨率特征。因此语义来自 MLLM，像素边界仍由专家模型负责。[论文](https://arxiv.org/abs/2308.00692)、[代码](https://github.com/JIA-Lab-research/LISA/blob/main/model/LISA.py) | 冻结 SAM 编码器，训练 mask decoder、投影、语言 LoRA/新 token；损失是语言 CE 与 mask BCE/Dice。训练混合 ADE20K、COCO-Stuff、PACO/PartImageNet、RefCOCO、VQA；常被引用的 **239** 是补充 ReasonSeg 训练样本，不是全模型训练量。[训练数据代码](https://github.com/JIA-Lab-research/LISA/blob/main/utils/refer_seg_dataset.py) |
| **PixelLM**（2023/24） | CLIP 不同层与生成的 mask-codebook token 接入多尺度像素解码器；论文称无需独立 SAM 图像编码器，但**仍有自训练的 SAM 风格 TwoWayTransformer/mask decoder**，不是纯 token→4 坐标。[论文](https://arxiv.org/abs/2312.02228)、[补充材料](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ren_PixelLM_Pixel_Reasoning_CVPR_2024_supplemental.pdf)、[模型](https://github.com/MaverickRen/PixelLM/blob/main/model/PixelLM.py)、[解码器](https://github.com/MaverickRen/PixelLM/blob/main/model/segment_anything/modeling/mask_decoder_multi_scale.py) | CLIP 冻结、LLM LoRA 与像素解码器训练；多尺度监督。MUSE 约 239k QA/约 0.9m mask，又混合其他分割/VQA。附录报告 8×A100 约 1.5–2 天。实现中计算 BCE+Dice，而论文提的 overlap loss 在公开 `PixelLM.py` 约 601–615 行被注释；复现时要以执行代码为准。[模型源码](https://github.com/MaverickRen/PixelLM/blob/main/model/PixelLM.py) |
| **Sa2VA**（2025） | LLaVA 一枚 `[SEG]` 的隐藏态投影为 SAM2 prompt，SAM2 图像编码、记忆和 mask decoder 完成图/视频掩码；与 LISA 同属**专家解码器路线**。[论文含附录](https://arxiv.org/abs/2501.04001)、[前向源码](https://github.com/bytedance/Sa2VA/blob/main/projects/sa2va/models/sa2va.py)、[SAM2训练包装](https://github.com/bytedance/Sa2VA/blob/main/projects/sa2va/models/sam2_train.py) | 主损失语言 CE+掩码 Dice/BCE；SAM2 encoder 和 memory 冻结，decoder、投影、语言 LoRA 可训练。混合约百万级分割/视频数据和 QA；附录表明兼顾 QA 与 grounding 有数据配比冲突。训练规模约 16×H100、24 h。它证明 `[SEG]` 含有对象语义，但不证明一个只在当前 157 图专项池适配的 bbox 头可学会实例身份。[配置](https://github.com/bytedance/Sa2VA/blob/main/projects/sa2va/configs/sa2va_in30_8b.py) |
| **Pixel-SAIL**（2025） | 原始像素 token 和语言 token 共用 single-transformer 主体；`[SEG]` 对视觉 token 作 mask 读出，加可学习上采样和可选专家蒸馏；推理无需 SAM2。其底座 SOLO/EVEv2 的 token 化与当前 Qwen3.6 模块式视觉塔不同。[论文含附录](https://arxiv.org/abs/2504.10465) | 关键消融：0.5B plain baseline 的 RefCOCO/+/g cIoU 64.5/57.3/60.1；只加上采样到 76.2/69.6/73.8；扩大分割及视觉提示数据继续涨；同时用 SAM2+Mask2Former 蒸馏平均只加约 0.6/0.3/0.5。训练约 1.7m 混合数据、32×A100，0.5B/3B 12/24 h。论文所指公开仓库没有与正文等量的 Pixel-SAIL 方法实现，因此源码级细节不可假设已复现。[论文](https://arxiv.org/abs/2504.10465)、[所指仓库](https://github.com/bytedance/Sa2VA) |
| **SELF1E**（2026） | InternVL 的压缩前 ViT 特征与压缩后 MLLM 图像 token 残差重填充，再用一个 `[SEG]` 向量与每个空间位置做点积成掩码，无外部 SAM decoder。它还改图像↔图像、图像↔`[SEG]` 注意力；因此“原生网格+单个查询 token+点积读出”已有明确先例。[论文](https://arxiv.org/abs/2603.19026)、[完整补充](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Rethinking_MLLM_Itself_CVPR_2026_supplemental.pdf)、[模型](https://github.com/ANDYZAQ/SELF1E/blob/master/model/InternVL3_self1e.py) | `InternVL3_self1e.py` 直接以 ViT/LLM 图像特征和 `[SEG]` 点积，BCE/Dice+文本 CE 联训；[collate](https://github.com/ANDYZAQ/SELF1E/blob/master/utils_internvl/dataset.py) 构造四维特制注意力。附录：base 约 561k 样本，SEG 版经 20× RefCOCO、6×其他分割重复到 2.4m **采样条目**，不等于 2.4m 独立图。作者明确说新注意力掩码妨碍自回归/多轮推理，VQA 尤其 OCR/知识任务下降；8B 的 TextVQA 78.9→72.3、MMB-en 83.4→74.6。[补充材料§6–8](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Zhang_Rethinking_MLLM_Itself_CVPR_2026_supplemental.pdf) |
| **Ferret**（2023） | 外部给定 region 的离散坐标和视觉采样的**连续 region 输入特征**共同进入语言模型；模型 grounding **输出**仍是量化坐标 token、按语言 CE 训练。把它引为“连续 bbox 输出已有论文支持”是方向错误。[论文](https://arxiv.org/abs/2310.07704)、[region 接口](https://github.com/apple-aiml-research/ml-ferret/blob/main/ferret/model/ferret_arch.py)、[生成头](https://github.com/apple-aiml-research/ml-ferret/blob/main/ferret/model/language_model/ferret_llama.py) | GRIT 约 1.1m 指令样本，含约 95k 难负样本，三轮阶段训练。其价值在同图相似对象与 region 语义辨别数据，而非当前任务可直接照搬的输出头。[训练入口](https://github.com/apple-aiml-research/ml-ferret/blob/main/ferret/train/train.py) |

**机制分界：** LISA/Sa2VA 的 `[SEG]→SAM` 是外部专家像素恢复；PixelLM 虽舍弃独立 SAM 编码器仍有专用 mask decoder；Pixel-SAIL/SELF1E 把空间预测更多放回多模态主体，却依赖充足 mask 监督或重设注意力。Ferret 是区域*输入*连续，区域*输出*离散。没有一篇直接证明“原生 Qwen3.6 三图、可扩充的实例/关系训练数据、单一 RGB bbox 评测”会成功。以上是从论文和代码推得的项目适配判断。

补充近邻：2026 年 7 月的 [Motto](https://arxiv.org/abs/2607.24407) 已把空间锚 P/R token、Qwen3-VL 原生网格读出、视觉信息再注入语言生成与 DETR 式连续框头组合；不能再把其中任何单项称为本项目新发明。其固定提交的[源码审读、训练 GT 锚与自由推理边界](motto-code.md)单列，本文后续方案仅作为现役三传感单次框预测的待验组合。

## 三、对现役 `qwen3_5` 的实际约束

本地配置是语言 64 层、宽 5120；视觉宽 1152、27 层、patch16、空间 merge2，`deepstack_visual_indexes=[]`。视觉模型有 merge 前 `last_hidden_state` 与 merge 后 `pooler_output`；后者接成语言 image token。merge 前 patch 顺序经过局部 2×2 重排，不能直接按 `H×W` reshape；三幅图的 `image_grid_thw`、裁切缩放和 RGB/IR/Depth 到原 RGB 坐标必须逐图恢复。[接口源码取证](../2026-10-25-architecture-rethink/qwen-interface.md)、[Transformers v5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

现役 `qwen3_5` 包含混合的线性注意力/全注意力层；SELF1E 以标准 Qwen3 层改四维 mask 的做法不能直接移植，也可能破坏当前自回归生成路径。证据位置与最终 `[LOC]` 放在全部三图及完整 Query **之后**，依因果注意力读取先前输入；在一处语言层边界显式回写证据，不改变底层注意力协议。若只在末层取一个 token 再接回归头，它与仓库旧 `auxiliary_bbox_enabled:false` 的头太近，无法检验实例身份、关系与主体回注。若把外部高分辨率 RGB+IR+Depth 再接一个大解码器，又偏向双塔专用 grounding，不能声称无需额外视觉系统。[旧可选 bbox 头和当前链路](history-and-current.md)

## 四、建议实验假设：实例关系证据参与第一次出框

**假设 H：** 当前错误的一部分来自目标与同图竞争实例、参照及集合关系没有在第一次出框前稳定绑定。让完整 Query 条件下的软实例记忆同时保留区域位置、视觉内容、有效 Depth 与 IR 证据，令目标/参照绑定和成对关系在语言中段更新，再将证据回注语言主体；最终框损失若能沿这条路径反传，就可能净改善错实例与范围错题。它不接收建议框 A、DINO 匿名候选 ID 或训练时 GT 区域输入。**这是待实验的任务化组合，不把 token、区域读出、实例槽、关系或连续框中的任一单项称为原创。**

完整计算图为：

```text
RGB/IR/Depth + 完整 Query + 若干证据位置 + 最后的 [LOC]
       │ 现役 Qwen3.6 视觉塔及语言前 k 层，正常因果掩码
       ├───────────────────────► 三图 merge 前视觉网格
       ▼                              │  图名、RGB 坐标、有效深度
完整 Query 条件状态 ──────────────────┘
       ▼
软实例区域记忆：各实例的区域分布、空间布局、RGB/IR 内容、Depth 分布
       ▼
可变目标/参照/集合角色 ↔ 实例分布；实例间关系及序数更新
       ▼
更新后的证据回写对应语言位置 → 后 64-k 层 → [LOC]
       ▼
最终目标状态 + 全图 RGB 细网格 → 共享单框读出 → [x1,y1,x2,y2]
```

1. **实例内容和身份。** 取 Qwen 原生 merge 前网格，恢复顺序与 RGB 坐标后建立有限容量的可微区域记忆。每个活跃实例需有自己的空间分布和局部布局、视觉内容、有效深度分布与 IR 内容；空实例有单独状态。容量是计算上限，不表示图中恰有该数量的实体。用可得的真实多实例区域标注做集合匹配、区域范围及内容监督；只有目标 bbox 的样本不能伪造全场景实例真值。训练/推理均由模型预测的软区域读取内容，GT 区域只用于损失目标，不能作为回注输入。
2. **Query 角色与关系。** 从完整 Query 读出可变数量的目标、参照和竞争集合表示，软绑定到实例分布；一个角色可对应多个实例。成对关系使用相对位置、范围、区域视觉内容、Depth 有效率和有效数值顺序概率。左右/序数按实例集合而非像素面积计算；近远证据在 Depth 无效时保留缺失信息，不拿整框平均深度硬判主体关系。关系更新必须反馈给目标绑定，不能只成为旁路辅助分类器。
3. **主体回注。** 把更新后的各实例/角色证据写回 Query 后对应语言证据位置，最终 `[LOC]` 在其后；继续 Qwen 后半层。框损失要回到关系、绑定、区域读图与语言 LoRA，不能 `no_grad` 或 detach Query 条件主链。只在一处层边界显式回写，不套旧四视觉出口或 SELF1E 双向四维 mask。单纯只改 `[LOC]` 而让角色/实例状态不可追踪，不足以检验关系机制。
4. **唯一框读出。** 最终 `[LOC]`、目标相关图和**全图** RGB 细网格共同驱动一个共享单框模块；左右上下边界查询在小模块中交换信息，输出有序的连续区间，按完整 bbox 的 L1 + GIoU/IoU 联合训练。这样保留跨实例组合、部件和超出当前实例区域的答案空间。四个互不关联的边界分布即使同时前向，也不构成完整联合几何概率；不能把多峰边界均值误当实际对象。原生 AR（自回归）坐标出口是同状态强对照：若它更好，应采用 AR，不能为了新头牺牲 ACC@0.5。

相比 SELF1E，这里直接预测 RGB 框、保留三传感的实例与关系证据且不改因果掩码；相比 Motto，不先生成离散 GT 派生空间锚、停机 zoom 和再次生成；相比旧 8B，不给 A 后二次修正。**这些是设计和实验边界，尚不是成绩或全球技术新颖性证明。**

## 五、数据与因果对照

官方允许外部公开训练数据；现有 226 对/157 张不同三模态图是**当前专项适配与诊断池**，不是整个训练资源边界。[赛题材料](../../../../contest/基于大模型的多模态视觉理解与推理.pdf) 可从 RefCOCO/+/g 配合源 COCO 实例标注准备同图竞争集合和目标 Query；Visual Genome/GQA 等可信实体关系数据、PACO 部件与有可靠 2D 区域标注的 RGB-D 来源可进一步补关系、部件和传感证据。数据映射/清洗尚未完成，不把它们写成已可直接训练。Depth/IR 缺失的 RGB 样本保留缺失标记，不复制 RGB 伪造模态；同图相邻帧不可跨训练与验证。GT 实例区域只用于匹配、内容和角色监督，不能先给模型真值 crop 再考定位。外部数据增益和结构增益必须用同数据对照拆开。

第一轮直接运行**完整实例关系与回注模型**，以及能证伪其因果机制的四组配对；相同 Arm2 ck500 初始化、图尺寸/图序、Query、训练样本与预算：

| 臂 | 要回答的问题 | 关键约束 |
| --- | --- | --- |
| 同数据原生 AR | 新数据和普通续训能达到多少 | 原生三图直接生成框；新版 400 重新推理，旧官方 .8012 仅作历史值 |
| 完整模型 | 实例/角色/关系/回注联合能否提高最终 ACC | 查最终框损失到首次绑定、区域读取、语言 LoRA 的梯度 |
| 完整模型删关系更新 | 成对关系和集合次序是否有增量 | 保留区域内容监督、回注、相同单框出口与训练数据 |
| 完整模型删主体回注 | 改变主体推理还是旁路定位器独自工作 | 保留同容量区域/关系模块与全图读出 |
| 同一状态接原生 AR 出口 | 空间头是否必要、是否损害对象身份 | 尽量共享状态/辅助监督，说明两出口训练成本差异 |

末层简单头可另作诊断或训练时 AR 框级辅助监督，**不是完整模型的准入门槛**。之后用同一权重做 RGB-only、IR 无效化、Depth 无效化和区域证据干预，分别检验真实传感使用、关系依赖与潜在捷径；输入消融有变化只说明依赖，仍须看正确框净增。

主要报告独立场景逐题 ACC@0.5、救回/损害、错实例/范围/关系/序数/有效传感分组。反复使用的 City 新版 400 属开发诊断；27 条改 Query 要重新前向，不能用旧 321/412 拼出新版成绩。若同数据 AR 同样改善、删关系/删回注无损，结构因果主张不成立；若 AR 出口好于空间头，应保留有效状态而使用 AR。

## 六、裁决与前置风险

完整结构与同数据强对照应同时推进，简单末层头的结果不决定是否实施完整模型。论文中的成功空间读出通常依赖大量含区域/掩码的数据，现有 157 图不足以独自学会实例分离与关系；外部公开数据可扩充，但其实际清洗、标注适用性和训练成本尚未核定。另一个风险是强空间训练损害 OCR/知识问答，SELF1E 附录有实证。最终以本赛唯一框 ACC@0.5、独立场景、逐题净收益以及关系和传感干预裁决。
