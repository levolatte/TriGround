# 三传感器视觉定位：以 RGB 空间网格为锚的端到端融合架构

本文为该专题的独立方案，综合架构与实施优先级见[主报告](README.md)。

2026-10-25。本文只做结构研究和可执行实验设计。比赛事实来自 `F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf`：输入 RGB、热红外、Depth 和英文 Query；输出 RGB 图上的归一化 `xyxy` 框；Depth 是单通道 16 位毫米值，0 或过小值无效；官方指标是 `ACC@0.5`。允许外部训练数据和云算力。以下文献数值均是各自数据集上的结果，不能当作本赛预期分数。

## 决策

值得实现的主候选是 **RGB 锚点的 Query 条件三传感器 Grounder**（下称 QTS-Grounder）：从同场景三路图像提取不同表征，以局部跨模态注意力把 TIR 和有效 Depth 对应到 RGB patch 网格，再让 Query 决定在哪些区域采用辅助模态，用直接的空间定位头输出 RGB 框。这是独立的端到端定位网络，初次实验以 RGBT-VGNet 的 CLIP/视觉语言 Transformer 实现为可复核起点，**替换现有 27B 的三图自回归坐标生成作为被测预测器**。现有 27B Arm2 二倍分辨率的官方 0.8012 保留为外部强基线；只有同协议实测才比较成绩，不将 RGBT 文献成绩与本赛成绩横比。

结构创新点不是增加输入图像：当前 `scripts/train_qwen36_27b_swift.sh` 默认冻结 ViT 与 aligner，语言 LoRA 从三段独立图像 token 自学配准；`tools/prepare_qwen3vl_native_sft.py` 将 City 毫米 Depth 映为 8 位灰度三通道。QTS-Grounder 把传感器对应、Depth 有效性和 Query 条件传感器选择变成可训练的**预测路径**，并以空间损失直接优化框。它有机会改变「选中了 RGB 中相似邻居」「把无效深度或背景距离当目标距离」「夜间只凭 RGB 猜实例」这三种决定。它仍需实测证明，特别是本赛已有人工对齐但局部视差/遮挡未必完全消除。

## 完整论文与方法源码审读

| 方法与一手材料 | 对本题最有用的证据 | 实际实现与迁移边界 |
|---|---|---|
| [RGBT-GroundBench / RGBT-VGNet 论文](https://arxiv.org/html/2512.24561)、[官方仓库](https://github.com/crazyxiaoxi/RGBT-GroundBench/tree/e8869905d4f74a813a2fc667d922838db79f8acb) | 最接近本赛：RGB+TIR+英文指代表达→BBox。论文 pp.8–10 Fig.4 和式(1)–(10)是非对称 LoRA、语言感知视觉交互 LAVS、三先验融合 TPF；p.12 Table 4 的 RGB+TIR test ACC@0.5 为 RefFLIR 72.43、RefM³FD 74.89、RefMFAD 67.58，均优于表中相应可比双模态基线；p.13 Table 5 在 RefFLIR test 从无模块 46.19 到 AMA 71.17、AMA+LAVS 72.65、全模块 72.43，说明最后一层融合并非所有分组都增益；pp.20–40 附录含分辨率、MLLM、LoRA 分配与分组结果。p.21 Table 9 的原分辨率普通 MLLM 加 TIR 并不稳定。 | `models/mmvg_fusion.py` 是所报 RGBT-VGNet 核心：CLIP-B/16 的 RGB/TIR 两分支、RGB r16/TIR r48、视觉层内文字引导/跨模态交互、IAFv3 逐 patch 融合、`[Reg]` 框头；`models/vl_transformer.py` 是实际的视觉文字联合编码器。`datasets/data_loader.py` 把 RGB 三通道与 IR 单通道打包成四通道，使 `datasets/transforms.py` 同步几何变换；`datasets/__init__.py` 选择数据/变换。`script_train/RGBT_VGNet/ref_m3fd/train.sh` 经 `train_val/mmvg_train.py`→`engine.py`→`utils/loss_utils.py` 训练，`script_eval/RGBT_VGNet/evaluate.sh`→`train_val/mmvg_eval.py` 推理/打分。实际损失还含 CLIP 对比、区域文字 focal+dice 和 box 派生 mask 的 focal+dice，论文式(3)则只说明框预测。论文 p.10 的 illumination prior 写 min-max 归一；代码 `mmvg_fusion.py:1181-1188` 实际做 patch 灰度均值/标准差归一后 sigmoid。不可把论文的 TPF 公式直接说成当前代码完全相同。源码中的分割 mask 由框派生，并非真实轮廓标注。 |
| [DualVision 论文全文与补充材料](https://cdn.amazon.science/c4/f8/f6b6ab48441c8f2aab9fdbb47bfe/scipub-approval152129-42174719-dualvision-rgbinfrared-multimodal-large-language-models-for-robust-visual-reasoning.pdf)、[官方仓库](https://github.com/abrarmajeedi/DualVision/tree/37c85d9f52015b7f38c2c6edea0b2cc18c7b3bdf) | p.3 Fig.2 是 RGB patch 查其附近 IR patch 的多尺度局部 cross-attention，半径 1/2/3；pp.6–8 把 25k 配对图像的约 204k 问答做退化训练，p.8 Table 3 显示弱光/模糊/雾条件下稳健；pp.11–13 补充材料含消融、数据制作和实现，补充 p.11 说明多尺度局部版在 12 个退化条件中赢 8 个，干净图的固定半径版本略高。 | `core/llava/model/multimodal_encoder/{clip_encoder.py,local_xattn.py,fusion_encoder.py,builder.py}` 实现冻结 CLIP ViT-L/14 双图塔和 3 层 RGB→IR 局部注意力；`core/llava/model/llava_arch.py` 接 projector 与 LLaVA-1.5-7B；`core/llava/train/train.py`、`scripts/train.sh` 训练 LoRA 和 fusion，`core/llava/eval/my_model_vqa.py`、`scripts/eval.sh` 做生成式问答。它**没有框监督或框头**。实现固定把两图方形填充到 336，假定同形 token 网格；`local_xattn.py` 建立稠密 `Nq×Nk` 浮点 mask，再屏蔽非邻域，局部语义不等于二次内存自动消失；2D RoPE 的 docstring 在 forward 中没有对应计算。可借邻域匹配思想，不能直接搬问答训练、方形固定网格或宣称稀疏复杂度。 |
| [RDTTrack NeurIPS 论文及附录](https://papers.nips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf)、[官方仓库](https://github.com/xuefeng-zhu5/RDTTrack/tree/794de41ff3f52ed100e4f449cf22ceb4932e0d36) | p.5 说明 RGB/Depth 同一 ZED，独立 TIR 经手工特征点变换对齐；p.6 Fig.3 是 Depth+TIR 辅助提示注入冻结的 RGB OSTrack；p.8 Table 2 报 RGB+D+T 跟踪 AUC 0.752；p.10 Table 4 的同方法同数据二模态重训是 RGB+D 0.737、RGB+T 0.734，三模态确有增益，Table 5 移除 Depth-TIR 模块降到 0.733；pp.14–22 是附录/清单，无语言定位实验。 | `lib/train/dataset/rgbdt.py` 读三路，Depth **逐帧 min-max→8 位 JET**，物理值被抹除；`lib/models/rdtt/{vit_ce_prompt.py,vit_prompt.py,ostrack_prompt.py}` 实现 Depth/TIR 融合、逐层 prompt 和跟踪头；`lib/train/actors/rdtt.py`、`lib/train/base_functions.py` 管训练，`lib/test/tracker/rdtt.py` 从首帧 bbox 裁剪 template 并推理。论文叫 orthogonal projection，但 `DepthIR_ort_block.forward` 使用逐元素乘法除通道范数，未作向量点积投影，不能保证严格正交。其输入有上一帧/初始框而本赛只有文字，不可把跟踪 AUC 或模块原封迁移。RGBDT500 可作为三传感器配对数据线索，**其 Depth 数值物理单位未核实**。 |

上述三篇的主文、公式/图、实验表与补充材料均已读；图形证据另将 PDF 实页渲染核对：RGBT pp.8–13/21，DualVision pp.3/8/11，RDTTrack pp.5–6/8/10。只读了与方法相关的自定义训练、模型、数据和推理路径，未审计仓库中其它模型的全部实现，也未运行作者训练、下载权重或复现分数。

## QTS-Grounder 张量流

在第一版沿用 RGBT-VGNet 的 CLIP-B/16 预训练视觉/文字骨干与对照训练设置，以便复用发布的 RGB+TIR 定位监督。统一几何变换首先把三传感器与 RGB bbox 一起变换；保留原始宽高与仿射映射，输出时逆变换回 RGB 像素再归一化。输入并非把 Depth 伪彩或重复三通道送入 RGB ViT。

```text
RGB B×3×H×W ─ CLIP ViT-B/16 + RGB LoRA ─ R_s: B×h_s×w_s×C ───────┐
TIR B×1×H×W ─ 单通道输入映射 + 同源ViT/TIR LoRA ─ T_s ─ local-align→T̂_s ─┤
Depth B×1×H×W ─ [log(毫米),valid,局部相对深度/边缘] ─ CNN stem ─ D_s ─ local-align→D̂_s ─┤
Query ─ CLIP text encoder ─ token语义 S:B×L×C ───────────────────────────┤
                Query条件、有效值与模态可靠性门控：F_s=R_s+g_T ΔT_s+g_D ΔD_s ─┘
                多尺度 F_s + 文字 token → VL Transformer → RGB 网格定位头 → xyxy
```

其中 `s∈{16,32}` 是相对输入的特征步长，具体 ViT 中间层/merger 取点须按实际 tensor 形状确定；`h_s≈H/s,w_s≈W/s`，非固定方形。IR 单通道的输入映射可用 RGB patch embed 三通道权重求均值初始化；RGB/TIR LoRA 容量先遵循论文 r16/r48，是否适合本赛由消融决定。Depth 的 City 输入为 `valid=(d>0 且在设备有效范围)`、`log(1+d_mm)`、局部有效邻域相对深度及缺失掩码；对 `0` 的数值卷积贡献显式清零。**跨来源不能统称毫米**：RoboRefIt 固定 359–3662 可视映射的物理单位尚未知，RGBDT500 `sensor_linear_20000` 只是现有可视映射名，不能据此宣称其原图单位毫米。外部源若无校准/单位，只用有效值与场景内相对秩，带来源/单位标记；City 才使用米制通道。现有错误的 3854 行原始 Depth 直接按 RGB 读取所造成的截白数据必须重建输入，不能直接作为数值监督。

对每个 RGB 查询 patch `(u,v)`，只在 TIR/Depth 对应网格的 `k×k` 邻域计算交叉注意力；小的可学习偏移 `Δ(u,v)` 有明确像素上界，接收配准残差而不自由跳到远处。Depth 的注意力 key/value 用 patch 内有效率掩码；若没有有效深度，`g_D=0`。融合门控 `g_T,g_D∈[0,1]` 看 Query token、RGB 与对应传感器局部特征、TIR 对比度、Depth 有效率和匹配相似度；残差输出层零初始化，使刚开始训练时是同一 RGB 特征，随后仅在可提供证据的 patch 引入辅助信息。同步增强三图；另对单路 IR/Depth 做小幅已知位移增强，用来学习局部配准，记录位移真值并检查注意力是否跟随。遮挡时不能强迫每个 RGB patch 在 IR/Depth 都找到对应物。

定位头不生成数字字符串：文字条件 `[Reg]` token 对多尺度 RGB 锚定特征做 cross-attention；在 RGB 网格上预测 Query 相关的中心概率与四边距离，取最高分中心及其回归边界，输出一个 `xyxy`。训练 `L=λ_h L_heatmap + λ_1 L1 + λ_g L_GIoU`，其中框损失在逆几何映射后的同一 RGB 坐标系计算；有真实 mask 才加入 mask loss，不把 box 派生填充掩码当成真实轮廓。对同图多 Query 的语义相似目标，额外用实例区域与 Query 的对比损失，负样本仅来自**已标注不同目标**，避免把未标注相似物误判负例。最终 `ACC@0.5` 仍由原样 `xyxy` 计算。

这一路把 **Query→模态证据→RGB 网格→框** 连成可微路径。普通视觉 LoRA 只改变特征，而生成式 LLM 仍可能在长序列中忽略 Depth；本结构的 Query 门控和直接定位头会收到 IoU/中心损失的梯度。因此结构机制与既有继续 SFT 不同。

### 27B 的具体位置

首个实验以 CLIP-B/16 + 视觉语言 Transformer 为**替换基座**，从作者公开的 MMVG 预训练检查点及其 RGB+TIR 训练代码开始，先拿到可检验的三模态直接框模型；不把当前 Arm2 有错误 Depth 表示的 LoRA 迁给新网络。若 QTS-Grounder 的模态条件与配准机制在独立场景确实救回实例，迁回 27B 的方式是：保留训练好的三传感器编码器与 RGB 空间头，取 `F_s` 经小型 projector 压成一组带 RGB 二维位置的视觉 token，替换 Qwen3.6 的三张独立图像视觉输入；Query 仍交给 Qwen3.6 语言骨干，语言末层 Query 表征与 RGB 空间头联合给框。训练只开放 projector、传感器 LoRA/融合、语言 LoRA、空间头，用框损失与少量原生答案 CE。必须按 Qwen3.6 的实际 `image_grid_thw`、视觉 merger 和位置编码实现新的加载/前向/推理入口；旧 Qwen3-VL DeepStack 代码不能机械复用，当前 vLLM 普通 LoRA 入口也不会自动支持自定义视觉融合。第一轮不把这条尚未实现的迁移算入预期收益。

## 第一轮完整实验

**假设：** 对真正需要 TIR/Depth 的 Query，显式 RGB 对应与 Query 条件门控，相比等数据/等骨干的三路独立 token 融合，提升单次直接框 `ACC@0.5`，且 IR/Depth 打乱会选择性破坏相关样本；普通 RGB 主导题不大幅倒退。

1. **材料与切分。** 用 RGBT-GroundBench 官方 train 先训 RGB+TIR 分支和定位头，val 选 checkpoint，test 只作一次独立外部诊断；三来源子集不要跨图/邻近帧泄漏。再用本项目已持有、描述与框经核对的训练材料做三模态调优，其中 City 保留原始 16 位毫米 Depth，RGB/IR/Depth 和框统一增强；RoboRefIt 等来源若单位不清用相对 Depth 通道并记录来源。按**场景/图组**分组留内部验证，不能随机按 Query 拆分同图。City412 已是长期开发集，只供机制与回归诊断，不把它叫独立证据；原 91 人工复核题中尚未确认的框/Query 不进训练真值。
2. **主对照。** A：RGB-only、同一骨干/文字编码/定位头；B：RGB+TIR+Depth 各自编码但仅同位置直接加/拼后投影，无局部对齐或 Query 门控；C：完整 QTS-Grounder。A/B/C 同初始化、可用训练图、输入尺寸、优化步数与框损失；B/C 使用相同 Depth 数值表示且总训练参数/计算预算尽量匹配，必要时给 B 同容量无跨模态 MLP。RGBT 预训练相同，Depth stem 初始化相同；至少三随机种子报告 scene bootstrap 区间。现役 27B Arm2 官方 0.8012 作为实用门槛，单列比较，不能把它与 A/B/C 的结构因果差混淆。
3. **过程与观察。** 先实现数据、RGB/TIR/Depth stem、两尺度局部匹配、门控及直接头；在数十个三图样本做视觉坐标/valid mask/逆变换人工核对，再训练 A/B/C。以分组验证的 `ACC@0.5`、mean IoU 和同图配对救回/破坏为主；按夜间/弱光、小目标、相似邻居、有效深度比例与含距离/序数关系 Query 分层。对 C 分别将 TIR、Depth 置零或**跨场景错配**重推理，观察有证据样本是否损失、无证据样本是否稳定；记录门控热图与匹配偏移，人工检查其是否落到目标主体而非背景。对与 B 有分歧的框，归因为实例、范围、配准还是语义错误。
4. **裁决。** C 若在独立场景上的配对收益显著高于 B 且没有大规模 RGB 回归，尤其在 TIR/Depth 有效层的提升与错配破坏相吻合，即投入 27B token 迁移或直接用 C 参赛；若只在 City412 涨、跨场景不涨，先审计传感器数据域和标签，再调容量；若 A≈B≈C 且模态错配无效，说明没有学到传感器证据，停止堆复杂融合。正式官方得分必须以提交后回传为准。

## 主要困难与先验检查

- **对齐误差和深度空洞。** 官方人工空间对齐不保证任意目标轮廓像素级重合，RDTTrack 的跨机手工变换是反例；先用可视化估计真实局部位移尺度，再定 `k`/最大偏移。无效深度不能当近物体，深度统计要绑定前景主体。
- **数据规模和域差。** RGBT-GroundBench 的 FLIR/M3FD/MFAD 是配对 RGBT、无本赛 Depth；RGBDT500 有三路但只有跟踪初始化框、无英文指代。二者都不能直接替代本赛 Query→框。RGBDT500 的深度单位、对齐文件和使用许可需要在真正接入前逐项核对；首实验不依赖它产生伪英文标签。
- **小物体与算力。** RGBT-VGNet 论文在 224×224 训练；本赛高分 27B 使用更高每图像素预算。初次方法比较以同一较高长边/patch 网格进行，保留原分辨率逆映射；局部 attention 要用窗口 gather 或 deformable sampling 实现，不能照搬 DualVision 的稠密 `N×N` mask 再声称降显存。
- **模型接口。** 首实验作为独立 Grounder 有明确训练/推理链；若迁入 Qwen3.6，必须单独验证 patch 顺序、merger、二维位置、LoRA 命中和自定义 HF 推理与提交一致性。结构性融合的难点不应被包装成一个配置开关。

