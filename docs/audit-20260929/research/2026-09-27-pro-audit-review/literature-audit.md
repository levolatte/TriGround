# Pro 审计所引文献核查（2026-09-27）

本报告核对用户所附 Pro 审计文本涉及的五篇工作，并选取一篇与测试时自我修正直接相关的原始论文作交叉参照。**五篇均可在作者预印本核实；SpatialRGPT 另有正式 NeurIPS 页面，SpatialRGPT 与 Rex-Thinker 有可访问的作者代码仓库。**依据为论文正文、表格、附录和这些项目页；文中的数字均是**论文报告值**，不是本项目复现实验。没有下载权重、运行模型或修改生产代码。对 AIC 任务的适用性判断还参照了本仓库已保存的原生 Qwen3-VL-8B LoRA 执行链与 City 412 条离线评估；该本地证据另见 [执行链审计](../2026-09-27-takeover-audit/execution-audit.md)。

## 核查结果总览

| 工作与原始来源 | 核实的任务、方法和证据 | 对本赛可借用的部分及边界 |
|---|---|---|
| **RGBX-R1**，Jiahe Wu、Bing Cao、Qilong Wang、Qinghua Hu、Dongdong Li、Pengfei Zhu；[arXiv v1，2026-01-31](https://arxiv.org/html/2602.00504) | 正题名 *RGBX-R1: Visual Modality Chain-of-Thought Guided Reinforcement Learning for Multimodal Grounding*。将五个 RGB-X 视频跟踪集改为多图定位；输入文本、RGB 与 X 模板及搜索帧，通常输出 **6 帧的 6 个框**，X 涵盖热红外、深度、事件。指标为每搜索帧 IoU>0.5 的正确率再等权平均，表 1 在**模态已知**提示下评估。先用带框引导生成并筛选的视觉模态思维链做冷启动监督微调，再做 GRPO 式时空奖励训练。表 1 的同榜平均值：Qwen2.5-VL-7B 直接坐标 SFT **31.04**；只做思维链冷启动的 `-cs` **19.64**；直接 SFT 后再强化的 `-sft+st` **32.74**；完整 RGBX-R1-7B **46.53**。 | 可借“先识别 RGB 目标—与其他模态对齐—验证模态是否真有补充信息”的**监督数据设计**和分模态对照。其视频模板、逐帧多框、Qwen2.5-VL、强化阶段均与 AIC 单场景 Query→**RGB 坐标单框**不同；表中绝对准确率不可横比。论文附录对热图亮暗和深度亮暗写有固定成像解释，迁移前须按本赛实际编码核验。本赛不应直接照搬长思维链或强化训练规模。 |
| **SpatialRGPT**，An-Chieh Cheng、Hongxu Yin、Yang Fu、Qiushan Guo、Ruihan Yang、Jan Kautz、Xiaolong Wang、Sifei Liu；[arXiv v3，2024-10-15](https://arxiv.org/html/2406.01584)、[NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/f38cb4cf9a5eaa92b3cfa481832719c6-Abstract-Conference.html)、[作者代码](https://github.com/AnjieCheng/SpatialRGPT) | 正题名 *SpatialRGPT: Grounded Spatial Reasoning in Vision-Language Models*。用检测、分割、单目公制深度估计及相机几何构造 3D 场景图，生成区域级空间问答；模型将 RGB 区域和**相对深度图**区域特征接入语言模型。推理时可给框/掩码作区域输入，主要输出空间问答，**并非直接生成本赛目标框**。 | 可借候选区域、前后/远近/左右关系的**显式监督与诊断题型**。其 3D 场景图依赖提案、分割、深度和相机估计质量；论文深度接入不是“把本赛深度图当第二张图片就自然增益”，也不能把其问答成绩当 AIC ACC@0.5。 |
| **Rex-Thinker**，Qing Jiang、Xingyu Chen、Zhaoyang Zeng、Junzhi Yu、Lei Zhang；[arXiv v1，2025-06-04](https://arxiv.org/html/2506.04034)、[作者代码](https://github.com/IDEA-Research/Rex-Thinker) | 正题名 *Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning*。开放词汇检测器先给候选框，模型依编号对候选作计划、逐一核对、总结；HumanRef-CoT 有 **90,824** 条经 GT 答案筛选的思维链，随后做 SFT 与 GRPO。论文在**不加思维链监督**的 box-hint 消融里，提示候选框相对无提示平均 Recall/Precision/Density-F1 分别高 **13.2/11.7/10.8 个百分点**。 | 与本赛最贴近的是**保留强候选器、训练可核验的候选选择器**，并报告候选召回上限、选择正确率、最终框分数三层。原任务为单 RGB 人物引用，可多实例或无目标；其 box hints 有检测器召回前提，且 HumanRef 大规模 CoT+GRPO 的提升不能外推为本赛少量辅助模态数据有效。 |
| **RGB-Th-Bench**，Mehdi Moshtaghi、Siavash H. Khajavi、Joni Pajarinen；[arXiv v3，2025-03-30](https://arxiv.org/html/2503.19654) | 正题名 *RGB-Th-Bench: A Dense benchmark for Visual-Thermal Understanding of Vision Language Models*。**29 对 RGB/热图、1,624 道人工 Yes/No 题**，14 项能力、每对每项 4 问；其中 7 项只给 RGB，7 项给 RGB+热图。问级准确率与“四问全对”的技能级准确率分别以 50%/6.25% 为随机基线。论文评测 19 个可接多图的 VLM；这是一套**诊断基准而非训练方法或 BBox 数据集**。 | 可借“配对图像、同一技能多道正反题”的测试设计，分别检验热图对齐、目标温度属性、相对冷热与异常，而不只看总 ACC。它不证明红外输入能提升 AIC 定位；场景偏工业/住宅且仅 29 对，热色图及温度标尺也不同于本赛红外成像。 |
| **LFPR**，Bo Ma；[arXiv v1，2026-08-20](https://arxiv.org/html/2608.19553) | 正题名 *Where Grounding Accuracy Lives on the IoU Curve: Label-Free Inference-Time Boundary Refinement*。冻结 Qwen3-VL-8B，按**预测框**是否小于 128 像素决定更高分辨率重推；对局部裁图另生候选，用纯几何门槛决定接纳，再与原框取坐标中点。Ref-L4 回顾性 31,921 条：Acc@0.5 **88.531→89.725**、Acc@0.9 **55.788→61.142**、mAcc **72.947→76.013**。冻结的 RefCOCO 家族迁移 Acc@0.5 **89.212→90.029**，Flickr30K 前瞻 merged-box **74.864→75.678**。 | 可借**无需 GT 的候选接纳门槛**、小框分辨率分流、分阶段同样本消融，以及 ACC@0.5/0.7/更高 IoU 曲线一起看。局部重定位不是无条件收益：该论文自己的不同数据集上，crop/guard/fusion 的边际方向有差别；本项目已有无条件局部细化 221/412 对 C 292/412 的负结果，不能直接复制其阈值或预期增益。 |

## 容易误读的比较

**RGBX-R1 的 19.64 与 31.04 是同一论文表 1、同为 Qwen2.5-VL-7B 和 RGBX-Grounding 平均分，所以可以描述为该表里“第一阶段思维链冷启动不如直接坐标 SFT”。**但这不是只改变“是否输出思维链”的严格单因素实验：监督目标、答案长度、训练阶段不同。论文称 RGBX-Grounding 总共约 7.4k 组、筛后保留逾 5k 条思维链，同时将 SFT 基线描述为在 RGBX-Grounding 微调；正文不足以证明两臂训练样本、token 预算和目标损失完全匹配。因此不能从这 **11.40 个百分点差** 推断“思维链本身使本赛单框分数下降 11.40 点”，也不能用完整两阶段的 46.53 推断本赛只加解释文本就会涨分。[任务定义与思维链构造](https://arxiv.org/html/2602.00504#S3)、[表 1 与基线定义](https://arxiv.org/html/2602.00504#S4)。

**LFPR 对无 guard 的表述必须限定数据集与对照臂。**RefCOCO 家族 30,969 条中，同一候选的无 guard 替换臂 `CR†` 在 Acc@0.5/0.75/0.9/mAcc 上确实都低于原始 `B`；Flickr merged-box 的同型替换臂亦如此。但 Ref-L4 附录 2×2 表中，无 guard **中点融合** Acc@0.5 为 88.262（低于 B 的 88.531），而 Acc@0.9 为 **59.350（高于 B 的 55.788）**、mAcc 为 **73.864（高于 B 的 72.947）**；无 guard **直接替换**的 Acc@0.9 **62.059**、mAcc **74.491** 也高于 B。故摘要里笼统的“无 guard 对每项指标都劣于原方案”不能覆盖全部 Ref-L4 表格；可以稳妥说 guard 在相同候选的迁移拆解中防止大幅回退，且 Ref-L4 上受 guard 保护的中点融合优于无 guard 中点融合。[迁移五臂表](https://arxiv.org/html/2608.19553#S5.SS2)、[Ref-L4 2×2 附录表](https://arxiv.org/html/2608.19553#S13.SS2)。

**GT 边界要区分训练标注、分析与部署。**RGBX-R1 和 Rex-Thinker 均在构造训练思维链时利用真值目标筛选，这属于训练监督，不等于推理泄漏；若把类似标签、候选 oracle 或最佳迭代步用于本赛验证阶段选择，则不能叫无标签方法。LFPR 的路由依据预测框而非 GT；其 Ref-L4 阈值与设计受同一测试集开发过程影响，论文也自行标为回顾性证据。另有 [*Iterative Visual Thinking and the Self-Correction Mirage in VLM Grounding*，Tripathy 与 Krishnan，arXiv v2 2026-07-06](https://arxiv.org/html/2606.13156)：作者报告的 +2.4 点单 RGB Grounding 自我修正增益依赖按 GT 选最佳迭代步；改用可部署、无标签停止规则后增益消失。这支持本赛把“GT 只评分”和“推理选择策略”分开审计；该论文同样不是本赛复现证据。

## 面向现有 AIC 证据的具体取舍

1. **优先把问题改写为候选覆盖与决策问题。**现有 C 为 292/412，B/G★/U★ 为 287/284/286，少量辅助模态训练未超 C；先用已有预测离线量目标是否落在候选集合中，再在**训练集**上构建“query、RGB/IR/Depth、带编号候选、正确候选 ID/无候选”的小型监督任务。与 Rex-Thinker 相似之处是对候选作受监督选择；其优势是否在 AIC 成立只能用完整 412 条冻结验证，且候选生成和选择不得读取验证 GT。
2. **先做分模态配对诊断再扩大辅助数据。**对红外/深度确有判别力的 Query 单独列出，比较 RGB-only、RGB+IR、RGB+Depth、全模态在同一 ID 上的改进和伤害；参考 RGB-Th-Bench 的热图对齐、属性与关系分类及 SpatialRGPT 的区域关系题型。现有 B→G★ 同时换图、Query、答案，G★→U★ 同时换 Query 和答案，不能据净分归因某种模态“学会了”。新数据应同时保留只改输入模态、只改监督目标等可解释配对。
3. **局部细化只作受控的后续假设。**先拆开高分辨率重推、crop 候选、无标签 gate、坐标融合各阶段，在同一 412 条上保留失败、无框和解析错误分母；提前固定基于预测框的门槛，再看 ACC@0.5、ACC@0.7、IoU 分布和成对翻转。LFPR 提示其门槛可能必要，也显示数据集转移时严格 IoU 可与宽松阈值相反；它不能抹掉本项目 221/412 的已观察失败。

## 查阅范围与未证实项

- RGBX-R1：读方法 3.1–3.4、实验 4.1–4.3、表 1 和附录数据统计；没有找到并核实作者公开代码/权重，依据为作者预印本。样本/训练 token 匹配程度不明。
- SpatialRGPT：读方法 3.1–3.4、实验与附录中的数据/架构说明；核实 NeurIPS 页面及作者 GitHub 仓库存在，未下载权重或运行代码。
- Rex-Thinker：读数据构造、方法、实验消融与附录说明；核实作者 GitHub 仓库存在，未复现 HumanRef 或 GRPO。论文的候选框消融指标属于 HumanRef，不是 IoU ACC。
- RGB-Th-Bench：读数据、14 技能、评价公式、实验与限制；正文提供数据链接且称评测代码“将公开”，此次未核实公开代码。未检查原始 29 对图片，也未复算答案。
- LFPR：读方法、主表、迁移拆解、Ref-L4 附录 2×2 和证据分层；预印本内提到的项目代码仓库此次未能核实可用，故无法独立审计其原始预测或置信区间。上述 Ref-L4 表格矛盾来自同一作者正文/附录的对照范围，不是重新实验所得。
- 自我修正补充论文：读摘要、评价协议与无标签停止规则；仅作为方法学交叉参照，未纳入五篇主证据的分数比较。

以上所有跨论文比较仅比较**机制与评估设计**，不将不同数据集、单框/多框任务、模型版本或 IoU 口径的分数拼成统一榜单。
