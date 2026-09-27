# G/U退化调查：论文检索与可迁移结论

检索日期：2026-09-26。范围覆盖三模态跟踪、RGB-T语言定位、深度空间问答、模态适配、多任务干扰、生成式框精度及捷径学习。检索使用24个不同搜索问题，并定向阅读作者仓库、arXiv全文、CVF和NeurIPS论文；以下保留直接相关的15项及1项补充，不把检索结果数当研究深度。本文件只整理研究证据，不宣布新训练方案获批。

## 最贴近本项目的证据

| 工作与原文 | 实际研究内容 | 对本项目的启发 | 不可直接外推之处 |
|---|---|---|---|
| [RGBT-GroundBench / RGBT-VGNet，2025预印本、2026修订v2](https://arxiv.org/html/2512.24561v2) | 原文§4、§5.3、补充§7：针对TIR的非对称视觉LoRA、语言引导交互、可靠性融合；其MLLM结果显示直接附加TIR的收益依模型与数据而变 | 读入两图不等于有效融合；优先测模态贡献，再决定是否需要视觉适配 | CLIP定位模型与我们的Qwen语言LoRA不同；论文分数不是比赛预期。原始分辨率实验与224输入也不能混比 |
| [RGBDT500 / RDTTrack，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf) | 三模态跟踪，训练代表帧标注；RGB-D同步、TIR映射对齐；刻意包含几何变化很小的平面对象及模态扰动 | 数据集有三种传感器，不代表每帧适合深度前后关系教学；跟踪目标不等于完整竞争对象标注 | 跟踪已知首帧目标与自然语言选实例不是相同任务；论文不能证明本地PNG的单位/方向 |
| [SpatialRGPT，NeurIPS 2024](https://arxiv.org/html/2406.01584v3) | §3.1–3.4：对象掩码/3D关系图派生问答；共享视觉编码器外增深度到语言连接器，并有分阶段适配 | 区域对应、关系标签和深度表征适配应形成连续链条；仅输出辅助图上的框是不完整监督 | 区域提示与专用结构带来额外条件；不能把有GT区域的关系准确率当端到端定位 |
| [SpatialBot，2024](https://arxiv.org/html/2406.13642v3) | §3：深度编码、像素读数、对象对应、近远比较、应用任务分层；可调用原深度读数工具 | 先确认深度值含义与对象取样，再教比较；框内地面/背景不能替代物体读数 | 各下游指标并非全面提升；它的特定编码及部分工具调用结果不能视为当前灰度图输入已具备的能力 |
| [SpatialVLM，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_SpatialVLM_Endowing_Vision-Language_Models_with_Spatial_Reasoning_Capabilities_CVPR_2024_paper.html) | 从对象、分割和深度构造空间问答，研究数据及训练因素 | 应把监督落在对象间关系，而不只堆图像描述；规则派生关系可降低逐题生成成本 | 大规模RGB空间问答不等于真实传感器Depth融合；不能据其规模要求本项目也收亿级题 |
| [DepthLM，2025](https://arxiv.org/html/2509.25413v1) | §3：像素指代与跨相机尺度歧义，视觉标记、内参条件增强；SFT与RL比较 | 必须先查“模型是否找准读取的区域”；不要遇到数值任务便直接上RL | 研究从RGB估计度量深度，并非读取本赛题传感器Depth；不能据此用估计图替换原始观测 |

## 视觉适配与精细定位

| 工作与原文 | 关键方法/结果类型 | 本项目如何使用这条证据 |
|---|---|---|
| [Thermo-VL，2026预印本](https://arxiv.org/html/2605.21882v1) | 冻结Molmo RGB/语言主干，训练部分热红外编码器及按Query调节的残差融合。§5是问答评估，非本比赛BBox验证 | 支持“保住已有RGB能力、让辅助分支学如何补充”的结构假设。不是立即换Molmo的依据；其80GB H100训练资源也不等于24GB可直接复现。论文含合成热图，不能当真实额外观测 |
| [IRGPT，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_IRGPT_Understanding_Real-world_Infrared_Image_with_Bi-cross-modal_Curriculum_on_Large-scale_ICCV_2025_paper.pdf)；[官方数据仓库](https://github.com/WheatCao/ICCV2025-IRGPT) | 真实IR语言数据及由可见光到IR的课程适配，多项IR任务评估 | 后续IR表征确有瓶颈时，可参考真实IR语言数据与难度组织；IR单模态能力仍不能证明三模态联合收益 |
| [Qwen3-VL技术报告，2025](https://arxiv.org/html/2511.21631v1) | §3.2.4采用0–1000归一化坐标；§3.2.5把对象定位和空间关系数据分开建设 | 核查本地坐标与提示，避免将关系能力视为框监督的自然副产品。官方预训练配方不能直接用来解释本轮某5题退化 |
| [VGent，2025预印本](https://arxiv.org/html/2512.11099v1) | 把高层指代理解与候选框选择分开；讨论目标粒度/标注范围的不一致 | 本轮边界翻转提示可先诊断框精度与候选覆盖。未来候选来自真实检测；不能用GT候选或跨实例平均框假装提分 |
| [Hi-Token，2026预印本](https://arxiv.org/html/2608.03471v1) | 分层坐标token与几何奖励，研究输出表示对框精度影响；附录列较重的训练配置 | 常规token损失与框几何误差不等价，是合理研究方向；不据此断言我们掉分就是token化造成，不直接改词表或引入昂贵RL |
| [GroundingME，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_GroundingME_Exposing_the_Visual_Grounding_Gap_in_MLLMs_through_Multi-Dimensional_CVPR_2026_paper.pdf) | 从相似目标辨别等多维度测试定位，强调简单基准不能代表复杂指代 | 数据验收要包括竞争对象与Query约束，不只看能否找到某个框；先检查我们真实错误构成再挑训练题 |

## 解释机制的参考，不能当本实验因果证明

| 工作 | 能支持的机制 | 在此处的限制 |
|---|---|---|
| [What Makes Training Multi-Modal Classification Networks Hard?，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html) | 不同模态学习/泛化速度不同，更多输入未必自然提分 | 其增加容量/分类实验不同于固定Qwen主干语言LoRA；不能直接说本轮过拟合已证实 |
| [Balanced Multimodal Learning via On-the-Fly Gradient Modulation，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Balanced_Multimodal_Learning_via_On-the-Fly_Gradient_Modulation_CVPR_2022_paper.html) | 强模态主导优化，弱模态表征可能学不足 | 独立分支梯度调节不宜直接套在共享Qwen三图tokens；我们尚未测到分模态梯度因果证据 |
| [Gradient Surgery for Multi-Task Learning，NeurIPS 2020](https://arxiv.org/abs/2001.06782) | 多任务梯度冲突可使共享参数训练互相干扰 | U变更输出模态可能产生任务干扰，但未测梯度夹角，不能把这当已确定根因 |

补充阅读：[VQA-CounterExamples，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Dancette_Beyond_Question-Based_Biases_Assessing_Multimodal_Shortcut_Learning_in_Visual_Question_ICCV_2021_paper.html)说明答对可依赖视觉与文字共同捷径。对本项目的推论：跨模态框坐标相似时，可借RGB定位/几何对应完成辅助答案；因此需要独立输入组合诊断，而不是用辅助输出数量证明利用了IR/Depth。此处只提出可检验假设，未断言模型实际走了该捷径。

## 研究结论对下一步的约束

1. 三模态主线应保留，但“在辅助图上再框一次”与“辅助图改变了目标选择”要分别定义与测量。
2. 优先增加真实竞争对象、可靠关系及辅助模态解决的具体歧义；现有对应数据保留为基础能力材料，不继续把它算作已完成的深度推理数据。
3. 对新方向先用已有检查点和冻结诊断题测试输入组合、辅助任务学习和最终框精度；结果再决定训练目标或轻量视觉适配，不能一次修改多个因素。
4. 论文启发不能替代本地复现；VQA、跟踪、3D框与本比赛2D语言定位有边界，所有外部数字不作为收益承诺。

另检索但未列为近期实施建议：N3D-VLM（3D专用表示）、MILES（分支学习率平衡）、SpaceMind（相机引导融合）、UZ3DVG（3D场景任务）、GUI zoom类方法。它们有研究关联，但任务条件或工程成本与当前实验不够接近。
