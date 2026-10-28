# 对象槽、参照物与可学习出框：替代三图直接生成坐标的结构方案

本文为该专题的独立方案，综合架构与实施优先级见[主报告](README.md)。

2026-10-25。研究问题是：现役 Qwen3.6-27B 在 RGB、红外、深度三图上直接生成四个框坐标，若错误主要来自对象边界、同类实例选择和关系绑定，是否应把“对象”变成模型内部可监督的单位，并由专用定位头输出框？本文是论文与源码审读及首轮实验设计，没有运行新模型或声称新成绩。竞赛事实以 [官方赛题](../../../../contest/基于大模型的多模态视觉理解与推理.pdf) 为准：输入空间对齐的 RGB、红外、深度与英文 Query；输出 RGB 坐标系的归一化 `xyxy` 唯一目标框；指标是 IoU≥0.5 的 ACC。深度为毫米，0 或过小值无效，数值越大越远；红外文件为三通道重复灰度。

## 结论先行

**建议首测“文本调制对象槽 + 目标/参照物关系打分 + 直接框回归”**。用公开的 MDETR 类定位预训练权重初始化对象槽，模型给每个槽同时预测框、短语匹配和目标概率；对于关系句，再对有序的“目标槽、参照槽”打分，最终选一个目标槽的连续框。红外和原始深度以 RGB 对齐坐标接入对象特征，深度无效区域带有效掩码。这样既保留对象级选择，又允许框头在训练中学习边界；不需要语言模型学会精确吐出四个数，也不把最终框限制在冻结检测器的候选框集合。它是**待检验假设**，不能从公开 RefCOCO 成绩直接推出本赛会提分。

第一阶段只比较同一初始化、同一训练数据与预算下的目标槽基线、显式关系头、关系头加辅助模态。现役 27B 固定为业务基线，独立场景上的 ACC@0.5 才决定是否替换或组合。City412 已长期用于开发；其旧失败 91 条、标签修订后的 53 条和新增/遗留错误只用于机制诊断，不承担独立提分证据。现有实验事实见[下一轮计划](../2026-10-24-next-plan/README.md)和[结构审查](../2026-10-24-next-plan/review-architecture.md)。

## 三篇论文给出的结构证据与边界

| 工作 | 对象如何形成，如何出框 | 与本赛最相关的证据 | 不能直接外推之处 |
| --- | --- | --- | --- |
| [Groma，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01023.pdf)及[补充材料](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01023-supp.pdf) | DINOv2-L 在 448² 图上取特征；类无关 Deformable DETR 先提 300 区域，NMS 后最多 100 个；多尺度 RoIAlign 把区域变成 `<rN>` token，Vicuna-7B 选择 token，再返回该区域框。RGB patch 每 2×2 合并。 | 将“看哪个物体”和“框的数值”分离；论文 RefCOCO 系平均 86.52，LVIS-Ground 平均 AR 28.8。 | 小目标 AR@S 只有 8.7；最终框受 proposal 召回约束。更关键的是官方实现训练 `groma/model/groma.py:220-303` 在 `no_grad` 感知器中把 `ground_boxes` 注入 proposal，然后让语言模型学习选 `<gbox>` 对应区域，推理没有 GT 注入。不能把这种训练设定解释为推理候选源已能覆盖目标。 |
| [Grounding DINO，ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)及[补充材料](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319-supp.pdf) | 图文特征先双向融合，图像特征与词特征相似度选前 900 个 query anchor；6 层 decoder 依次做槽自注意力、图像交叉注意力、文本交叉注意力，由 MLP 逐层修框；词 token 对比得分决定短语。 | 论文第 12 页表 5：只用 O365/GoldG 的 RefCOCO val 为 50.41，加入 RefC 训练到 73.98，RefC 专项微调到 89.19。补充材料第 12 页图 8 有同类人物左右选错实例。第 9 页表 16，BERT-L 对比 BERT-B 没有稳定收益。 | 强开放词汇检测并不自动得到唯一指代表达能力。其 [公开仓库](https://github.com/IDEA-Research/GroundingDINO)在审读版本中主要是推理模型，`groundingdino.py:353` 的辅助训练输出被注释，仓库无论文对应完整 criterion/训练实现；论文中的 Hungarian、focal、L1、GIoU 损失不可误称已由当前仓库验证。 |
| [MDETR，ICCV 2021](https://arxiv.org/pdf/2104.12763) | CNN 图像 token 与 RoBERTa 文本 token 投影到同一空间、拼接后经过联合 encoder；100 个学习对象 query 经 decoder 直接预测归一化 `cxcywh` 框与 token 分布。Hungarian 匹配后用 soft-token、对象↔token 双向对比、L1、GIoU 训练。 | 文本短语和对象对齐是可监督量。CLEVR-Ref+ 额外 `isfinal` 头区分最终目标与关系里的中间参照物；论文合成唯一目标子集 100%（不是自然场景证明）。附录第 18 页表 8：合成 CLEVR-Medium 类无关 AP 基线 99.0，去掉对比对齐 83.2，去掉 soft-token 87.7。 | 公开自然图像模型是 RGB 的；其 `isfinal` 使用显式参考对象标注，本赛训练数据是否有这种标注须先核对。官方代码 decoder 的独立 text cross-attention 被注释，文字信息通过联合 encoder 进入 decoder；不能误描成与 Grounding DINO 完全同构。 |

共同指向：**先生成可学习的对象表示，再做目标和参照物选择，最后由框头回归坐标**。Groma 证明区域 token 可供语言模型选择，但其硬 proposal 可能漏小目标；Grounding DINO 证明只会按类别提物体远不够；MDETR 提供可复用的训练式对象槽和明确的目标/中间对象区分。三者都没有证明本赛的红外/深度收益。

### 审读覆盖与源码落点

主文/附录逐页阅读：Groma 主文 19 页+补充 3 页；Grounding DINO 主文 18 页+补充 15 页；MDETR arXiv PDF 22 页，含附录。对关键图、公式、表格另渲染核对：Groma 第 5、7、11 页；Grounding DINO 第 5、6、7、12、14 页及补充第 2、8、9、12 页；MDETR 第 2、3、4、18 页。渲染文件留在 `F:/AIC/tmp/architecture_rethink_20261025/objects/`，未复制进仓库。

对方法相关自定义源码的审读范围：

- [Groma 源码](https://github.com/FoundationVision/Groma/tree/d3a95b6c2657dcb5c05f326da8895f521cf60009)：`groma/model/{groma,ddetr,ddetr_transformer,roi_align}.py`，`train/{train_det,train}.py`，`data/{collator,datasets/refcoco_rec,datasets/det_data,configs/vl_pretrain}.py`，`eval/{eval_rec,model_refcocog}.py`，检测预训练脚本。检查了 proposal 后处理、GT 注入、区域 token 到框的映射和训练阶段。
- [Grounding DINO 源码](https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44)：`groundingdino/models/GroundingDINO/{groundingdino,transformer,fuse_modules,bertwarper,utils}.py`，`groundingdino/util/inference.py` 与 `config/GroundingDINO_SwinT_OGC.py`。检查了 900 query 初始化、图文融合、逐层框修正、token 相似度输出及公开训练代码缺口。
- [MDETR 源码](https://github.com/ashkamath/mdetr/tree/ea09acc44ca067072c4b143b726447ee7ff66f5f)：`models/{mdetr,transformer,matcher,postprocessors}.py`，`datasets/{coco,refexp}.py`，`engine.py`，`configs/refcoco.json`，README 的公开权重/RefCOCO 指南。检查了联合 encoder、`SetCriterion` 的 soft-token/双向对比/isfinal/框损失、Hungarian 分配和推理分数。源码中的 RefExp evaluator 用 GIoU 条件，不能替代本赛的普通 IoU≥0.5 评估。

## 方案 A：目标与参照物对象槽定位器（优先）

### 前向与输出

1. **语言和对象槽。** 用 MDETR 类 RefCOCOg 公开权重起步：RGB 多尺度特征与 Query token 经联合 encoder；100 个对象 query 经 decoder 得到 `(h_i,b_i)`，`b_i=(cx,cy,w,h)`。训练和推理都从模型自己预测的 `b_i` 出框。目标名词、属性、关系词无需人工规则硬切；token 对齐头保留短语监督。公开 checkpoint 只作为初始化，首轮先运行冻结推理测域偏差，再做本赛训练适配。
2. **目标与参照物。** 对每个有序对 `i≠j` 形成 `z_ij=[h_i,h_j,Δx,Δy,log(w_i/w_j),log(h_i/h_j),IoU(b_i,b_j),q_rel]`；`q_rel` 是 Query 的关系 token 注意力汇聚。对 `j` 做 masked softmax 聚合，得到每个 `i` 被关系支持的分数。另一条 `s_i^direct` 从对象 token 与目标短语对齐给出目标分数；最终 `s_i=s_i^direct+λ·relation_i`，`argmax_i s_i` 的 `b_i` 经 `cxcywh→xyxy` 输出。无显式关系的 Query 令 `λ=0`。序数/左右/前后可以由几何特征和文本学习；不要把“最近”“第三个”写成确定的手工规则。`i≠j` 防止自指。
3. **红外/深度接入。** 以 RGB 坐标为锚，分别用轻量 IR stem 和原始毫米深度 stem 形成对齐特征；深度输入包含有效掩码，0/过小区域不被当作近物。各模态与 RGB 特征通过零初始化门控残差融合，提供给同一对象 decoder；针对每个预测框再汇聚有效主体深度的分位数、有效比例，形成 `d_i`，仅在 `i,j` 都有足够有效值时把 `d_i-d_j` 送入关系头，否则关系头看到显式无效标记。低可信深度不得支配顺序。红外用于区域外观/轮廓，深度用于数值距离；它们是否有实际贡献由单独消融证明。
4. **提交接口。** 唯一目标框先在 RGB 原图做坐标还原，再裁剪到 `[0,1]` 的合法 `xyxy`；推理不输入 GT 或人工标框，不输出中间参照框。若同一图多 Query，RGB/IR/Depth 特征可缓存，文本和对象 decoder 逐 Query 运行。

### 监督、梯度与失败模式

公开阶段利用 MDETR 的带短语框数据预训练对象槽与 token 对齐；有可靠目标/参照物框的公开关系数据可训练配对头。**本赛训练数据只有最终目标框时，不伪称参照物真值存在**：Hungarian 在对象槽和唯一 GT 之间分配，匹配槽做 L1+GIoU，所有槽做目标/非目标排序；参照槽的注意力是潜变量，只由最终目标监督间接更新。若公开关系数据不足，关系头可能坍缩到任意背景或 `s_i^direct`，需报告参照注意力可视化并做 `λ=0` 消融。对近似同类但不同实例的模型预测槽，使用目标排序损失，但不把未核实候选框当 GT。训练期 GT 只能用于匹配和损失，不注入推理对象集合。梯度应能到关系打分、对象 decoder 和框头；首轮可冻结大部分 RGB backbone，逐层解冻由独立验证决定。

全模型的框损失只监督最终目标时，**场景完整对象召回**不一定增长；若目标在 100 个槽中没有任何可用局部特征，关系头无从补救。故先测 `oracle@100`（预测槽中最佳框 IoU≥0.5 的比例）、GT 与最大 IoU 框分布、所需对象尺度；若召回差，先提高分辨率/改多尺度主干或补对象检测预训练，而不是叠加关系推理。另一方面，若 oracle 高但 top-1 低，关系/短语对齐才是主要杠杆。

## 方案 B：区域 token 选择器（便宜的否证实验）

冻结公开 Grounding DINO 或 Groma 式 proposal，从每个候选区域抽 RGB/IR/Depth 特征，给小型文本交叉注意力网络选唯一目标，并回归候选框残差。这样可快速判断“对象级候选 + 关系特征”是否有价值，训练成本小，也可用作方案 A 的诊断上界。但它不是最终主推：候选源漏目标时无法恢复，GT 框注入只能用于训练标签构造，不能计入推理 oracle；还须独立报告候选 `recall@K`，不能用旧 8B oracle、论文 RefCOCO 分数或标签修订后错题覆盖替代。若它的独立场景 `recall@K` 低于现役直接框可达到的覆盖，应停止把硬候选作为主路径。

## 第一轮可证伪实验

**假设 H1：**在相同公开初始化和训练预算下，显式目标/参照槽关系比纯目标槽提高“同类多实例、序数、相对方位”题的 ACC@0.5，且不明显损害简单外观题。**假设 H2：**有效区域绑定的 IR/Depth 在确有辅助信号的题上进一步提高；如不存在独立增益，不把融合模块带进正式推理。**关键替代解释：**收益可能来自新权重/训练样本/更高分辨率、有效框修订、单纯更好 bbox 回归，未必来自关系或模态。

| 臂 | 唯一改变 | 作用 |
| --- | --- | --- |
| A0 | 冻结公开 MDETR RefCOCOg 权重，RGB 直接 top-1 出框 | 先确认公开模型域迁移和可运行性，不与训练臂做结构因果比较 |
| A1 | 同一公开初始化，RGB 对象槽 + target score + 框损失，按本赛训练清单微调 | 可学习出框的内部对照 |
| A2 | A1 加目标/参照配对头，使用同样训练图、Query、GT、分辨率、步数和可训练参数预算尽量匹配 | 测 H1；若加了额外公开关系数据，A1 也给同一数据和预算 |
| A3 | A2 加对齐 IR 与有效深度支路；A2 以等量通道/算力的 RGB-only 支路对照 | 测 H2，分开统计 IR、Depth 及双模态消融 |

数据按**场景/原始图**分割，禁止同一图不同 Query 跨 train/val；全部预训练公开集与本赛验证/测试图做已有 ID/来源排查，不将官方测试用于训练或伪标。先固定一个从未反复调参的独立场景验证集，City412 只做错误分类和回归检查。对 A1–A3 使用同一初始化、训练/验证图、seed、输入尺寸、训练步数、优化器与早停规则；模块差异导致参数量无法完全相同须记录。需保留当前 27B Arm2 直接框预测，用同一修订后 Query 与 GT 重新推理/评价；它是**实际业务基线**，不作 A1–A3 的结构因果对照。官方测试集只推理。

主指标是独立场景逐 Query 的普通 IoU≥0.5 ACC；报告按场景配对的救回/损失、平均 IoU、无效框率。机制指标是预测槽 `oracle@100`、相似实例 top-1 准确率、完整物体边界误差、关系子集与有效深度子集的 ACC；每类样本数和 95% 场景 bootstrap 区间一起给，避免把 412 条当独立图。定义成功为 A2 相对 A1 在预定关系子集净救回且全量 ACC 不降，A3 的辅助信号子集对 A2 有净救回并在遮蔽/交换 IR 或 Depth 后消失；若只在开发 City412 赢，或 `oracle@100` 低、配对头无增益、辅助模态消融不改变结果，则该机制不成立，停止扩展。需要先保存每题的对象框、目标分数、参照分数、有效深度比例以供逐题核验。

这轮的实现顺序是：冻结 checkpoint 推理核对坐标与域差 → 建立 scene split 和 A1 → 加 A2 并分析 oracle/选错实例 → 仅当关系定位值得继续时加 A3。不要先训练 27B 与专用对象网络的复杂端到端联合结构；其部署接口、梯度路径和显存成本会遮蔽这轮要回答的核心问题。

## 仍需核实的事实

- MDETR/其他公开权重对本赛 RGB 摄像域的零样本 `oracle@100` 与 top-1 表现，及许可和本地训练显存。README 提供 RefCOCOg checkpoint，但该文件尚未下载或运行。
- 本赛训练清单中关系词、同类实例、可靠 IR、有效主体深度的样本量；这些决定 A2/A3 能否学到信号。尤其需要按 RGB 框核对三模态对齐误差和深度无效比例，不能仅因官方称空间对齐就假设每个像素完全一致。
- 方案 A 在只有目标框、无参照框标注时可能学不到语义关系；若 A2 无效，先用少量**训练集内**人工核实的参照框或合法公开关系数据证明监督可行，再决定是否扩大，而不是自造伪标签掩盖问题。
