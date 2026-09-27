# 三模态视觉定位提分机制调查（仅研究，2026-09-25）

## 结论先行

当前最值得验证的是把**真实辅助模态变成可检查的对象证据**：先列全同类实例，再比较目标与干扰对象的原始毫米深度、有效像素和红外局部信息，把可靠的近远顺序表达成简短文本或候选特征。与此并行，给 IR 一条独立候选通路，防止 RGB 候选漏掉暗处目标。训练方向以同图困难实例判别、可信距离关系辅助任务、针对模态内容的反事实约束为先；视觉侧小范围适配可作为后续单卡试验。它们都是待证伪假设，不是已实现结果或涨分承诺。

本调查仅阅读本地材料、代码与公开资料；未下载权重、未运行 GPU、未用复赛测试集训练或人工标注，也未改模型/提交。官方 [赛题 PDF](F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf) 第 2–4、6–8 页规定：输出是 RGB 坐标系中的归一化 BBox，ACC@0.5 为主指标；训练可用外部公开数据并需报告来源，测试数据只能推理；禁止商业闭源模型在线 API 与手工干预测试答案。官方称三图空间对齐，仍需测目标级局部配准和有效性。红外 8 位强度方向为亮热暗冷，**不是已标定摄氏温度**；深度 PNG 原始整数单位毫米，0 或过小可能无效。现有复赛测试还混有 8 位 JPEG 深度可视化，不能把其灰度当毫米；这来自 [本地 handoff](F:/AIC/code/HANDOFF.md)，任何原始毫米策略只能对确认为 16 位的图适用。

本地实证边界：[R2/M2 执行记录](F:/AIC/docs/research/2026-09-22-rematch-execution.md) 是同一 412 条、78 图组上 R2 与 M2 各 288/412（69.90%）；M2 是 Qwen3-VL-8B 三图、仅语言 q/k/v/o LoRA、两轮 928 更新。M2 三图正常与 IR/Depth 错配均 288，双黑 291，六组干预的图组 bootstrap 置信区间均跨 0；换机混杂进一步限制几条差异的解释，详见 [干预结论](F:/AIC/docs/research/2026-09-24-modality-intervention-conclusions.md)。RGB 正确样本在双错配后仍有 279/288 正确，说明当前附加图像内容尚未带来可观测稳定净收益，**不能推成 IR/Depth 无用**。已退役的 EGM 和 LocateAnything 不纳入新方案；Locate 历史结果仅作为“候选上限与实际选择有差距”的证据，其 R2+Locate 前 8 候选离线覆盖 340/412、实际 R2 仍为 288，见 [结果报告](F:/AIC/docs/research/2026-09-24-locateanything-results.md)。

## 从问题独立生成的假设

文献检索前先写出以下互不等价的可证伪假设：H1 同类候选的**对象内有效毫米深度排序**比深度整图灰度更能回答 near/far/front/back；H2 红外对弱光对象可以补 RGB 候选召回，但需要自己的提案通路；H3 框内背景、遮挡和局部错位会制造伪深度/伪热度，掩码比整框更合适；H4 bbox token 监督不能保证学到模态依赖，同图负例或关系问答可改变这一点；H5 分辨率使小物体在全图输入中消失，候选裁剪能帮助实例选择或修框；H6 视觉侧冻结使 IR/Depth 域差难适应，但直接增加参数也可能过拟合；H7 当前许多错误可能是候选缺失而非排序失败。初始假设是研究提案；下文引用是相邻任务或工程机制的**定位证据**，不是本赛成绩证据。

## 候选机制与最小实验

| 优先级／机制 | 相对当前 M2 真正改变什么、依赖 | 最小证伪实验与停止条件 | 单 24GB 成本／主要风险 |
|---|---|---|---|
| **A：双源对象候选 + 掩码内原始深度** | RGB 候选之外另从 IR 生成候选，统一映射回 RGB；对每个实例用掩码内部原始 16 位深度取有效比例、中位数、分位差、与相邻对象的相对秩，而非把全图灰度交给 VLM。RGB/IR 图像、类别词、候选框/掩码、原深度和对齐信息是依赖。 | 冻结两路候选，在完整 412 条统计 RGB、IR、并集 Recall@1/4/8 和新增可覆盖的 R2 错误；只用训练集 GT 建立对象证据。独立计算含 near/far/front/back 的有效目标中“正确实例与同类干扰物能否按深度区分”，报告有效像素率。若 IR 无新增目标、有效深度不足或对象间秩不稳定，则不训练排序器。 | 需顺序加载检测/分割模型，缓存每个图组的候选，不能与 8B 同时常驻。掩码错误、RGB↔IR 局部错位、深度空洞、远处截断会造成反向证据；RGB 候选单独漏目标构成硬上限。**SAM 3** 官方可文本找全实例并输出 mask/box/score，但需 gated checkpoint、Python≥3.12、Torch≥2.7/CUDA≥12.6，实际 24GB 峰值未测；先做少量场景烟测。来源：[SAM 3 官方仓库](https://github.com/facebookresearch/sam3)。|
| **B：对象级 RGB/IR/Depth 排序器** | 用真实候选的 RGB/IR 裁剪、相对位置和上面的数值特征选候选 ID；不让生成式模型直接再猜坐标。训练集同图同类非目标实例作困难负例，验证候选绝不由 GT 补入。先保留 R2 框作为一个候选。 | 固定候选来源和划分，比较 RGB+位置、再加原始深度、再加 IR 裁剪的成对 412 ACC@0.5、mIoU、纠正/破坏数及 78 图组区间；先做冻结模型或小排序头，只有真实候选上限足够才训练。若新增辅助特征仅在 GT 候选的条件诊断有效、端到端不增，停止。 | 缓存特征后小头训练成本低；逐候选跑 8B 则吞吐昂贵。选择器可能破坏原本正确的 RGB 框，训练集仅 3707 Query/681 图组，必须按图组而非按 Query 防泄漏。相邻依据：[FineCops-Ref 官方候选选择代码](https://github.com/sleepyshep/FineCops-Ref)、[Cops-Ref 论文困难负例损失](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Cops-Ref_A_New_Dataset_and_Task_on_Compositional_Referring_Expression_CVPR_2020_paper.pdf)；它们不保证本赛收益。|
| **C：可信对象对的关系辅助监督** | 训练答案除了最终 BBox，再教模型“候选 2 比候选 1 近”“目标是第 n 个同类”“左右/遮挡关系”。只从**同图、不同实例**且深度有效、差距超噪声的训练样本构造，IR 属性仅在局部证据可靠时用。相对于现有单一 bbox token 损失，它直接监督需要辅助模态的判别。 | 固定同一基座、总更新数、RGB/BBox 原样本数，做 RGB 辅助任务与三图辅助任务对照；报告距离词子集、序数子集与全体成对结果、打乱深度后的任务准确率。若训练辅助题提高而最终框或干预敏感性不变，说明旁支无效。 | 构造可靠对象对比训练轻，8B 两轮历史约 2.5 小时可作预算参照，但新增监督长度会变。自动把 bbox 视为真实物体 mask、把所有近远词都解释为相机距离，会造伪标签。RGB-D 单视图先区域后对象匹配见 [Refer-it-in-RGBD](https://haolinliu97.github.io/Refer-it-in-RGBD/)；其 3D 指标与本赛 2D 框不可直接比较。|
| **D：模态反事实训练** | 对同一训练 Query 保留 RGB，配真 IR/Depth 与图组错配／置黑版本；只在高置信依赖辅助证据的训练样本上施加“真图优于假图”的候选级 margin 或排序损失。对普通样本不强制模型看噪声。 | 先审计训练集里有足够“RGB 难、辅助真能区分”的对象对；分别量训练/验证上真辅助相对错配的正确候选优势和最终 412 ACC，必须与同训练量普通增强及 RGB-only 对照比较。若只学会辨别错配痕迹却不改善正确目标，停止。 | 多前向训练约增加 2–3 倍计算；背景、场景风格、图像黑边等捷径可能使模型只识别真假图。当前 M2 干预的无净差已提示风险。模态主导是已有研究问题，[OGM-GE 官方代码](https://github.com/GeWu-Lab/OGM-GE_CVPR2022)以单模态分支贡献调节梯度，但其音视频分类设计**不能直接套在只有共同语言 LoRA 的 M2 上**。|
| **E：IR 专属视觉适配 + 可靠性选择** | 为红外设小视觉适配分支/LoRA，按目标处 RGB 质量、IR 对比与 Query 选择证据；深度分支只在有效时进入。当前 M2 视觉编码器全冻结，三图共用 RGB 预训表示；新增视觉适配才真正改变 IR 表征。 | 先从训练/验证划分中按弱光、尺寸、遮挡、目标 IR 对比做分析。比较相同步数和可训练参数量的 RGB 视觉 LoRA、IR 视觉 LoRA、双分支；在正常/错配 IR 上成对评估，真正的辅助学习应提高正常相对错配的可解释优势与最终 ACC。 | 单卡视觉 LoRA 是否放得下需一更新显存验收，宜先小分辨率或 2B 分支，不假定 8B 三图可放。RGBT-GroundBench 的 [论文/代码](https://github.com/crazyxiaoxi/RGBT-GroundBench) 确有非对称模态 LoRA、语言引导和照明先验；完整 RGBT-VGNet 还要 MMVG 预训、warmup 和 HiLoRA 多阶段，**不能声称原样迁移可单卡快速训练**。该公开基准有 26,604 训练条但只有 RGB-T，不含深度。|
| **F：候选定向高分辨率裁剪与局部修框** | 全图仍负责上下文/关系，候选区域加 1–2 个原图分辨率 RGB/IR crop；选出实例后才进行局部边界精修。区别于把三图整体像素预算从 602112 同时抬高。 | 先按目标面积与 IoU 错误类型分层，固定 412 候选，比较原图、原图+候选 crop；在候选已覆盖的样本分开报告“选对实例”与“框精度”；坐标必须从 crop 映回原 RGB。若只在含 GT 的理想 crop 有效，或正确样本退化抵消收益，停止。 | 每加图像增加视觉 token、延迟与显存；裁剪可能去掉关系上下文，过窄框会截断对象。Qwen3-VL [官方仓库](https://github.com/QwenLM/Qwen3-VL)支持按图控制像素与 image zoom 工具，但这不是本赛 crop 方法的实证。|
| **G：公开外部数据的分阶段蒸馏／难例扩充** | 先用公开 RGB-T GroundBench 或 IR-TD 辅助学 IR 表示/关系，再回本赛训练集做 bbox；难例由训练集模型错误与已知 GT 筛出，教师提供候选解释或概率，不把测试图做伪标注训练。 | 小规模公开数据 1–2 千条与等步数目标域重复训练、同模型对照；检查跨域后真 IR 的干预优势、目标域 ACC 和是否遗忘 RGB。若外域提高自身指标却使 412 退化，终止。 | 外部许可/格式/域差和算力是主要成本。[IRGPT 官方资料](https://github.com/WheatCao/ICCV2025-IRGPT) 提供红外定位/关系标注说明和早期部分数据，但完整图像并未全部直接发布；[RGBT-GroundBench](https://github.com/crazyxiaoxi/RGBT-GroundBench) 有数据入口与训练代码。公开 ICCV 2025 VG-SMART 队伍通过筛选难例、IoU 奖励、72B→7B 蒸馏获相邻赛事冠军，但其 [官方仓库](https://github.com/xuetf/ICCV-2025-MARS-ActiveAlphaAgent-Solution) 写明 8×80GB 训练配置，不能照搬到单 24GB；仅借数据筛选/蒸馏思路。[美团技术说明](https://tech.meituan.com/2025/10/27/ICCV-2025.html)。|

以上 A/B/C 可以组合，但实验应按新增机制逐项对照，先证明候选覆盖和辅助证据可分性。反对意见必须保留：**若真实目标只在 IR 中可见，RGB 检测候选池和从 RGB 框启动的裁剪会先验排除它**；因此 A 中的独立 IR 提案与“RGB+IR 并集覆盖”是 B 的前置条件。反过来，IR 热强度与目标语义并非一一对应，特别是无热目标、热背景、饱和和视差情况下，独立 IR 分支可能增加假阳性。深度排序也不能用于空洞、混合 JPEG 或大框背景主导的区域；必须保存可靠度而不是静默填值。

### 零训练补充：对象近远顺序文本提示

[Depth-Ordinal Prompting（DOP）原论文](https://arxiv.org/html/2607.11173) 直接比较“把深度作为第二张图”与“只把被问对象的近远关系写成一句文本”。它使用 Qwen3-VL-8B，报告空间问答中短文本线索常优于深度图，和当前 M2 的输入接口瓶颈相关。但**主结果的对象框来自评测基准已有标注**，没有运行检测器；不能拿主表当本赛端到端视觉定位证据。论文提到附录另有检测框压力测试，主表仍是已知区域的条件实验。其深度主要是单目估计，而本赛部分数据有传感器毫米值；实验用 H200 141GB，未证明本项目 24GB 可运行。

最小无训练试验：只从冻结 RGB+IR 候选器在验证集生成的真实框读取可靠深度秩，为含近远含义的 Query 添加一句对象关系，固定模型入口测全部 412、适用子集及错误翻转。GT 框仅用于训练构造或事后分层，不准进入验证推理。对照“候选表无深度”“数值深度”“一句顺序”“深度图”四种接口；候选未覆盖和秩反转都计失败。若收益只存在于 GT 给框条件下，则拒绝作为端到端路线。[ByDeWay 论文](https://arxiv.org/abs/2507.08679) 与 [公开代码](https://github.com/Rajarshi12321/ByDeWay) 也将深度分层转成文字，但公开指标是 POPE/GQA 问答而非 BBox，且另需深度估计和区域描述；适合作为文本接口对照，不优先照搬。

## 文献可运行性与迁移边界

| 来源 | 核实到的实际可用资源 | 对本项目的边界 |
|---|---|---|
| [SAM 3](https://github.com/facebookresearch/sam3) | 官方代码、图像文本提示示例、mask/box/score 输出、可下载但需申请访问的 checkpoint | 可先测试候选/掩码；未实测单卡内存，不能保证在当前 8B 同进程运行。 |
| [RGBT-GroundBench/RGBT-VGNet](https://github.com/crazyxiaoxi/RGBT-GroundBench) | 真实训练/评估脚本、数据链接、MMVG 初始化、多阶段 HiLoRA | 是 RGB-T 指代而非 RGB-D-T；源码可以借机制，训练脚本不能直接替换本赛入口。此前本仓 [2026-09-19 调查](F:/AIC/docs/research/2026-09-19-structural-optimization-roadmap.md) 已审核心损失，本文不把普通门控重复包装为新发现。 |
| [Refer-it-in-RGBD](https://haolinliu97.github.io/Refer-it-in-RGBD/) | 论文、项目页、SUNREFER 数据与 GitHub 链接 | 3D 单视图 grounding，结果不构成本赛 ACC 预期；借对象匹配思路即可。 |
| [RDTTrack/NeurIPS 2025](https://github.com/xuefeng-zhu5/RDTTrack)、[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b4962fcd5d4410a9f43ef70f528eedd8-Abstract-Datasets_and_Benchmarks_Track.html) | 三模态 RGBDT500 与官方跟踪代码，README 给单 GPU 命令，但原报告以 4×3090Ti、batch16 训练 | 同传感器组合提供可靠性/融合参考；目标跟踪依赖时间模板，没有文本 Query，不能声称直接可用或证明本赛融合收益。 |
| [IRGPT](https://github.com/WheatCao/ICCV2025-IRGPT) | 官方红外文本资料、grounding 与关系任务，早期 80k+ 图像；完整数据部分需自行从原公开源取得 | 数据准备高成本，先确认授权与可获取子集，不能写成一键复现。 |
| [FineCops-Ref](https://github.com/sleepyshep/FineCops-Ref) | 候选生成/选择、难负例数据与训练说明 | 原任务可有无目标拒答，本赛每 Query 唯一目标，需改为选实例并按所有 Query 评分。 |
| [Qwen3-VL 官方微调](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune) | Grounding 格式、多图格式、vision_tower_lr/tune_mm_vision 等训练开关 | 本地运行用的包装器实测仅语言 LoRA；视觉 LoRA 需另外核查真实可训练参数、单步峰值与输出。 |
| [DOP 论文](https://arxiv.org/html/2607.11173)、[ByDeWay 代码](https://github.com/Rajarshi12321/ByDeWay)、[DIST²Loss](https://github.com/JiwanChung/dist2loss) | DOP 给出对象近远顺序单句接口；ByDeWay 有深度分层文字代码；DIST²Loss 仓库 README 仍写训练与推理代码“Coming Soon” | DOP 主实验框由基准标注提供且用 H200，不是本赛端到端框证据；ByDeWay 是 POPE/GQA，DIST²Loss 当前不能即插即用。 |

## 对公开参赛方案的核查

“其他队伍”只指可核实的公开资料。本赛题同队公开技术方案的定向检索**没有找到可验证的完整代码和成绩**；搜索结果里的 AIC 城市场景目标检测是**另一赛题**。可借鉴的 [ActiveAlphaAgent](https://github.com/xuetf/ICCV-2025-MARS-ActiveAlphaAgent-Solution) 是 ICCV 2025 MARS2 的 VG-RS 队伍，不是本届 AIC 同赛道对手；冠军与 0.6671 是它自己的比赛成绩，不能移植为本赛预测。不能声称了解任何私有队伍方法或排行榜因果。

## 检索覆盖记录与证据强度

检索日 2026-09-25。先查本地 [官方 PDF](F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf)、两页已存官方 HTML、技术报告模板、[执行](F:/AIC/docs/research/2026-09-22-rematch-execution.md)、[模态干预](F:/AIC/docs/research/2026-09-24-modality-intervention-conclusions.md)、[Locate 结果](F:/AIC/docs/research/2026-09-24-locateanything-results.md)、[旧结构调查](F:/AIC/docs/research/2026-09-19-structural-optimization-roadmap.md) 与现有 Qwen 数据转换代码；再进行多轮网页检索。关键词族与实际筛选如下：

| 轮次 | 搜索词族（原文要点） | 主要命中／排除 |
|---|---|---|
| 1 | `RGB thermal visual grounding RGBT GroundBench`; `RGB D referring expression`; `multimodal visual grounding depth thermal` | 保留 RGBT-GroundBench、Refer-it-in-RGBD；排除普通 RGB REC 综述作为直接三模态证据。 |
| 2 | `RGB thermal depth visual grounding github`; `infrared language grounding reliability`; `depth distance ranking object-level`; `modality dropout counterfactual` | 保留 RGB-T 专项、RGB-D 空间数据；2026 Condition Dropout 论文代码尚未公开，未当作可运行路径。 |
| 3 | `hard negative contrastive referring expression`; `FineCops Ref`; `high resolution crop grounding`; `Qwen3 VL vision lora` | 保留 FineCops/Cops 负例、Qwen 官方像素/训练开关；通用“裁剪提分”未找到能直接外推本赛的对照。 |
| 4 | `AIC 基于大模型的多模态视觉理解与推理 参赛方案`; `site:github.com AIC 红外 深度 Visual Grounding` | 找到官方页和**相邻城市场景目标检测**，没有确认本赛其他队伍的公开完整方案；搜索不完整不等于不存在。 |
| 5 | `Collaborating Vision Depth Thermal RDTTrack`; `SAM3 text all instances`; `IRGPT grounding`; `MARS ActiveAlphaAgent VG-SMART` | 逐个打开作者官方仓库/论文与竞赛技术文章，核对权重、训练脚本和硬件；把跟踪与别赛方案标为迁移参考。 |
| 6 | `OGM-GE modality imbalance official`; `RGBT-GroundBench train script`; `SAM3 prerequisites checkpoint` | 对训练不平衡机制与资源可用性反查；发现 OGM 分类结构并不匹配现有共享 LoRA，SAM3 权重访问和新环境是明确门槛。 |
| 7 | `When Depth Is Better Told Than Shown DOP`; `DOP detector supplied boxes`; `ByDeWay code`; `DIST2Loss code` | 读 DOP 正文 5.1 节确认主结果使用基准已给对象框、检测框仅在附录压力测试；核对 ByDeWay 有代码但问答指标不同，DIST²Loss 代码未发布。 |

证据分级：本地 412/78 实验是本题直接证据，但仅一开发集且硬件迁移混杂；官方论文/代码证明机制存在和相邻任务可运行，**不证明本题单卡可复现或提分**；A–G 的排序是研究判断。缺口包括真实 IR 提案召回、对象级配准误差、有效深度与 Query 关联、SAM3/视觉 LoRA 的 24GB 峰值、外部数据域差。下一个执行者若进入实验，应先把 A 的无训练诊断作为决策门，再决定 B/C/D/E；本报告本身不授权启动 GPU 或覆盖既有基线。


补充核对：FineCops-Ref有EMNLP2024作者仓库liujunzhuo/FineCops-Ref及TPAMI2025扩展sleepyshep/FineCops-Ref，两者版本不同；数据规模与实验指标应按对应版本引用，不混算。SAM3的实际图像推理显存待本机验收，本报告不按参数总量估算可运行性。


## 收尾补检：图像层融合与显式模态监督

第8轮追加`Text-IF`、`TarDAL`、`infrared visible image fusion grounding`检索并核对作者仓库，避免只考虑语言层或对象选择。

- [Text-IF](https://github.com/XunpengYi/Text-IF)、[TarDAL](https://github.com/JinyuanLiu-CV/TarDAL)、[RIS-Fuse](https://github.com/GMY628/RIS-Fuse)研究先把RGB/IR信息融合成图像，再供人或下游任务使用。可考虑保留原RGB，再添加融合视图的低成本接口对照，深度仍单独提供。融合图视觉好看不证明指代定位更好；配准误差、颜色改变、热目标伪影可能损害答案。首版宜用已有冻结推理资源做小样本机制检查，不能先重训整套融合网络或把颜色词对应关系破坏。没有本题ACC证据，列为备选。
- [DualVision](https://github.com/abrarmajeedi/DualVision)将RGB局部区域与对应IR区域进行局部交互，并设计模态相关问答和RGB退化增强。原代码基于LLaVA7B、8GPU，非Qwen3-VL即插即用；其数据见外部地图补充第32项。最值得借的是有意构造RGB难而IR仍有信息的监督，但人工退化不应破坏颜色等Query依据，更不能只让模型学会识别合成退化。

这些是新增可检验备选，尚未取代主报告的数据修订与对象证据路线，不启动任何实验。
