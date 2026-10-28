# 真实深度、二维几何与语言指代：结构路线调查

2026-10-25。本文基于论文正文、附录、作者公开源码与项目历史；没有新训练或推理成绩。赛题以[官方 PDF](../../../../contest/基于大模型的多模态视觉理解与推理.pdf)第 2–4、6–8 页为准：输入空间对齐的 RGB、热红外、16 位 Depth 和英文 Query，输出 RGB 上唯一归一化 bbox，按 IoU≥0.5 的 ACC@0.5 计分；Depth 的零值和过小值无效。项目实际 City Depth 可能全零，且未提供相机内参。因此可可靠利用像素位置、有效深度及同图前后顺序，不能直接宣称世界坐标、真实横向尺寸或三维欧氏距离。

内部[全史与现役底座](history-and-current.md)、[DINO 复盘](../2026-10-25-dino-retrospective/README.md)形成负约束：旧 8B R/S/S+G 已实现数值 Depth、对象条件读出和连续几何头，但它们读取冻结 A 建议框做第二次修正，City A/R/S/S+G 为 296/296/298/296，55 道 Depth 正常或置空均为 45。冻结 DINO/SAM 候选选择和局部细化由 292 降至 273、221。单独再加 Depth stem、普通门控或连续头，不是新的预测机制。

## 四项主工作

### 1. Refer-it-in-RGBD：真实单视角 RGB-D，却不能照搬三维点云

[CVPR 2021 正文](https://arxiv.org/pdf/2103.07894)、[补充材料](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Liu_Refer-It-in-RGBD_A_Bottom-Up_CVPR_2021_supplemental.pdf)、[作者训练源码](https://github.com/HaolinLiu97/Refer-it-in-RGBD)。SUNRefer 从 SUN RGB-D 得到 7,699 张 RGB-D 图、38,495 条描述；ScanRefer 版本从 ScanNet 抽单帧。这是**真实传感器 RGB-D**。公开数据准备中有点云、相机 pose/内外参、完整三维目标框及 point-vote 标注。目标是被遮挡物体的**完整三维 AABB**，而非 RGB 中任意物体、部件或组合的二维框。

张量流：带 RGB 颜色的三维点云→0.05 m 稀疏体素 U-Net；GRU 编码 Query 并在瓶颈融合，产生点/体素相关热图；按热图加权的最远点采样，PointNet++ 从 20,000 点读 256 个种子；融合 Query 后 VoteNet 形成 32 个三维候选框，选最高分。补充材料说明体素网络的 3 个输入通道**只有 RGB 颜色**，Depth 通过点的 XYZ 坐标进入，不能误述为“原始 Depth 图送三通道卷积”。热图、投票、匹配、框分别有损失；先热图 10 epoch，再整体 60 epoch。作者报告 SUNRefer/单帧 ScanRefer 三维 Acc@0.5 为 35.9%/31.5%，均匀采样消融为 28.6%/25.4%。公开 [ref_net.py](https://github.com/HaolinLiu97/Refer-it-in-RGBD/blob/master/network/ref_net.py) 和 [数据 loader](https://github.com/HaolinLiu97/Refer-it-in-RGBD/blob/master/dataset/sunrefer_dataloader.py)证实热图引导采样与点级投票。

可迁移的是“完整语言先产生全图密集相关性，再读取细粒度空间特征”的监督顺序。不能迁移 0.05 m 体素、三维采样/投票或三维框：本赛无内参，原始深度可能全零，框内前景也可能不是主体。它的三维 Acc 不能作为本赛单 RGB 框收益预测。

### 2. SD-VLM：Depth 改视觉 token 的位置表示，成绩仍是空间问答

[NeurIPS 2025 正文及附录](https://papers.neurips.cc/paper_files/paper/2025/file/30bc3a3a44c9d2e3f32e6dd1cd18f552-Paper-Conference.pdf)、[作者源码](https://github.com/cpystan/SD-VLM)。将深度按 RGB patch 平均，用多频正弦/余弦编码，再与该位置 CLIP 特征逐元素相加，送入 LLaVA-1.5-7B。它不增加视觉 token 数，与伪彩另输入 Depth 图不同；具体实现见 [llava_arch.py](https://github.com/cpystan/SD-VLM/blob/main/llava/model/llava_arch.py) 中 depth_sincos_encoding、resize_depth、encode_depth。

MSMU 来自 ScanNet/ScanNet++ 的 2,000 场景、25,000 图、700K QA，任务含存在、计数、尺寸/距离、相对位置和少量二维坐标问答。训练可用真实 Depth；缺深度的推理使用 Depth-Anything-V2 从 RGB 估计，故主成绩**并非纯真实传感器 Depth 效果**。代码对每图深度池化后做 min-max 归一到 0–100，丢失原始毫米尺度。LLaVA 视觉塔冻结、LoRA 一 epoch；论文记 8 张 V100、32 GPU 小时。同一论文设置下 MSMU-Bench 的 VQA 成功率由 RGB-only 46.73% 到正弦 DPE 56.31%；Depth 另作图为 22.64%，另作 token 为 35.72%。这支持测试同格点几何编码，**不支持本赛 ACC 直接增加 9.58 点**。附录明确三维反投影仍需相机 K。

可迁移同格点深度编码，但 City 全零或无效 patch 必须让几何增量为零并保留有效掩码；不能让 sin(0)/cos(0) 等同“已测得近物体”。City 可保留毫米通道，单位未知外部源只用场景内秩/相对差。DPE 本身没有解决 Query 的目标和参照主体绑定。

### 3. SR-3D：高分辨率区域特征和“伪三维”位置，依赖预给区域

[ICLR 2026 正文](https://arxiv.org/pdf/2509.13317)、[作者源码](https://github.com/AnjieCheng/SR-3D)。单视角 RGB 由 Depth-Anything-V2 估计**相对深度**，再给视觉 token 加空间位置编码；动态切片、视觉特征拼回、按**已给** mask/box 池化出区域 token。单视角约 700 万图文/区域混合样本预训。多视角阶段才使用真实深度、相机 K 与 pose，将各帧对齐到共同空间。其单视角/多视角指标是空间问答及已知区域分类，非自由 Query→bbox。

论文说“反投影到三维”，但发布源码 [mm_utils.py 无内参路径](https://github.com/AnjieCheng/SR-3D/blob/main/llava/mm_utils.py#L56-L125)用归一化图像坐标 u、v 和深度 z 构造 (uz,vz,z)，并不使用该函数的 intrinsic 参数；[process_depth](https://github.com/AnjieCheng/SR-3D/blob/main/llava/mm_utils.py#L727-L742)先把 8 位相对深度除 255。这是可学习的 **2.5D 位置特征**，不是米制相机坐标。论文消融还表明，不经大规模单视角预训，位置编码增益有限。

可迁移保留二维格点、高细节切片及位置编码接口；不能照搬 8 位相对深度伪 XYZ，也不能把模型在预给 mask/box 后的区域能力当作自主出框。区域必须由完整 Query 在**第一次出框前**自主且可微地产生。

### 4. GR3D：自动定位文本实体并回读区域，但训练/推理区域不同

[CVPR 2026 正文及附录](https://arxiv.org/pdf/2605.30307)、[作者项目页](https://anjiecheng.me/gr3d/)。这是 SR-3D/NVILA 的后继；**输入仍是 RGB，空间底座来自估计相对深度/位置编码**。新增三项：语言头直接输出二维 bbox；推理文字提及实体时先生成其框，再从已编码图像抽取区域 token 插回语言序列；给已定位区域预测三维框。训练二维坐标用 teacher forcing，回插的是**GT 区域 token 且 detach**；推理用预测区域，因此有定位错误传递和训练/推理分布差。三维框部分还用焦距归一和深度/点图监督；无内参的本赛不能直接用。

论文使用 97K grounded CoT、780K 三维检测、272K 点图监督；第一阶段训视觉塔外模块约 4 天、8 节点 A100，第二阶段只训 LLM 约 4 小时。MM-GCoT 的中间区域 Acc@0.5 为 74.2%，更噪的区域监督会降至 62.8%；但它不是本赛唯一最终框 ACC。作者项目页截至本轮明确标注 **Code/Weights Coming Soon**；不能声称检查或复现 GR3D 训练源码。其前身 SR-3D 的区域基础代码已发布，但不是自动回插实现。

可迁移的是“目标与参照实体各自绑定图像证据”。不能照搬长文字推理链、GT 区域回插、硬框二次修正和需焦距的三维头。更值得检验的是首次框前的连续软区域交互，让错误绑定本身可反传。

### 旁证：SpatialRGPT 的区域能力不等于自由定位

[NeurIPS 2024 正文及附录](https://papers.neurips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf)与[官方源码](https://github.com/AnjieCheng/SpatialRGPT)用 RGB 估计相对 Depth 的额外连接器、反卷积＋给定 mask 池化。离线 OpenImages 空间 QA 标注使用 GroundingDINO/SAM、Metric3Dv2、WildCamera 和姿态估计做伪三维场景图，约 1M 图、5M 区域、8M 模板 QA、700K 复杂 QA。这不是推理时输入本赛原始 Depth。区域问答须预给框/掩码；无掩码时还会调用 SAM。公开 [区域提取](https://github.com/AnjieCheng/SpatialRGPT/blob/main/llava/model/region_extractor/base_extractor.py)与[视觉嵌入替换](https://github.com/AnjieCheng/SpatialRGPT/blob/main/llava/model/llava_arch.py)验证此限制，不能以其问答分数恢复旧 DINO/SAM/ID 路线。

## 本赛的新机制假设

**假设：**部分可救错例发生在**第一次目标身份/参照绑定**，而非第二次框几何修正。真实 Depth 的有效前后关系若在第一次目标选择时进入目标 patch 与参照 patch 的相互注意，可改正同类竞争、前后遮挡等题；单纯三图输入、DPE、普通门控或读取 A 建议框的框头不会产生同样净收益。这是待实验假设，不能写成已超过官方 .8012。

建议复用现役 qwen3_5 的完整 RGB/IR/Depth/Query 路径，**不先生成 A 框**。新结构与对照按[综合方案](README.md)从公开基座及清理后的合法混合建立共享起点，Arm2 只作冻结历史参考。从实际视觉塔抽 RGB 二维格点特征，从读完完整 Query 的语言状态提取目标及关系条件。原始 Depth 在相同 RGB 网格提取有效率、有效值中位/分位、场景内秩、局部深度跳变；City 可加 log(毫米)，单位未知源不冒称米制。IR 只在 RGB 对应小邻域读取特征。Depth 全零时几何增量为零。

**区别于旧结构的是二元关系场：**对低分辨率 RGB 网格每对位置 i、j，在视觉注意力 logits 外加入由完整 Query 控制的相对二维位移、有效深度顺序、局部深度边界及有效性偏置。目标与参照是两种可学习的**软读出**，在同一层交互；目标相关图遍历整张图，不从 DINO 类别池、SAM mask 或原 A 框截取。随后以高分辨率 RGB 边界特征和目标相关图**一次**预测 RGB 全图归一化框。部件或组合不强制属于某一检测类别，目标图可跨不连通区域，最终框按标注范围学习。该方案把 RGB-D 论文的密集热图、同格点深度表示和 GR3D 的实体回读用于首次出框，并新增**Query 条件的目标—参照成对几何交互**；旧 R/S/S+G 未做首次选择，也没有这个成对关系偏置。

主损失必须落在最终单框的 IoU/L1/GIoU 目标。GT RGB 框只能派生弱目标热图监督，不能当主体 mask，亦不能从整个矩形的平均深度造“主体距离”真值。原生成框 CE 可辅助保留语义，但其高低不是比赛结论。仅有目标框时参照读出可能塌缩到目标；首轮应在确有多个可靠对象框/区域标注的合法训练图组（例如 SUNRefer）做参照绑定，再用本项目三图来源适配。没有参照真值的图只训练最终框，不伪造参照框。现有 226 对/157 图专项池足以诊断，不能默认足以从零训练高自由度关系模型。

工程门槛：必须核对当前 qwen3_5 的 merger、视觉 patch 顺序、放大/切片和 RGB 坐标逆映射，不能机械套旧 Qwen3-VL DeepStack hook，或把展平 token 当普通矩形网格。若现役接口无法保留所需二维高细节特征，先在同数据的小骨干对照结构因果，不能声称保留了现役 27B 完整能力。

## 最小对照和否证

1. 固定训练图组、骨干、分辨率与步数：A 为现役直接框实用门槛；B 为同格点 Query 热图＋一次框；C 在 B 上加二维成对关系；D 再加真实 Depth 顺序与有效性；E 再加 IR 局部证据。B→C 测关系结构，C→D 测真实深度增量。不同数据和骨干的绝对分数不得归因于几何。
2. 同权重将 Depth 置无效、同图错位、跨图交换，按近远/前后 Query 查看目标热图和最终框；全零样本不应随填充值翻转。IR 也需做空间错配。同权重干预只证明依赖；只有重训 D/E 相对 C 的单框净收益才说明有益使用。
3. 最终看单框 ACC@0.5、逐题救回/损害、错实例与范围错误。以场景/序列留出，不能按同图不同 Query 随机拆分。City400 和外部94已长期使用，只是开发诊断；独立结论需合规新留出或后续官方反馈。
4. 若 B 有收益而 C/D 无增量，结论是首次空间读出而非几何关系；若 D 在无效/错位 Depth 下同样提升，几何解释失败；若热图对而最终框破坏原本正确题，需查目标范围与回归监督，不恢复局部细化；若仅救极少数近远题却伤普通题，不作主线。

**裁决：**优先检验第一次出框前的目标—参照软绑定和有效 Depth 成对顺序约束。点云伪三维、预给区域问答、长文字推理链或现成 Depth＋连续头不能包装成结构突破。最终是否接管输出，只由本赛唯一框净收益决定。
