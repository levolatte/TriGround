# 外部数据集地毯式检索：RGB＋热红外＋深度语言定位

日期：2026-09-25。任务：只做公开来源核查和数据选型，不下载大数据，不训练，不把文献指标当成本赛题指标。本文件使用“图组”指同一时刻/场景的配对图像，使用“表达实例”指一条 Query 与其目标的标注；视频帧数不能充当独立场景数。

## 判断基准与本地边界

本地[赛题 PDF](F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf)第 2–4、7–8 页规定：每条输入有空间对齐的 RGB、热红外、深度和英文 Query，输出 RGB 坐标系归一化 `xyxy` 框，ACC@0.5 是主指标；公开外部训练数据允许使用，技术报告需说明来源、用途及预处理。官方测试集不得人工标注或用于训练。文档只评价**候选外部数据**，不重新认定团队已核验的 city 目标域素材是否合法或可否继续使用。已知团队有 city 新标签约 3707 条、复核/开发 412 条，复赛官方测试 5690 Query 无 GT；大多数本地深度为毫米 `uint16`，另有 97 组 JPEG 可视化深度。这些是后续转换和评估的边界，外部数据的伪深度、立体视差或 3D 点云不能直接当同量纲毫米图。

**核心发现**：本轮未核实到一个同时提供“真实且配对的 RGB＋TIR＋毫米深度、单目标英文表达、RGB 2D 框、贴近城市室外场景、可直接取得作者训练划分”的公开数据集。最接近的路线是分工互补：RGB–TIR 指代数据补热模态语言绑定，真实 RGB–D–T 数据补传感器对齐，RGB–D 指代数据补距离/关系语义；用现有 city 目标域数据检查是否产生可迁移增益。这个结论是**本次检索范围内未发现**，不是断言此类数据绝不存在。

## 优先审阅的五类

| 顺位 | 数据及用途 | 为什么排在这里 | 最小转换与主要限制 |
| --- | --- | --- | --- |
| 1 | [RGBT-GroundBench](https://github.com/crazyxiaoxi/RGBT-GroundBench) 的公开 train：RGB–TIR＋英文表达＋框 | 21,535 配对图组、38,760 表达实例，含城市道路、弱光、小目标；作者 [HF 仓库](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/tree/main) 已列出图片与标注 tar | 只取官方 train 的 RGB/TIR/Query/框；缺真实深度，不能以复制灰度或生成深度充当三模态。先核对 FLIR/M3FD/MFAD 源数据重合与许可，再与现有 city 做图组去重。 |
| 2 | [RGBDT500](https://xuefeng-zhu5.github.io/RGBDT500/) 的真实三路图像＋跟踪框 | 500 序列、约 203.7k 帧图组、≥66 类和 >100 场景，作者页提供 Google Drive/百度链接，三路对齐；与赛题三传感器最接近 | 没有自然语言；不能将类别词自动伪装为唯一指代。可按**序列**抽稀并为少量多候选帧人工写可核实的英文表达，先检验真实三路输入。页面先展示 research-only 数据协议，下载可达性需在同意协议后另验。 |
| 3 | [SUNRefer](https://haolinliu97.github.io/Refer-it-in-RGBD/)、[RoboRefIt](https://luyh20.github.io/RoboRefIt.github.io/) 与 [SUN-Spot 旧版](https://github.com/crmauceri/refer_sunspot)：RGB–D＋关系语言 | 它们直接提供 Query 与对象对应关系；SUNRefer 38,495 表达/7,699 RGB-D 图，RoboRefIt 50,758 表达/10,872 RGB-D 图，SUN-Spot 旧版 7,987 表达/1,948 RGB-D 图且偏空间介词 | 取有明确 2D 框的记录；只有 3D 目标的须投影并核实可见性。全部以室内为主，不能取代城市三模态训练；深度制式先查再接入。 |
| 4 | [PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)／[MM5](https://github.com/martinbrennernz/MM5-Dataset)：真实三路＋语义/实例 mask | PST900 作者仓库明确列出 RGB、`uint16` 立体深度、8/16 位热图及像素标签，894 图组/1.4 GB；MM5 的 aligned/cropped 版本同时有 RGB、16 位深度、热图及实例 mask，并有 figshare 下载入口 | PST900 只标 4 个地下场景目标类；MM5 主要室内/物体，图组数未在本次官方页确认。mask 可转框，但“类别模板 Query”只适用于唯一实例且经核验，否则需人工语言。PST900 深度由双目估计，不能先验等同赛题毫米测量。 |
| 5 | [RefRT](https://huggingface.co/datasets/jz-fan/RefRT) 的 RGB–TIR＋双语描述＋轨迹框 | HF 显示 46.7 GB 文件，作者称 72 序列、388 描述、1,250+ 目标、166,147 语言-帧组，含校园道路与低可见度；比只含检测类别的 RGB–TIR 集更有直接 Query | 描述往往指一段轨迹或多个目标，不能将 166,147 帧当独立 Query。先筛单目标、逐帧可见、目标唯一的记录；按原始 LasHeR/VTUAV 视频分组，查看两层数据许可。缺深度。 |

这五类是**调研与小规模试验优先顺序**，不是下载或全量训练指令。只选作者训练划分与独立目标域验证，逐图组确认路径、框坐标、目标唯一性、真热图和深度来源。尤其 RGBT-GroundBench 建自 FLIR/M3FD/MFAD，不能把它与这三个母集相加计算“新增数据量”。

## 逐项候选审计

状态含义：“公开文件/入口”表示作者或托管页展示了可取得的文件或下载链接，**未下载验证整包内容或可持续访问**；“申请”须提交表单/同意协议；“仅论文/待发布”不能投入当前训练。未找到明确数据许可时写“未核实”，不能用代码仓库的 MIT/GPL 代替数据授权。

| # | 候选、主要一手来源 | 实际模态；标注；规模单位 | 场景、取得与许可；结论/最小转换 |
| ---: | --- | --- | --- |
| 1 | [RGBT-GroundBench 作者仓库](https://github.com/crazyxiaoxi/RGBT-GroundBench)、[HF 文件树](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/tree/main) | RGB＋TIR；同对图的表达、2D 框、条件标签；21,535 **图组**/38,760 **表达实例**，train 26,604 实例 | FLIR/M3FD/MFAD 派生，城市/夜间适配强；HF 列出 `ann_*.tar`、`data_*.tar` 共 10.4 GB，Viewer 解析失败但文件树可见，无 dataset card/清晰数据许可。优先 train；按母集查原始许可与重叠。 |
| 2 | [RefRT 作者 HF 卡](https://huggingface.co/datasets/jz-fan/RefRT)、[代码仓库](https://github.com/yan-qiu-yu/RT-RMOT) | RGB＋TIR＋中英文本＋MOT 框；72 **视频**、388 **描述**、166,147 **语言-帧组** | 城市道路/校园/无人机；HF 46.7 GB 文件和下载命令，Viewer 部分失败；数据页标 MIT 但也明确须遵从 LasHeR/VTUAV 底层条款。逐帧转框，仅筛唯一目标；序列隔离。 |
| 3 | [SUNRefer 作者主页](https://haolinliu97.github.io/Refer-it-in-RGBD/)、[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Refer-It-in-RGBD_A_Bottom-Up_Approach_for_3D_Visual_Grounding_in_RGBD_CVPR_2021_paper.pdf) | RGB＋D＋语言，3D 目标框；7,699 **图组**/38,495 **表达** | SUN-RGBD 室内；主页有 `SUNREFER_v2` 下载链接，图像及 SUN-RGBD 原始使用条款仍需核实。若有 2D 框直接用，否则从 3D 投影并剔除不可见目标。 |
| 4 | [RoboRefIt 作者主页](https://luyh20.github.io/RoboRefIt.github.io/) | RGB＋D＋Query＋2D 框＋mask；10,872 **图组**/50,758 **表达** | 室内操作，6 生活场景；主页提供 Google Drive 数据与 JSON 字段 `text,bbox,rgb_path,depth_path,mask_path`；网页 CC BY-SA 是**网页**许可，数据许可未单独核实。可直接映射 2D 框，需检查深度编码。 |
| 5 | [SUN-Spot 2019 论文](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CLVL/Mauceri_SUN-Spot_An_RGB-D_Dataset_With_Spatial_Referring_Expressions_ICCVW_2019_paper.pdf)、[作者代码](https://github.com/crmauceri/refer_sunspot) | RGB＋D＋空间指代表达/目标；1,948 **图组**/7,987 **表达** | 室内；作者链接 Zenodo 标注，底图来源与深度包需分别检查。适合 `left/right/behind` 诊断，小样本不宜泛化为城市主训练。 |
| 6 | [SUN-Spot v2／Spatial-LLaVA 论文](https://arxiv.org/abs/2505.12194) | 约 90k **image-caption pairs**，含空间指代与 landmark、Set-of-Marks 提示；不是旧版 7,987 表达 | 本次只核实论文描述，未核实完整图像＋深度＋2D 框的可下载包或字段，故暂不列为可用 RGB-D 训练集；尤其不能把 90k 当独立 RGB-D 图组。 |
| 7 | [OCID-Ref 作者仓库](https://github.com/lluma/OCID-Ref) | RGB＋原始 OCID RGB-D/点云＋指代 mask；305,694 **表达**/2,300 **场景** | 室内桌面遮挡；原始 OCID 与 Ref 注释须分别下载，仓库提供 GDrive；Ref 仓库称 MIT，原始 OCID 条款另查。可由 mask 得框，检查重复同场景表达。 |
| 8 | [RefSpatial 作者 HF](https://huggingface.co/datasets/JingkunAn/RefSpatial)、[论文](https://arxiv.org/abs/2506.04308) | RGB-D 与 31 类空间关系、QA/指向；论文称 20M **QA pair**，大量为 2D/3D/模拟混合 | HF 有数据包但 20M 不是实拍独立图组，任务也不全是 2D BBox；可抽具有真实 RGB-D、对象 mask/坐标、单目标指代的子集，不做整包无差别混训。许可卡 Apache-2.0，底层素材另核。 |
| 9 | [DRSet／DRMOT 作者仓库](https://github.com/chen-si-jia/DRMOT)、[论文](https://arxiv.org/abs/2602.04692) | RGB＋D＋语言＋跟踪框，旨在深度关系定位 | 作者 README 明言“论文接收后一个月完全开源”，当前未提供数据下载；**仅论文/待发布**，不列入近期训练。 |
| 10 | [RGBDT500 作者页](https://xuefeng-zhu5.github.io/RGBDT500/)、[NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf) | 真实 RGB＋D＋TIR、对齐、跟踪对象框；500 **序列**/203.7k **帧图组** | 多场景/66+ 类；作者页列 Google Drive/百度全量下载，页面先提示 research-only 协议，具体获取需接受协议；无 Query。对视频抽帧须按原视频和场景分组，少量人工补语言。 |
| 11 | [PST900 作者仓库](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) | 对齐 RGB＋立体估计 D (`uint16`)＋TIR（8/16 位）＋语义 mask；894 **图组** | 地下挑战环境，4 前景类；Google Drive 1.4 GB 入口。代码 GPL-3.0 不自动说明数据许可。可由单实例 mask 得框并人工描述；近域不足，作为真实三路管线验证。 |
| 12 | [MM5 作者仓库](https://github.com/martinbrennernz/MM5-Dataset) | RGB＋16 位 D＋16/8 位 LWIR，另有 UV/NIR；对齐裁剪版有语义和实例 mask；**图组数未在本次主页核实** | 室内物体/多照明；作者列 figshare 原始、对齐、标注下载，数据 CC BY-NC 4.0。按同文件名配三路，实例 mask 得框，语言需新增；注意作者称深度与热图在不同 RGB 光照设定间复用，不能误当独立场景。 |
| 13 | [Trimodal Human Segmentation Zenodo](https://zenodo.org/records/7996570)、[论文](https://www.dagm-gcpr.de/fileadmin/dagm-gcpr/pictures/2023_Heidelberg/Paper_MainTrack/040.pdf) | 注册 RGB＋D＋TIR，人体 mask/动作；15,618 **帧**、101 **shots**、10 环境 | 人体行为/多视角，非城市多类别；Zenodo 记录为公开下载，许可未在本轮核实。人体 mask 可转框；按 shot 切分，语言需要人工构造。 |
| 14 | [TRansPose 论文](https://arxiv.org/abs/2307.05016)、[作者下载页](https://sites.google.com/view/transpose-dataset/download) | 双目 RGB-D＋TIR＋透明物实例 mask/位姿；论文 333,819 **图像**（不能当图组） | 机器人桌面/透明器皿，域差很大；作者页要求申请表，分卷每序列数 GB。可由实例 mask 得框，但优先级低。 |
| 15 | [L-CAS RGB-D-T 作者页](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-rgb-d-t-re-identification-dataset/) | Kinect RGB＋D、Optris 热图、2D 激光；每 rosbag 约 1,000 **连续帧** | 室内单人 ReID（身份再识别），非全场景指代；作者提供 rosbag 入口，框/Query 不明确。数据前处理重，暂排除。 |
| 16 | [M3FD／TarDAL 作者仓库](https://github.com/JinyuanLiu-CV/TarDAL) | RGB＋TIR、目标检测框；4,200 **配对图组**、约 34,407 **目标标签**、6 类 | 大连校园/道路，几何已配准；Google Drive/百度入口；无深度和 Query。可从多目标框中挑唯一属性目标写语言。注意它已被 RGBT-GroundBench 复用，不再相加。 |
| 17 | [FLIR ADAS 官方页](https://oem.flir.com/solutions/automotive/adas-dataset-form/) | RGB＋热图＋检测框，官方现列 9,711 热/9,233 RGB 训练验证图、15 类 | 道路/夜间强匹配；下载需填表，RGB 与热画幅并非当然像素对齐，须按所用版本和配准方法核查；无深度/Query。也与 RGBT-GroundBench 的 RefFLIR 可能重叠。 |
| 18 | [MFAD，RGBT-GroundBench 作者子集说明](https://github.com/crazyxiaoxi/RGBT-GroundBench) | RGB＋TIR＋检测原框；衍生 RefMFAD 21,500 **表达实例**，原母集图组数本次未从作者一手确认 | 衍生数据可用 HF tar，母集独立入口/许可本次未确认；不单列增量训练推荐。需先按 RefMFAD 的图像、表达、框对应做源谱系审计。 |
| 19 | [LLVIP 作者仓库](https://github.com/bupt-ai-cz/LLVIP) | RGB＋TIR＋行人框；15,488 **配对图组** | 夜间安防贴域，但目标类别单一；作者给主页/下载入口，未在本轮确认数据专属许可。可做热图行人定位，但若一帧多人，仅 `person` 模板不足以唯一指代。 |
| 20 | [FMB／SegMiF 作者仓库](https://github.com/JinyuanLiu-CV/SegMiF)、[ICCV 论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Multi-interactive_Feature_Learning_and_a_Full-time_Multi-modality_Benchmark_for_Image_ICCV_2023_paper.pdf) | 对齐 RGB＋TIR＋语义 mask；1,500 **配对图组** | 室外全天候道路，有 Google Drive/百度入口，数据许可本次未核；无深度/Query/实例 ID。只把单实例类别 mask 转框，其余需实例化或人工写描述。 |
| 21 | [MFNet 作者实现](https://github.com/haqishen/MFNet-pytorch)、[项目页](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) | RGB＋TIR＋语义 mask；1,569 **图组**（820 日/749 夜） | 道路域较好；作者页可下载，但 4 通道 PNG 要按实际通道拆，语义分割无法天然区分同类多目标。无深度和 Query。 |
| 22 | [DroneVehicle 作者仓库](https://github.com/VisDrone/DroneVehicle) | RGB＋IR＋旋转检测框；28,439 **配对图组**、5 类车辆 | 城市道路/停车场但无人机俯视，尺度和视角与赛题不同；百度训练/验证/测试入口，数据许可未核。旋转框取外接 `xyxy`，仅可筛唯一目标造 Query；注意原图有 100 px 白边。 |
| 23 | [KAIST 多光谱行人作者仓库](https://github.com/SoonminHwang/rgbt-ped-detection)、[官网](https://sites.google.com/view/multispectral/home) | RGB＋TIR 行人框；约 95k **配对视频帧** | 车辆视角城市/校园，官网 CC BY-NC-SA 3.0；平台还采 stereo/LiDAR，但**标准行人发布包不能因此被认作现成配对毫米深度图**。长视频高度重复，无语言。 |
| 24 | [DIODE 作者页](https://diode-dataset.org/) | RGB＋高精度 D＋有效掩码／法线；规模按官网具体分包核 | 真 RGB-D 户外，缺 TIR、目标框与 Query；适合深度值域/无效像素预处理而非直接指代训练。页面有下载入口，许可本次未核。 |
| 25 | [ScanRefer 作者仓库](https://github.com/daveredrum/ScanRefer)、[ScanNet 官网](https://www.scan-net.org/ScanNet/) | 室内 RGB-D 扫描＋3D 对象/语言；51,583 **表达**、800 **ScanNet 场景**（论文） | ScanRefer 需表单获标注链接，ScanNet 需学校邮箱和条款同意；3D 对象不自动是每帧可见的 RGB 2D 框。转换成本高，次于 SUNRefer/RoboRefIt。 |
| 26 | [CityRefer 作者仓库](https://github.com/ATR-DBI/CityRefer) | 城市点云/彩色＋3D 实例＋语言；约 35k **3D 描述** | 城市语义好，但基于 SensatUrban 空中点云，不提供本题配对 RGB/TIR/D 单视角图组；作者提供标注 Google Drive，底层 SensatUrban 须另取。只作关系词语料，不直接训 2D 框。 |
| 27 | [RefDrone 作者 HF](https://huggingface.co/datasets/sunzc-sunny/RefDrone)、[论文](https://arxiv.org/abs/2502.00392) | VisDrone RGB 航拍＋Query＋2D 框，包含多目标/无目标样本 | 作者 HF 提供标注，图片须另取 VisDrone；英文城市小目标，但仅 RGB 且一条文本可映射多个框，筛单框才符合本题。许可卡 CC BY 4.0，底图另查。 |
| 28 | [Refer-KITTI 作者页](https://referringmot.github.io/) | KITTI RGB 视频＋语言＋多目标跟踪框；18 **视频**/818 **表达**，每表达平均 10.7 对象 | 城市道路关系语义有用，但多目标表达与单框赛题不合；作者页有代码/数据入口，须筛每帧唯一目标且避免视频帧泄漏。标准包没有同步热图。 |
| 29 | [RefCOCO/+/g 作者 API](https://github.com/lichengunc/refer) | COCO RGB＋指代表达＋2D 框/mask | 通用指代预训练已成熟，但无 IR/D、域差；作者 API 说明原下载服务器故障，需要其 issue 镜像且 COCO2014 图像另取。代码 Apache-2.0 不等于 COCO 数据许可。只建议作为语言/框格式基准。 |
| 30 | [FineCops-Ref 作者仓库](https://github.com/liujunzhuo/FineCops-Ref) | GQA RGB＋组合关系 Query＋2D 框，正负样本；论文称 9,605 **正表达** | 适合“关系词是否真的起作用”的离线诊断；作者 figshare/百度只提供标注与负图，GQA 底图另取；无热/深。负样本无目标，不能直接用本赛题单目标正例输出格式。 |
| 31 | [LanguageBind VIDAL-10M 论文](https://arxiv.org/html/2310.01852)、[作者数据说明](https://github.com/PKU-YuanGroup/LanguageBind/blob/main/DATASETS.md)、[HF](https://huggingface.co/datasets/LanguageBind/VIDAL-Depth-Thermal) | 视频/文本与**生成**热图、**生成**深度；百万级配对但无单目标框 | 论文 §4 明确以 sRGB-TIR 生成热图、GLPN 生成深度；HF 已有 23.4 GB 文件但全量 10M 可达性未核实，README 数据许可 CC BY-NC 4.0 与 HF MIT 标签不一致。**排除为真实三传感器训练证据**，最多作表征预训练线索。 |

## 检索路径、纳入与排除

本轮先读本地 PDF 与已有[多模态文献审计](F:/AIC/docs/research/2026-09-11-multimodal-literature.md)，再交叉检索 arXiv/CVF/NeurIPS 论文、作者主页、GitHub、Hugging Face 文件树与卡片、Zenodo、figshare 入口、ModelScope，以及另一个 AIC [城市检测官方页面](https://www.aicomp.cn/tracks/tracks-1/3700.html)。核心关键词组：`RGB thermal referring expression grounding dataset`、`RGBD referring expression SUNRefer RoboRefIt SUN-Spot`、`RGB depth thermal aligned detection segmentation`、`RGBDT500 download license`、`RGBT GroundBench RefRT`、`site:huggingface.co/datasets RGBD referring`、`site:modelscope.cn/datasets RGBDT500/RGBT/SUNRefer/PST900`、`AIC 2026 城市场景 RGB 红外 深度`。ModelScope 的四组精确名称检索没有返回可核实一手数据页；不能据此断言站内没有镜像。对 HF Dataset Viewer API 的只读 `/is-valid` 请求也显示 RGBT-Ground-Dataset `preview=false,viewer=false`、RefRT `preview=true,viewer=false`，故不能把页面预览失败误写成“无文件”，更不能以预览代替抽样验包。来源审计优先作者仓库/主页与托管文件树，综述仅用于发现候选，不用其二手统计替代原始数据页。

纳入标准依次是：①与本题至少共享一种**真传感器**模态或语言定位标注；②图像、Query、2D GT 是否同场景同对象可追溯；③可否见到实际文件或下载入口；④图组/序列独立性；⑤许可与原始母集边界；⑥城市外景和距离/关系句的匹配度。未确认的量填“未核”，绝不以论文图片张数推导图组数。RGB-T 中的 **T 是 thermal（热红外），不是 depth**；“三模态”有些论文把文字也算一个模态，此处视觉三路必须逐项实查。

本次排除直接推荐的典型原因：DRSet 尚未发布；LanguageBind 热/深为生成图且无框；CityRefer 是城市点云而非本题四元组；TRansPose/L-CAS 是室内专门任务且申请/转制成本高；ScanRefer 需双重申请且 3D→2D 可见性难；KAIST 平台虽有 stereo/LiDAR，常用检测包不等于现成 RGB-D-T 毫米图；公开的另一 AIC 城市检测赛题具有类似三路格式，但其比赛材料和团队现有 city 素材应按各自来源记录，不因“test”字样推断质量或训练适用性。当前 city 可继续作为目标域素材和修标池，本报告不对它另作裁决。

## 下一步可证伪的小规模核验

1. 先用 RGBT-GroundBench 官方 train 抽一个来源、场景、光照均分层的小子集，逐样核对 RGB/TIR、Query、框的同源性和目标唯一性；与 city 现有样本做图像/场景级重叠排查，记录任何剔除。
2. RGBDT500、PST900 各取少量可用且有**不同对象可区分**的帧，只在许可和获取条件满足后确认真实图像类型、对齐、框/mask 与深度值域。对 RGBDT500 优先人工写少量表达；PST900 先作为三路读取/分割转框诊断，不能凭模板虚增指代样本。
3. SUNRefer/RoboRefIt/SUN-Spot 选含 near/far/front/behind、左右、序数的正例，确认其深度单位和 2D 框定义，用本地既定 412 条和独立错误分桶检验迁移；任何收益只在固定验证集报告 ACC@0.5 与逐 Query 变化。复赛官方 5690 条无 GT，只能推理提交，不用于阈值或权重选择。
4. 全量下载或训练之前，由实际操作者记录数据版本、条款、图组数、训练划分、坐标变换、人工语言生成规则和文件大小，供技术报告引用。本轮**没有**作整包下载、训练或可达性保证。

## 交接

优先从本文件第 1、2、3 类挑一项做**小规模文件级核验**；第 4 类真三路数据用于管线真实性检查，第 5 类 RefRT 受 46.7 GB、视频重复和多目标表达限制。不要把任一双模态数据称为本题完整样本，也不要把分割 mask 的类别名直接包装成经人工验证的 Query。下一位需要实测下载可达性和样例字段后，才可把表格中的“公开入口”升级为“本地已可用”。


## 收尾补检：第32项 DualVision 的模态问答监督

[DualVision官方仓库](https://github.com/abrarmajeedi/DualVision)说明DV-204K含约25k对真实RGB/IR、约204k模态相关问答；基于LLVIP/HDRT，底图另取，不算新的25k独立图源。另发布更密集v2，论文成绩仍对应v1。作者仓库说明提供标注，本轮未下载抽样；它没有Depth，不能当成本题现成BBox指代训练集。价值是补“哪个问题需要哪个模态”的辅助监督与数据构造方式。官方训练脚本是LLaVA7B、8GPU，不能按原样排入本机轻量复现。此项来自最后一轮图像融合/红外语言模型补检，因此全文覆盖为前表31项＋本项32个候选，不改变首批优先级。
