# 8B 三模态：结构性优化调查与优先级

日期：2026-09-19。用户要求调查 GitHub 前沿实现及其他队伍公开方案，寻找比位置编码/初始化更有潜力的方向。本轮进行了原作者仓库、训练脚本和实际模型/损失源码核查。下文的优先级是依据当前错误与实施成本提出的判断，不是已证实的提分。

## 新得到的本地实验依据

以下统一取独立 `evaluate.py` 结果，119条相同验证集，同一检查点关闭部分辅助模态；不是分别训练的双模态模型。

| 模式 | C4固定第4轮命中 | C4第2轮best命中 |
|---|---:|---:|
| RGB | 76 | 76 |
| RGB+IR | 77 | 78 |
| RGB+Depth | 77 | 78 |
| RGB+IR+Depth | 73 | 78 |
| 事后用真值总能选中四种输出里正确的一个：诊断上限 | 84 | 80 |

C4末轮三模态相对RGB：纠正7条，同时破坏10条，净减3条。best三模态纠正4条、破坏2条，净增2条。末轮四种模式全部失败35条，best全部失败39条。说明存在相互干扰，也存在所有现有输出都无法解决的错误；仅学习切换现有答案有明确上限，新增候选或改善定位能力更有意义。此上限使用验证真值，仅用于诊断，不能当作可执行模型分数。

证据：`8b-run-20260918-evidence/results/c4_modality_complementarity.json` 和两份逐样本 JSONL。当前训练中的T4于02:11左右启动，未改动其配置。

**评估口径待确认**：训练内C4 best为79/119，独立重载为78/119；last命中都为73但mIoU略有差别。源码发现 train.py 对CUDA设置 FP32 matmul precision=`high`，evaluate.py未设置（通常默认`highest`），且融合模块为FP32、两处评估都没有autocast；原生BF16 RGB结果一致。此为有代码依据的解释，尚未用GPU短复验确认。当前独立C4/T4入口一致，主比较保持同一入口，禁止把79与78混用。T4必做序列结束后，用同一C4 best只改变matmul精度验证，不重训、不覆盖既有结果。

## 当前结构真正存在的限制

- `adapters.py::JointQueryAwareStageFusion` 已有样本门、token门、跨模态相似度、模态丢弃和零初始化残差；再加一个普通门控不构成新的主方向。
- 在第8/16/24/26视觉层把融合残差写回RGB，后续层继续处理该结果。结构没有保证训练后的辅助模态只纠正错误、不会破坏原先正确的RGB判断。
- 模态注意力在对齐空间token位置局部执行；全局关系可经ViT和文字间接建模，但没有显式对象列表、候选关系和度量深度比较。
- 当前深度以log距离、有效掩码、零通道编码成8位图，再由共享Qwen视觉层和适配器处理；不是直接利用候选目标的毫米距离或近远排序。不能因此断言它完全丢失深度信息。
- 本轮主损失为坐标回答token交叉熵。仓库虽已有可选L1/GIoU辅助框头，本轮未开启；不能误称从未实现几何监督。损失下降而ACC回落，支持调查目标函数与定位质量的错配，但不足以证明它就是根因。

## 优先方向一：候选对象级融合与选择

**提案**：RGB产生候选框，保留原生Qwen预测，再把候选对应的RGB/IR局部证据、Depth统计和Query交给对象选择器。选择器输出候选ID，程序取回候选框。必要时仅对选中的区域做局部放大精修。

借鉴 [GroundingDINO推理实现](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py) 的框与文本分数；借鉴 [SeeGround对象选择实现](https://github.com/iris0329/SeeGround/blob/master/inference/inference_scanrefer.py) 的对象表、标号图及选ID机制。[SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT) 也提供面向区域的空间推理代码与模型。SeeGround是3D场景且默认72B，不直接移植整套、不引用其成绩作为我们8B的预期。

最小实施顺序：

1. 冻结候选检测器，对相同119条验证生成候选；用短目标类别描述和完整Query分别观察候选召回，加入原生RGB框，绝不把真值加入推理候选池。
2. 测候选池中至少一个框IoU≥0.5的覆盖上限、43个RGB错误中的新增可解数量。若新增候选连5个旧错误都覆盖不到，先解决召回，不投入重排训练。这是研发停止门槛，不是显著性标准。
3. 同候选池比较 A=RGB+对象几何；B=A+Depth；C=B+IR局部信息。深度采用有效值中位数、有效比例与排序，优先中心区域或掩码，避免框内背景污染；IR强度不当成绝对温度。
4. 若零样本选框显示空间，再在923训练样本学习轻量排序器，用同图同类其他对象作困难负例，候选IoU给出监督，缓存特征减少GPU成本。

**为何优先**：原生RGB43个错误中29个IoU<0.1，严重选错或定位失败比小幅框边界误差更突出。对象选择直接针对这个现象。深度关系组当前只有极少量样本，不能预设仅靠“远近”就能提高全体成绩；IR局部证据及候选召回同样重要。

**限制**：候选漏检会限制结果，标号/裁剪可能干扰VLM；选择器也可能破坏正确RGB答案。保留RGB候选并不保证不降分，需要真实成对结果。

## 优先方向二：显式定位与区域对齐监督

[RGBT-GroundBench/RGBT-VGNet](https://github.com/crazyxiaoxi/RGBT-GroundBench) 当前有真实训练入口、核心模型及权重链接，不再沿用早期“只有README”的判断。已核对 [train.sh](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/script_train/RGBT_VGNet/ref_flir/train.sh)、[模型](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/models/mmvg_fusion.py)、[损失](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/utils/loss_utils.py)。

它不只使用融合门，还使用模态专属视觉LoRA、连续框回归、图文/区域对齐以及条件mask监督。真实执行的损失含L1、GIoU、图文对比、patch区域focal/dice与mask focal/dice。源码全局对比项实际使用RGB图文矩阵，不能宣称IR也有独立全局对比监督。

**最小迁移**：先复用已有可选辅助框头，或由训练bbox自动构造目标区域监督，让Query对应的IR/Depth特征更明确地指向目标区域；困难负例来自同图相似对象。采用对应的单一损失消融，不一次照搬全部损失。当前batch=1，不能直接复制依赖批内负例的对比学习，梯度累积16不等于同一前向有16个负例。

这是训练目标和中间表示的改造，比词序编码更直接；但现成框头若只拟合旁支而不改善最终生成框，就不算完成提分。验证需始终报告实际输出框的ACC。

## 后续方向：定位奖励或适量主干适配

[PR1](https://github.com/linkangheng/PR1/blob/main/src/open_r1/rewards.py) 的定位奖励是真实IoU²，[Visual-RFT](https://github.com/Liuziyu77/Visual-RFT/blob/main/src/virft/src/open_r1/grpo_lisa.py) 的对应定位实现使用GIoU+1。它们提供定位质量驱动训练的可借鉴机制；现有多卡、多次采样脚本不能直接承诺在4090 24GB上跑8B。优先候选排序监督，后考虑定位奖励训练。离散生成坐标后计算GIoU并直接加loss并不能自动反传。

如果冻结主干难以吸收新表示，可用小范围视觉或语言LoRA，但要补同数据、同预算的RGB适配对照，区分“学了目标域”与“辅助传感器实际有用”。不以增加可训练参数本身作为高收益保证。

## 其他参赛队伍与证据边界

本次限定公开检索，没有找到能确认属于本赛题的其他队伍完整公开方案及成绩证据。不要把同名AIC、相邻检测赛道或我们自己的TriGround算作外队方案。已经找到的可用证据来自以上论文作者实现；未找到不等于不存在。

附录：[公开队伍及融合实现核查](2026-09-19-public-teams-fusion-sources.md)、[候选与定位奖励源码核查](2026-09-19-structural-grounding-sources.md)。

## 决策

当前C4/T4继续，保留完整比较。研究建议优先级：**候选召回及对象选择短实验 → 显式区域/定位监督 → 资源允许再做定位奖励训练**。本次交付是源码调查、已完成四模态结果分析和下一步具体实验建议，没有擅自用新架构覆盖运行中的已批准对照，也没有承诺固定提分幅度。
