# 公开队伍与融合实现核查（2026-09-19）

## 结论

本次按赛题完整中文名称、AIC/RGB/Depth/Qwen、GitHub 限域等检索，未找到可以独立确认属于本赛题的其他队伍完整公开方案。搜索结果中的越南 AIC 视频检索、汽车问答、城市目标检测不能当作本赛题实现；本项目自己的 levolatte/TriGround 也不作为外部证据。这个结论是“本次没有找到”，不是断言其他队伍没有公开。

最相关的可借鉴实现是 RGBT-GroundBench 的 RGBT-VGNet；CMX、CMNeXt 是相邻的多模态语义分割研究，不是本比赛队伍或获奖方案。它们支持把精力放在模态适配、局部可靠性和有效空间特征，而不是只改文字位置编码。没有任何来源能承诺迁移到当前 119 条验证集后提高多少。

## 1. RGBT-VGNet：目前已存在训练入口和模型发布

- [作者仓库](https://github.com/crazyxiaoxi/RGBT-GroundBench)
- [论文](https://arxiv.org/abs/2512.24561)
- [实际训练入口](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/script_train/RGBT_VGNet/ref_flir/train.sh)
- [模型构建](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/models/__init__.py)
- [融合实现](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/models/mmvg_fusion.py)
- [下载脚本](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/script_train/RGBT_VGNet/prepare_pretrained.sh)
- [公开权重目录](https://huggingface.co/JiawenXi/RGBT-Ground-Model/tree/main)

实读当前 main 的脚本和源码确认：训练入口调用 MMVGFusion + IAFv3，开启 LAVS，RGB LoRA rank=16、IR rank=48，还启用对比、RTCC 和 mask 相关损失。不是仅将两路特征相加。IAFv3 利用局部亮度、RGB/IR 特征差异、位置与通道门来决定融合比例；该模型立足 CLIP，输出定位框。模型目录实际存在 models、pretrained_weights；本次没有下载大权重或复现论文成绩。

需要更正可能的旧判断：不能因仓库没有名为 VGNet.py 的文件就声称核心代码未发布。build_model 的 MMVGFusion 分支与上述启动脚本相互对应，代码以不同类名组织。当前已验证到训练路径和权重下载路径，尚未实际验证完整环境和端到端数值复现。

可迁移的最小想法（本项目建议，非作者已经验证的 Qwen3 方案）：

1. 给 IR 更有针对性的编码适配，不默认一个冻结 RGB 编码器加小型输入变换即可解决光谱差异。
2. 把全局固定融合量改为位置相关、由 RGB/IR 差异和亮度共同决定的补充量，判断辅助信息在哪些区域值得使用。
3. 用定位或区域对齐监督帮助辅助支路，而不是只依赖生成四个坐标时的文字交叉熵。

不能直接复制其 120 轮、默认 8 卡设置到本轮 4090；也不能把不同任务集上的论文成绩当成当前模型预期增益。

## 2. CMX：先校正两路，再融合

- [作者仓库](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
- [实际模块源码](https://github.com/huaaaliu/RGBX_Semantic_Segmentation/blob/main/models/net_utils.py)
- [论文](https://arxiv.org/abs/2203.04838)

实读 FeatureRectifyModule、ChannelWeights、SpatialWeights、FeatureFusionModule：两路特征共同产生通道与空间权重，分别给原特征加上加权的另一模态信息；之后用 CrossPath 和 ChannelEmbed 融合。其局部选择、独立支路和区域结构值得借鉴，但它仍会修改 RGB 特征，不能说原生 RGB 输出绝对不受影响。

本项目最小化方案：保留现有编码器，在对齐空间网格上产生小型辅助残差及位置门，并用现有四模态预测统计检验“纠正多少、破坏多少”。真正保证有原生 RGB 候选可用，需要额外保留原生 RGB 预测路径，并在后端选择；单纯 residual/skip connection 并不保证答案不退化。

## 3. CMNeXt：RGB 主路、辅助模态先选择再融合

- [作者仓库，旧地址会重定向](https://github.com/InSAI-Lab/DELIVER)
- [实际主干源码](https://github.com/InSAI-Lab/DELIVER/blob/main/semseg/models/backbones/cmnext.py)
- [CVPR 论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Delivering_Arbitrary-Modal_Semantic_Segmentation_CVPR_2023_paper.pdf)

实读源码：RGB 有独立的 patch embedding 和注意力模块；辅助模态使用另一组模块；PredictorConv 产生各辅助模态的空间分数；tokenselect 加权后跨辅助模态选择，再经 FRM/FFM 与 RGB 交互。是多尺度分割主干，不是可以原样插入 Qwen 的生成定位插件。

该作者公开表也说明“增加模态必升分”并不成立：KITTI360 的 RGB mIoU=67.04、RGB-D=65.09，而 DELIVER 上 RGB=57.20、RGB-D=63.58。场景与辅助数据的互补性会改变收益方向。这些数字是语义分割 mIoU，不是本题 ACC@0.5，不能横向比较。

最小借鉴：不要把 Depth 当另一幅 RGB 用同一套语义融合。优先显式提取有效深度、候选区域距离与前后排序，或者在空间特征上做可靠性选择；没有相机标定/地面信息时不应假称已经得到完整 HHA 几何编码。

## 对当前实验的优先级建议

### 结合当前实现后的修正

主 Agent 已核实本项目 JointQueryAwareStageFusion 已有 token＋sample 双门、余弦一致性、模态 dropout、零初始化残差，因此“再加一个门控”不是新结构方向。上面的门控描述只解释外部实现，不能当作我们缺失该能力的证据。更有实质区别的是外部方法在视觉编码层做模态专属 LoRA，以及直接给定位与区域对齐提供监督。

RGBT-VGNet 的损失链已继续核到源码：`train_val/mmvg_train.py` 导入 `engine.train_one_epoch`；`engine.py` 的 `'MMVG' in args.model_name` 分支调用 `utils/loss_utils.py::trans_vg_loss`。该函数实际总和如下，不只是脚本上列了未使用参数：

- 框 L1×2 与 GIoU×2，直接优化连续框的位置和重叠。
- contrastive：`clip_loss(text_eos)`，双向批内图文对比。
- RTCC：Query/视觉 patch 相似图对目标区域 mask 做 focal×20＋dice×2。
- mask：Query 条件分割输出对目标 mask 做 focal×20＋dice×2。

源码：[engine](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/engine.py)、[loss_utils](https://github.com/crazyxiaoxi/RGBT-GroundBench/blob/main/utils/loss_utils.py)。`MMVGFusion.forward` 在 `models/mmvg_fusion.py:1359`，末尾明确产生框、全局图文 logits、patch 相似图、Query 条件 mask；它不是让大模型把四个坐标当文字生成。

有一处不能扩大解释：RGBT forward 虽计算 IR 的全局图文 logits，但 trans_vg_loss 对比项只用返回的 RGB text_eos logits，传入的 img_cls 列表并未用于该项。不能宣称该实现已经对 RGB 和 IR 都施加了独立全局对比损失。此外当前 batch=1 若直接照搬批内 CLIP 对比则没有有意义的负样本；优先区域 patch/候选负样本监督，比机械复制 contrastive 标志更实际。

当前可借鉴最小改造是：在已有 Query 与辅助视觉特征上加区域定位辅助头，用已有 bbox 生成区域训练信号，保留现有生成定位目标，与仅生成坐标的基线做对照。辅助 mask 是框区域信号，不能冒称新增加了精细分割人工标注。是否加独立连续框头应单独验证，避免一次堆上所有损失。

1. 先使用已保存预测做四模态互补上限诊断：辅助模态确实纠正的样本有多少？若 oracle 选择也几乎不增益，后端路由不是主要方向；若净收益被大量对变错抵消，才优先做后端可靠性选择。
2. 若错误集中于多同类、距离和序数，优先候选框＋显式深度距离/空间排序，使 Depth 的作用可观察、可监督。这是本项目提出的改造，不是声称上述分割仓库已经替我们解决。
3. 若辅助支路本身没有语义定位能力，再考虑模态专属适配和区域定位监督。结构改造后与原生 RGB 保持独立对照。
4. 不建议仅因查到论文就把 CMX/CMNeXt 整网换入当前 8B；现有 Stage1 权重可作为小改造的起点。

本次仅查公开资料、保存研究笔记与两份源码快照，未改训练代码或运行中的实验。
