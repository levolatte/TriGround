# 8B 三模态定位：结构性方向与公开源码核查

日期：2026-09-19。范围：公开原作者 GitHub／论文；只研究，不更改正在运行的训练。现有参照 RGB 76/119，C4 各轮 74/79/77/73。下文优先级是基于机制与预算的判断，不是已经观察到的新模型增益。

## 三个待验证假设

1. 主要瓶颈可能是选错对象，而非框坐标的小误差：原生 RGB 的 43 个错误中有 29 个已解析预测 IoU<0.1。候选对象选择比全图像素级融合更直接对应这个问题。但低 IoU 本身不能证明语义选错，仍可能是严重坐标误差。
2. 深度最可靠的价值可能是对象之间的远近／遮挡关系，而非要求一个小融合器从整张深度图学习所有几何语义。将深度变成候选对象的有效像素中位数、有效比例与距离排序，可以让原生 Qwen 保持 RGB 感知能力，同时获得直接可用的补充信息。
3. 文字 token 交叉熵与 ACC@0.5 不完全一致。直接监督候选排序或框的几何质量，比继续拟合坐标字符串更贴近任务；但 GRPO 在 24GB 单卡上的成本明显高于轻量排序头。

## 1. GroundingDINO：先保证候选召回，再解决选谁

- [原作者仓库](https://github.com/IDEA-Research/GroundingDINO)
- 已阅读实际源码：[groundingdino/util/inference.py](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py)
- `predict` 从模型取 `pred_logits.sigmoid()` 和 `pred_boxes`，用每个候选的最大文本 token 分数过滤框；可保留低阈值候选而非直接只取最高分。其框为归一化 cxcywh，必须转换成我们的 xyxy。
- **资源状况**：公开 Swin-T/Swin-B 预训练权重和推理代码；此官方仓库 README 的完整训练代码仍是待办，不把它称为训练全链条齐备。我们优先冻结检测器、缓存候选，暂不需要重新训练它。
- **可迁移部分**：用目标类别短语生成若干实例框；将原生 Qwen RGB 预测也作为候选，避免仅依赖检测器漏检；最后由关系推理／重排模块选框。
- **限制**：GroundingDINO 的最高置信度不等于最符合复杂描述，原生 RGB 上 76/119 的能力也不会自动被继承。小目标、遮挡目标漏检会限定排序上限。不能以检测 mAP 替代本题 ACC。

## 2. SeeGround（CVPR 2025）：把深度／空间关系变成对象表，模型输出 ID

- [原作者仓库](https://github.com/iris0329/SeeGround)
- 已阅读：[inference/inference_scanrefer.py](https://github.com/iris0329/SeeGround/blob/master/inference/inference_scanrefer.py)、[prepare_data/object_lookup_table_scanrefer.py](https://github.com/iris0329/SeeGround/blob/master/prepare_data/object_lookup_table_scanrefer.py)
- 实际流程：Mask3D 提供预测对象及三维包围框；解析 Query 的 target 和 anchor；生成对象空间说明与带对象 ID 的视角图；Qwen 选择 `Predicted ID`，程序从对象表查框。真值框只在最后评价环节取用。`OpenAI` 类在源码中连接的是 localhost 的 Qwen 服务，不应误读为其必须调用商业模型。
- **资源状况**：对象表预处理、推理、评价代码和基础 Qwen 权重链接已提供；本身是零样本方法，不是另有一套端到端融合训练权重。默认配置 Qwen2-VL-72B／8卡，不能套用其性能数字预测我们的 8B 单卡效果。
- **可迁移部分**：只借“对象表＋图像＋ID选择”。我们的单帧 Depth 不需要先重建完整三维场景：每框提供归一化中心、尺寸、有效深度中位数、有效深度比例、近远排序。序数由候选中心排序；相对距离由测量值与参考对象支持，而不是让融合网络从零学这些规则。
- **限制**：完整 SeeGround 依赖三维点云、实例与视角渲染，不能原样套进 RGB/IR/Depth 单帧。框内背景会污染深度统计，遮挡／缺失深度需要有明确的有效比例；框中心附近统计或对象掩码可作为后续改进。IR 适合作为弱光下候选补充或局部识别证据，不预设直接迁移 RGB 检测器到 IR 就有效。

## 3. Perception-R1 / PR1（NeurIPS 2025）：直接优化定位几何奖励

- 注意有同名项目；这里明确是 [linkangheng/PR1](https://github.com/linkangheng/PR1)，不是几何数学推理的另一个 Perception-R1。
- 已阅读：[src/open_r1/rewards.py](https://github.com/linkangheng/PR1/blob/main/src/open_r1/rewards.py)、[训练脚本](https://github.com/linkangheng/PR1/blob/main/local_scripts/train/train_qwen2_2b_vl_grounding.sh)
- `pr1_grounding_reward` 解析框后直接返回 **IoU²**；格式奖励另加。源码不是仅在 README 中声称做定位奖励。
- **资源状况**：训练代码、grounding 评价代码、数据链接和 `Kangheng/PR1-Qwen2-VL-2B-Grounding` 权重均在作者仓库提供。
- **限制**：现成脚本为 7 个训练进程、8 次组采样、Qwen2-VL-2B；不是适配当前自定义三模态 Qwen3-VL-8B 的即插即用训练器。单卡额外生成多条候选的时间／显存代价需先测。IoU² 在全错、无交集组中仍可能几乎没有学习信号。
- **我们的优先迁移**：先做轻量候选排序，用真实框自动产生正例、同图困难负例和 IoU 偏好；只有召回和选框已显示明确上限，且组内奖励有差异，再投入 GRPO。直接加 GIoU 到离散生成后的框不会自动获得有效反向梯度，不能把奖励函数当成可微回归损失随手相加。

## 4. Visual-RFT（ICCV 2025）：小数据定位奖励的具体实现与运行成本

- [原作者仓库](https://github.com/Liuziyu77/Visual-RFT)
- 已阅读：[grpo_lisa.py](https://github.com/Liuziyu77/Visual-RFT/blob/main/src/virft/src/open_r1/grpo_lisa.py)、[2B_lisa_grounding.sh](https://github.com/Liuziyu77/Visual-RFT/blob/main/src/scripts/2B_lisa_grounding.sh)
- LISA 定位分支 `compute_giou` 实际返回 **GIoU+1**，并有输出格式奖励；它相较纯 IoU 能区分部分没有交集的框。不要把检测分支、LISA 分支的奖励混为一谈。
- **资源状况**：训练代码、数据构造与评价代码已提供；README 链接一个 200 多条 LISA 样本训练出的推理模型。
- **源码限制**：所读 LISA bash 仍指向 `grpo_gui_grounding_lisa.py`，而源码归档中的实际文件是 `grpo_lisa.py`；有硬编码数据路径。例程还默认 8 卡、8 次组采样。它证明机制可参考，不证明我们的单卡 8B 直接可运行。

## 最优先实验：RGB 候选＋对象级 Depth 关系选择

这是一项结构性推理原型，先不重训 8B、不覆盖现有 C4/T4。

1. 固定 119 条验证 Query，不挑错误样本单独报总成绩。对每幅 RGB 缓存类别候选框，加入原生 RGB 的预测框。候选列表只来自图像和 Query，不把 GT 加入推理列表。
2. 先报告候选池的 oracle ACC@0.5（至少存在一个 IoU≥0.5 的候选）及 RGB 43 个错误中可被新候选覆盖的数量。这是诊断上限，不是模型成绩。
3. 固定同一候选池做两个对照：A=RGB 图＋候选 ID/几何；B=A＋对象 Depth 统计。相同 Qwen8B、提示结构、生成长度，仅多出深度字段。先不要同时加入 IR、掩码、RL，以免不知道增益来源。
4. 输出一个合法候选 ID 并查原始候选框；记录正确→错误／错误→正确、解析率、推理秒数。可在 RGB 选择器基础上，再试 IR 对应区域图用于夜间／可见光混淆，不预先强行改全局特征。
5. 如果零样本选 ID 有潜力，再用 923 条训练集学习小型排序器：候选正例由 GT IoU 给出，同图同类别且 IoU 低的框做困难负例；训练目标用列表排序或成对偏好。训练集上候选未覆盖 GT 的样本必须记录为召回失败，不能从最终验证分母删去。

**预设失败判据／停止条件（研发判断，不是显著性标准）**：

- 若新增候选在 RGB 的 43 个错误中最多只能覆盖不到 5 个，暂不训练重排器，先解决召回或改走其他结构。
- 若 A 选择器比原生 RGB 明显掉分，先检查标号图/候选框限制是否损害感知，不把 B 的结果直接归因于深度无效。
- 若 B 相比 A 净增不足 3/119，或者所有收益靠新增错误抵消，当前不足以成为主线；保留原生 RGB。
- 即便首轮净增达门槛，也只称开发信号，需训练种子／提示固定复验。119 条、24 场景规模不能承诺高收益。

## 当前结论

比 Query 位置编码更值得投入的是：让辅助模态作用于明确的对象选择和几何关系，或把学习目标改成实际定位质量。优先级建议 **候选＋对象关系短实验 → 排序监督 → 资源允许时定位 RL**。这比直接宣布再堆一个融合 Transformer 更容易定位失败环节，也更可能在现有 24GB／48小时约束内得到明确答案；“更可能”是工程判断，尚不是收益保证。

源码读取方式：网页原作者页面＋GitHub codeload 源码归档；未执行第三方训练代码。归档暂存于 `.work/research_SeeGround.zip`、`.work/research_PR1.zip`、`.work/research_Visual-RFT.zip`。
