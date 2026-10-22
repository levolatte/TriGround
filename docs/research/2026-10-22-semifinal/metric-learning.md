# 半决赛指标对齐训练研究：从 IoU 阈值到候选错选

2026-10-22。本文只研究训练目标与输出方式，不改动业务代码，也未运行 GPU 训练。比赛事实取自 `contest/基于大模型的多模态视觉理解与推理.pdf` 全文：每条查询以预测框和真值框的 IoU 是否 **≥0.5** 判对，最终 ACC@0.5 为正确查询比例；提交框为归一化 `xyxy`，测试数据不得用于训练。以下论文效果均是**外部数据集结果**，不能当作本赛成绩。

## 先由计分公式确定可优化量

设真值框为 $b_i$、预测框为 $\hat b_i$，则

$$\mathrm{ACC@0.5}=N^{-1}\sum_{i=1}^{N}{\bf1}[\mathrm{IoU}(\hat b_i,b_i)\geq0.5].$$

这意味着把已经达到 0.8 IoU 的框再推到 0.9，直接得分为零；把 0.49 推到 0.51 才有收益。训练不能只优化平均 IoU，也不能只优化格式。当前交付记录的 5690/5690 题解析成功、0 兜底，因此格式奖励大概率已饱和。最终官方 0.8012 是 Arm 2 checkpoint-500、三图、2 倍放大**已记录分数**；本地没有官方逐题真值可复算，且训练材料审计指出同帧重叠，不能把它当作无泄漏泛化基准。参见 `code/docs/research/2026-10-22-takeover/current-pipeline.md`。

对历史 City412 真值与存档的旧8B C0直接框、候选集合，我只读计算得到以下**本地观测**（不是官方 5690 题）：直接框 292/412 命中，120 题未中；候选 oracle（每题知道真值后选择最佳候选）339/412 命中，理论上最多可多 47 题。直接框 IoU 分段：0 为 61 题，(0,0.1) 8 题，[0.1,0.3) 17 题，[0.3,0.5) 34 题，[0.5,0.6) 30 题，[0.6,0.8) 98 题，[0.8,1] 164 题。未中题中，零重叠 61 题有 23 题候选 oracle 可救；[0.3,0.5) 的 34 题有 13 题可救。34 题近阈值失误中，22 题预测/真值面积比大于 2 或小于 0.5，提示尺度回归值得单独检查。这些只说明候选集合的上限，**不是候选选择器可实现的提升**；历史选择器在同一 City412 的实测 273/412，低于直接框。City412已被多轮选模复用，重新切分不能恢复独立性；需从尚未参与选模的合法外部图组建立留出集，并结合规则允许的官方反馈判断迁移。这些旧8B统计不代表当前27B的错误组成。

计分误差可拆为两类：①目标身份错选，如同图相似车辆/人、序数和方位关系错，框平移往往 IoU 很低；②已选对目标但框尺度或边界错，较可能集中在 0.3–0.5。前者可研究**跨候选辨别**，后者可研究**阈值附近的几何精修**。IoU区间本身不能证明对象身份正确，需在有合法标注的开发图像上另行核对。仅为所有错框添加更多语言推理不能直接证明解决任何一类。

## 原论文与原源码审阅

| 项目 | 原论文结论与边界 | 源码核对、实际可复用部分 |
|---|---|---|
| [VLM-R1 论文](https://arxiv.org/html/2504.07615)、[代码](https://github.com/om-ai-lab/VLM-R1/tree/90052f478646e38bc67363862f699b7afca6b337) | 全文读方法、REC 奖励及实验；以连续 IoU、输出格式奖励做 GRPO。其 RefCOCO 系列特定设置中，3B 模型 RL 结果优于文中 SFT 基线（如 LISA-Grounding 63.14 vs 54.82）；不是本赛三模态或当前 27B 的证据。 | commit `90052f4`，仓库 Apache-2.0。核读 [`qwen_module.py`](https://github.com/om-ai-lab/VLM-R1/blob/90052f478646e38bc67363862f699b7afca6b337/src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py#L88-L177) 的格式/IoU、坐标变换；[`grpo_rec.py`](https://github.com/om-ai-lab/VLM-R1/blob/90052f478646e38bc67363862f699b7afca6b337/src/open-r1-multimodal/src/open_r1/grpo_rec.py#L113-L190) 的单图样本和奖励接线；[`grpo_trainer.py`](https://github.com/om-ai-lab/VLM-R1/blob/90052f478646e38bc67363862f699b7afca6b337/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L522-L585) 的多次采样、图像处理、优势计算和 PPO 式裁剪损失（同文件约 660–684、749–771 行）；[`run_grpo_rec_lora.sh`](https://github.com/om-ai-lab/VLM-R1/blob/90052f478646e38bc67363862f699b7afca6b337/src/open-r1-multimodal/run_scripts/run_grpo_rec_lora.sh) 的接口。IoU 函数只用第一张 `image_path`；原数据按一图输入，坐标从生成格点经原图宽高转换，并使用像素框的 `+1` 面积习惯，不能原封不动搬到三图归一化框。模块只显式实现 Qwen2/2.5 VL。 |
| [MedGround-R1 论文](https://arxiv.org/html/2507.02994)、[代码](https://github.com/xu-huihui/MedGround-R1/tree/37b210dd7d6ee71179b3013d7f8042af1de0d5d3) | 全文读 Chain-of-Box、IoU>0.5 奖励与 MedCLIP 语义奖励、实验。文中医疗定位结果与方法启发有价值，但胸片领域、7B、4 候选 rollout 与本赛差距大。论文称预测 ROI 的连续语义相似度有帮助。 | commit `37b210d`；仓库根无 license 标记，嵌套 `open-r1-multimodal/LICENSE` 为 Apache-2.0，复用时应核准适用边界。核读 [`grpo_rec.py`](https://github.com/xu-huihui/MedGround-R1/blob/37b210dd7d6ee71179b3013d7f8042af1de0d5d3/open-r1-multimodal/src/open_r1/grpo_rec.py#L236-L339) 与 [`grpo_trainer.py`](https://github.com/xu-huihui/MedGround-R1/blob/37b210dd7d6ee71179b3013d7f8042af1de0d5d3/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L363-L491)。实际 `medclip_reward` 参数名为 `solutions`，而数据/训练器交付 `solution`；实际对**整张图**取嵌入，没有裁剪预测 ROI，并把相似度 `>0` 二值化；运行脚本默认只有 accuracy/format，未启用 medclip。代码用严格 `>0.5`，而官方是 `>=0.5`；代码用 3B/8 rollout，与论文 7B/4 rollout 不一致。因此只取“阈值 + 语义辨别”假设，不能把仓库脚本视为已复现论文。 |
| [Visual-RFT 论文](https://arxiv.org/html/2503.01785)、[代码](https://github.com/Liuziyu77/Visual-RFT/tree/2ffad63b25ddd79bfe25d3e046645401201c89d6) | 全文读视觉检测与 grounding 的奖励、实验；其 LISA 推理定位实验提升的是其他任务的 mIoU，例如 2B 37.6 vs SFT 28.3、7B 43.9 vs 39.1，**不是 ACC@0.5 保证**。 | commit `2ffad63`，Apache-2.0。核读 [`grpo.py`](https://github.com/Liuziyu77/Visual-RFT/blob/2ffad63b25ddd79bfe25d3e046645401201c89d6/src/virft/src/open_r1/grpo.py#L87-L429) 的 IoU、confidence、格式奖励及脚本入口；[`grpo_lisa.py`](https://github.com/Liuziyu77/Visual-RFT/blob/2ffad63b25ddd79bfe25d3e046645401201c89d6/src/virft/src/open_r1/grpo_lisa.py#L56-L143) 的 GIoU/LISA 奖励；[`grpo_trainer.py`](https://github.com/Liuziyu77/Visual-RFT/blob/2ffad63b25ddd79bfe25d3e046645401201c89d6/src/virft/src/open_r1/trainer/grpo_trainer.py#L355-L461) 的单图 rollout、组优势和损失。其多框匹配在低于 0.5 时把值置零，接近本赛阈值思想，但默认还有主观置信度输出，增加本项目解析面。`src/scripts/2B_lisa_grounding.sh` 引用不存在的 `grpo_gui_grounding_lisa.py`，实际文件名为 `grpo_lisa.py`；不能复制脚本就跑。 |

补充筛选：[RGBX-R1](https://arxiv.org/html/2602.00504) 面向 RGB-X 推理，但其多图形式是顺序选模板/搜索图，并非本赛同步 RGB、红外、深度三视图；公开代码未定位到，暂不选作实现底座。[DPO 原论文](https://arxiv.org/html/2305.18290) 给出无需在线 rollout 的偏好目标，但其证据来自语言偏好，并未证明数值坐标字符串适合 DPO。[IoU-PD](https://arxiv.org/html/2607.15732) 讨论定位目标与离散化优化差异，可作阅读线索，尚无本赛验证。上述项目只用于机制对照，主训练管线仍以当前已验证的 ms-swift/Qwen3.6 路径为接入点。

## 奖励和损失：该先试什么

**第一优先是有真值的直接框 SFT 对照。** 保持目前语言 LoRA、三图预处理、短 `xyxy` 输出、训练数据边界，针对错例加权或分层采样：同图相似对象的身份误选、近阈值尺度误差分开计数。所有增强只用许可的训练/验证标注；先设一个未参与选模的图像组评估，不能继续在 City412 上反复选方案后把 City412 增益解释为泛化。训练时精确输出坐标的交叉熵是 token 目标，与 IoU 不完全对齐，但目前最省算力、稳定且容易证伪。

**候选相关方向：先训练辅助判别，再有条件地用于推理评分。** 总体顺序以主报告为准：最小增量是直接框SFT加同图A/B/both/neither判别辅助目标，第一轮推理仍直接框；只有新候选对当前27B有独有正确覆盖，才投资推理选择器。不要直接寄希望于oracle。 对同一三视图和 query 产生候选框，取真值 IoU≥0.5 的候选为正、同类对象或相邻位置候选为硬负，训练一个针对候选的对比分数或 pairwise 排序。正负应尽量同图，避免模型只学颜色/场景捷径。先报告候选召回（oracle 上限）、身份错选率、排序后 ACC@0.5 三个数；候选召回不足时训练评分器无用。若复用当前 27B，也可把候选编号/坐标送入短输出语言头；更小的图文 reranker 可能更便宜，但必须保证三图输入与准确的候选坐标映射。当前存档说明候选 oracle 相比直接框有 47 题理论空间，而历史手工选择器反而更差，恰好支持“选候选必须学”的实验假设，并不保证提升。

**第三优先才是直接优化阈值的 GRPO。** 一个可试的奖励为 $r(b)=\mathbf1[\mathrm{IoU}(b,g)\ge0.5]+\lambda\min(\mathrm{IoU}(b,g),0.5)$，无效框给低奖励；也可用 0.5 附近平滑斜坡，需通过同预算消融确定。硬阈值部分与官方指标一致，细小连续项让同一 query 的失败采样之间仍有排序信号，但过大 $\lambda$ 会退化回平均 IoU。VLM-R1 的纯 IoU 奖励偏向把已命中的框继续磨精；MedGround 的纯 0/1 阈值奖励又可能在一组全失败/全成功时没有组内优势。若单次采样命中概率为 $p$、每 query 抽 $G$ 个独立样本，纯二值奖励全同概率为 $p^G+(1-p)^G$；例如 $p=0.8$ 时 $G=4$ 约 41%，$G=8$ 约 17%。这里的 $p$ 不是 City412 确定性 ACC；相关采样还会让有效多样性更低。先记录 rollout 解析率、每组 IoU/奖励方差、全同组比例、有效梯度和单位 GPU 小时新增过线题，再决定是否扩大。27B×三张高分辨率图×多 rollout 显著昂贵，先用小样本、较低 G 和 LoRA 验证机制，再扩张。

**偏好学习（DPO）可作为离线备选。** 对同一 query 用现有模型采多个短框，只有较大 IoU 间隔、特别是跨越 0.5 的偏好对才用于 chosen/rejected；固定图像和提示词，避免数据泄漏。相比在线 GRPO 节省 rollout 训练期开销，但整条数值字符串的 token 对数概率受坐标位数、格式和长度影响，不等价于空间几何；训练后若只看到输出词形改变、IoU/过线率不涨，应停止。这一方向没有在本项目的实证依据。

**连续定位头是结构性更大的改动，暂不优先。** 坐标 token 对一个数字的小变化没有连续几何梯度；一个可学习的四维回归头或候选 RoI head 可以直接用 L1/GIoU/阈值近邻损失。但它需要改模型接口、训练器、推理和提交解码，可能与已验证的 Qwen3.6-27B LoRA checkpoint 不兼容。先用短框输出上的 SFT/重排/奖励证明瓶颈确实是数字回归，再考虑独立小头；若近阈值几何错误占比稳定且候选身份基本正确，该投入才可能值得。

## 直接短框与显式思维链（CoT，逐步推理）

当前生产解析仅需归一化 `xyxy`，交付记录解析失败为 0。长思维链会增加生成 token、训练目标分散和格式解析风险；如果仅用最终 IoU 奖励，它对中间“推理”文字没有直接监督，可能学会与定位无关的解释。MedGround 的 Chain-of-Box 也存在论文/公开代码不一致，不能据此宣称本赛需要 CoT。身份错选的语言关系难例理论上可能受益于先解释对象关系再出框，但这是待测假设。

公平对照：从**同一**基座/三图数据/训练预算、同一切分分别训练 A=直接短框 SFT，B=简短结构化中间步骤+框；推理分别算固定计算预算和相同最大 token 两种设置。报告总 ACC@0.5、身份错选子集、近阈值尺度子集、格式率、延迟、平均 token。若 B 只在被多轮调过的 City412 上涨、独立图像组或官方反馈不涨，或收益不足抵消算力与延迟，则保持直接短框。不要把 VLM-R1/Visual-RFT 外部“reasoning”实验结果直接外推到 Qwen3.6 三模态。

## 最小实验顺序和停止条件

1. 固定现有最高**已记录**交付路径作格式/推理参考；新训练数据先按图像组去重，留一个不参与任何调参的组。保存逐题 ID、真值、预测、IoU、来源与子集标签。官方测试不用于训练、补标或人工改框；规则允许的官方反馈可以用于模型选择，不能把反馈变成逐题训练标签。
2. 获取当前27B短框SFT基线的逐题结果并建立错误分解；旧C0仅保留作历史参照。分别构建身份硬负与尺度近阈值训练切片；一次只加一个机制，并用逐题翻转表、按图像组配对 bootstrap 的差值区间比较。若只提高平均 IoU 而 ACC@0.5 不动，停止该分支。
3. 在同一个候选生成器上训练候选评分；先验收候选召回和可救题覆盖，若排序 ACC 不超过直接框或增益只出现在 City412，则放弃/重做特征。
4. 先小规模采样检查IoU/奖励多样性；存在可学习差异时，以现有训练栈的小规模自定义奖励接口试阈值+弱 IoU；与纯 IoU、纯阈值、原 SFT **同预算**对照。若持续没有有效奖励差异、坐标解析下降或独立评估没有净收益，应停止或修正GRPO；纯二值全同不等于加连续项后仍必然无信号。
5. CoT、DPO、连续头按前述前提排序进入，不同时叠加。每次实验留清楚的模型、数据、提示词、图像分辨率、输出协议、检查点与逐题结果，以便区分训练目标和采样波动。

## 代码移植边界

现有三图 Qwen3.6-27B 使用 ms-swift 训练、vLLM 推理，视觉塔和 aligner 冻结、语言 LoRA。三个外部仓库的原创训练器主要围绕 Qwen2/2.5 VL 与**单图** REC：即便某段 trainer 的张量列表可接多图，其数据类、reward 输入和坐标约定也没有端到端证明适配本项目。因此推荐复用奖励**公式**、候选构造、rollout 诊断与 paired 评估设计，在当前栈上小步接入；不要 fork 外部 trainer 替换已验证路径。移植时唯一必要的边界检查是三视图顺序/有效图片、模型输出的归一化 `xyxy`、IoU 与官方 `>=0.5` 完全一致，以及训练/验证/测试图像组的隔离。

本地报告源：`code/docs/research/2026-10-22-takeover/current-pipeline.md`；City412 原始文件为 `code/docs/audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json`、`code/docs/audit-20260929/evidence/evaluation/C0/city412/predictions.jsonl`、`code/docs/audit-20260927/evidence/aux_selection_refine_20260927/city412/candidates.jsonl`。只读核算并未训练或改写上述数据。
