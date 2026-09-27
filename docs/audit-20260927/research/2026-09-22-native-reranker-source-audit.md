# 原生多图与候选重排路线源码审计

日期：2026-09-22。范围是源码、论文和本地已存在实验资料的只读审计；没有启动 GPU、下载权重或修改训练代码。本文的“可确认”指来源直接给出的机制或本地源码实际行为；“待验证”指对本赛题 923/119 划分的实验假设，不能当作预期涨分。

## 结论先行

1. **原生 Qwen3-VL 三图 LoRA 是可实现的路线，但官方微调入口的“只训练语言 q/k/v/o、冻结视觉 Merger（合并器）”是依赖模块命名的实现结果，不是独立的显式开关。** 官方脚本先冻结全模型，再以 `q_proj/k_proj/v_proj/o_proj` 注入 PEFT LoRA；Qwen3-VL 视觉注意力使用融合式 `qkv/proj`，而语言注意力使用这四个名字。因此在当前 Qwen3-VL 架构上 LoRA 落到语言层，视觉主干和 `visual.merger` 保持冻结。开跑前必须打印实际 LoRA 参数名和可训练参数名，不能只相信配置。
2. **Qwen3-VL-Reranker 的现成 `process()` 是严格推理接口，不能直接拿来训练排序。** 它将 `yes` 与 `no` 的 LM head（语言模型输出头）权重相减，取最后 token 隐状态做一维线性分数并取 sigmoid；`compute_scores` 有 `@torch.no_grad()`，最后还 `.cpu().detach().tolist()`。训练版必须保留原 `lm`、关闭该 no-grad/脱图路径并直接计算 `logit_yes-logit_no`，随后才可对 LoRA 做 listwise（列表排序）或 pairwise（成对排序）损失。
3. **把 RGB、IR、Depth 原生作为三张图送给语言模型，会让语言模型实际接收三份视觉 token；当前融合代码只把一份融合后的 RGB token 序列送入语言模型。** 在项目当前 `max_pixels=602112` 的上限下，Qwen3-VL 的每图上限约为 `602112 / 32^2 = 588` 个合并视觉 token；三图约为 1764，旧融合路径约为 588。旧路径仍会对三支流运行视觉块，不能把它说成“三图视觉免费”；差异在于原生路线还把三倍视觉 token 放入语言模型上下文。
4. **RGBX-R1 可借“先建立多模态推理信号、再做强化”的研究问题，但不能作为本赛题的直接配方或“BBox SFT 主要提升 RGB”的充分证据。** 它是时空对齐的 RGB-X 跟踪视频改写出的多图任务：通常两张模板图后接六张搜索图，并使用 5k+ VM-CoT（视觉模态思维链）和 GRPO。论文明确说只用 BBox 监督的 MLLM 仍难理解 X 模态，支持“裸 BBox SFT 对 X 不充分”；没有给出能把“只/主要提升 RGB”严格归因给 BBox loss 的单变量消融。审计时未找到作者可确认的官方代码库或权重链接。
5. **候选排序仍比直接上 GRPO 更适合先做结构筛选，但门槛是候选 Recall@K（候选召回）。** 先测 RGB 候选是否覆盖当前 RGB 错误的真框；没有正候选时，任何“在候选中选最优”的损失都无法纠正该样本。Rex-Thinker 是有官方实现与 7B 权重的相关候选范式，不过它依赖 GroundingDINO、面向 RGB 指代表达，不能代替三模态候选覆盖测试。

## 1. 与赛题和现有实现的对应关系

官方赛题是**单帧、空间对齐的 RGB/红外/深度与一个英文 Query**，输出 RGB 图中的一个归一化框；一组图可对应多条 Query，指标是 IoU>=0.5 的 ACC@0.5。深度的 0/过小值无效，数值越小越近，单位毫米。它不是 RGBX-R1 的模板追踪任务。依据：本地官方 PDF《基于大模型的多模态视觉理解与推理》页 2--6；尤其页 3 的输入、标注与深度定义，页 4 的 ACC@0.5。

当前项目的 `NativeGroundingCollator` 在对话内容中只放 `sample["rgb"]`，并以这一张图调用 processor；IR 与 Depth 被另行编码为 `ir_pixel_values`、`depth_pixel_values`。`MultiModalGrounder._qwen_inputs()` 最终只向 Qwen 原生接口传入 `pixel_values=fused` 及一份 `image_grid_thw`。并行融合视觉前向虽对 RGB、IR、Depth 都执行 `patch_embed` 和每个视觉 block，最终返回的却是 `vision.merger(rgb_tokens)`。所以现有路径是“三流视觉计算 + RGB token 长度的语言模型”，不是原生多图路径。[本地 collator](../../code/src/mm_grounding/data.py) 的 174--224 行与 [模型](../../code/src/mm_grounding/model.py) 的 87--221、601--626 行是直接证据。

这个区别有两层含义：

* 原生三图路线可能让语言层直接做跨图 token 注意力，属于真实的结构改变；它并非把现有融合门改名。
* 原生路线也同时改变了语言上下文长度、视觉 token 次数、输入模板和训练分布。即使结果变好，也不能自动归因于 IR/Depth 内容；须有同预算 RGB 直接 SFT 对照以及 IR/Depth 内容置换对照。

## 2. 官方 Qwen3-VL 微调：LoRA、视觉冻结与多图

### 2.1 已确认的官方训练逻辑

[Qwen 官方 `train_qwen.py`](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py) 的 LoRA 分支先执行：

```python
for p in model.parameters():
    p.requires_grad = False
LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], ...)
model = get_peft_model(model, lora_config)
```

同一文件的非 LoRA 分支则分别用 `tune_mm_vision`、`tune_mm_mlp`、`tune_mm_llm` 对 `model.visual`、`model.visual.merger`、`model.language_model` 整体设 `requires_grad`。因此：

* 在 LoRA 分支中，**基础权重**均先冻结，包含视觉编码器、视觉 Merger 和 LM head；脚本没有再将 Merger 打开。
* `q/k/v/o` 定位是按**模块尾名**匹配，并不是 API 中名为“language-only”的参数。Qwen3-VL 当前视觉注意力的线性层命名为融合式 `qkv`/`proj`，文本语言层才使用 `q_proj/k_proj/v_proj/o_proj`；于是目标列表只会命中文本层。这是当前版本的可复现机制，但依赖 Qwen/Transformers/PEFT 版本。
* 该脚本本身没有一个“仅语言 LoRA”的断言。实际训练前要在模型构建后输出：所有 `lora_A/lora_B` 参数名、`model.visual.merger` 的 `requires_grad`，以及总可训练参数数。若出现任一 `visual.*lora_*` 或 Merger 参数可训练，便不满足本路线假设。

这里的结论是“**可限定且 Merger 会冻结，但要用实际参数清单验收**”，不是“官方接口有无条件保证”。`rank=32` 只决定新增低秩参数的大小，不能证明整次前向、反向或优化器必然放得进 48 GB。

### 2.2 多图和 patch/merge 后 token 数

[Qwen 官方 Reranker 源码](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py) 把 `IMAGE_BASE_FACTOR=16`、`IMAGE_FACTOR=32` 写为图像对齐因子，并注释 `MAX_PIXELS=1800 * 32^2` 对应 1800 token。其 README 也明确接受 image 的列表。由此可用以下**处理器层面的计数**审计每张图：

\[
t_i = \frac{H'_iW'_i}{32^2},\qquad 4 \le t_i \le t_{\max},
\]

其中 `H'_i,W'_i` 是处理器按 32 对齐、缩放后的尺寸。32 可理解为 16-pixel patch 经 2x2 空间合并后的每个语言视觉 token 的边长；精确取整仍应由实际 `image_grid_thw` 日志给出。当前项目使用的 602112 像素上限对应 `t_max=588`。若三张同样接近上限：

| 路线 | 视觉编码支路 | 输入语言模型的视觉 token 上限 | 语言层序列长度近似 |
| --- | --- | ---: | --- |
| 当前并行融合 | RGB、IR、Depth 各一支 | 588（融合后的 RGB） | `T_text + 588` |
| 原生 RGB/IR/Depth 三图 | RGB、IR、Depth 各一支 | 1764（三张分别合并） | `T_text + 1764` |

这解释了“原生多图的语言层更贵”，却不足以给出速度倍率。FlashAttention、每张实际 `image_grid_thw`、文本长度、梯度检查点、分辨率、冻结支路是否保留激活、LoRA 放置层数、L40 与 4090 的显存/带宽/算力都不同。**不能由 48 GB L40 和旧 4090 日志推断吞吐、训练时长或必然可放下。** 唯一有效验收是固定版本、固定三图上限和真实最长样本，完成一次 forward/backward/optimizer step，记录 token 数、峰值 allocated/reserved 和记时；这也与项目已有 8B 升级文档的预检原则一致。

## 3. Qwen3-VL-Reranker 2B/8B 的真实打分机制

[官方 README](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/README.md) 说明 2B/8B reranker 为 query-document 单塔、pointwise 重排；二者类与打分实现共用同一 [`qwen3_vl_reranker.py`](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py)。源码事实如下。

1. 初始化加载 `Qwen3VLForConditionalGeneration`，保存 `lm.model`（去掉语言输出头的主干），设置 `model.eval()`。
2. 从 tokenizer 词表取字面 token `yes`、`no` 的 ID，并从 `lm.lm_head.weight` 取两行权重，构造无 bias 的线性层 `w_yes-w_no`。
3. 对“系统要求只回答 yes/no + Query + Document”的单个 pair，取 `last_hidden_state[:, -1]`，算 `sigmoid((w_yes-w_no)h)`。
4. `compute_scores` 标记 `@torch.no_grad()`，返回前 `.cpu().detach().tolist()`；`process` 对 documents 逐个 pair 循环，而非一次把候选在上下文中联合比较。

当 LM head 无 bias 时，分数正是只在 `{yes,no}` 两项归一化的条件概率：

\[
s=\operatorname{sigmoid}(z_{yes}-z_{no}).
\]

它不是完整词表 softmax 中的 `P(yes)`，也不是候选间的 listwise softmax。这个 logit-difference（对数几率差）是有源码依据的可靠机制；它对“哪一个框的 IoU 更高”是否有效，则没有本赛题证据。

### 3.1 改造成可训练排序器的最小变化与代价

不建议复制官方 `process()` 再把返回分数套损失。训练实现至少要：

* 保留含 LM head 的 `Qwen3VLForConditionalGeneration`，改为 `train()`；不走 `@torch.no_grad()` 和 `.tolist()`；
* 对每个 pair 直接从可求导 logits 得到 `z_yes-z_no`，或复用差权重但保留计算图；
* 在语言 q/k/v/o LoRA 上求梯度，视觉与 Merger 保持冻结，并在训练开始时检查可训练参数名单；
* 重新设计 batch、候选编码、无正例和最终 fallback（回退）路径。官方 2B/8B 权重和推理封装并没有给出这些训练数据管线、loss 或显存配置。

若每个候选包含 5 张图，官方 pointwise 格式对一条 Query 的计算量约为 `K` 次前向：

\[
\sum_{j=1}^{K} C\bigl(T_q+\sum_{m=1}^{5}t_{j,m}\bigr).
\]

它不是把所有 `K` 个候选放进一次上下文，但可训练 listwise loss 必须在反传前保留所有 `s_j` 的计算图。逐候选前向后再 `torch.stack(scores)` 并不释放前面图，峰值仍随 K 增长；把 `K` 个 pair 直接 batch 化则会受 padding 和 batch 激活影响。梯度检查点能以重算换激活，不能使 K 份语言/视觉计算消失。最省显存的备选是先冻结大模型、缓存候选特征并只训练小排序头；但这测试的是另一条路线，不能宣称为“端到端 LoRA reranker”。

## 4. 排序监督的边界：多正例、无正例与伪对应

### 4.1 多正例 listwise loss

按官方 ACC@0.5，候选集可有多个同样合格的框。令 `P={j: IoU_j>=0.5}`，一个与指标一致的多正例 listwise 目标为：

\[
L=-\log\frac{\sum_{j\in P}\exp(s_j/\tau)}{\sum_{j=1}^{K}\exp(s_j/\tau)}.
\]

它要求至少一个合格候选排在负例前，避免把 NMS 不充分造成的同一目标重复框互相当假负例。若需要偏好更高 IoU，可在此基础上以 IoU 构造温和软标签或在正例内再排，但不要把 ACC@0.5 的“任一合格”与“唯一最高 IoU”混为同一监督目标。重复候选很多时应先按重叠分组/抑制，否则正例总质量会被重复框放大。

### 4.2 无正例不是可忽略的实现细节

`P` 为空时上式分子为零，不能损失化为“挑一个最不差的框当正例”。这类样本反映候选召回失败：

* 用于纯候选排序训练时跳过 listwise loss，但单独报告其比例和 Recall@K；
* 若业务/比赛输出允许拒答，加入一个明确的 none-of-the-above 候选并对它监督；本赛题必须输出框，因而最终仍需一个已验证的回退，例如原生直接 SFT 预测或候选检测器最高分框；
* 无正例的测试样本不能被 reranker 修复，故候选召回必须先于排序训练验收。

### 4.3 “同坐标三模态”伪对应的风险

赛题材料说明三模态做过人工对齐，这允许把同一 **真值 RGB bbox** 投影为 IR/Depth 的起始区域假设。但从 RGB detector 候选框直接复制坐标到 IR/Depth，不自动成为三模态对象对应真值：剩余视差、裁剪/缩放、时间不同步、深度无效点、边界背景和 RGB 检测框偏移都会使另两图局部区域偏离目标。把它写成同一对象的强正例，会把传感器不一致误训成“模态语义”。

较稳妥的最小处理是：训练阶段先用已知真框核验注册误差与 Depth 有效比例；候选阶段把 IR/Depth 同坐标 crop 视为带不确定性的辅助证据，Depth 用中心/掩码内有效值中位数、有效比例和相对顺序，不把 IR 强度当绝对温度。若要训练跨模态对应，正例须由真框 IoU、实际对齐质量或人工复核支持；同图同类对象是排序难负例，不是跨模态同坐标的证明。

## 5. RGBX-R1：支持什么，不支持什么

[RGBX-R1 论文 HTML](https://arxiv.org/html/2602.00504v1) 的关键原文证据是：

* 它从五个 RGBX **tracking video** 数据集构建 RGBX-Grounding，关键帧间隔 24--29 帧，每四个关键帧生成 8 张 RGBX 图构成一条 MIG 样本；形式上有 RGB template、X template 和通常 `N=6` 张搜索图。
* 97.24% 样本是“两张模板图 + 六张搜索图”，且 VM-CoT 的 Associate 步骤使用模板框的空间对应，Validate 步骤逐帧检查退化与互补。
* 论文说即使用 BBox 做 SFT，MLLM 仍难以感知理解 X 模态；随后以 5k+ VM-CoT 做 cold-start SFT，再以 GRPO/MuST reward 做时空强化。

所以以下表述是严谨的：**RGBX-R1 提供了“裸 BBox SFT 不足以建立 X 模态理解”的论文证据；其完整体系在时空对齐模板追踪数据上有效。**

以下表述超出了证据：

* “BBox SFT 只/主要改善 RGB”——论文动机与基线支持 X 仍弱，但没有仅改变 BBox supervision、固定其余训练数据/序列/推理链的因果消融来证明“主要”或“只”。
* “把 UAV 模板和多帧 CoT 移到本赛题会提高分数”——本赛题没有模板目标图和同对象视频序列；照搬会人为生成一个不同任务。
* “有官方可运行代码和权重”——本次查阅 arXiv v1 与公开检索未发现作者可确认的 GitHub/Hugging Face 代码或模型链接。应记为“未找到”，而非断言未发布。

它能给本赛题留下的较小启发是：先用直接 BBox SFT 验证原生多图能否产生信号；若确有稳定提升，再针对有限且可验证的空间/模态解释加入辅助目标。不要在第一轮同时引入长 CoT、序列模板和 GRPO。

## 6. Rex-Thinker 作为候选路线参照

[Rex-Thinker 官方仓库](https://github.com/IDEA-Research/Rex-Thinker) 与 [ICLR/OpenReview 论文](https://openreview.net/forum?id=btWHQoSZZ1) 均存在；仓库给出 GroundingDINO 安装、SFT/GRPO 训练指引，并链接 [Rex-Thinker-GRPO-7B 权重](https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B)。它将直接坐标生成改为开放词汇检测候选 + MLLM 对候选进行思考/选择，故可作为“候选先验 + 对象级判断”的真实公开机制来源。

可迁移的是顺序：先冻结候选生成器、量 Recall@K、把同类对象放入困难候选集，再评价选择器。不可直接迁移的是 RGB detector 召回、其语言/CoT/GRPO 数据、7B 权重及成绩：它没有处理本赛题的 IR、16-bit 深度有效值和三模态对齐误差。引入它的代码或权重会额外需要 GroundingDINO 检查点和环境，当前审计未执行下载或运行。

## 7. 923/119 下的最小可解释对照

当前 923 训练、119 验证和旧冻结融合不稳定，优先级应是先分辨“原生三图直接适配是否值得继续”与“候选排序是否有可达上限”，而不是一次叠加 R/RID、长解释、候选和强化。

### 7.1 原生路线第一轮

先做同一 Qwen3-VL 规模、同一 LoRA target/rank、同一图像上限、同一 923/119 划分、同一 optimizer updates 与种子的两臂：

| 臂 | 视觉输入 | 每条样本的主要监督 | 目的 |
| --- | --- | --- | --- |
| D-RGB | RGB | 直接 bbox JSON SFT | 排除“目标域适配本身”带来的增益 |
| D-RID | RGB+IR+Depth 三图 | 同一直接 bbox JSON SFT | 测原生多图的完整可行性与净收益 |

报告 ACC@0.5、mIoU、解析率、逐样本错变对/对变错和峰值显存；D-RID 另做 IR/Depth 内容置换。该对照回答的是完整原生路线是否有信号，尚不声称单独来自某一模态或某个提示词。

若“60/25/15”是 R、RID 与直接 bbox 三种**样本格式份额**，它从一开始会减少直接坐标监督覆盖，进而混淆方法与监督量。最小公平办法有两种，优先第一种：

1. 两臂的 923 条样本都保留 100% 直接 bbox loss；扩展臂只在同一条样本上增加 R/RID 辅助 loss，令每段损失先按被监督 token 数归一化，`L=L_bbox+lambda_R L_R+lambda_RID L_RID`。两臂保持相同 input visits、optimizer updates、图像/文本上限；报告辅助 token 数和 `lambda`。这能检验辅助任务，不把 bbox 样本量偷换掉。
2. 若工程上只能使用三种互斥响应格式，则设一个与完整臂使用**同一 60% direct 子集**的 direct-only 对照，并保留 100% direct 的参考臂。前者匹配直接框覆盖，后者量化少用 40% direct 数据的代价。不能把 60/25/15 混合臂和 100% direct 臂的差异解释为 R/RID 单独效果。

只有 D-RID 对 D-RGB 有稳定、可复验的信号后，再拆 IR 与 Depth、R 与 RID；否则复杂辅助目标只会扩大变量数。

### 7.2 候选路线的先验筛选

不训练 reranker，先对 119 条和 923 条建立相同候选池，报告：每条 K、Recall@1/5/10、无正例率、RGB 直接 SFT 错误中被候选新增覆盖数、候选数分布及耗时。候选池禁止在推理时插入真框。若现有直接 RGB 错误里几乎没有新可覆盖目标，停止排序 LoRA，先改善候选生成或回到定位目标；listwise loss 不可能穿透候选上限。

若通过覆盖门槛，再做一个轻量排序器或 2B reranker 的冻结/小参数 screen；确认候选内容、标签和 listwise loss 有效后才评估 8B LoRA。48 GB L40 是可用试验资源，不是 8B、rank32、五图乘 K 一定能 fit 的证据；第二张卡不保证时，不应把初轮设计成依赖双卡。

## 8. 可靠机制、外推和下一步验收

| 结论 | 证据级别 | 原因/验收 |
| --- | --- | --- |
| 当前融合只把 RGB token 长度送入 LLM | 已由本地源码确认 | collator 的单 RGB message、`_qwen_inputs` 和 `vision.merger(rgb_tokens)`；训练日志记录实际 token 数复核 |
| 原生三图会增加语言视觉 token | 处理器/模型机制确认 | 记录三张 `image_grid_thw`，而非凭原分辨率估算 |
| q/k/v/o LoRA 冻结视觉 Merger | 当前官方脚本+模块命名支持 | 每次依赖升级后检查 trainable parameter names |
| Reranker 是 yes/no logit difference 的 pointwise 推理器 | 官方源码确认 | no-grad、差权重、逐 document 循环均已见源码 |
| 原生三图、候选 rerank 或辅助 CoT 会超过现有最佳 | 待验证 | 923/119 的配对对照、第二 seed 和置换证据 |
| RGBX-R1 的涨幅可迁移到本赛题 | 不支持 | 它是模板+多帧跟踪改写任务，训练和奖励都不同 |
| BF16 rank32 在 48 GB 可训练 | 不支持 | 真实最长输入的预检才给答案，且候选 K 保留图会放大激活 |

建议的执行顺序是：**D-RGB vs D-RID 直接 SFT预检与显存测量 -> 同预算成对验证和内容置换 -> 候选 Recall@K -> 仅在候选覆盖足够时做可训练 listwise reranker。** 这保留了“原生多图”和“对象级选择”两个结构假设，但不会提前把论文的时空 CoT、候选质量和硬件容量误写成已验证结果。

## 来源索引

* 赛题：本地 [`contest/基于大模型的多模态视觉理解与推理.pdf`](../../contest/基于大模型的多模态视觉理解与推理.pdf)；已逐页读取。技术报告提纲为本地 [`contest/【技术报告】基于大模型的多模态视觉理解与推理-0415.docx`](../../contest/【技术报告】基于大模型的多模态视觉理解与推理-0415.docx)。
* Qwen 微调：[官方训练脚本](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/train_qwen.py)，[官方训练参数](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/train/argument.py)，[Qwen3-VL 8B 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)，[Transformers 的 Qwen3-VL 模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。
* Qwen 检索模型：[Qwen3-VL-Embedding/Reranker README](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/README.md)，[Reranker 源码](https://github.com/QwenLM/Qwen3-VL-Embedding/blob/main/src/models/qwen3_vl_reranker.py)。
* 论文/候选参照：[RGBX-R1](https://arxiv.org/html/2602.00504v1)，[Rex-Thinker 官方仓库](https://github.com/IDEA-Research/Rex-Thinker)，[Rex-Thinker OpenReview](https://openreview.net/forum?id=btWHQoSZZ1)。
* 本地实现：[数据输入](../../code/src/mm_grounding/data.py)，[并行融合与 Qwen 输入](../../code/src/mm_grounding/model.py)，[8B 升级/预检说明](../../code/QWEN3_VL_8B_UPGRADE.md)。
