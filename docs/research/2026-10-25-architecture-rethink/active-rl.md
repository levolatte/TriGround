# 主动感知与可学习视觉动作：一次双区域三模态观察策略

本文为该专题的独立方案，综合架构与实施优先级见[主报告](README.md)。

2026-10-25。研究问题：让 27B 模型主动取得能改变目标身份和边界判断的新视觉证据，再直接优化赛题的最终框。本文是架构提案，尚未训练或运行新模型。

## 结论先行

**推荐训练一个“全图初判 → 一次双区域三模态观察 → 全图最终框”的有限动作策略。** 同一 27B 模型先从 RGB、IR、Depth 全图和 Query 给初判框，再决定直接结束或发出两个连续坐标区域：目标疑似区和竞争对象／参照区。环境从三张原始图按 RGB 全图坐标同步取两个高细节区域，第二次模型调用保留原全图上下文，并输出唯一的全图归一化框。观察框只是取证动作，绝不充作最终提交框。策略通过短轨迹 SFT 冷启动，再以最终 ACC@0.5、连续 IoU、实例身份和少量动作成本训练；用替换局部图的反事实测试确认它确实读取新证据。

它与退役的 EGM、Locate、旧全量多轮 Agent 有实质区别：不调用外部定位教师／检测器作推理依赖，不做树搜索或不限轮数的工具对话，不把“候选已给出”当成搜索完成；仅保留一个可学习的取证决策和最多一次额外观察，最终框可跳出两处观察区域。初判、观察位置、继续／结束和最终框共用同一可训练策略，主指标直接约束全链。

赛题官方材料 [PDF](../../../../contest/基于大模型的多模态视觉理解与推理.pdf) 第 2–4、6–8 页要求 RGB、热红外、深度和英文 Query 输入，输出 RGB 坐标系下 `[0,1]` 的单一 bbox，指标是 IoU≥0.5 的 ACC@0.5；测试集不得用于训练，禁止商业闭源大模型在线推理 API。本文训练方案只用合法训练来源，不用 City412／官方测试图及其人工修订做训练，也不依赖 GPT-4o 轨迹蒸馏。

## 一手文献与源码：已发表结果和边界

| 方法 | 正文、附录中与本任务相关的实证 | 公开源码核查 | 对本赛的判断 |
|---|---|---|---|
| [DeepEyes](https://arxiv.org/html/2505.14362)（ICLR 2026） | Qwen2.5-VL-7B 的视觉问答主动裁剪；V* 从 71.2 到 90.1，HR-8K 从 65.3 到 72.6。其 RefCOCO/+/g 为 89.8/83.6/86.7，相对复现基座仅 +0.7/+1.0/+0.6，**不是单框定位的大幅证据**。无工具奖励 HR-8K 55.4，无条件工具奖励 71.8，有条件工具奖励 72.6。附录列 47k QA 来源、工具协议及失败案例。 | [`parallel_env.py`](https://github.com/Visual-Agent/DeepEyes/blob/main/verl/workers/agent/parallel_env.py) 串接“生成→执行→加入局部图→继续生成”，`action_mask` 排除观察 token；[`visual_toolbox_v2.py`](https://github.com/Visual-Agent/DeepEyes/blob/main/verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py) 从原图像素坐标裁剪；[`vl_agent.py`](https://github.com/Visual-Agent/DeepEyes/blob/main/verl/utils/reward_score/vl_agent.py) 计算 `0.8*acc+0.2*format+1.2*(acc且用工具)`；[`core_algos.py`](https://github.com/Visual-Agent/DeepEyes/blob/main/verl/trainer/ppo/core_algos.py) 按同题组标准化终局奖励。 | 证明视觉动作可端到端训练；其主要问答收益不可直接外推到本赛。源码的工具回报只看调用和答对，并未证明局部图内容起因。 |
| [Pixel-Reasoner](https://arxiv.org/html/2505.15966)（NeurIPS 2025） | 7,500 条 SFT 轨迹（5,500 视觉、2,000 纯文本），含错误观察后的纠正；再用每题 8 条 rollout 的 GRPO。正文表 1：V* 84.3、TallyQA-Complex 73.8；去掉 RL／好奇奖励／纠错轨迹均下降。附录 B/C 给出工具协议、单次/重复裁剪比例、奖励推导和超参 `H=.3,N=1`，附录 D/E 为失败模式和提示。 | [`experience_maker.py`](https://github.com/TIGER-AI-Lab/Pixel-Reasoner/blob/main/curiosity_driven_rl/openrlhf/trainer/ppo_utils/experience_maker.py) 执行裁剪、回填图片、建立 assistant/observation loss mask，并计算同题奖励与优势；[`sft_tooltrainer.py`](https://github.com/TIGER-AI-Lab/Pixel-Reasoner/blob/main/instruction_tuning/sft_tooltrainer.py) 只监督 assistant 片段。公开裁剪会额外加 0.1 归一化边距；公开好奇奖励实现还有按当前 `RaPR` 除法、按视频／图像分别缩放等细节，不能把论文公式视作源码逐字复刻。 | 冷启动须教会动作语法、真正依据局部图修正和直接结束；不能奖励“调用次数”代替单框收益。其附录自己记录了错误工具返回后模型假装看到了图的失败。 |
| [LFPC](https://arxiv.org/html/2603.27494)（CVPR 2026） | 对 DeepEyes 在 HR-8K Lite、16,384 全图视觉 token 下替换裁剪：原预测裁剪 68.0、噪声图 68.5、真值裁剪 69.5，说明形式调用与使用证据可分离。第一阶段给低细节全图、从原图高细节取局部；第二阶段用 256 条人工区域标注的覆盖率门控 IoU＋L1 奖励抑制过大裁剪。1024 token 下 HR-8K 72.1（其复现 DeepEyes 62.0）。附录 8–14 节给训练动态、反事实、消融、418.5 A100 小时开销和限制。 | [官方仓库](https://github.com/XuanPu-Z/LFPC) 在本次核查时**仅有 README 和 LICENSE**，README 明言代码待发布，不能称已审其 rollout／reward 实现。论文公式中的 `IoU_i` 分子分母仍写 `B1`，与文字“对所有嵌套框取最大值”不一致；具体实现未公开，不能猜。 | 本任务最重要的反例：需要测真实局部图替换为噪声后的最终框变化；信息差训练有价值，但全图仍须保留对象集合与参照关系。 |

DeepEyes 论文正文、附录使用本地 `active_rl/deepeyes.pdf`、`deepeyes.txt`（25 页），Pixel-Reasoner 为 `pixel.pdf`、`pixel.txt`（30 页），LFPC 为 `lfpc_arxiv.pdf`、`lfpc_arxiv.txt`（17 页，含补充材料）；均位于 `F:/AIC/tmp/architecture_rethink_20261025/active_rl/`。论文来源为上述作者／CVF 官方页面。源码浅克隆同目录，修订号 DeepEyes `11d20c6`、Pixel-Reasoner `a3a6c3a`、LFPC `e51808a`。DeepEyes 核查了 README 指向的实际 `visual_toolbox_v2`、`parallel_env`、奖励分发 `reward_score/__init__.py`、GRPO 优势和训练脚本；`vl_agent_v2` 是另一环境且其 `get_bbox_2d` 存在无条件 `return None`，不把它误认成 README 指定的 DeepEyes 主工具。Pixel-Reasoner 核查了主 rollout、动作执行、奖励、优势、观察 mask、SFT 监督与训练脚本。没有审计两仓所有通用 veRL/OpenRLHF 文件；LFPC 没有可读训练源码。

**实际阅读范围。** DeepEyes：PDF 正文第 1–10 页的状态、奖励、数据筛选、表 1–9；附录第 15–25 页的工具协议、训练数据、消融、成功／失败案例及限制；第 11–14 页为参考文献。实际源码为 `README.md`、`verl/workers/agent/parallel_env.py`、`verl/workers/agent/envs/mm_process_engine/visual_toolbox_v2.py`、`verl/workers/agent/envs/mm_process_engine/prompt.py`、`verl/utils/reward_score/vl_agent.py`、`verl/utils/reward_score/__init__.py`、`verl/trainer/ppo/core_algos.py`、`verl/trainer/ppo/ray_trainer.py`、`examples/agent/final_merged_v1v8_thinklite.sh`、`eval/eval_vstar.py`、`eval/eval_hrbench.py`、`eval/judge_result.py`；额外排除了不适用的 `vl_agent_v2.py`。Pixel-Reasoner：PDF 正文第 1–10 页、附录技术内容第 15–19 页和动作提示第 20–23 页；第 11–14 页为参考文献，第 24–30 页为会议检查表。源码为 `instruction_tuning/sft_tooltrainer.py`、`curiosity_driven_rl/openrlhf/trainer/ppo_utils/experience_maker.py`、`curiosity_driven_rl/openrlhf/trainer/evaluator.py`、`curiosity_driven_rl/openrlhf/trainer/ray/evaluator2.py`、`curiosity_driven_rl/scripts/train_vlm_single.sh`、`curiosity_driven_rl/scripts/eval_vlm_new.sh`。LFPC：arXiv PDF 全 17 页，其中正文第 1–10 页、补充材料第 12–17 页，第 9–11 页包含参考文献；仓库实际只有两文件，缺少可读源码。

**图表复核。** 使用 `pdftoppm` 渲染并用 `view_image` 核对 Pixel-Reasoner 第 4／5／7 页的轨迹图、好奇约束和消融表，DeepEyes 第 5／6／8 页的奖励公式、定位表、工具奖励消融，LFPC 第 4／5／8／14 页的反事实表、信息差架构图、奖励目标与追加实验。渲染图均在上述临时材料目录，文件名如 `lfpc_arxiv-p5.png`。这些图表数字已按可见原图核对。

## 可训练结构：对象竞争驱动的一次观察

**状态。** 原 RGB 像素尺寸 `(W,H)` 是唯一全局坐标系，所有内部框均用 `[0,1]` 表示；输入为 Query、三张全景图的较低但足以识别对象集合的视图，原分辨率三图由环境持有。深度进入模型时按项目已有正确视觉表示；原始 16 位深度及有效像素掩码供环境裁剪，不把深度零值当成近物体。全图不在第二步删除：模型需继续判断“左数第三个”“桥旁那人”这类全局关系。

**动作。** 第一次生成固定短格式 `b0=[...] ; decision=FINAL|LOOK ; roi_target=[...] ; roi_context=[...]`。`FINAL` 直接交 `b0`。`LOOK` 最多一次，允许 `roi_context=null`；它表示竞争实例或关系参照对象，**不是预设检测候选池**。动作区域由模型自主回归，可与 `b0` 不同。环境按相同归一化区域从原 RGB、IR、Depth 同步切取，给目标和参照各留确定的少量周边；裁剪不能只在缩小后的模型输入上做。将两个 ROI 在每个模态中作等比例留白拼版，形成最多三张局部观察图，并附 `roi_target/roi_context` 的全图绝对归一化坐标、模态名与有效深度标记。这样一组观察可以比较同类对象、看可见部件是否属于同一实体，也能把 IR/Depth 证据绑定到所选区域。

**第二步。** 同一 27B 保有第一次的全图上下文和动作记录，读取三模态局部图后仅生成 `bbox=[...]`，它仍是全图 RGB 坐标。推理入口把它写入官方原 JSON 的 `bbox`，并使用现有单框合法性检查。即使两个 ROI 均选错，模型也允许在最终框中跳出 ROI；训练指标会惩罚最终错误。动作框、初判框和最终框需分别记录，防止局部坐标被误解释为全局坐标。

**架构创新是可学习的信息分配与对照观察**：固定一次观察上限保留了干净的优化问题，双区域把目标与竞争／参照物并列展示，跨三模态同坐标取证；动作回归和最终框回归均由同一策略受赛题奖励约束。不会在部署时等待额外专家给候选。比单次三图直接框多出的能力是取回被全图缩放压低的局部细节，并在一个视野里比较“哪个实例、哪个可见部件”。这还只是机制假设，没有本项目成绩。

## 真实训练数据、冷启动与 RL 闭环

1. **真实监督来源。** 优先用仓内合法的 RGBDT500 训练图组（已有准备子集 992 行及本轮独立审核的 79 个训练序列）、原生多模态训练来源和带真实实例／引用框的公开数据。完整三模态 RL 只用确有三图的训练样本；RGB+IR、RGB+Depth、RGB-only 可用于单独预热动作格式，但缺的模态不伪造。需要从同图多 Query、真实实例框／可靠对象标注形成“目标—竞争实例”对；只给单框且无可信竞争对象时训练最终定位和目标观察，不计算身份对照项。与 City、官方测试图逐帧／序列排重，评估图组不回灌训练。引用公共集时记录数据来源和预处理，符合赛题 PDF 第 7–8 页。
2. **SFT 冷启动。** 从现有 Arm2 27B 作为共同起点：保留直接框样本；对同图可靠目标／竞争实例，构造短 `b0→LOOK(目标疑似区,竞争／参照区)→最终GT框` 轨迹。`b0` 可取冻结 27B 在训练图的交叉拟合预测；动作区域取真实目标／竞争位置附近带随机偏移的观察窗，而非把 GT 最终框原样作为动作输出。覆盖“初判错实例→观察后纠正”“初判漏部件→观察后扩完整”“初判正确→直接结束”三种行为。只监督模型自己生成的字段，局部图和环境提示 token 不计 loss；最终答案仍是短框，禁止长篇推理成为主要学习目标。若项目 27B 的动作语法可靠，也可以把 SFT 压到极短，但第一轮保留它来避免 Pixel-Reasoner 所见的无效操作循环。
3. **在线 RL。** 每个训练 Query 采样 4–8 条完整轨迹：初判、是否 LOOK、ROI、环境执行、继续生成最终框。奖励同时关心身份和几何，工程初值可取 `R = 1[IoU(final,GT)≥.5] + 0.25·IoU(final,GT) + 0.05·(1−L1_norm(final,GT)) + 0.15·identity_margin − c·1[LOOK]`；其中 `identity_margin = max(0, IoU(final,GT)−max_j IoU(final,distractor_j))`，**仅在有真实竞争实例框的样本启用并按是否存在该标注分层统计**。`L1_norm` 用图像归一化四角距离，只作零重叠早期的微弱方向信号。为避免奖励缩小到目标一部分，主项永远是最终 GT 全框的阈值和 IoU；ROI 本身不拿高 IoU 代替最终框。`c` 按算力预算定成小的正值，并对直接答案保留无工具成功样本。取同 Query 组内相对优势优化所有 assistant 动作和答案 token；局部图与环境文本只进入状态，梯度 mask 为 0。格式错误记零主奖励并单列，而不是静默改框。
4. **信息差和真实利用。** 训练全图视图可适度降细节，局部图始终从原图提取；下采样不能让全局关系和目标集合不可辨。训练后在同一条已定动作轨迹上把局部图分别替换为噪声、错位置同面积裁剪和真实局部图，比较最终 ACC 与实例选择翻转。若真实／噪声无差异，就按 LFPC 的证据判为“尚未学会使用观察”，即使工具调用率很高也不算新能力。可在第二轮加入同动作前缀的真／错观察配对继续训练，但第一轮先量出缺口。

## 首轮可执行实验与裁决

| 臂 | 参数起点和数据 | 推理预算 | 回答的问题 |
|---|---|---|---|
| A：现役 Arm2 单次框 | 冻结归档权重 | 原生三图一次 | 参照成绩 |
| B：等预算直接再看全图 | 与 C 相同初始化及训练图组；不提供局部高细节图 | 两次 27B 调用，第二次重复全景 | 提升是否只是多一次生成／更多训练 |
| C：本文策略 SFT → RL | 同初始化，合法训练图组；一组可学习的至多一次观察 | LOOK 时两次调用、最多三张拼版局部图 | 新证据是否带来净增益 |

先在合法训练来源按图组／序列留出的验证集做冻结 27B 的“GT 附近高细节 ROI 是否能纠正最终框”信息价值试验，再建动作 SFT；此试验的 GT 只用于分析或训练模拟，不进入评估推理。C 的 RL 首轮用真实训练框与可核实竞争实例，不依赖大量手写视觉推理。主评估是 City400 已审口径与保留的历史外部 94 条诊断，以及按场景／序列隔离的训练源验证组；City 是已使用开发集，不把它当独立官方分数。记录直接结束率、LOOK 率、实际局部图替换后的 ACC 变化、ROI 对目标／参照覆盖率、最终框 ACC@0.5 和 IoU、错实例／漏部件救回与损害、每题时延和视觉 token。主比较 C 对 A 的真实单次提交框指标；C 对 B 判断图像内容贡献；噪声／错裁剪干预判断模型是否真正使用工具。

推理成本上限清楚：`FINAL` 与现役近似一轮；`LOOK` 多一轮 27B 生成和最多三个双 ROI 拼版图的视觉编码，期望轮数是 `1+p_LOOK`，期望新增图像数是 `3·p_LOOK`，实际 token 和时延由首轮记录。训练则是 4–8 倍 rollout 加入图像环境处理，需要云端 27B 训练资源；本机 8GB 仅能做 CPU 数据、坐标与提示检查，不能据此排除路线。代码应在当前 `qwen3_5` 处理器和现役 vLLM／训练栈上实现最窄的二步状态机；DeepEyes 的 Qwen2.5-VL veRL、Pixel-Reasoner 的 OpenRLHF 均不可直接搬进 Qwen3.6-27B。

**首轮成败判据：** 若 C 的最终 ACC 无净收益，或真实裁剪替换成噪声后几乎不变，则放弃“主动观察已增强定位”的解释；若主要增益发生于小目标、遮挡、同类竞争和有效 IR／Depth 专项，且不牺牲原本正确的普通题，继续扩大可靠多模态训练源与优化 ROI。该裁决针对实际最终框，不以动作准确率、GT 最佳候选覆盖或高调用率代替。
