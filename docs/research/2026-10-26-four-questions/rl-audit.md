# 强化学习路线：做过什么，下一步怎样真正检验

2026-10-26。此文核对仓内代码、历史记录和上游接口；本轮未调用 GPU。赛题以[官方 PDF](../../../../contest/基于大模型的多模态视觉理解与推理.pdf)第 4、6–8 页及[本地官网规则](../../../../contest/官网赛题规则_20261022.html)为准：每条 Query 交 RGB 坐标系的归一化单框，IoU **≥0.5** 算命中，主指标 ACC@0.5 以**所有 Query** 为分母，无效框计失败；官方测试集不能用于训练。

## 对“为什么没有做”的可核对回答

**做过研究和接口准备，没有做过本项目的在线强化学习实验。** 9 月的[历史路线记录](../../audit-20260927/research/2026-09-19-structural-optimization-roadmap.md)已读定位奖励论文，优先写为候选排序监督、随后才考虑 GRPO（组相对策略优化）；[10 月 22 日研究](../2026-10-22-semifinal/metric-learning.md)比较 VLM-R1、MedGround-R1、Visual-RFT 的奖励和迁移边界，并提出阈值加弱 IoU；[10 月 23 日接入设计](../2026-10-22-semifinal/swift-integration.md)明确当时尚未实现插件或跑 GPU；[10 月 24 日计划](../2026-10-24-next-plan/review-supervision.md)仍把短框 GRPO 放在第一阶段模型能读取证据后的分支。[HANDOFF](../../../HANDOFF.md)的 10 月 23–24 日记录明确脚本存在不能推断训练，也记载 GRPO 当时仍未实施。[10 月 25 日结构研究](../2026-10-25-structural-search/history-and-current.md)再次明确新数据/训练脚本不等于已训练。

仓内现在有[数据转换器](../../../tools/prepare_grounding_rl.py)、[奖励函数](../../../tools/grounding_reward.py)、[ms-swift 插件](../../../tools/swift_grounding_reward.py)、[训练脚本](../../../scripts/train_grounding_rl.sh)、[离线测试](../../../tests/test_grounding_rl.py)。转换器剥离旧 assistant 答案，单列原精度 `solution`；奖励严格解析单个 `bbox_2d` JSON，非法框 -0.1，主臂 `1[IoU≥.5]+0.2 min(IoU,.5)`，另有纯 IoU/纯二值对照。现有离线测试覆盖阈值、非法输出、标签精度和数据转换；本轮没有重跑。**离线测试不证明三图 rollout（采样生成）、真实反向传播、检查点或 ACC 净收益**。仓内未找到本项目 GRPO 训练日志、RL 检查点或配对评价结果。因此应明确承认路线停在准备阶段，不把它表述成已试验失败。

历史文档能证明这条路线被考虑过且曾被有条件地排后；它们不能证明当时每次没有开跑的个人心理原因。现在回看，**缺少一次小规模真实三图冒烟和有效奖励诊断，是执行缺口**。算力、组内奖励全同和错实例风险都属于需要测量的实验问题，不能当成永久不试的理由。

## 现有入口离“可跑通并可比较”还差什么

1. `train_grounding_rl.sh` 使用 `swift rlhf --rlhf_type grpo`、SFT `--adapters` 与 `--ref_adapters`，后者符合[ms-swift v4.5.3 官方参数说明](https://github.com/modelscope/ms-swift/blob/v4.5.3/docs/source_en/Instruction/Command-line-parameters.md)。[官方奖励插件文档](https://github.com/modelscope/ms-swift/blob/v4.5.3/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)也支持从独立 `solution` 列向 ORM 传值。脚本有静态字段预检，但没有在目标机上证明 Qwen3.6-27B（内部 `qwen3_5`）三图生成、图像网格、参考适配器、反向传播和保存/重载全链成功。Qwen3.5 多模态 GRPO 曾有[图像 token/加载故障报告](https://github.com/modelscope/ms-swift/issues/9287)，上游后来关闭该问题；它是冒烟必须核查真实权重加载和有效输出的依据，不是认定现环境一定坏。
2. 现 RL 默认 `MAX_PIXELS=602112`；现役[27B SFT 脚本](../../../scripts/train_qwen36_27b_swift.sh)默认 1204224。分辨率改变会影响极小目标；先以相同输入预算做机制对照，若显存迫使降档，就让基线与 RL 都在降档下重评，另报与现役高分辨率模型的差距。
3. 默认 `dynamic_sample=true` 会按组奖励方差替换全同组；即使 `max_resample_times=1`，采样账与原定固定样本也不再同义。[v4.5.3 训练器源码](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/rlhf_trainers/grpo_trainer.py)明确该逻辑。第一轮诊断应改为 `false`，保留全失败/全成功比例和原样本分母。`overlong_filter`、`truncation_strategy=delete` 的丢弃数也需记录。
4. 目标原始浮点框优先；仅从 SFT 量化答案反推的目标须显式允许并分层报告。`solution` 不可混入模型可见 `messages`。按图像组/场景划分训练与验证，任何官方测试图或其标注都不能进 RL。现奖励衡量最终框与 GT，不能保证模型选对完整主体；每次评价应拆出错实例、漏部件、空框、仅磨框等翻转。

## 最小完整实验：短答案单框 GRPO

当前脚本仍强制要求 `SFT_ADAPTER`，同时写入 `--adapters/--ref_adapters` 并检查文件。因此下述“公开基座直接起步”是本轮修订的实验路径，实施时须先把适配器参数改为可选，让未挂适配器的公开基座成为共同策略起点与参考；本轮没有修改脚本。现有 `ALGORITHM=dapo` 使用 GRPO 入口下的 DAPO 式损失和裁剪，也应明确记录，不能把所有变体混写成同一个原始 GRPO 设置。初次诊断要同步关闭当前硬编码的动态采样并保留丢弃计数。

**假设。** 从合法清洁的同一起点采样，同一 Query 能产生不同质量的单框；对过线框给主奖励、对不过线框给弱连续几何信号，可提高单次最终 ACC@0.5。若只升平均 IoU 或只改善 JSON 解析，而独立场景 ACC 不升，该假设不成立。奖励要以最终完整 GT 框为准；纯二值组可能全同，纯 IoU 可能把已过线框继续磨紧。已有 [VLM-R1](https://arxiv.org/abs/2504.07615) 的单图连续 IoU GRPO、[Rex-Omni](https://arxiv.org/abs/2510.12798) 的大规模 SFT→GRPO、[EGM](https://arxiv.org/abs/2601.13633) 的视觉定位 SFT→RL 表明机制有先例，但模型、模态、数据量和指标都不能直接外推为本赛涨点。

**执行。** 优先审计是否已有训练清单经官方测试图组排重的清洁适配器，可将它作为所有训练臂的共享起点；若没有，不必先完整重训 SFT，可直接从公开 Qwen3.6-27B 基座做 100–200 条合法训练 Query 的采样诊断。若基座能输出合法短框且组内奖励有差异，就从同一基座独立做短 GRPO 与同数据 SFT 对照；仅当格式或动作基础不足时，才加有限的共享 SFT 热启动。[HANDOFF](../../../HANDOFF.md) 的污染审计记载旧 Arm 2 有 508 个训练行与官方测试帧重叠，它只能冻结为历史参考，不能直接作为新实验初始化。GRPO 不以新结构或辅助 SFT 先涨分为前提。训练样本保留原浮点 GT 与三图顺序，按场景留验证组。每题先采 G=4、`dynamic_sample=false`、短答案温度 0.8，仅生成不更新，保存原始框、IoU、每组奖励方差、全零/全同组率及无效框率；不能以确定性 City ACC 代替采样命中概率。接着以同一小数据做 **2 步真实反传冒烟**，检查图像、权重初始化、非零 LoRA 梯度、KL、输出与保存后重载；再运行固定预算短 GRPO 与同起点、同图像的 SFT。两臂同时报告相同步数和相同 GPU 小时的比较，避免把多次 rollout 的成本藏在步数里。首轮主奖励用现有阈值+弱 IoU，纯二值和纯 IoU 只在主臂有可解读结果后做同预算奖励消融。记录 27B 每步峰值显存、秒数与总 GPU 小时后再决定是否从小数据扩大；本地 8GB 和已记录约 46GB L40 无法验证 BF16 27B 训练，单 80/96GB 也必须由三图真实冒烟确定能否容纳激活及参考策略。

**判据。** 保持同一推理协议、同一图像预算、单次直接框，按独立图组报告 ACC@0.5 及配对救回/损害、平均 IoU、无效率、身份/范围错误、耗时；另列保留的旧开发集诊断，但不将已反复分析的 City 当独立测试。对比冻结起点（零新增训练成本）与等预算同起点 SFT；若 RL 的净过线不增、有效组太少或误选主体更多，应定位奖励/数据问题并停止扩大。阈值奖励是可验证信号，不会凭空教会模型看见未编码的目标。

## 与量化的接口边界

当前 RL 脚本是 BF16，`use_vllm=false`，未启 `quant_method` 或 `quant_bits`。[ms-swift v4.5.3 参数](https://github.com/modelscope/ms-swift/blob/v4.5.3/swift/arguments/base_args/quant_args.py)确有 `bnb`/`fp8` 与 4 位设置，且[官方 GRPO QLoRA 例程](https://github.com/modelscope/ms-swift/blob/v4.5.3/examples/train/grpo/internal/qlora.sh)确实用 `--quant_method bnb --quant_bits 4`，但例程是 Qwen2.5-3B 文字数学、多卡 vLLM，**不能证明 Qwen3.6-27B 三图、已有 SFT LoRA 或官方 FP8 基座可直接执行**。官方 Qwen3.6-27B-FP8 是另一权重，不能把旧 BF16 底座上训练的 LoRA 未经兼容性检查直接视作同一策略。量化可省底座内存，但 rollout 采样分布和训练 logprob 的精度/权重同步必须核对；[FP8-RL 论文](https://arxiv.org/abs/2601.18150)讨论重要性校正来缓解这种差异，不能据论文推定 ms-swift 本例已经自动校正。先建立 BF16 小步正确性，量化作为独立算力臂再比同样单次 ACC、KL、显存与速度。
