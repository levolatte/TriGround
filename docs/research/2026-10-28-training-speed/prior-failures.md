# 既往训练加速试验复核

2026-10-28。依据本地源码、2026-10-26 固定回答试验和本轮留存的运行快照。本文未运行新的 GPU 试验；数值根因未被隔离的地方明确保留为待验证。

## 结论

之前没有采用加速配置，不等于已经证明“训练无法大幅加速”。当时主要尝试了调度、同步、局部内核和用显存换重计算，**尚未消除 GRPO（组相对策略优化）中按回答反复计算相同图像、长提示和旧策略的结构性工作**。最终选择回到原配置的直接原因是短测预算内没有候选同时通过数值、显存和完整训练链路门槛，见[机器决议](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/selection.json:1)。

本次复核得到四项判断：

- **确定：原非确定性基线自身就不满足当时的数值门槛。** 因而不能用聚合审计的一次失败，证明它改变了训练目标。
- **确定：微批2和卷积组合改变了完整模型的前向数值。** 但测试将候选的原生旧策略／参考概率覆盖为基线概率，尚缺候选原生链路对照，不能据此直接归咎于梯度实现错误。
- **确定：GC2 的那个具体配置耗尽显存。** 它是关闭一半层的检查点，不能推广为所有减少重计算方案都不可行。
- **优先方向：删除满足条件时多余的旧策略前向，再共享冻结视觉特征和同题长前缀。** 这些方案直接减少计算次数；聚合梯度审计只能作为附带整理。

## 1. 实际测过什么

固定基准只用4道真实三图题、每题8个既有回答，共32答，从共同 checkpoint-50 和新优化器执行一次更新。它替换生成结果，仍执行编码、原生旧策略／参考前向、损失、反传和优化器；见[基准定义](F:/AIC/code/tools/benchmark_quant_grounding.py:57)、[固定回答注入](F:/AIC/code/tools/benchmark_quant_grounding.py:261)。所有固定试验的 `generation_seconds` 都为0，例如[原基线资源记录](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/baseline/resource_steps.jsonl:1)。因此下表不是在线生成训练的完整吞吐。

| 配置 | 固定更新秒数，剔除梯度快照复制 | 梯度相对 L2 差异／余弦 | 参数更新相对 L2 差异 | 能支持的结论 |
|---|---:|---:|---:|---|
| 原基线，微批1 | 370.549 | 基准 | 基准 | 单次参考 |
| 同配置重复 | 368.155 | 0.014763／0.999891 | 0.072587 | 原链路存在运行间波动 |
| 聚合审计 | 359.869 | 0.014527／0.999898 | 0.069676 | 单次快2.88%；数值差异没有超过基线自差异量级 |
| 微批2／累积16，已补偿损失分母 | 376.147 | 0.631978／0.775352 | 0.807229 | 当前组合单次未加速，且前向／梯度明显改变 |
| 确定性基线 | 402.094 | 基准 | 基准 | 确定性有约8.51%的固定步成本 |
| 确定性基线重复 | 402.405 | 0／1 | 0 | 此输入与实现上可重复 |
| 确定性＋聚合审计＋卷积内核 | 383.661 | 0.749410／0.709562 | 0.738270 | 比确定性基线快4.58%，比原基线慢3.54%；组合未通过 |
| 确定性 GC2 | 未完成 | 未取得 | 未取得 | 此配置显存不足 |

原始数据：[原比较记录](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/comparison.json:69)、[确定性比较记录](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/comparison.json:69)。当时门槛为梯度和更新余弦均至少0.999、相对 L2 均不超过1%，[判定实现](F:/AIC/code/tools/benchmark_quant_grounding.py:454)。表中时间为耗时减少比例，不能与 `baseline_seconds / candidate_seconds` 的速度倍率混写。

## 2. 为什么同配置也“不等价”

### 已证实的事实

原基线与重复运行的32个微批损失逐项相同，原生旧策略和参考 log probability（对数概率，下称 logp）最大差都为0，但累计梯度仍有1.476%的相对 L2 差异，首次参数更新有7.259%的差异；[重复运行结果](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/baseline_repeat/result.json:42)、[梯度与更新比较](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/comparison.json:137)。这至少说明，只检查标量损失和旧／参考前向不足以解释全部差异；当前策略前向、反传、累积和更新链路仍需分别定位，不能仅凭相同标量损失锁定反传算子。

启用 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、PyTorch 确定性算法及 cuDNN 确定性之后，重复运行的梯度和更新都完全相同；[设置实现](F:/AIC/code/tools/train_quant_grounding.py:197)、[重复比较](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/comparison.json:137)。这证明该组设置在此单步上足以消除已观察波动，**没有隔离到底是哪一个设置、算子或反传内核造成波动**。

### 不能越过的推断边界

不能把根因直接写成 cuDNN 卷积、SDPA（缩放点积注意力）或 FLA（快速线性注意力）中的某一个。原配置使用 SDPA，同时使用 FLA 和 PyTorch 因果卷积，见[实际配置](F:/AIC/code/tools/quant_grounding_config.py:149)、[内核快照](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/a-runtime-audit.json:7)。PyTorch 官方说明，固定种子、关闭 cuDNN 自动择优和使用确定性算法解决的是不同层面的可重复性问题；其中关闭自动择优本身并不保证算子确定性。[官方说明](https://docs.pytorch.org/docs/2.12/notes/randomness.html#cuda-convolution-determinism)。

聚合审计只读取梯度来计算有限性与非零标志，没有在实现中修改梯度；见[实现](F:/AIC/code/tools/train_quant_grounding.py:210)。原基线自差异与聚合审计差异相近，故“审计改法破坏梯度”证据不足。正确状态应为**未完成确定性审计单项复测**，而不是数学目标已被证伪。确定性比较文件明确将 `audit` 列为未运行，见[缺失项](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/comparison.json:242)。

首次优化器更新比梯度更敏感也是合理候选解释：接近零的梯度分量对符号及自适应归一化敏感，但目前没有逐参数符号翻转、优化器状态和小梯度分位数证据，不能据7%的更新差异另行断言优化器错误。

## 3. 微批2为什么没有通过

### 已排除的简单解释：未补偿的两倍缩放

Swift 的 DAPO（动态采样策略优化）损失按该次生成批的完成词元总数归一化，[实际源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:1140)。微批从1改2，梯度累积从32改16，若沿用该分母，更新会放大两倍。因此训练入口把分母乘以微批大小，见[补偿实现](F:/AIC/code/tools/train_quant_grounding.py:234)。实际微批2审计记录了 `factor: 2` 和32答／更新，见[运行审计](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/mb2/runtime_audit.json:51)。

所以现有0.632的梯度偏差不能简单归因于“忘了除2”。FP32（32位浮点）的小模型与训练步骤测试只说明归一化代数成立，不能证明真实多模态拼批、BF16（16位脑浮点）和 NF4（4位量化）实现一致。

### 明确的定位线索：偏差已经出现在无梯度前向

微批2原生旧策略和参考 logp 相对微批1的最大绝对差均为0.247890；[原始结果](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/mb2/result.json:38)。该差异在反传和优化器运行前就已存在，所以只查梯度累积没有抓住第一处分歧。

当前测试校对回答词元、题目身份、有效词元数、奖励、优势和截断状态，[校对代码](F:/AIC/code/tools/benchmark_quant_grounding.py:132)，**没有保存并逐项对齐完整提示的 `input_ids`、`attention_mask`、位置编码、图像网格、图像张量及 DAPO 分母**。因此不能排除拼批／填充引入的语义差异；也不能认定一定是它们出错。

候选解释还包括批大小改变后的低精度矩阵运算、注意力或线性注意力计算路径。PyTorch 明确不保证同一运算在整批与逐样本调用下逐位一致，[数值精度说明](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html#batched-computations-or-slice-computations)。这解释了为什么差异“可能正常”，**没有证明本例0.247890的幅度可接受，更不能证明没有实现错误**。现有微批2没有确定性复测，无法将其系统偏差与原运行噪声拆开。

### 重要测试边界：原生目标被基线覆盖

候选先运行自身旧／参考前向，然后 `_replay_targets` 把这些 logp 覆盖成基线张量，见[覆盖代码](F:/AIC/code/tools/benchmark_quant_grounding.py:153)、[调用位置](F:/AIC/code/tools/benchmark_quant_grounding.py:307)。于是测试比较的是：

`候选当前 logp − 基线旧 logp`，而候选原生训练本应使用 `候选当前 logp − 候选旧 logp`。

对于当时的同步、单轮使用、更新前策略，前向实现发生变化时，覆盖会把实现差异额外注入重要性比率和 KL（相对熵约束）。Swift 后续恰好计算该差，[比率计算](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:1022)。单个0.247890的有符号差可对应约0.780或1.281的概率比率；只记录最大绝对差，无法确定有多少词元实际越过裁剪边界。

**固定旧／参考目标有价值：它检验候选是否维持原数值计算。缺陷在于仅有这一项，却没有候选使用自身目标的原生链路对照。** 因此现在能裁定“严格固定目标测试未通过”，还不能裁定“候选原生更新必然有同等偏差”。后者仍需实测，不能据此放宽门槛直接采用。

### 速度同样没有显示收益

微批2单次固定步慢1.51%，其当前策略前反向从212.167秒升至224.790秒；旧策略／参考前向基本没变，分别约65.9／64.8秒与65.6／64.5秒。见[微批1分段](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/baseline/resource_steps.jsonl:1)、[微批2分段](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/mb2/resource_steps.jsonl:1)。它减少的是串行调用次数，未减少总回答、长提示计算或量化底座反传工作量，因此“微批翻倍必然明显加速”的假设不成立。

## 4. causal-conv1d（因果卷积快速内核）为什么没有通过

小内核试验确实调用到 `causal_conv1d_interface`，BF16 前向相对 L2 为0.002893，输入／权重梯度为0.002740／0.002416，解码更新为0.001700，余弦均超过0.99999；[小内核记录](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/causal_conv1d_check.json:1)。它证明接口、该输入上的前反向和状态更新基本一致。官方实现定义的目标确实是带截断的逐通道卷积，[官方仓库](https://github.com/Dao-AILab/causal-conv1d)。

完整模型候选的原生旧／参考 logp 最大差则达到0.241962，梯度相对 L2 为0.749410；[完整结果](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/audit_conv/result.json:42)。小内核误差经过多层传播可能放大，但目前没有真实层输入的逐层误差记录，不能认定这是唯一原因。

当时完整候选同时启用聚合审计／事件计时、检查点重新启用包装、独立环境与新卷积实现；[变体设置](F:/AIC/code/tools/benchmark_quant_grounding.py:220)、[运行审计](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/audit_conv/runtime_audit.json:1)。缺少确定性 `audit-only` 以及同一环境里只切换卷积函数的单项对照。再加上前述旧／参考目标覆盖，**不能把完整模型全部偏差归因于 CUDA 卷积实现错误**。

确定性设置不意味着第三方自定义算子自动具备相同的确定性保证；完整卷积候选也没有自身重复运行。因此它的可重复性仍未知。现有数据只支持“这个组合未通过，收益较小”，不支持“卷积内核已被证明不可用”，也不支持“小内核通过所以可以上线”。

## 5. GC2 为什么耗尽显存

训练入口把语言层的检查点开关设为 `index % every_n_layers == 0`，[实现](F:/AIC/code/tools/train_quant_grounding.py:160)。`every_n_layers=2` 是保留0、2、4等层的检查点，并让1、3、5等32层保留正常反传所需激活；**它不是把每两层包装成一个检查点段**。

固定回答运行以退出码1结束，每秒显存采样最高48,469 MiB、整卡49,140 MiB；[采样记录](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/deterministic/gc2_oom_measurement.json:1)。既有总结记录异常时 PyTorch 已分配45.74 GiB，仅余39.88 MiB，84 MiB申请失败；[异常摘要](F:/AIC/code/docs/research/2026-10-26-quant-rl/OPTIMIZATION.md:83)。留存采样本身没有异常调用栈，因此本次无法独立定位是哪个层／算子的分配失败，也不能把每秒采样峰值当作异常瞬间峰值。

梯度检查点本来就是通过反传重算减少激活存储，[PyTorch 说明](https://docs.pytorch.org/docs/2.12/checkpoint.html)。关闭一半层会显著增加长序列激活，但不能从权重约20 GB、原峰值约27 GB推算有20 GB可以任意分配；还要容纳优化器、临时工作区、量化状态和缓存。此次结果排除了**此输入、此精度、此逐层掩码**的配置，不排除只释放少数层，或通过选择性检查点保留某些昂贵算子输出。后者仍须按真实算子与显存测量；PyTorch 提供按算子控制保存／重算的接口，[选择性检查点](https://docs.pytorch.org/docs/2.12/checkpoint.html#torch.utils.checkpoint.create_selective_checkpoint_contexts)。

## 6. 旧试验还缺少哪些裁定条件

| 未决项目 | 现有不足 | 最小补证 |
|---|---|---|
| 聚合审计是否等价 | 非确定性基线本身失败；没有确定性单项候选 | 确定性基线与仅改审计各一次，再各重复一次；不要捆绑卷积 |
| 微批2第一处分歧 | 只有最终 logp 最大差，无完整模型输入对齐 | 一道题的两个固定回答，比较逐样本与拼批的有效提示词元、掩码、位置、网格、像素，再比较逐层输出 |
| 微批2原生训练是否等价 | 候选 old/ref 被基线覆盖 | 同一固定回答分别保留自身 old/ref、回放基线 old/ref，报告两条梯度结果与裁剪比例 |
| 卷积究竟改变什么 | 组合试验，小内核输入与真实激活不同 | 同环境同批形状只切卷积函数，输入取真实一层激活；对比前向及输入梯度，再追踪首个明显偏离层 |
| GC2 显存消耗位置 | 只有设备采样与异常摘要 | 相同输入一次短剖析，记录语言层边界和保存激活规模；不要再次盲跑半数层关闭 |
| 完整吞吐／生成／重载 | 资格失败后未执行 | 数值通过的单项候选再测包含生成的完整更新，另核对保存重载后行为 |

还有四个计量问题：

1. 每组只有一个冷启动固定更新，缺少相同输入的稳定期中位数。确定性两次稳定不等于候选速度已重复验证。
2. 候选专属的目标覆盖包含 `.item()`、CPU／GPU复制和写回，仍处于计时区内；基线没有相同工作，见[覆盖实现](F:/AIC/code/tools/benchmark_quant_grounding.py:153)。这会污染小幅速度差，但不能据此假定微批2实际更快。
3. 基线重复运行梯度快照复制耗时188.724秒，而首次仅4.784秒；最终固定步已剔除该段，见[重复快照时间](F:/AIC/code/docs/research/2026-10-26-quant-rl/evidence/optimization/original/baseline_repeat/result.json:51)。这段不能算训练耗时，也提示当时机器条件并非所有时段一致。
4. `numeric_pass` 实现只检查梯度和更新，虽然报告损失差异，却没有将损失或原生 logp 差异纳入判定，[实现](F:/AIC/code/tools/benchmark_quant_grounding.py:454)。本次没有因此误接纳候选，但未来报告应明确每个指标的判据，不能让单个布尔值代替原因分析。

## 7. 真正减少重复计算的最小方案

### 7.1 第一项：满足当前策略条件时省去旧策略独立前向

Swift 快照无条件计算每个微批的旧策略 logp，[源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:777)，但损失已支持 `old_per_token_logps=None` 时用当前 logp 的脱离梯度副本，[源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:992)。同文件还有描述何时需要旧策略的 `old_policy()`，却未用于上述计算点，[辅助条件](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:1825)。上游 TRL 对生成与优化边界对齐的场景已经采用相同省略思路，[官方源码](https://raw.githubusercontent.com/huggingface/trl/main/trl/trainer/grpo_trainer.py)。

当前参数为同步 HF 生成、`num_iterations=1`、G8、不用 vLLM；LoRA dropout（随机失活）为0，[配置](F:/AIC/code/tools/quant_grounding_config.py:156)。微批1的生成批8／累积32意味着每更新4题；应在实际对象上再确认 `steps_per_generation=8`、一批只消费一次、生成后到消费前没有优化器更新，以及底座所有有效 dropout 均为0。满足这些条件才把旧 logp 留空，**参考策略前向与 KL 保留**。改变 vLLM、异步生成、多轮使用或跨更新复用后必须重新审视条件。

最新留存A第60—80步的旧策略前向约52.6—65.7秒，与参考前向几乎相同；[逐步原始记录](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/a-resource-steps.jsonl:1)。在约403—495秒的常规更新中，这是约13%的直接可删工作；该比值是观测归因上限，不是实测速率承诺。比起梯度审计约0.06—0.08秒，这一项的量级更值得优先验证。

**可证伪最小测法：** 同一 checkpoint、同一模型实现、相同4题32答，确定性下只改变是否计算旧 logp；两侧均用自身原生参考 logp，不做跨实现覆盖。首先在不更新参数时记录原旧 logp 与当前 logp 的差、重要性比率和裁剪比例；再比较一次累计梯度／更新。若旧／当前并不对齐或梯度不满足门槛，假设即失败，先找策略版本／随机性／编码差异。通过后测3个完整更新的稳态墙钟和实际旧策略调用次数，预期32次降为0；记录生成词元数，避免把随机回答长度波动当成加速。

### 7.2 第二项：冻结视觉特征按图像复用

视觉编码器和对齐模块已冻结，[配置](F:/AIC/code/tools/quant_grounding_config.py:152)。Qwen 的 `get_image_features` 每次调用视觉网络并输出合并后的图像特征，[源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/modeling_qwen3_5.py:1397)；存在像素输入时，模型前向会再次调用它并填入图像占位词元，[源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/modeling_qwen3_5.py:1529)。G8的8个回答共享同一组图像，旧策略／参考／当前策略也用同一个冻结视觉网络，因此有明确重复计算机会。

最小实现是一个题组生命周期内只计算一次该组三图的冻结特征，在三条路径按原顺序复用；先不做全数据磁盘缓存，不引入管理框架。冻结权重并不自动证明训练与推理模式的输出相同，必须确认无随机增强、视觉有效 dropout 为0、预处理／网格一致，且保存的是包含冻结对齐部分的最终特征。不能把整个语言 `inputs_embeds` 脱离梯度，误删仍需训练的路径。

**可证伪最小测法：** 一组三图G8固定回答，比较原调用和复用时图像特征、有效词元 logp、LoRA 梯度与视觉调用次数。调用计数未减少，或复用引入额外拷贝后完整步没有明显减少，即否定速度假设。既有日志没有独立视觉耗时，当前不能为它报加速倍数。

### 7.3 更大的结构机会：同题长前缀只处理一次

同题8答只在短答案处分叉，而每个微批的旧策略、参考和当前策略都重新输入整个提示；训练 logp 路径明确设 `use_cache=False`，[源码](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/swift_grpo_trainer.py:1560)。因此微批2没有消除重复长前缀。

先从**无梯度的冻结参考策略**做题组内前缀共享最容易验证：提示处理一次，为8个答案分支复制完整状态，再只处理答案；不跨当前策略参数更新复用缓存。Qwen3.5是混合注意力模型，状态包含普通注意力缓存、线性注意力递归状态和因果卷积状态，不能只复制普通 KV（键值）缓存；[卷积与递归入口](F:/AIC/code/docs/research/2026-10-28-training-speed/evidence/modeling_qwen3_5.py:444)。三图位置编码和回答起始处的预测对齐同样须保留。

理想情况下，参考分布的长提示处理从每题8次降至1次；“该部分最多减少7/8的提示计算”只是计算量上界，解码、状态复制和串行开销会降低净收益。当前策略若也共享带梯度的前缀，需要把8个分支的梯度正确汇入同一前缀图，不能直接脱离梯度；这是独立改造，不应与第一项混在一次试验里。

**可证伪最小测法：** 固定一道题、两个不同长度答案，先仅对参考策略比较整段前向与前缀分叉的所有有效答案 logp，并核对分支间状态没有互相写入；再扩至G8，记录处理的提示词元量、状态复制耗时及峰值显存。数值不通过就先定位混合状态或边界词元，不扩展训练框架。通过后才接回完整更新衡量端到端收益。

## 8. 下一次试验顺序

先完成“省略旧策略前向”的单项数值与完整步计量，再按测到的视觉／前缀占比选择下一项。聚合审计可以补一个确定性单项对照，但不作为大幅加速主线。微批2和卷积若继续调查，先在单题／单层上查第一处分歧；不再用一轮27B更新同时判别拼批、内核、目标覆盖和计时改动。

保留两种用途明确的证据：**固定共同目标的严格数值比较**，以及**每个实现使用自身旧／参考目标的原生训练比较**。前者定位实现差异，后者判断真实更新是否改变。两者都需要，任何一方都不能代替完整训练的速度与任务效果。
