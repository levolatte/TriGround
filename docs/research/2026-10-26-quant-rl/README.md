# 27B NF4 与在线强化学习：实现与运行记录

2026-10-28 02:36，四份全量补评完成，932个新增答案均合法、无截断，标点后处理没有改变这些新增回答。以下统一采用标点规范化评分；mIoU以全部题目为分母，无效答案记0。

| 状态 | 新增训练呈现 | City命中 | City mIoU | 外部命中 | 外部 mIoU |
|---|---:|---:|---:|---:|---:|
| 共同warmup50 | 0 | 291/378 | 0.661608 | 85/88 | 0.867701 |
| A59 | 236 | 295/378 | 0.669715 | 85/88 | 0.867802 |
| B59 | 236 | 300/378 | 0.673123 | 85/88 | 0.867480 |
| C29 | 232 | 283/378 | 0.656196 | 85/88 | 0.882139 |
| C550 | 4400 | 303/378 | 0.682655 | 85/88 | 0.876148 |

- 相对共同初始化，A在City救回10题、退步6题，净增4题；B救回13题、退步4题，净增9题。外部A/B均救回1题、退步1题，命中持平。A外部mIoU变化+0.000101，B为−0.000221，不能称为所有指标均提高。
- A救回的10题中8题、B救回的13题中10题原IoU处于0.45—0.5，收益主要来自临界框修正。C550仍比B多命中3题，当前最高City成绩基线继续保留。
- 近似同曝光下，A/B的City命中分别比C29多12/17题；C29的外部mIoU高于A/B约0.0143/0.0147。C29与A/B相差4次呈现，只称近似同曝光，单种子结果不足以认定方法稳定胜出。
- C29→C550在City救回23题、退步3题，净增20题，mIoU提高0.026460；外部命中题目完全相同，mIoU下降0.005991。更长监督训练主要改善了City。
- 严格口径仅B外部有1题标点无效，命中84/88；统一后处理恢复至85/88。其余状态两种口径一致。完整五状态各自覆盖City378与外部88，逐题ID和真值一致。

见[五状态完整逐题比较](evidence/punctuation/five-state-paired-development.md)及[逐题明细](evidence/punctuation/five-state-paired-development.json)、[训练输入顺序证据](evidence/training-prefix-alignment-20261028.json)、[补评完成记录](evidence/fair-evaluation-completed-20261028.json)。四份补评含加载共1.078030 GPU小时；此时账本已结算39.094726小时，费用因单价未知保留为空，不再按72小时或阶段额度截断实验。

D已对齐语言混合精度并接入Swift原生三策略采样校正，完整128动作回放的概率差为0。1024长度对照32/32有效、23命中、无截断，但随后真实训练批次只有30/32有效、23命中、1截断；三组仍有定位奖励差异。04:55:06在优化器更新前按协议停止，实际更新0，没有新检查点、12题重载、正式D或D/D0结果。两条无效均在强制关闭思考后继续输出非规范内容，不能将一次长度对照通过当作可靠输出。见[原始轨迹诊断](evidence/d-policy-replay-20261028/diagnosis.json)与[D修复及停止原因](d-repair-run.md)。05:21累计占机41.837954 GPU小时；实例电源保持不变，下列运行状态保留为历史。

补评前状态见[2026-10-28 01:05汇总](status-20261028-0105.md)，该文保留当时缺少共同初始化对照及旧小时额度的历史，不代表当前进度。

2026-10-27 23:08，D修复版四题小测完成：31/32有效、无最终答案截断，合法框均命中且奖励饱和。未通过定位学习信号门槛，流程已停在训练前，实例保持开机。见[四题结果与诊断](d-repair-gate4-result.md)。

2026-10-27 22:57，按用户提供的新实例完成清盘与部署，D修复版四题G8小测已启动。当前不操作实例电源，最新状态与预算见[新机清盘与续跑](newmachine-redeploy-20261027.md)。

2026-10-27，D修复版已接入受控思考、单框格式与真实更新门槛。Agent擅自关机释放GPU，用户确认当前无空闲卡，4题小测尚未开始。已移除自动关机逻辑，后续未经明确指令不改变实例电源状态。最新流程见[D修复实现与限时试跑](d-repair-run.md)。

2026-10-27 13:27，按用户最新要求停止原D的无效长评价，保留City270条原文，暂停原队列与自动续行。CPU逐元素比较确认D第6步的992个LoRA张量与warmup50完全相同。截断、最终JSON与零优势根因已核实，修复方案改为分阶段生成、统一单框规范化及先验证学习信号；尚未启动修复版GPU训练。当前以[D根因与修复方案](d-rootcause-repair.md)为准，以下阶段记述保留为历史。

本阶段依据 [实施约定](PLAN.md)，从公开 Qwen3.6-27B 建立独立的四臂训练路径。A/B/C均已完成两套全量开发评价；D训练触发硬时限后，已从最后完整检查点恢复评价。优化短测没有合格加速组合，后续沿用原工程配置。C的匹配曝光评价和共同初始化完整基线仍待补齐，尚无完整四臂对照。

2026-10-27 11:18，D在3小时硬时限退出124：内层停止判断未预估下一步约21分钟耗时，导致第9步中断。已完成8步/32次呈现，完整权重只保存到第6步/24次呈现。11:24从checkpoint-6恢复D两套全量评价，之后顺序继续C匹配曝光；D不追加训练，失败与等待时间继续计账。停止判断已修复并同步云端，活动日志为`d_evaluation_recovery.log`。原条件下已完成256答全无效、230答截断，8步梯度及KL均为0；不把训练诊断当开发成绩。见[原因、恢复与证据](d-timeout-recovery.md)。

2026-10-27 08:18，C完成checkpoint-550的全部466题评价：**City303/378（80.16%），外部85/88（96.59%）**。原严格评分经CPU逐项复现；统一标点后处理没有修改任何回答，466答均合法、无截断。C在City比A净多8题（救回16、退步8），比B净多3题（救回12、退步9）；外部与A/B均为85题，但各有1题救回和1题退步。C训练4400次呈现，A/B各236次，当前分差不能单独归因于训练方法。见[完整记录](evidence/c-completed.json)、[逐题配对](evidence/punctuation/abc-final-paired-development.md)、[City后处理](evidence/punctuation/c-final-city378/summary.json)和[外部后处理](evidence/punctuation/c-final-external88/summary.json)。

C训练含加载与保存6.760202 GPU小时，两套评价0.547720小时，整臂含间隔7.307944小时。550步均记录有限非零LoRA梯度。D已于08:18自动接续；08:18快照仅一个活动作业，总占机估算29.2474/72小时、数据盘余13.04GiB。C第29步的232次呈现状态仍受保护，待D结束后全量评价；随后按既定顺序补共同warmup50基线。见[阶段账本](evidence/budget-20261027-c-completed.json)及[队列快照](evidence/queue-20261027-c-completed.json)。

2026-10-27 07:45，C训练按时结束并保存checkpoint-550：550次更新、4400次题目呈现，含加载和保存6.760202 GPU小时。用于匹配曝光的第29步状态（232次呈现）仍保留。随后顺序运行City378和外部88，完整成绩见上文。见[训练指标](evidence/c-final-training-metrics.jsonl)、[逐步资源](evidence/c-final-resource-steps.jsonl)、[最终训练状态](evidence/c-final-trainer-state.json)及[训练阶段账本](evidence/budget-20261027-c-training-complete.json)。

2026-10-27 01:57已恢复远程监控：C完成77步、616次题目呈现，同曝光第29步状态已保存，梯度正常，队列持续运行。此前访问失败来自旧终端权限快照与当前工作区不一致；现改为每轮新建当前权限下的连接。定时任务中文也已修复，读取采用明确的编码与保存后逐字核对。见[故障原因与修复记录](operations-repair.md)。

2026-10-27已按用户要求加入全组统一的JSON标点后处理。A/B全部932个回答复算后，仅B外部一题的坐标对分隔符被修复，四个坐标不变，IoU=0.903616。**后处理口径：A为City295/378（78.04%）、外部85/88（96.59%）；B为City300/378（79.37%）、外部85/88（96.59%），即City多5题、外部持平。**原始回答及以下严格成绩保留；后续C/D、同曝光与共同初始化也统一报告双口径。见[修复规则、全量证据及运行入口](format-repair.md)。

A最终checkpoint-59：City378命中295题（78.04%），外部88命中85题（96.59%），466个回答全部合法、无截断。训练236次题目呈现、1888个回答，整臂含训练、评价与阶段占机共7.406961 GPU小时。[完整记录](evidence/a-completed.json)、[City原始回答](evidence/a-final-city378-raw.jsonl)、[外部原始回答](evidence/a-final-external88-raw.jsonl)。这些是当前A成绩；尚不能据此判断相对共同初始化的增益或四臂优劣。

B于10月27日00:59完成两套全量开发评价：City378命中300题（79.37%），外部88命中84题（95.45%）；比A分别多5题、少1题。外部集有1个严格格式无效回答，全部466答均未截断。两组均为59步、236次呈现、1888个训练回答，B最终checkpoint-59与受保护的同曝光状态均已保存；最终状态已与A曝光匹配，后续队列会复用B全量成绩。B训练含加载与保存6.761816小时，整臂含失败、间隔和评价7.338528小时。[完整记录](evidence/b-completed.json)、[City逐题评分](evidence/b-final-city378-scored.jsonl)、[外部逐题评分](evidence/b-final-external88-scored.jsonl)、[全部训练回答及奖励](evidence/b-final-reward-metrics.jsonl)。

C于10月27日00:59从共同warmup50接续，训练入口和实际量化设置见[运行审计](evidence/c-launch-runtime-audit.json)。[原严格配对核对](evidence/ab-paired-development.md)中的外部唯一退步来自JSON标点错误；[统一后处理配对](evidence/punctuation/ab-paired-development.json)后，B在City救回5题、退步0题，外部无救回或退步。City救回题中4题原IoU为0.478—0.498，另1题从0升至0.687。当前只有一个种子，尚不能认定删除负框奖励会稳定改善；相对共同初始化的强化学习增益仍须等待补评。

四臂各自的比较目标及A臂前49步耗时分析见[训练效率复核](performance-review.md)。[优化决议与完整数值证据](OPTIMIZATION.md)已记录：短测占机1.377097/2 GPU小时，微批2和卷积组合数值不合格，隔层重计算耗尽显存；确定性设置建立了可重复参考，但没有作为加速配置采用。实际新队列为`runs/optimized_queue_state.json`，日志`optimized_queue.log`，B/C/D均从共同warmup50开始，使用原`train_env`、微批1、逐层重计算和原同步审计。B/C分别额外保存236/232次题目呈现状态，之后在独立2小时预算内全量复评。旧队列已结束，不应恢复或重复启动。

后续续行优先读云端`runs/optimization_state.json`、`runs/optimized_queue_state.json`及唯一账本`runs/budget.json`。优化和曝光复评各2小时、extend8小时，总预算72小时不变。定时检查保持30分钟。旧完整检查点已在本地归档后释放约14.0GB；14个固定回答数值快照也已全部归档至本地`F:/AIC/.work/quant_rl/optimization_tensor_archives`，共13.08GB，核对大小后释放对应云端临时文件，见[归档记录](evidence/optimization/tensor-archive-report.json)。各臂最终和匹配曝光检查点、共享warmup50与A最终状态须保留。

[最近运行快照](evidence/monitor-latest.json)记录已完成更新、学习信号、预算与磁盘余量。其中运行中GPU时间为检查时估算，未写入正在执行作业的唯一账本；最终以作业结算记录为准。

2026-10-26 20:47核对：B已完成27次更新、108次题目呈现，单更新平均约403秒。按当前单卡顺序排程，B完整成绩预计10月27日01:00前后，四臂首轮及匹配曝光复评预计10月27日17:00—20:00；这是无新增故障条件下的估算，不包含后续独立种子复现和部署比较。72 GPU小时是总上限，不要求用满。

A的强化学习增益尚未证实：共同初始化checkpoint-50缺少完整City378与外部88的同口径单答贪心评价；已有32题G8结果属于训练内诊断，不能作为开发集基线。确认增益需补齐这两套全量评价，并逐题统计相对A的救回与退步；按A评价实际耗时估算约需33分钟，须纳入既有总预算。

补评入口已准备为云端`evaluate_warmup50_baseline.py`，见[源码快照](run-scripts/evaluate_warmup50_baseline.py)。30分钟续行任务会在当前B/C/D及匹配曝光队列完成后、复现或扩展前运行；当前尚未启动补评。脚本要求队列完成且账本无运行任务，保持A的NF4/HF、短答案、单答贪心与并发2口径；上限1 GPU小时取自既有final阶段6小时，总72小时不变。完整结果写入`runs/warmup50_development_baseline/baseline_completed.json`，未覆盖378/88题不能报告部分分数。

2026-10-26 14:32实时检查：四臂队列正常运行A臂，已完成48步，最近步骤保持有限非零梯度、每步4题/32回答；GPU利用率100%，当前占用30711MiB。已保存checkpoint-18与checkpoint-36，B/C/D仍待队列依次执行，完整开发集成绩尚未产生。已结算5.804745 GPU小时，加当前A臂约5.673388小时，账面合计约11.48/72小时；数据盘余28.51GB。按用户要求重新建立 `aic-rl`，检查间隔改为30分钟，仅在有实质结果或故障时通知。

2026-10-26 08:52续行状态：准备阶段已结束，累计5.804726 GPU小时。完整四臂队列已启动，当前为A，队列PID180935，状态`runs/initial_seed2026/queue_state.json`，日志`initial_arms.log`。A/B/C各预留7小时训练、1小时开发评价，D预留3小时训练、5小时开发评价；每臂上限8小时，按实际占机分别计账。四臂均从共享预热checkpoint-50开始，HF生成及评价并发2，G8不变；训练内层提前0.25小时停止以留出保存余量。启动参数见[记录](evidence/initial-launch.json)。更早租机费用及单价仍待平台账单核对，尚无完整四臂成绩。

训练边界已修复两处接口问题：Swift将`negative_boxes/kind`误读为负对话前缀，现改用`grounding_negative_boxes`与`grounding_negative_kind`，奖励和统计同步读取，冻结源数据不变；并发2的两组三图应有6个图像网格，探针改为按请求数检查并统计提示输入。旧失败全部保留计账；负框接口相关测试54项通过，并发检查修订后的训练测试20项通过。

全词表检查的实际HF参数为temperature1、top_p1、top_k0；32题中19组零优势、13组存在奖励差异。训练诊断首答29/32、全部回答229/256命中IoU0.5，仅用于判断格式与奖励信号，不是开发集成绩。生成共910.62秒，含加载占机0.26282小时；[统计](evidence/warmup50-diagnostic32-full-vocab-summary.json)、[原始回答](evidence/warmup50-diagnostic32-full-vocab-raw.jsonl)、[评分](evidence/warmup50-diagnostic32-full-vocab-scored.jsonl)已保留。

A路径探针于08:13:12结束，3步均为4组、32个回答，LoRA梯度有限且非零，冻结参数梯度数0。共96个回答全部合法；12组中7组奖励相同，5组存在奖励差异。峰值分配29,144,232,960字节（27.14GiB），峰值保留28.64GiB；生成517.49秒、参考前向192.74秒、训练前反向634.99秒，含加载、保存与重载总占机0.48313小时。完整checkpoint-3含适配器、优化器、调度器、随机状态和训练状态。记录见[逐步资源](evidence/probe-a-resource-steps.jsonl)、[训练指标](evidence/probe-a-training-metrics.jsonl)、[运行审计](evidence/probe-a-runtime-audit.json)、[奖励记录](evidence/probe-a-reward-metrics.jsonl)及[重载后状态](evidence/probe-a-resumed-trainer-state.json)。现有`logits_to_keep`钩子只观察到生成阶段的1词元调用，不能单凭它证明训练前向也裁剪了词表输出，需按Swift实际计算路径核对。

D工程检查沿用共享checkpoint-50，不使用A探针权重。一次训练步骤于08:44:20完成，总占机0.41114小时；32回答全部无效，31个384词元截断，4组零优势，loss与grad_norm均为0，没有有效参数更新。生成984.27秒、参考前向69.94秒、训练前反向230.52秒，峰值分配25.75GiB；完整step-1状态已保存。见[资源](evidence/probe-d384-resource-steps.jsonl)、[训练指标](evidence/probe-d384-training-metrics.jsonl)、[全部奖励记录](evidence/probe-d384-reward-metrics.jsonl)。这确认并发2下长回答链路可运行，也暴露当前384词元和共同初始化组合缺少奖励差异；不代替完整D臂。

随后固定8道City开发题的并发2贪心评价也全部无效，7个截断；生成251.72秒，平均31.46秒/题，峰值分配21.10GiB。按466道开发题线性估算约4.07小时，故D预留5小时全量评价，保留384词元和共同初始化。见[统计](evidence/d384-cross-query-cap2-summary.json)及[原始回答](evidence/d384-cross-query-cap2-raw.jsonl)。该小样本仅用于速度与接口检查，不能代表整个开发集成绩；四臂最终比较需同时报告实际GPU时间、Query曝光和有效优势组数。

## 当前机器与恢复情况

用户新机为 AutoDL `aic新`，实例 `wtcnbd3acn-d78fcd9c`，SSH端口36581。密码只用于交互认证，不写入代码、文档、命令参数或自动化配置。克隆等待已结束，当前工作根为 `/root/autodl-tmp/quant_rl_20261026`。

| 项目 | 只读检查结果 |
|---|---|
| 实际GPU | RTX4090，49140MiB，计算能力8.9 |
| 驱动 | 580.105.08 |
| 主环境 | `/root/miniconda3/bin/python`，Python3.12.3 |
| 主训练包 | Torch2.12.1+cu130 / Transformers5.16.1 / PEFT0.20.0 / Swift4.5.3 |
| NF4依赖 | 独立 `train_env` 继承既有主环境，新增 bitsandbytes0.50.2 |
| GDN优化 | FLA/fla-core0.5.2；闭包实际绑定 `fla.ops.gated_delta_rule.chunk` 和 `fused_recurrent` |
| 因果卷积 | 仍为Transformers的PyTorch实现；不能用函数非空认定已启用优化 |
| 推理环境 | `/root/vllm_env` 保留；Torch2.13.0 / Transformers5.18.0 / vLLM0.30.0，未装Swift、PEFT |
| 模型目录 | `/root/autodl-tmp/rematch_models/Qwen3.6-27B` 已出现15个权重分片；FP8目录未发现 |
| 数据目录 | City的visible/infrared/depth各759个文件；复核标注与新增媒体已合入，正式索引已冻结 |
| 磁盘 | 数据盘200GiB，恢复过程中已用约156GiB，余约44GiB；系统盘约9GiB可用 |

小型BF16矩阵运算、NF4线性层前向与反向、完整27B NF4推理、C路径SFT及A路径RL保存续训已运行。内核判断读取装饰器闭包的 `implementation`，不依据表面 `__module__`：后者会因 `functools.wraps` 保留Transformers名称。

AutoDL页面曾完整显示原 `aic推理` 实例仍在并已关机，原 `aic训练2` 处于克隆锁定。早先的空列表不足以判断原实例丢失，该判断已更正。后续Chrome连接不可用，不能仅凭磁盘不再增长或容器内没有复制进程宣布克隆完成。

当前环境沿用Torch、Swift与已生效的FLA0.5.2；`/root/autodl-tmp/quant_rl_20261026/train_env` 只补入bitsandbytes0.50.2。vLLM仍在独立环境中，FP8部署能力待后续补齐。

## 首批GPU结果

公开基座、NF4、每图1204224像素上限、HF并发1；两道真实RGB/IR/Depth题各采样8次。16个回答均先输出英文解释，并在64词元处截断，严格合法率0/16；这是格式失败，不能据此判断目标定位能力。生成共157.90秒，PyTorch峰值分配19,928,180,736字节（18.56GiB），采样观察到的整卡占用20,847,984,640字节（19.42GiB）。[统计](evidence/base-smoke-g8-summary.json)与[原始回答](evidence/base-smoke-g8-predictions.jsonl)保留全部样本。

已经直接编码实际输入：短答案模式末尾为 `<think>\n\n</think>\n\n`，与权重附带的Hugging Face模板一致；首题3501个输入词元，三图网格均为 `[1,50,90]`，未触及8192上限。128题诊断完成后，若合法率仍不足95%，执行实施约定中的共享格式热启动，不修改严格解析器。

C路径的独立工程探针已完成两次更新、完整保存、重载与第三次更新。三步均有非零有限LoRA梯度，优化器参数集合与语言LoRA完全一致，冻结参数无梯度；视觉、GDN小门控、词嵌入和输出头均保留BF16。可训练参数233,455,616个，保存的适配器约0.934GB，完整单检查点约2.80GB。峰值分配28,166,405,632字节（26.23GiB），峰值保留30,284,972,032字节（28.20GiB）；含两次模型加载及保存共370.19秒。此探针仅使用8帧City，所得适配器不作为四臂初始化。[参数审计](evidence/probe-c-model-audit.json)、[逐步资源](evidence/probe-c-resource-steps.jsonl)、[重载后的状态](evidence/probe-c-resumed-trainer-state.json)已保存。

相同两题G8的HF并发2试跑共90.23秒，峰值分配21,191,100,416字节（19.74GiB），16个回答仍均不符合严格格式。不同批次会改变随机采样轨迹，此试跑只用于判断显存和吞吐，不比较两次定位分数。[并发2记录](evidence/base-smoke-g8-cap2-summary.json)。完整正式128题×8回答诊断已于2026-10-26 04:52左右启动，使用并发2，结果目录为 `runs/public_base_diagnostic128_g8`。

完整诊断于06:04:32结束，128题含City51、RGBDT45、Robo32，共1024个回答，严格合法率0，528个回答截断，128组奖励全部相同且优势为0。常见完整输出是带代码围栏、列表及额外 `label` 的JSON，另有解释文字；这些仍按既定严格规则判无效。由此可以判断公开基座直接接当前奖励没有定位奖励差异，不能据此判断模型不会找目标。[完整统计](evidence/public-base-diagnostic128-g8-summary.json)、[全部原始回答](evidence/public-base-diagnostic128-g8-raw.jsonl)、[逐回答评分](evidence/public-base-diagnostic128-g8-scored.jsonl)均保留。生成58399词元、4313.73秒，含加载总占机1.20654小时；峰值分配21,195,415,552字节（19.74GiB）。

采样条件更正：上述公开基座诊断及50步后的32题G8诊断虽然请求 `top_k=-1`，Swift4.5.3最终合并配置时却使用了模型默认 `top_k=20`；`top_k=0`也会被回填。因此它们只能作为top-k20条件下的诊断，不能冒充计划中的全词表采样。共享SFT预热不使用随机生成，不受此问题影响。正式RL须在底层明确关闭top-k，并记录最终生成配置后重新运行。

修复将HF引擎的基础生成配置明确设为 `top_k=0`，请求层仍用 `-1`；经过Swift与Transformers5.16两级合并后最终仍为0，真正关闭候选截断。仅清成None仍会被Transformers回填50，故未用于GPU实验。训练同时要求 `do_sample=true`，记录最终temperature=1、top_p=1、top_k=0；本轮训练/评价相关32项检查通过。vLLM保留其原生 `-1` 语义。云端补入[msgspec](https://pypi.org/project/msgspec/0.21.1/) 0.21.1，实际TRL0.29.1的原生DataLoader亦已在无导入替身情况下重验，[远端记录](evidence/sampler-order-audit-remote.json)。

## 正式冻结数据

云端 `data/quant_grounding_20261026_formal` 已完成构建，实际训练4905题（City2882、RGBDT159、Robo1864），封存519题（296、17、206）。正式19200次呈现保留既定40%常规City、5%常规RGBDT、5%常规Robo、25%错实例、25%范围比例；共覆盖4055个不同Query。Robo有849道训练题未进入这一轮固定呈现，统计中明确保留该覆盖限制。[构建汇总](evidence/formal-data-summary.json)。

开发集使用City378/74帧和外部88题。外部原94题中，`rgbdt:297` 的一张RGB图与官方测试 `visible/007897.png` 逐字节相同，因此整序列6题剔除；原94题所有开发组仍禁止进入训练。`city400_dev.jsonl`、`external94_dev.jsonl`保留旧文件名，不能按文件名误报分母。四份实际索引的6543个媒体文件与官方6005个图像文件比较，未发现逐字节相同文件；这不声称排除了重编码同画面或未知物理场景重用。[媒体比较](evidence/formal-test-media-audit.json)。

迁移复用了4113个完全相同的云端媒体或标注文件，新增629个媒体。构建时先合并复核和执行排除，再检查最终保留媒体；旧图被复核图替换后不再因其旧深度路径缺失而误报。正式重建须同时提供City帧排除清单与 `--official-test-excluded-groups data/quant_grounding_source/official_test_excluded_groups.json`。

## 数据净量与边界

恢复的官方测试排除清单来自旧训练机系统盘 `/root/sprint_qwen36/excluded_city_only.json`，本地为 `F:/AIC/data/quant_grounding_20261026/recovered_city_test_exclusions.json`。

| 来源 | 封存前净量 | 已处理内容 |
|---|---:|---|
| City原浮点标签 | 3178题 / 571帧 | 从3707题/681帧排除529题/110帧；114个清单键中4个不在该训练集；与City开发78帧再交集为0 |
| RGBDT | 176题 | 合并98条既有记录与复核表达；15条旧框被复核框覆盖 |
| RoboRefIt | 2070题 | 2000条已有数据与70个新增表达；同图同题去重，只有RGB＋Depth |

3178/571是**封存前标签净量**，对应媒体已完成正式构建。恢复的排除清单还覆盖City开发集4帧；逐字节比对确认其分别对应官方测试可见光图009367、010410、012370、012447。400道有效开发题再剔除22道后，统一使用378题/74帧；`city400_dev.jsonl`沿用旧文件名，但报告称City378，既有City400分数不能直接混比。原浮点标签仍记录原标注来源，不伪称全部人工复核。

已建立的本地 `F:/AIC/data/quant_grounding_20261026/` 为 `nonformal_local_pool`：

- 训练索引RGBDT159、Robo1864，共2023题；封存RGBDT17、Robo206，共223题。
- 1920次呈现用于本地链路；常规RGBDT、常规Robo、错实例、范围各480次。此比例明确只用于缺City时的工程检查。
- 外部94开发集保留，训练与封存、训练与外部94/辅助40的图组交集均为0。
- `diagnostic128.jsonl` 按固定呈现顺序取前128个不同Query，用于G8采样诊断，避免把重复呈现当作不同题。
- 正式数据默认19200次呈现，常规City40%、常规RGBDT5%、常规Robo5%、错实例25%、范围25%；汇总记录每个来源及池的实际题目/图组覆盖与重复曝光。
- 逐行保留 `build_mode`；正常训练要求同目录 `summary.json` 为 `formal`，非正式池只能进入 `--probe`。

本地迁移媒体清单为 `F:/AIC/code/.work/quant_rl/local_media_manifest.json`：4734个唯一文件，共2,139,441,307字节，覆盖训练、封存、外部94及辅助40。清单只列相对路径、字节数和用途，没有复制媒体或增加完整性哈希。

正式构建必须提供完整City媒体与测试排除清单。媒体根下应有 `visible/infrared/depth`，原标签中的 `../visible/...` 等路径以 `target_v2` 为相对基点；depth读取原整数深度并按既有毫米策略生成固定灰度图。每个媒体最终相对 `data_root` 保存。

## 新入口

| 文件 | 职责 |
|---|---|
| `tools/prepare_quant_grounding.py` | 浮点标签合并、复核负框、图组封存、固定采样与深度渲染 |
| `tools/quant_grounding_config.py` | 明确JSON配置、媒体路径解析、SFT/GRPO参数构建 |
| `tools/train_quant_grounding.py` | Swift训练扩展、NF4范围/精度/共享参考检查、G8分批、累计计时、真实保存重载 |
| `tools/quant_grounding_reward.py` | 唯一严格JSON、thinking终止、截断判错、A/B/D奖励 |
| `tools/swift_quant_grounding_reward.py` | Swift ORM桥接、逐回答与逐呈现计数 |
| `tools/eval_quant_grounding.py` | HF及vLLM生成、独立IoU复算、全Query分母、G8诊断、逐题翻转 |
| `tools/run_quant_budget.py` | 72 GPU小时总额、阶段及实验臂记账，包含失败、加载和评价 |
| `tools/export_quant_submission.py` | 唯一回答导出，保持源字段，只添加或替换bbox |
| `configs/quant_grounding/arm_a.json` 至 `arm_d.json` | A完整短框RL、B删负框项、C普通SFT、D有限思考RL |

正式训练默认 `max_steps=1000000` 作为安全上界，主要由实际GPU时钟停止，不再以200步误当完整8小时。配置仅按约2/4/结束时刻保存，保留3份可恢复检查点，避免每100步保存及每臂12份检查点挤满新机磁盘。扩展和独立种子应使用新输出目录；保留必要的完整状态后再安排后续空间，不删除历史模型。

`--probe` 取真实三图样本，完成两次更新、检查优化器/调度器/训练状态文件、重载后再更新一次；探针仅保留最后一份完整检查点，避免单次工程检查占用三份约2.8GB状态。它不能代替8小时实验。正常恢复使用 `--resume-from-checkpoint`；只有LoRA权重的初始化不等于恢复训练。

## 运行顺序与命令

以下在代码仓库目录运行；本地为 `F:/AIC/code`。云端相对媒体与清单已放到同一数据根，命令中的模型和数据根换成实际路径。

### 1. CPU准备

```powershell
python -m tools.prepare_quant_grounding --output F:/AIC/data/quant_grounding_20261026 --local-pool
python tools/train_quant_grounding.py --config configs/quant_grounding/arm_a.json --dry-run
python tools/train_quant_grounding.py --config configs/quant_grounding/arm_d.json --probe --dry-run
```

本轮云端实际正式构建参数：

```bash
python -m tools.prepare_quant_grounding --data-root /root/autodl-tmp/rematch_20260922/data --source-dir /root/autodl-tmp/rematch_20260922/data/quant_grounding_source --city-media-root /root/autodl-tmp/rematch_20260922/data/city/train --city-test-exclusions /root/autodl-tmp/rematch_20260922/data/quant_grounding_source/recovered_city_test_exclusions.json --official-test-excluded-groups /root/autodl-tmp/rematch_20260922/data/quant_grounding_source/official_test_excluded_groups.json --output /root/autodl-tmp/rematch_20260922/data/quant_grounding_20261026_formal
```

缺媒体会明确写出 `missing_media.json` 并中止。正式训练、封存及开发清单共用一次冻结构建，不把另一聊天的新标注中途并入。

### 2. 未更新采样与真实量化链路

```bash
python tools/eval_quant_grounding.py --dataset DATA/diagnostic128.jsonl --data-root DATA_ROOT --model PUBLIC_BF16_MODEL --engine hf --precision nf4 --samples-per-query 8 --output-dir runs/quant_grounding_20261026/base_g8
python tools/train_quant_grounding.py --config configs/quant_grounding/arm_a.json --model PUBLIC_BF16_MODEL --dataset DATA/schedule.jsonl --data-root DATA_ROOT --output-dir runs/quant_grounding_20261026/probe_A --probe
```

HF每题的8次生成实现为8个请求、每请求 `n=1`、默认并发上限1。本轮纯推理已实测并发2，A路径已完成并发1的完整RL与重载检查，D路径正在实测并发2。不能用单请求 `n=8` 替代，否则Transformers会同时建立8份生成缓存。诊断报告区分全错、混合、全中、奖励全同、零优势、无效及截断，并记录生成词元与耗时。

生成文本从真实输出token恢复，剥离框架自动拼接的提示前缀；核心解析器仍拒绝模型自行生成的思考或多个答案。每题结束立即保存 `predictions_raw.jsonl`；中断不会丢掉之前生成。未覆盖完整清单时拒绝计算正式ACC，防止隐式缩小分母。

合法率不足95%时才使用计划允许的共享格式热启动，上限100步且不超过一个数据遍历。所有臂必须共用同一起点；RL将该同一适配器另加载为冻结 `ref_adapter`，仍共享一份27B底座。该条件流程尚需GPU验证，不能用公开基座参考冒充热启动参考。

已准备的条件预热从正式固定呈现中按首次出现顺序取800道不同训练题，先运行50步、每步8题；如需增至100步，从第50步完整状态继续，合计不超过800题的一遍。最终选用同一份第50或100步适配器，不能给不同实验臂挑不同起点。

云端预热配置位于 `runs/shared_format_warmup/warmup50.json`、`warmup100.json`；800题含City506、RGBDT146、Robo148，均来自正式训练呈现。[准备记录](evidence/shared-warmup-preparation.json)。运行仍须通过唯一预算账本，并覆盖模型路径为本机公开27B目录；首段可预留外层1.8小时、内层1.65小时，若实际准备阶段余额更少则相应缩短。达到时间上限但不足50步时，不把它当作已完成50步继续。先检查50步的格式改善及实际耗时，再决定是否续到100步。

50步预热实际读取400题，于06:43:14完成，含加载与保存共0.62326小时，训练前向及反向合计2147.56秒。50步均记录有限非零梯度，峰值分配28,188,807,168字节（26.25GiB）；实际输入最大3536词元，监督回答只计算25至26个位置的logits。第50步完整适配器、优化器、调度器、随机状态和训练状态已保存。[运行审计](evidence/warmup50-runtime-audit.json)、[逐步资源](evidence/warmup50-resource-steps.jsonl)、[训练状态](evidence/warmup50-trainer-state.json)。预热后诊断固定取原128题的前32题；这属于训练内格式检查，不用于声称开发集或泛化提升。

50步后诊断在实际top-k20条件下256/256合法、无截断，9组同时有命中和未命中回答，13组奖励有差异、19组零优势。首回答命中29/32、全部回答命中229/256；这些均为训练内数值。[统计](evidence/warmup50-diagnostic32-g8-summary.json)和[原始回答](evidence/warmup50-diagnostic32-g8-raw.jsonl)。生成905.51秒，含加载共0.26046小时；已选择同一第50步适配器作为四臂候选起点，正式RL前仍核对修正后的采样行为。

已关闭 `dataset_shuffle` 与 `train_dataloader_shuffle`。真实Swift4.5.3/TRL0.27.0小数据加载检查确认RL仍使用原生RepeatSampler：每次生成取同一呈现的8个副本，跳过32个训练微批后接续第5次Query呈现。C按索引顺序读取，800题跳过400个微批后从第401题恢复。此检查没有运行模型前向，实际GPU上的G8归属仍由训练记录确认。HF单次评价已支持相邻两题并发，题图与回答对应已做模拟检查，尚待实际双题GPU检查。

### 3. 四臂与预算

每个8小时槽包含模型加载、训练和必要开发评价。示例先预留1小时给评价，训练累计时限设7小时；具体预留量按首轮实测调整，D的384词元回答应按实际耗时单独留足完整开发评价时间。外层预算器和训练自身计时均需保留，同一实验臂的训练及评价复用相同 `--stage/--label`。

```bash
python tools/run_quant_budget.py --ledger runs/quant_grounding_20261026/budget.json --stage initial --label A_seed2026 --gpu-model RTX4090-48GB --max-hours 7.2 -- python tools/train_quant_grounding.py --config configs/quant_grounding/arm_a.json --model PUBLIC_BF16_MODEL --dataset DATA/schedule.jsonl --data-root DATA_ROOT --max-gpu-hours 7
python tools/run_quant_budget.py --ledger runs/quant_grounding_20261026/budget.json --stage initial --label A_seed2026 --gpu-model RTX4090-48GB -- python tools/eval_quant_grounding.py --dataset DATA/city400_dev.jsonl --data-root DATA_ROOT --model PUBLIC_BF16_MODEL --adapter CHECKPOINT --engine hf --precision nf4 --output-dir runs/quant_grounding_20261026/A_city400
```

外部88另运行一次，仍计入同一标签。B/C/D替换配置与输出路径；D评价加 `--answer-mode thinking`，B诊断奖励加 `--reward-mode acc05`。不以A或C先涨分为D的前置条件。各臂主比较相同硬件和时间，次比较接近的实际Query曝光。

阶段上限：`prepare=6`、`initial=32`、`replicate=16`、`extend=12`、`final=6`；总上限72。初始及复核每标签最多8小时，扩展每标签6小时。超时只终止预算器自己启动的进程树。正常应由训练回调先保存并停止，外层时限仅兜住卡死或超时命令。

预算器只观测它启动的进程。云机已经发生的克隆、空闲或外部下载占用应按平台账单通过 `--record-hours` 补记，不能称为“没有训练所以没有成本”。真实单价通过 `--price-per-gpu-hour` 填入；缺单价时费用记为null，不能当作免费。GPU型号分开记录，不将不同卡的一小时当成相同算力。

正式检查点暂按C探针测得的每份约2.80GB估计空间。已删除本轮安装完成且本地仍保留的2.14GB传输包，云端余约42.56GB；四臂、共享预热和后续复核不能无限累积检查点。每臂结束后检查空间，必要时将本轮旧完整检查点下载到本地归档，再移除云端副本；不删除历史模型或仅剩的一份恢复状态。

### 4. 独立评价与最终格式

```bash
python tools/eval_quant_grounding.py --dataset DATA/city400_dev.jsonl --predictions RUN/predictions_raw.jsonl --baseline BASELINE.jsonl --output-dir RUN/rescored
python tools/eval_quant_grounding.py --dataset DATA/city400_dev.jsonl --data-root DATA_ROOT --model PUBLIC_FP8_MODEL --adapter CHECKPOINT --engine vllm --precision fp8 --output-dir RUN/fp8
python tools/export_quant_submission.py --queries OFFICIAL_QUERIES.json --predictions SINGLE_PASS_RAW.jsonl --format-policy punctuation --output-dir RUN/submission
```

基线JSONL每条使用 `id` 及归一化 `prediction` 或 `bbox`。FP8必须是预量化基座；NF4和BF16要求原版权重，不允许FP8上再次套NF4。HF和vLLM分别检查适配器文件及加载范围；不支持的GDN目标层必须报错，不能忽略后继续报成绩。最终仍需BF16/HF与BF16/vLLM对照，分离后端差异。

导出不执行比赛上传。它要求每题唯一完整回答，不能从G8中选最好结果，也不会给无效输出制造替代框；不满足条件时明确拒绝打包。评价和导出均支持显式`--format-policy punctuation`，仅修正明确的坐标对标点，保留四个坐标；后处理后仍无效或截断的回答计失败。训练奖励保持原严格规则。

## 已有证据与待完成工作

真实安装ms-swift4.5.3的CPU环境完成本轮6个测试文件的联合检查：**83项通过、0跳过**，另有2条上游SWIG弃用警告。已实际导入奖励注册项、Qwen模板并解析四臂参数；常规Python环境也可运行无Swift的数据和几何检查。奖励覆盖IoU=0.5、双方都过阈值的范围负例、越界/空框/NaN/重复键/多答案/截断；预算覆盖失败计费和自身进程超时；导出覆盖源字段不变、完整覆盖及禁止G8选优。

四个配置均为NF4，包括C的普通SFT。C不是BF16训练对照；BF16仅用于计划中另行短时安排的80/96GB参照，不能误在48GB新卡加载完整BF1627B权重。

目前仍缺：D路径及并发2实测、完整四臂和独立种子结果、FP8适配器实装，以及最终封存评价。C路径SFT与A路径RL的NF4、有限非零梯度、完整保存重载已完成；A已记录真实G8归属、参考前向和训练分段耗时。训练回答区词表输出的裁剪已按Swift及Qwen源码核对，旧运行钩子因PEFT直接调用底座forward而漏记该路径，后续正式臂补记实际输出shape。工程探针通过不代表比赛成绩提升。

后续结果应记录每臂City378/外部88的全部Query ACC@0.5、平均IoU、逐题救回与损害、格式失败、生成词元、实际GPU小时与费用。只有两套开发诊断均不退且至少一套有非纯格式净救回，并通过独立种子核对，才进入扩大实验。研究结论目前保持未定。
