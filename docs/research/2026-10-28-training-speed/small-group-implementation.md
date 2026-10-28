# G2与连续IoU独立候选

**2026-10-29用户目标纠正：以最终ACC@0.5为唯一成绩主目标，连续IoU候选不再优先，也不因平均IoU或非零优势组增加自动安排训练。下文配置仅保留为未运行候选；新的G2纠错分流方案见[ACC重新分析](../2026-10-28-rl-reassessment/README.md)。**

**2026-10-29接续修正：以下配置尚未训练，不可直接当作只改G的公平对照。** [实际损失审计](../2026-10-28-rl-reassessment/reward-audit.md)发现原生分母与累积另含1/G：同一重复分布G2梯度是G8的4倍；组均值自包含还改变策略项相对KL的系数。固定G8的现役B不受此次候选修正影响。后续先在固定G8下比较奖励，再为独立G8/G2明确统一有效分母与优势估计器；修改后的G8需单独对照，不能直接拿旧B曲线代替。下文为原候选接线与历史安排，不能据CPU微批等价推断G间目标可比。

2026-10-28。本轮新增两个算法候选，仅进行CPU实现检查，尚未启动GPU实验。原A/B至550步队列及A/B/C/D配置继续使用原口径。

## 最小改动

候选继承B的短框训练、同一正式数据顺序、NF4、学习率2e-6、KL系数0.01、`scale_rewards=none`（不按组内标准差缩放）和共同warmup50起点。每步仍呈现4道题，将每题回答数从8降到2。不会把GT框加入模型提示，也不增加课程、回放或动态补采样。

| 候选 | 配置 | 每步题数/答数 | 奖励 |
|---|---|---:|---|
| G8原B对照 | 原B训练配置 | 4/32 | `1[IoU>=0.5]+0.2*min(IoU,0.5)` |
| G2原奖励 | [arm_b_g2.json](../../../configs/quant_grounding/arm_b_g2.json) | 4/8 | 与原B相同 |
| G2连续IoU | [arm_b_g2_continuous_iou.json](../../../configs/quant_grounding/arm_b_g2_continuous_iou.json) | 4/8 | `1[IoU>=0.5]+0.2*IoU` |

两个新配置仅在奖励选项与输出目录上不同。配置从共同warmup50开启新的优化器过程，引用相同预热权重作为参考策略，不从B59恢复。总共550步、2200次题目呈现；共同预热步数不算入550。每50步保存，不设置小时截止。

## 参数与默认兼容

[quant_grounding_reward.py](../../../tools/quant_grounding_reward.py) 的 `score_completion`、`score_completions`、`grounding_rewards` 增加关键字参数 `iou_reward_mode`，默认`capped`，可显式设为`continuous`。合法回答的命中主项不变，格式非法或答案截断始终为-0.2。连续项权重固定0.2，IoU达到0.5仍有1分的命中跃升；任何合法命中的奖励仍大于任何合法未命中。

[swift_quant_grounding_reward.py](../../../tools/swift_quant_grounding_reward.py) 新增独立注册 `quant_bbox_acc05_continuous_iou`。既有四种注册保持`capped`；新注册复用原始生成词元还原、严格单框解析和终止状态检查，未改变输出格式容忍度。

配置层新增 `num_generations`（默认8）及 `geometry_reward`（默认`capped_iou`）。只有显式`continuous_iou`选择新注册。G2的`generation_batch_size=2`，每次生成1题的2个回答，与HF并发上限分开。单卡`gradient_accumulation_steps`（梯度累积次数）必须为`4G / micro_batch`：微批1为8，微批2为4。G8默认仍为32或16；C的原口径不变。

DAPO损失继续使用原生每组回答词元分母及现有微批补偿机制；变更G时分母随真实组内词元数变化，不把G8的32答累积次数带到G2。G2本身改变组内均值和采样估计器，因此不能以固定答数的梯度一致性门槛声称与G8等价。

## 可证伪假设与成绩对照

1. **少采样假设：** 同样4题/步时，G2将生成及逐答模型计算的逻辑回答数减少75%，成绩下降是否可接受需要真实训练曲线判断。接近4倍仅是这部分工作量比例，固定开销和收敛变化会影响实际总耗时。
2. **连续几何反馈假设：** 不饱和IoU可区分已命中的好坏框，改善G2的有效反馈。它修改训练目标，可能强化无益的微小框抖动，不能称为等价加速。当前`scale_rewards=none`，组内小差异不会被标准差归一化自动放大。

后续有独立GPU时段时，按G8原奖励→G2原奖励→G2连续IoU比较。三个过程都从同一warmup50开始，使用同一正式题序和seed2026；每步4题，比较50、100、150、…、550步时相同题呈现量的曲线。现有G8的同档结果可复用，其优化器与随机状态必须有完整来源。G2改变采样数后会改变后续随机数消耗，不要求各候选的随机答案逐token相同。

评价使用完整City378及外部88，分别列ACC@0.5和mIoU、按难例/小目标/阈值附近分层、逐题命中翻转，以及完整步耗时与累计耗时。除均值奖励外，还记录同奖励组比例、组内奖励范围、有效格式率和KL，防止把轻微几何变化当作稳定成绩提升。运行耗时只作效率指标，不设新的小时截断。

## CPU检查与接续

- 奖励单测：`F:/AIC/code/.venv/Scripts/python.exe -m pytest tests/test_quant_grounding_reward.py -q`，49通过、4跳过。跳过的是依赖本机未安装ms-swift的原有真实模板检查。
- 连同后处理和奖励信号诊断回归检查：`-m pytest tests/test_quant_grounding_reward.py tests/test_quant_grounding_postprocess.py tests/test_quant_grounding_signal.py -q`，70通过、4跳过。
- 新增CPU插件检查仅替换`swift.rewards`注册依赖，执行实际插件的词元还原、评分与截断路径；它不证明真实Swift训练或GPU链路正确。
- [test_quant_grounding_small_group.py](../../../tests/test_quant_grounding_small_group.py) 固定两个配置仅奖励/输出不同，并检查微批1/2的4题8答、梯度累积、参考权重、保存与无小时截止参数。
- 配置接线完成后，候选集成合同的5个用例通过。两份配置的`train_quant_grounding.py --config ... --dry-run`均退出0：生成数和生成批次均为2、微批1/GAS8、550步，奖励分别为原`quant_bbox_acc05`及新`quant_bbox_acc05_continuous_iou`。
- 候选与训练入口合并回归为56通过、2跳过、2失败：本机缺PEFT依赖；既有`MiniDapoTrainer`测试替身缺少本机Transformers要求的`use_apex`属性。未把这些失败写成实现已通过的证据。
- 扩展检查奖励、预算、基准和D修复入口并排除上述两项后，132通过、6跳过、2未选、1失败；失败为既有基准测试直接导入本机未安装的ms-swift。没有修改原测试来隐藏依赖缺失，也未安装依赖或执行GPU。
- 随后使用已有`F:/AIC/.work/grounding_aux_round1/cpu_preencode_env/Scripts/python.exe`完整运行共享前缀安装器合同及训练入口：71通过、无跳过或失败。训练测试替身补齐真实Trainer字段`use_apex=False`，微批1/2/8的原损失及梯度守恒均通过；真实Swift采样与PEFT前向hook也通过，未安装新依赖。
- [test_quant_grounding_shared_prefix_integration.py](../../../tests/test_quant_grounding_shared_prefix_integration.py) 的14个合同覆盖G2/G8完整组、一次视觉编码复用于旧策略/参考/当前策略、原始mRoPE、首答移位、原分母微批补偿、显式题目/呈现/媒体一致性与方法恢复。它保留真实分组规划器，仅用CPU替身隔离Swift导入、视觉特征接口和共享前向；不能代替真实GPU核及成绩检查。

接续在独立GPU时段执行三图探测和完整成绩对照。此文不意味着已运行新采样或测得加速倍数。主handoff由主Agent统一更新。
