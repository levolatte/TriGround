## 2026-09-29 审计包与Git交付

用户要求打包现有关键代码、技术报告、逐题对照图与开发证据，并提交推送。用户重新以CPU模式打开原AIC后，已将四组正式20份预测、最终预检/训练记录/图册/实际执行源码下载到results/triground_abv_execution_20260928/cloud_final；本地重算评分一致，三组4800条消费顺序均验证通过。此前C0/V原件缺失已补齐。GPU任务未重启，triground定时任务仍PAUSED，第二种子未执行。

审计入口code/docs/audit-20260929/README.md与AUDIT_REQUEST.md；包输出F:/AIC/audit_exports/TriGround-audit-20260929.zip。包内保留原件与失败证据，排除权重、优化器大文件、重复tokenizer与原始训练图批量，提供缩略预览且不替换正式清单。并行experiments/visual_agent仅快照参考，本次提交不纳入其未提交代码。CPU相关测试81通过1跳过；修正旧400步测试取整预期，不改变600步训练实现。提交号以Git和包内GIT_STATE.txt为准。

## 2026-09-29 用户要求结束GPU使用并关闭定时任务（最新指令）

用户要求V审计结束后通过Chrome关闭名称恰为AIC的实例，绝不能操作AIC2。原计划A第二种子暂不启动。已复核V全部评估/报告完成，nvidia-smi无计算进程；V City289/412、专项54/70，IR净效用-1（M3FD000698），Depth净效用-1（Robo0007227），双置空对照净-2，未显示额外模态收益。共享GPU账本23674.615233秒，running=null。

自动化triground已通过工具确认为PAUSED。浏览器控制多次读取Chrome标签页均报nodeRepl.fetch request failed，重建会话仍失败，尚未取得实例页面，更未点击关机。AIC关机未完成/未确认；AIC2未操作。后续只能在浏览器连接恢复后核实AIC行并关闭，不要把已暂停自动化误写为实例已关机。

# 每30分钟接续A/B/V实验

## 最新覆盖状态：2026-09-29 首轮全部完成

以下早期训练进度仅为历史。`seed2026_600_deterministic` 的 A/B/V 均完成600步，C0/A/B/V的City412、70题四条件评估及报告/图册均已完成；账本 running=null，累计23674.615233秒（6.576小时），剩19525.384767秒。勿重启旧恢复脚本或重跑首轮。

City命中：C0=292，A=296，B=289，V=289。A对C0救回8、损害4，救回覆盖7图组，达到原计划第二种子入围门槛；第二种子尚未启动。下一GPU阶段按原授权检查预算、生成seed2027独立清单和初始化，执行A的必要预检及复跑，不能复用seed2026优化器。B/V不入围，不扩大训练。当前用户要求先快速并行完成A/B逐题与模态审计。

本地完整A/B预测：`F:/AIC/results/triground_abv_execution_20260928/verification/AB_analysis`。City B对A救回3、损害10；专项正常57→54，0救回3损害。IR置空无命中翻转；实际去Depth的38题，正常输入相对置空净效用A +1、B -2。详见该目录city_audit及modality_audit；后者已独立从原始浮点GT重算。三组真实消费trace均4800条且逐位符合冻结清单，B/V专项题确已消费。

**用户已明确“继续”，暂停已解除。** 已核实原云端GPU空闲及完整检查点，启动`resume_after_user_pause.py`（父进程1422，日志`resume_after_user_pause.console.log`），A128→600、B32→600、V32→600，随后自动完整评估。自动化triground已ACTIVE，每30分钟检查。启动时共享账本4343.769169秒（72.4分钟），主动暂停不计失败重试。不要重复启动该脚本。

本次恢复点：A/main/checkpoint-128，B/main/checkpoint-32，V/main/checkpoint-32，MAX_STEPS仍600。Trainer继承save_steps=16；不能用原runner固定checkpoint32恢复A。已核对样本trace自动备份与截断逻辑：A暂停前1082条，只保留128×8=1024条，其余58条存为uncommitted_after_checkpoint_128.jsonl后重跑。沿用同一预算和优化器状态，不重新初始化。任何后续中断仍须重新查最新完整检查点，不能重跑当前固定恢复脚本覆盖新进度。

用户已授权自动检查、有限重试和成功后推进。每次唤醒只检查一次；进程仍正常运行则结束本次检查，不留在对话里轮询。状态不变不发消息；阶段完成、实质问题、最终结果才通知。不要恢复历史定时任务。

## 当前真实状态

- 已冻结368批准中的232专项训练题、70诊断题；600与400步清单均CPU验证通过。70诊断中的Depth6只作个案。
- 云端根目录：`/root/autodl-tmp/rematch_20260922/results/triground_abv_execution_20260928`。
- 当前代码目录：该根目录`code_snapshot`；配置`../execution_config.cloud.json`；清单`../deployment_600_seed2026/release.json`。
- 当前有效运行：`../runs/seed2026_600_deterministic`。600步曲线，seed2026，A/B/V三组。
- 22:00心跳确认预检全部完成，已启动顺序train→evaluate，日志`../train_evaluate.console.log`。训练阶段内部串行A→B→V，全部成功后自动接四组完整评估。实际状态以进程及账本为准，不要另启重复阶段。
- 新目录零步8题完全一致；A/B/V连续32与16＋16恢复全部pass，LoRA分别288/288/304张量、Adam动量分别576/576/608全部逐值一致，最大差0；样本/RNG/调度一致。V语言及视觉BA增量均非零。
- `seed2026_600`是旧失败证据，不继续其检查点。其首段出现非确定性差异；现启动器开启`--full_determinism True`。`determinism_probe`双单步也完全一致。不能放宽恢复容差。
- 既有交互SSH会话为21267（旧81074已断开）；若不可用，按本对话已授权SSH连接，凭据不得写入文件。GPU为单4090。23:55检查A585/600，进程存活、日志无异常，含正在运行部分累计GPU约154.7分钟；正常继续，不重复启动。

## 共用预算与推进

唯一权威账本：`/root/autodl-tmp/rematch_20260922/results/triground_abv_20260927/gpu_budget/gpu_budget.json`，上限43200秒。预检结束已用3230.5124秒（53.8分钟），剩39969.49秒。实测预算预测600步训练＋完整评估22936.92秒（6.37小时），满足10%余量；不需要降400。后续必须重读账本。`running`字段及实际进程要同时检查；不能重复启动GPU任务，也不能因旧会话失联认定进程已死。

预检结束后读A/B/V的`resume_comparison.json`，全部pass才能运行train。命令在code_snapshot目录：

```bash
python -m tools.run_triground_abv --config ../execution_config.cloud.json --release ../deployment_600_seed2026/release.json --output-dir ../runs/seed2026_600_deterministic --phase train --execute
```

该阶段自动顺序A→B→V，先生成budget_forecast；必须保留10%余量并覆盖完整评估。600不容纳则依原计划导出已冻结400清单，使用新目录重新初始化及完整恢复预检，不能修改600曲线终点。400仍不容纳即结束本轮并报告。

train成功后运行同命令的`--phase evaluate --execute`；自动完成C0/A/B/V全412、70诊断四条件、报告及翻转图册。每阶段用nohup及独立日志启动。可使用一个简单顺序进程串起已通过门槛的train与evaluate（前一成功才执行下一），避免等待下一次唤醒。不要重跑已完成阶段；`--resume`只用于确认已完成阶段后的跳过，不代表失败训练会自动从最新检查点恢复。

完成首轮后按用户计划排名及入围门槛检查，预算足够才对最高入围组seed2027复跑。之后有余量才做固定异场景模态置换。保留历史C292/412与同环境C0区分；无提升如实记录。

## 每个任务最多2次自动重试

在`F:/AIC/results/triground_abv_execution_20260928/retry_state.json`记录逻辑任务（seed/步数/组/阶段）、尝试次数、失败原因和处理动作，发起前先递增。换目录不能重置同一逻辑任务计数。仅对已定位可修复的错误或临时执行故障重试；先核实原进程已经结束。已有历史失败和修复留档，不重复消费其失败检查点。

训练中断时使用最近完整检查点恢复模型、优化器、调度、RNG及实际样本位置；不得盲目用runner固定的checkpoint32重新覆盖部分训练。必要时调用现有train_stage/execute_stage，仍记入同一预算和阶段记录。精度、零步一致性、恢复比较、数据或预算门槛失败，先修根因再重验，不能绕过。最多2次仍失败则停止该任务和依赖任务，说明原因并等用户；不通过改名继续重试。

## 收尾

取回预测、配置、日志、比较报告、账本和图册；无需批量下载权重。核对实际消费清单及完整分母，翻转原因有证据才分类，否则未知。更新HANDOFF、研究README、status.json。完成全部可执行工作、预算耗尽或重试上限用尽时暂停本定时任务，交付结果和未完成项。不提交推送、不比赛上传、不删除历史数据模型。
