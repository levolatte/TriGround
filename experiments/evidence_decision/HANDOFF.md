# 真实证据决策改革交接

> **已停止（2026-10-01，用户明确要求）。** 定时任务aic已PAUSED。本机教师采集控制台26784及其CLI子进程已停止，核查无该控制台或子进程残留；三个审计子Agent已中断。新建一次性直接SSH只读核查：27135 GPU为1MiB、利用率0%。不再采集、预检、训练或评分，除非用户重新明确授权。冻结C、原始轨迹、审核、预算及已完成评分保留；最终80未评分。下文所有“下一步/继续”均为历史计划，不是当前执行授权。

停止时已下载完整城市412档案`F:/AIC/tmp/aic_untrained_city412_20261001.tar.gz`（448709976字节，scp退出0），含全部真实轨迹/工具图、四个阶段state/log、完整评分及能力诊断；未重新推理或修改原始记录。复盘见`POSTMORTEM_20261001.md`。

2026-09-30：用户批准完整改革计划，已开始实施。新代码experiments/evidence_decision由旧实验完整复制，包引用已更新。旧experiments/visual_agent及冻结C不改。

代码分工：input_decision_audit负责控制器/工具/教师状态；training_data_audit负责真实教师导出、数据课程与留出；student_behavior_audit负责评价与推理/观察遮蔽；root负责训练入口、云端同步与整合。

当前训练入口已统一默认8192，支持显式6144、真实最长样本capacity-only/probe；普通token均值CE保留。新增长上下文边界后，云端训练loss相关CPU验证11通过、1因未设置真实Qwen模型路径跳过；正在补真实Processor检查。其它改动仍在实施，尚未启动新训练。

云端27135以无卡模式运行：实查nvidia-smi为No devices found，无新训练进程。旧GPU总已结算12.22925588096848/20h，余7.770744119031519h。新GPU实验未开始，CPU工作不计GPU账。

新结果B=/root/autodl-tmp/rematch_20260922/results/visual_agent/evidence_decision_20260930。拟隔离代码B/code；旧素材在capability_rebuild_20260929，仅只读取素材；模型/root/rematch_models/Qwen3-VL-8B-Instruct；冻结C=/root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500。

已有50教师episode/109review；待重新整理实际合格行为。旧120留出和训练/复习/student200图组无交集；尚待冻结40dev/80final。不要直接训练旧1500固定指令能力行，不能把750BBox复习称成自主工具训练。

下一步：完成代码和CPU真实数据导出，运行最小流程验证，同步云端并完成可执行教师采集。必须真实取证补齐自主Depth/Search/恢复；云GPU恢复后才容量检查、新接口未训练对照和唯一C初始化训练。

## 完整审核轨迹合并入口

新增 `experiments.evidence_decision.merge_reviewed_data`，用于把 `collect_object_teachers` 各bucket最终产出的 `records.jsonl` / `decisions.jsonl` 合并到冻结种子 `data/train.jsonl`，默认生成 `data/train_full.jsonl` 和对应 summary JSON：

```powershell
python -m experiments.evidence_decision.merge_reviewed_data `
  --collect-dir <bucket目录1> --collect-dir <bucket目录2>
```

样本只要有至少一条明确审核决策即纳入，`complete` 状态只计入summary；因此未最终finish或最终错误的轨迹中，已审核的实际证据步骤仍可保留。新轨迹会整体替换该 `sample_id` 的旧教师行，不完整轨迹不会与旧轨迹拼接；无审核决策的样本不替换旧教师行。重复样本出现在多个bucket时直接报错。合并保留750条BBox复习，不包含冻结 `dev40/final80` 图组，并从新教师决策行移除离线IoU/GT字段。Summary报告实际行数、教师动作、数据源、图组/样本、备注覆盖和历史工具观测/模态覆盖；它不把4200行作为自动达标结论。执行入口的定向测试：`pytest tests/test_merge_reviewed_data.py`。

## 2026-09-30 真实集成进展

种子数据已导出并上传独立云目录：843行=93条经逐步复核的历史真实动作+750行BBox复习。历史动作分布inspect44、depth9、search3、正确finish37；13条错误finish未用GT改写。此规模不代表4200行课程已齐，尚不能启动正式训练。

冻结留出已完成：开发40组、最终80组，合计120组，与训练、复习及旧student200图组无交集。旧holdout候选中的C结果缺少权重来源，不能称作本轮冻结C基线；需在恢复GPU后重新跑。

真实Qwen Processor检查单项通过（9.47秒）；843行全部CPU预检通过（113.21秒），最长4325词元。早先预检暴露1条旧budget-finish tools容器为dict，已修复为列表并重新生成，ID/动作/来源不变。CPU长度文件只对应当前843行；追加新轨迹后必须重新全量预检。

已实跑5个GT盲教师样例：中间距离鹿先depth后finish；站立猴、最左苹果、相机近处人员直接finish；第五个机器人执行search/inspect并出现重复搜索。每个动作、执行前输入、模型实际缩放图、工具真实输出均落盘。LUNA-MAX通过本机ChatGPT登录Codex CLI产生动作和独立盲审，未使用付费API。重复动作保留为历史，复核拒绝后不监督；不能根据最后GT把先前错误动作重写。

有限教师循环collect_codex_episodes.py已实现并开始首批8个真实job集成，后续队列有128个优先job（包含全部35条可解释city_mm相机距离Query）及55个真实学生失败前缀恢复。采样类别不强制工具动作。每次新动作由教师看当前状态选择，真实执行后复核仍只接收冻结的执行前输入；最多8个新步骤，可从实际episode/动作/复核续跑。

已修复联调发现的步后视图语义：prepare_teacher_views的step_index表示历史执行前状态，采集循环步后必须不传该参数，取真实当前输入。深度工具相同请求在候选bbox/mask不变时缓存结果并明确new_evidence=false，重复仍消耗证据配额；26项工具/控制器测试通过。

GPU入口run_gpu_phase.py已保存完整命令，默认仅打印，--execute要求CUDA。当前云机仍无GPU，本轮GPU消耗0、剩余7.770744119小时；容量前后向、新接口未训练对照、正式训练、完整412及最终80评价尚未执行。详细数据与教师工件在F:/AIC/results/visual_agent/evidence_decision_20260930。

## 2026-09-30 晚间接续与真实阻塞

已修复本机教师历史转换漏掉tool_calls/name/tool_call_id的问题；重新盲审拒绝两次相同搜索，原始错误历史保留但不监督。同一路径的处理后图像附件去重，消息、调用和返回仍完整保留。

截至首轮合并，train_full.jsonl为859行=109条审核教师决策+750条BBox复习，动作depth15/finish45/inspect45/search4；新轨迹20条整体替换4条旧教师行。它仍是扩充中的训练工件，不是已冻结正式训练集。843行种子的CPU长度文件不适用于它。run_gpu_phase默认改为train_full.jsonl，正式训练前重新生成cpu_preflight_full_8192。

首7条相机距离轨迹均自主depth后finish，14个步骤盲审通过，终局离线IoU均≥0.5。这些是教师训练来源，不能称作学生提分。首35条相机距离队列继续采集中，实际有UNKNOWN深度返回，仍保留实际返回供后续决策。

两条实际搜索尚未得到工具返回：reviewed_0008的van/left非零退出；city_001211_013的robot/right独立诊断返回-9（SIGKILL），30.31秒，容器memory.max=2GiB、memory.events.max693→704、oom/oom_kill均0。未追加伪造ERROR步骤，原始执行前状态与计划动作保留在teachers/blocked_search_jobs.jsonl，待GPU或内存资源实际变化后接续。不能宣称已确认内核OOM，也不能通过改裁图掩盖它。

旧student200全部使用visual-agent-object-v1/paired-evidence-capability-training。恢复入口现在仅在显式--migrate-legacy-prefix时接受该已知组合，保持原候选与事件；未知版本仍拒绝。实际v1输入已验证；对象教师、原生历史转换、合并的13项定向检查通过。修复已同步独立云目录，首3条真实失败前缀正在接续，未把旧错误动作设为监督目标。

相机队列reviewed_0178发现新实际错误：两个reference角色候选进入旧compare_depth_pair被拒绝。正在独立vision_tools修正对象接口与旧角色语义的交界，随后从原动作接续；不修改共享tools。reviewed_0162、reviewed_0157已执行的动作和待盲审状态均保留。

GPU恢复后的完整阶段与三臂评分命令见GPU_RUNBOOK.md。直接C412应取implementation_20260928/manifests/city412_c_baseline.jsonl；旧c_reference412是225/412控制器，不能误作292/412直接C。120组直接C来源未知，仍需实际重跑。无GPU恢复授权或设备变化，本轮GPU训练和评测仍未启动。

## 用户准备重启开GPU时的停点

2026-09-30约22:30（香港时间），用户要求准备重启，准备后明确告知“可以重启”。已经停止安排新采集；当前有限调用自然结束。云端ps核查未见本实验教师、训练或推理进程，已取回全部存在的16个collection episode。没有启动GPU任务，也没有消耗本轮GPU预算。

最新train_full为865行=115条教师+750条BBox复习，教师depth17/finish48/inspect46/search4，120个留出图组零重叠。9条相机距离轨迹完整，终局均离线正确；reviewed_0162/0157各有已执行depth但尚未盲审，0178已存原始depth动作、等待执行修复后的接口。恢复3条中1条search资源阻塞；另2条完成，其中鸟的joint查看合理但finish错误，finish不监督；红衣人员finish正确。恢复共保留inspect1/finish1。教师新增26条，整体替换4条旧教师行；数据还未正式冻结，预测框和恢复覆盖仍需补齐。

已修复恢复采集将prefix_events从steps扣除的错误；object_teacher的steps仅含新教师动作，旧学生历史单独在prefix_events/turns中。回归验证首次恢复动作的计数和盲审不会漏掉。迁移时保留非空错误助手输出及协议反馈，只跳过未产生动作/输出的纯运行预算失败；旧错误仍不监督。深度两个reference对象现在沿用原可靠性与距离阈值，报告真实角色，27项工具检查通过。

接续索引在结果目录restart_status.json，包含未审核的真实步骤、原始待执行动作和远端episode路径。资源阻塞清单现3条：机器人right、van/left、自行车full。重启后先重连27135、核查CUDA/设备与容器内存变化，然后从真实保存动作补取观察并继续盲教师；不要重新初始化或改写未完成episode，不训练目前865行作“完整课程”。正式训练数据补齐后重跑全量CPU预检，再按GPU_RUNBOOK执行同接口对照、唯一训练和完整评价。

云端完整预算账本已重新核对，z_budget63行、t_budget39行、旧capability ledger4行，合计约12.229256小时，新ledger尚不存在。执行入口读取这三个完整账本；本地副本不完整不能用于推断云端漏账。

## 2026-10-01 恢复执行

用户已开机并要求继续，27135恢复4090/24GB与约90GiB容器内存，原3条资源阻塞搜索已全部真实执行。主机SSH参数延续27135，未改46057。所有模型取证通过run_capability_stage统一结算原20小时账；工具准备累计分配0.65小时，不另开账。实际执行序列见本轮ledger。

原5个待接续collection job已完成，包括0178 UNKNOWN深度后的补查及自行车EMPTY后的恢复，待逐步盲审/离线终局过滤后的汇总。机器人独立fresh轨迹已执行left搜索至第6次证据工具调用，模型接下来仅按真实配额终局；right区域盲审拒绝，原步骤照常保留。不能因取到新框就认定找到目标。

直接C120已实际完成，127.74秒、120/120新推理、无失败。新holdout120_candidates_frozenC.jsonl为两接口臂共同初始池；旧未知来源缓存只读保留。开发40 C=29/40、mIoU0.632311，最终80还未评分。score_inputs_dev40_reference和score_dev40_reference记录只含40题的合法评分输入/结果。

已修容量探针CPU设备问题，正式capacity前明确model.to(cuda)及输入同设备。新增align_baseline_candidates与prepare_evaluation_subset，GPU_RUNBOOK已更新真实C池对齐顺序与按40/80筛预测命令。云端代码已同步103个py/md文件，相关容量/损失3项和候选/子集5项检查通过。

根持久会话在三个独立集合并行运行GT盲CLI教师：collection（相机35，复用已完成项）、collection_recovery（55前缀按真实错误/未知排序，跳过此前首3个已采项）、collection_additional（首128剩余93+额外32）。线程图像缓存独立，工具观察和账本结算用一把设备调度锁串行；没有文件锁/重试框架。恢复reviewed_0155与相机轨迹同ID，最后合并须明确选完整的合格恢复轨迹，不能拼接前缀。数据还未正式冻结。

用户随后澄清：赛题复核聚焦技术任务，安全和合规由人工管理；用户确认已有离线教师标注的明确许可。教师采集已经继续，不能因此另开审批或阻塞实验。赛题技术口径仍是RGB全局归一化xyxy框、IoU≥0.5、完整查询分母；红外是温度相关灰度，深度毫米编码仅在实际适用数据上用于距离比较。最终提交保留除bbox之外的原JSON字段并压缩为zip。

短暂停止新CLI调用的状态保留在原episode、动作和审核文件中；camera集合已从原停点接续，另外两个集合继续运行，没有重新初始化或覆盖原始步骤。最近一次预算读取：本轮31个已结算阶段538.75秒，其中教师工具411.01秒；这只是该时刻快照，下一阶段必须读取实时完整账本，不能当最终消耗。

fresh的robot step6已完成GT盲终局审核，随后才离线评分。fresh_collected_gpu20261001包含5条完整轨迹，4/5终局正确、8条合格监督；机器人错终局不监督，合理inspect/search仍保留。合并时用这份新fresh工件整体替换同ID旧fresh行，不能同时纳入fresh_collected/fresh_supplement导致重复。

已启动untrained_holdout_8192，后台PID4793、实际子进程4798；启动日志untrained_holdout_launch.log、预算状态stages/untrained_holdout_8192/state.json。根设备调度锁在该GPU阶段结束前保持占用，教师CPU推理可继续，实际depth/search等待；必须调用read_gpu_phase看到终态才释放锁。禁止重新加载tmp/evidence_decision_sync.py使锁丢失。正式capacity仍用最终全量数据，早跑未训练对照不替代这一检查。

未训练120首段按800秒内部时限结束，815.68秒、68/120；execution_first_segment.json保留该段摘要。已用相同run_config加--resume接续，stage=untrained_holdout_resume1_8192，后台PID7195。根TASK_ACTIVE_GPU_PHASE=('untrained_holdout_resume1',8192)，仍需终态后释放设备调度锁；终态时核对execution.complete而非仅进程exit0。

临时容量快照916行，全量CPU预检78.43秒，最长5969词元（reviewed_0170:teacher_step:5）。第一次GPU探针在eval/无autocast模式OOM，耗15.89秒；此模式与正式Trainer不一致，原失败日志保留。train.py已显式model.train()并使用CUDA BF16 autocast前向、上下文外反向，LoRA参数仍FP32。修正后相同样本通过：前后向4.513秒、峰值21228442112字节（约19.77GiB）、优化器更新0，阶段总20.99秒。维持8192；这只是临时数据检查，最终课程仍重新全量预检与最长样本容量检查。

新增24条真实初始RGB候选未覆盖训练题（所有现有可提交RGB框离线IoU<0.5），来自训练GT的离线课程筛选。GT、IoU、目标框和筛选提示都不进入教师任务/输入，原候选池未改，不指定动作。collection_missing已排队，沿用三线程采集池，等待已有一个集合完成后开始；不是独立盲测成绩。最后合并须同时处理fresh与collection各集合的同sample重复，选一个原完整轨迹并写selection，不能拼接不同前缀。

截至2026-10-01约03:05香港时间，未训练120全部完成，接续阶段519.00秒，原两段共1334.69秒；完整execution.complete=true。全部120的原始轨迹、实际工具图、预测与两个预算段已取回本地。只评分dev40：C29/40、未训练24/40（mIoU0.528733、合法终局37/40、纠正2/破坏7），最终80尚未评分。camera集合35题已完成，其余recovery/additional/missing继续采集。

城市412未训练对照已启动untrained_city412_seg01_8192，PID8396，内部500秒、外部600秒。tmp/evidence_decision_sync.py新增start_gpu_segment函数，仅读取已有run_gpu_phase命令后缩短时限和更名，后续segment>1追加--resume；根会话只载入新增函数，没有重载设备锁。TASK_ACTIVE_GPU_PHASE=('untrained_city412_seg01',8192)，仍须read_gpu_phase读终态后释放锁、核对完整数，窗口之间允许教师真实工具执行。不能把分段的退出0当作412完整完成。

tmp/export_evidence_training_snapshot.py导出扩充中课程，保留已审阅合格部分前缀、按sample选择一份原轨迹、原750复习逐行不变；尚不覆盖train_full。冻结后再运行全量CPU预检和正式最长行GPU容量。diagnose_capabilities已加入ERROR恢复统计，2项定向测试通过，需同步该py文件到云端；独立能力切片清单正在CPU准备。

## 2026-10-01 每30分钟接续

用户明确要求创建本对话定时任务、每30分钟核对进度，避免长任务一直等待轮询。已创建并核实原生heartbeat任务`aic`（AIC证据决策实验进度接续），ACTIVE，绑定本对话。每次做有限核对，健康运行时返回等待下一次检查；完成时推进已批准的必要后续，不能重复启动已完成阶段或把改革宣告成功。

城市412未训练前三段完成143题，阶段耗时505.20/503.07/519.75秒；execution_seg02/seg03保存调用摘要。第四段已启动PID10466、stage=untrained_city412_seg04_8192，内部3500秒、外部3600秒，沿同一配置--resume补齐剩余题，不改变412分母。最近已完成账本合计13.334279小时，当前第四段另在运行；剩6.665721小时为该段开始前余量。教师工具累计946.94秒/2340秒分配，之后须读取实时账本。

为避免主动agent持续轮询，根会话20636新增task_gpu_watch_pool/task_gpu_watch_future。wait_gpu_phase_completion通过已核实GNU tail --pid在一个有限后台调用中等待第四段包装进程退出，再调用read_gpu_phase释放设备调度锁。因此只要watch future尚未完成，下一次检查只读stage文件或future状态，不并发调用read_gpu_phase抢先释放同一锁；future完成后核对其结果/异常及execution.complete。如watch实际失败，读取原因及阶段真实终态后再处理，不能整体重载helper。没有新实验调度框架，后台程序继续运行。

recovery一次盲审被根控制台Ctrl-C中断（错误证据保存在collection_recovery/cli_interrupt_20261001.json），已用原队列/episode/动作从待审点恢复，没有重做取证。不要再向此共享控制台发送Ctrl-C。camera已完成，recovery/additional/missing三个future仍运行且最后核查无异常。当前课程快照1018行=268教师+750复习，仍未冻结。自主bbox教师终局仍缺，750 GT框复习不能算自主框预测；保留真实采集结果，不能指定教师动作凑数。

diagnostic_dev40/final80/holdout120/city412已按Query和模态生成启发式切片，源清单不变；城市412实际GPU清单已取回city412_eval_manifest.jsonl，与诊断副本逐ID/Query/模态核对412/412一致。diagnose_capabilities的ERROR修正及PLAN/HANDOFF/RESULTS/GPU_RUNBOOK已同步一次；本节最新文字尚待下次资料同步。下一次检查从第四段和三个教师future的真实状态继续，课程达到实际覆盖后冻结、全量预检、正式容量和唯一训练；final80仍未评分。

## 2026-10-01 04:00定时检查与SSH接续

本次向20636持久Python控制台发送检查代码被自动审批拒绝（只读空输入仍可读）。这不表示SSH不可用。用户明确要求直接SSH，并重新提供原主机登录信息；已直接连接`ssh -p 27135 root@connect.westc.seetacloud.com`，新持久SSH终端20439。密码仅用于登录输入，未写入文件或回显。后续优先用该SSH终端的明确只读命令核查，不能把单一工具拒绝扩大成连接不可用，也不能据AutoDL显示名称猜实例对应关系。

实际SSH核查：untrained_city412_seg04_8192包装PID10466为运行中，已运行35分54秒；日志完成367/412，GPU RTX4090利用率79%、20539/24564MiB。完整已结算账本仍13.334279小时，当前段尚未结算；四账本为63/39/4/82行。前三段PID8396/8914/9503实查均已退出且没有残留进程，所以GNU tail --pid等待的正常退出机制仍可使用。未启动新GPU阶段，未提前释放在用锁。

课程快照刷新为1057行=307教师+750复习，留出图组仍0行；teacher动作depth48/finish113/inspect104/search42，全部113个教师终局仍为ID，自主bbox终局0。当前原始采集文件数量collection38/recovery38/additional52/missing6，仍未冻结，等待真实剩余轨迹。已完成的合理部分前缀照常保留，不重做工具，不将750GT复习称自主预测。定时任务保持ACTIVE，健康运行继续交给下一次有限检查；下一次先用SSH读第四段state/execution及实际GPU进程，再推进课程验收和训练。

## 2026-10-01 04:20 直接SSH与城市完整对照

用户追加要求精简定时任务提示词并排查审批拒绝。原生任务`aic`已原地更新，保留每30分钟、ACTIVE及原线程；训练参数和详细验收转引本目录文档，提示明确直接SSH的持续授权及连接20439。配置实查原本就是`sandbox_mode="danger-full-access"`、`approval_policy="never"`，本项目没有覆盖配置；没有理由再改配置。向20636旧Python控制台发送一条只读状态打印仍被工具返回“approval required by policy, but AskForApproval is set to Never”，空输入可读，20439的直接SSH命令可执行。未查得更具体拒绝原因，不能宣称底层审批问题已修复；直接SSH实际可用，后续沿获准命令继续，不再因旧控制台单次拒绝推断服务器不可达，不通过换装命令绕过限制。20636仍在运行教师采集及原串行锁，不得Ctrl-C或整体重载helper。

SSH实查第四段已正常完成，state.status=complete、exit_code=0、elapsed_seconds=2548.236337；execution.expected/completed均412，complete=true。四段城市推理的阶段总耗时约4076.26秒（1.1323小时），现不能再启动城市未训练对照。教师GPU工具随原后台watch释放锁后继续执行，空读20636已见新阶段和真实UNKNOWN返回。没有提前手动释放锁。

完整城市412 CPU评分已生成`score_city412_pretrain/summary.json`，能力切片和ERROR/UNKNOWN等实际恢复诊断已生成`diagnostic_city412_pretrain`。原冻结C292/412（70.8738%，mIoU0.613628）；新接口未训练217/412（52.6699%，mIoU0.468091），合法终局394/412，相对C纠正11、破坏86、净-75。初始池覆盖339题，搜索新增覆盖1题；73个初始未覆盖均未命中，自主bbox终局2次均错误。该结果是训练前对照，不能当改革提分；最终80未评分。

20:18 UTC完整四账本63/39/4/87行，已结算14.059788小时、剩5.940212小时；教师工具1010.54秒/2340秒。后续仍读取实时账本，保留训练后完整412（当前同接口阶段实测约1.1323h）、120（约0.3707h）、诊断及容量时间，再定唯一训练时限。不得占尽训练预算漏掉完整评测。

扩充快照目前1064行=314合格教师+750原框复习，depth49/finish113/inspect108/search44，留出0行。课程未冻结、全量最终预检/容量/训练尚未启动；恢复、补充、未覆盖队列继续。两位已授权审计Agent正只读核查实际课程覆盖及自主bbox缺口，不给教师GT，不强制选择某动作。城市全部真实轨迹、工具图、四段state/log和完整评分正取回本地；下载完成后注明确认状态。若采集仍健康运行，本轮核对后等下一次定时接续；下一轮先查已存工件与采集停点，不重复跑412，不触碰final80评分。
