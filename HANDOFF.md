# TriGround Handoff

## 9月19日02:32最新快照与外部审计材料

- 云端实查T4已完成第2轮：81/119（68.07%）、mIoU0.5744775590、解析100%、loss0.7264169030；已进入第3轮，最新日志120/232更新、1910样本。第1轮76/119。此为训练内指标，尚未独立重载，不替代固定第4轮。runner7068及训练进程仍运行，未更改配置。剩余纯训练约14分钟，验证/保存/独立评估另约18分钟，预计03:00–03:10完成，需按实际日志更新。
- 用户要求整理供网页版GPT Pro独立审计的材料：`../docs/research/2026-09-19-gpt-pro-audit-brief.md`、`2026-09-19-gpt-pro-audit-prompt.md`、`2026-09-19-gpt-pro-audit-package.zip`。完整prompt内嵌背景和Github链接；40文件包含相关源码、4份实际配置、Stage1/C4/T4指标、RGB/C4逐样本证据及源码研究附录。仅本地交付，未外发、未提交代码。T4第二轮原始metrics已下载至本地证据目录。

## 2026-09-18 8B 三模态计划实施（执行中）

### 9月19日结构优化调查及C4独立评估（最新补充）

- 用户新增请求已完成GitHub公开实现调查：主报告 `../docs/research/2026-09-19-structural-optimization-roadmap.md`，另有public-teams-fusion-sources和structural-grounding-sources两份源码附录。优先建议对象候选/关系选择、区域定位监督；未直接更换运行中架构。未找到可验证同赛题其他队伍公开完整方案。RGBT-VGNet现有真实训练代码及权重链接，不沿用旧未发布结论。
- 独立四模态评估已同步：C4末轮 RGB/IR/Depth/三模态=76/77/77/73；best=76/78/78/78。末轮三模态纠正7条破坏10条；best纠正4条破坏2条。四模式oracle覆盖上限84/80，不是可部署成绩。结果在 `results/c4_modality_complementarity.json`。
- **待短复验的重要口径**：训练内best79、独立best78。代码发现train.py设置FP32 matmul precision high，evaluate.py默认highest，融合层FP32而RGB主干BF16，符合仅融合预测有差别的观察。尚未GPU确认；在当前必做runner结束后以runpy入口设置high重评同C4best，保留独立新文件，不重训。evaluate.py目前没有仅三模态CLI，若不改入口则四模式约7分钟；禁止偷偷改已有C4/T4独立入口造成不公平。当前主对比统一独立入口默认精度，报告两种口径。
- 02:13云端T4已启动；C4固定末轮四模态耗396秒，best四模态407秒。现runner继续；原C4/T4配置不变。

### 9月19日02:03最新状态（优先）

- C4四轮训练及训练入口末尾best重载验证已完成，用时2295秒（38分15秒），232次更新。逐轮命中 **74→79→77→73 /119**；第4轮73/119、mIoU0.524671、解析100%，best为第2轮79/119、mIoU0.569477、解析100%。主对比必须用73，不能拿best79代替fixed4。
- 当前runner正在c4_fixed4_eval（四模态逐样本评估），之后c4_best_eval，再自动T4；尚无T4结果。评估进程的同名子进程为DataLoader workers，勿误判为重复任务。
- 原生RGB76/119；C4 best高3条是候选观察，C4第4轮反而低3条，延长训练未体现稳定收益。等T4固定第4轮结果后按既定条件继续消融/复验，不修改这次T4训练配置。
- 按C4训练38分钟和目前四模态评估速度粗估，C4剩余评估约10分钟，随后T4训练及评估约50–60分钟，首组成对结果约03:00–03:20，需随实测修正。剩余磁盘约10.7GiB。
- 本次heartbeat实际触发02:02:52，晚于此前数据库显示的01:37:20；只记录实际，不能保证下一次严格准点。每次被唤醒必须输出进度/ETA。

### 9月19日01:30最新状态

- Depth已完成：外部512验证428/512（83.59375%），mIoU0.7766499345，解析100%；阶段总耗时7179秒，于01:19:28结束。
- C4已开始第2轮；第1轮119条命中74（62.1849%），mIoU0.539917，解析116/119，暂低于原生RGB76/119。不要提前停训，继续固定4轮后与T4比较。60更新/955样本累计581秒（包含已完成的验证时间）。
- 用户要求每次30分钟检查必须有中文最终回复，含进度与预计剩余时间，不能空白回复。01:30排查应用SQLite只读状态：配置ACTIVE/30分钟，上次实际触发00:38，下一次被排至01:37:20，00:43修改提示后没有01:13触发。原因尚未确定，不能把配置成功当作按时触发的证据；不建立重复任务或替代cron。云端训练独立正常运行。

### 最新实际状态：23:57 UTC+8（优先于以下历史快照）

- **Stage1 IR 已完成**：23:19:49结束，13302条/832次更新，含验证保存共7026秒；外部固定512验证命中384（75.00%），mIoU0.640699，解析100%，best重载评估一致。该外部集成绩不可与目标域原生RGB的76/119直接比较。
- **Depth 正在训练**：6080/18458条、380更新，2.856条/秒，无缺失梯度组。预计9月19日01:10前后完成纯训练，另计验证保存；之后现有runner自动进入C4/T4，无需重新启动。
- 首次正式保存核算完成：IR last约339.57MiB、best约113.18MiB；last epoch1/global_step832，291份优化器参数状态及scheduler/scaler存在；best紧凑格式不含优化器。剩余12.257GiB。后续不再重复检查此首批结果，只看新阶段实际大小。
- 阶段日志、状态已同步本地 `docs/research/8b-run-20260918-evidence/outputs/`；空间记录在云端 `results/stage1_ir_checkpoint_sizes.json`。阶段汇总在本地证据目录 `results/stage1_summary.csv`。持续跟进频率为用户要求的每30分钟。

### 用户最新执行要求与效果诊断

- 用户明确 city 数据合法合规已由团队人工核验；停止追加来源、安全和工程完备性审计，不再追问，不因此阻碍训练。精力集中性能、效率和答案结果。原始比赛材料已经重新读取，后续只对齐任务、ACC@0.5 和输出格式，详见 `../docs/research/2026-09-18-contest-task-crosscheck.md`。
- RGB 错误诊断已保存 `native_rgb_error_diagnostics.json`：43 条错中 2 条回答无目标；其余 41 条能解析，其中 29 条 IoU<0.1。序数关键词组 9/24 命中，仅作诊断；优先完成原定 C4/T4，不另起格式或安全重构。
- IR 已观察到 100 次更新/1600 样本，吞吐约 2.115 条/秒，预留峰值约 22.74 GiB，无缺失梯度组；前 20 次更新的较低显存数据是历史快照。后续检查真实最新日志即可，不反复压力测试。

### 最新运行状态：21:27 UTC+8

- **已正式启动**：原生 RGB 119 条评估完成，76/119 命中（63.87%）、mIoU 0.542532、解析 117/119；逐样本证据同步至工作区 `docs/research/8b-run-20260918-evidence/results/`。Stage1 IR 训练中，20 次更新/320 条时约 2.10 条/秒；Depth→C4→T4 由当前 runner 自动接续。
- 真实压力曾 OOM，**所有配置已统一 max_pixels=602112**。修复训练循环上一批 logits 保留到下一次前向的问题，并对齐压力 autocast；为变长批次统一启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后全部通过。IR/Depth/Joint 峰值预留 22.732/18.551/20.105 GiB，各32微批次、2次更新，梯度有限、冻结符合预期、compact 保存/重载生成完全一致。不要重复压力测试或回调802816。
- runner 启动快照 PID7068，读取 `/root/autodl-tmp/experiments/8b_trimodal_20260918/runner.pid` 后核对真实命令。预算起点 **1789736801**（9/18 21:06:41）；24h判断 9/19 21:06:41，44h训练截止 9/20 17:06:41，48h截止 9/20 21:06:41，均UTC+8，不能重置。
- 心跳跟进已创建：automation id `8b`，每30分钟；每次检查都输出当前阶段、进度与预计剩余时间，即使阶段未完成。完成整个预算内实验、最后评估及报告后暂停该跟进。
- **远端运行脚本与本地最新版有意暂时不同**：当前跑 mandatory 版；本地新增了 `EXPERIMENT_STAGES` 可选分支、自动诊断和报告，已上传至运行目录 `validation_runner_v2/scripts/run_8b_experiment.sh`，真实 Bash 替身测试通过。等当前 runner 结束后，再把本地最新版部署到仓库 scripts；不要覆盖正在读取的脚本。若需诊断现有候选，可直接独立调用 diagnose 工具。
- 后续先看 `stage_status.tsv`、`logs/<stage>.log`、`outputs/<stage>/metrics.jsonl`、`results/`；稳定吞吐估计IR纯训练约1.76h，另计验证，其他阶段用实测重新估计。首次正式检查点尚待生成，需要查看 checkpoint_saved 的大小和 free_gib；弱监督四组磁盘预算也应提前计算，不删除历史权重。
- 当前云端工程完整测试121 passed；随后新 Bash 与报告测试4 passed，分次执行。新模型张量生命周期回归已通过。最终本地Python工具Ruff及diff检查通过；Windows下排除上游Slurm编码问题的测试114 passed/2 Linux测试skipped。
- `tools/report_8b_experiment.py` 用 `PYTHONPATH=.:src:tools /root/miniconda3/bin/python` 执行，输出 summary.csv/summary.md/decision.json，并调用现有成对/场景bootstrap。fixed4与best分开，不把3条门槛叫显著性。
- 新增用户附件 `F:/Downloads/target_v2.zip` 仅有四份标签JSON，与云端逐条逐字段一致，无需导入。923训练中323条Query已修正；人工复核v1仍有weak_label=true，不能按该旧字段过滤。详见工作区 `docs/research/2026-09-18-target-v2-labels.md`。

### 部署准备记录（早于以上运行状态）

- 用户已明确授权实施完整 48 GPU 小时计划，包括云端训练、24 小时判断、可选复验和最终评估；不需要再次请求启动许可。禁止自动 commit/push 或比赛提交。
- 本地与云端基准均为 `85cda24`。已有本地预检修复已由上游覆盖，原文件另存工作区 `docs/research/local-backup-2026-09-18/`，旧 HANDOFF 与测试保留。尾组缩放沿用上游，不重复修复。
- 云端仓库 `/root/autodl-tmp/TriGround`，Python `/root/miniconda3/bin/python`，模型 `/root/autodl-tmp/models/Qwen3-VL-8B-Instruct`，RTX 4090 24GB。SSH 地址和密码见用户本任务原始消息；密码没有保存到文档或脚本。
- **新实验根目录** `/root/autodl-tmp/experiments/8b_trimodal_20260918`。`runs` 指向旧 2B，不要使用或覆盖。当前可用空间约 13 GiB；紧凑保存前后保证至少 3 GiB。
- 新工具：`tools/prepare_8b_experiment.py` 生成绝对路径验证清单、C4/T4/W4 和后续配置；`scripts/run_8b_experiment.sh` 顺序执行并限制 44/48 小时；压力、融合诊断和报告工具同属本轮未提交改动。
- `train.compact_checkpoints` 默认关闭。本轮开启：last 保留完整状态；best 不含优化器，用于推理及 `initialization_checkpoints`，**不能放入 `init_checkpoint` 续训**；mIoU best 仅写指标。`train.log_every_updates=20` 提供稳定吞吐和显存记录；非有限损失/梯度立即失败。
- 云端外部路径检查已通过；IR 13302、Depth 18458，各自的固定 512 验证及目标域 119 验证与相应外部训练来源 ID/图像路径/场景标识重叠均为 0。复用前期目标域检查。来源标识检查不等于重新做图像内容去重，不引入哈希。
- 新运行配置已生成：Stage1 各 1 轮；C4/T4 各 4 轮同初始化、batch1/accum16/seed2026，差异仅输出目录、Query 位置和 partial warm-start。主对比第 4 轮 last，best 另行候选选模。
- 首批部署代码在云端 `pytest -q` **110 passed**。本地全套当时 114 passed/1 failed，失败为上游 Slurm 测试读取 Windows 非 UTF-8 stdout；同测试 Linux 通过。新增编排/压力/诊断工具的后续测试结果另行登记。
- 详细固定协议和最新实测表：工作区 `../docs/research/2026-09-18-8b-trimodal-experiment.md`。当前尚未产生正式定位成绩；不要把两步预检或工程测试成功写成提分。


## 2026-09-11 架构调查与预检阻断修复

- 已同步远端，`main/origin/main` 仍为 `2571a4c`，无更新。9 月 9 日合并提交已在用户授权后推送；下方旧“未推送”只描述当时状态。
- 本轮已确定并修复 `tools/preflight.py` 的 `FrozenInstanceError`：`ModelConfig` 为冻结 dataclass，原实现直接把 modality_dropout 赋为 0。现用 `dataclasses.replace` 创建预检专用副本，训练源配置与原 dropout 报告保留。
- 新增 `tests/test_preflight.py`，调用真实非 offline `main()`，在外部模型/数据替身下覆盖配置替换和报告路径；不会下载权重。preflight/config/Slurm 相关组 **15 passed**；Ruff、compileall、diff 检查通过。模型审计另有相关 **44 passed**，两组有重叠；未重跑全套，未做真实 GPU 验证。
- 本轮待提交修改仅预检修复、该回归测试及本交接段；没有改训练算法、配置、权重、数据，也未 commit/push。
- 完整报告位于工作区 `../docs/research/2026-09-11-triground-architecture-and-improvement-roadmap.md`，另有 architecture-audit、pipeline-audit、multimodal-literature 三份同日附录；若只复制本仓库，请同时取得工作区报告。
- 关键事实：历史 0.6404/0.6785 是 2B；没有当前 8B 成绩证据。Stage2 两个 Query Encoder 可训练，视觉 adapters 冻结；约 45.06M 参数，Joint/Weak/Clean 共 178 次更新。无 Query 位置、Stage1 旧融合器被新 Joint 替换、双门/BF16 残差信号是优先实验问题，尚无 8B 因果结论。
- 后续先复用集群现有权重/日志，运行真实预检和小样本拟合，再做原生 RGB 与同 checkpoint 四模式评估、Query 干预和场景分组统计。随后独立对比训练预算、Query 位置与 warm-start。候选生成加三模态选择先测 OracleRecall@K，不能默认候选召回足够。

## 2026-09-09 分支合并与 Slurm 流程

- 将 `main@770928e` 的 8B/DeepStack 与 `public-v1@e3e5ede` 的 Query 因果诊断、ACC 选模和逐样本评估合并。下方 Query A/B 状态是历史记录，尚无新实验成绩。
- 保留显式融合层 `[8,16,24,26]`、分离的 adaptor/fusion/query 维度、零初始化、稀疏初始化碰撞检测、checkpoint 格式与运行环境信息；同时保留位置编码与三种尺度干预。
- 新入口：`scripts/qwen3_vl_8b_smoke.slurm`、`scripts/qwen3_vl_8b_formal.slurm`。正式任务是五阶段训练后全量评估，操作说明见 `SLURM.md`。
- 冒烟使用独立临时配置，无需前序权重，覆盖 IR/Depth/Joint 两步优化与四模式生成解析。正式配置保留训练参数，前序权重全部指向本次运行目录。
- `tools/prepare_slurm_run.py` 负责路径配置、运行隔离和已有数据重叠审计；`scripts/slurm_env.sh` 使用已经激活的 Python 与共享模型缓存。
- 集群分区、账户、真实路径通过命令参数/环境变量提供；本地没有 Slurm/L40，未提交 GPU 作业。真实 GPU 预检与模型成绩仍须集群运行。
- 本地验证：Windows / Python 3.11 / PyTorch 2.8.0+cpu / Transformers 4.57.3，完整 `pytest -q` 为 **99 passed**。包含四项真实 Bash 编排测试（模型命令替身）、配置隔离/权重链和数据审计测试；新增脚本及预检 Ruff、Python compileall、Bash 语法检查、依赖一致性检查均通过。GPU 显存和训练效果未验证。
- 合并时将 main 中六份 2B 旧配置的本机模型绝对路径替换为模型 ID，并以 `backbone_revision` 保留原版本。所有真实预检临时关闭模态 dropout，以免随机丢弃导致梯度误报；训练配置不变。
- 本轮只完成本地合并，未推送远端。使用前按 `SLURM.md` 配置模型缓存、训练清单路径及集群分区；先冒烟，成功后再执行完整五阶段流程。

## 以下为历史 Query A/B 记录

## 当前阶段

Query 位置编码 A/B 的本地实现已完成；正式云端预检、训练、离线 119 条评估和两次官方提交尚未执行。完整命令和判定规则见 `EXPERIMENT_QUERY_POSITION_AB.md`。

## 已完成

- 新增无参数动态正弦 Query 位置编码，默认 `none` 保持旧模型兼容。
- 新增 v3 Control/Treatment 配置，只有输出目录与位置编码开关不同。
- 全训练链改为 `ACC@0.5 → mIoU → ACC@0.7 → parse_rate` 选模，并单独保留 mIoU 最优和 last checkpoint。
- 修复最佳 checkpoint 重载后误复用最后一轮指标的问题。
- 修复 Joint Fusion 诊断没有作用于真实 `joint_stage_fusions` 的问题；scale=0 直接跳过对应模态路径。
- 增加逐样本评估证据、A/B 成对比较、scene bootstrap、尺寸/类别/已有 Query 类型分层。
- 增加初始化点解析、配置等价检查、整条已知初始化训练链重合审计和梯度预检。
- `combined284` 已降为纯诊断，不允许输出主选 checkpoint。
- 提交工具支持 9555 条强制检查，并记录解析失败/fallback。

## 下一执行者必须先做

1. 在云电脑按 `EXPERIMENT_QUERY_POSITION_AB.md` 安装固定环境并运行完整测试。
2. 运行 `prepare_query_position_ab.py`，确认两组共同初始化来源。
3. 对所有已知祖先训练清单运行 overlap audit；任何 ID、图像或 scene/sequence 重合都必须停止训练。
4. 两份配置分别跑真实 Qwen forward/backward 预检，确认 Query Encoder、Joint Fusion 有梯度且冻结 adaptor 无梯度。
5. Treatment → Control 分别启动训练进程。
6. 每组三类 checkpoint 跑四模式评估；best-ACC 模型再跑 scale/mismatch 诊断。
7. 固定提交 Control best-ACC 与 Treatment best-ACC，返回两个官方分数。

## 当前验证状态

- 本机 pytest 环境缺少 PyTorch 与 `huggingface_hub`，完整测试在收集阶段因依赖缺失而停止，尚未进入测试断言。
- 已完成全仓 Python AST 语法解析、`git diff --check` 和不依赖 PyTorch 的成对比较测试。
- 云端 GPU 预检与完整 pytest 是正式训练的硬门槛，不得跳过。

## 暂不处理

Direct BBox Head 等九项后续方向只记录在实验文档的“待研究计划”中。本轮结果回来前不要实现，也不要提前修改 README、模型注册表或模型卡里的主模型定义。
