# LocateAnything 完整评估执行记录

## 约定与当前起点

用户批准先完成LocateAnything，再讨论其他路线。本轮只做冻结定位与离线候选覆盖，不追加三图训练、不训练选择器、不改变提交方案。当前多模态负结果不排除训练量、监督或局部数据质量问题。

云机connect.weste.seetacloud.com:28355，RTX5090；根目录/root/autodl-tmp/rematch_20260922。沿用models/LocateAnything-3B及envs/locateanything/bin/python，Torch2.8.0+cu128、Transformers4.57.1；现有模型完整标记存在。cv2/decord/lmdb及官方worker导入通过，但此前仅CPU/算子检查，真实模型8条检查尚待完成。旧加载失败日志保留。

## 固定执行与评分

- 固定expert_smoke_8.json检查后自动完成qwen_generation_val.json全部412条、78图像组。前几条准确率低不提前结束。
- 原始RGB、完整Query；官方ground_multi提示为“Locate all the instances that match the following description: {phrase}.”，官方batch runtime、hybrid、la_flash/flash_attention_2、严格注意力模式，BF16默认权重精度。生成上限2048、temperature=0，不额外resize，关闭TF32。
- 取原生输出第一个有效框为单框答案，保留全部有效候选；原生0–1000坐标转归一化xyxy。原始浮点GT仅评分，不用GT选框。没有有效框计失败，完整评估分母412。
- 原evaluator已支持按ID和相同配置恢复。补充逐条进度、样本耗时、GPU峰值、模型加载时间与runtime_history；不改变已有run_config，以保留此前失败检查的恢复兼容。
- 脚本只做smoke→l0→单模型报告→R2+L0报告，取消旧训练/下载PID等待。GPU不空闲时明确停止。明确瞬时故障最多额外重试两次；确定性或未知错误停并定位。每次独立日志，旧日志不覆盖。

## 比较与验收

L0与R2第二轮288/412比较ACC@0.5命中数、mIoU、解析率、错变对/对变错和78图组bootstrap。模型、训练方式、硬件均标注，属于完整方法比较，不归因单一结构因素。

分别报告L0自身与R2优先再L0轮流合流的候选Recall@1/4/8/不截断；IoU≥0.9去重。真实候选来自图像和Query，GT只评价覆盖，上限不是模型成绩。

10项CPU解析/路径/恢复/候选报告测试通过，其中多框顺序保持和失败样本分母有明确用例。真实8条检查、完整412与GPU速度尚未完成，不以CPU测试代替。

启动后恢复每30分钟跟进，按实际完成量估算包括处理和加载的ETA。保存JSONL、summary、run_config、runtime_history及报告，更新HANDOFF。交付后暂停跟进，不自动Git提交推送或比赛上传。

## 2026-09-24 01:56 LocateAnything 已完成8条检查，完整412按ID续跑

- 当前新5090 connect.weste.seetacloud.com:28355；runner PID4982、评估子进程5000，logs/locate_5090_runner.pid为准。运行根/root/autodl-tmp/rematch_20260922。01:55:07启动恢复，日志logs/locate_5090_resume_20260924T015507.log；当前评估日志l0_attempt1_20260923T175507354774798.log。不要重复启动。
- 8条真实检查全部解析，首框4/8命中，mIoU0.405226，仅工程检查、不推断全量成绩。合计136.15秒，峰值分配8.92GiB/预留8.94GiB，无OOM。原RGB/完整Query/ground_multi/hybrid/2048/temperature0/TF32关闭，正式重复惩罚保持现有1.0。
- 发现重复框长生成。曾中断全量做有界诊断，原12条完整结果保留，01:56已推进15/412。修改重复惩罚1.1的两条诊断仍约17.5秒且产生98/339个框，未采用为正式配置。固定首条24-token前缀la_flash与sdpa输出完全相同；代码审查未找到明显终止标记/KV错误，但短检查不能证明完整运行无实现问题。按用户要求继续完整评估，不因低准确率提前停。
- 当前约17秒/条，剩余约1小时50分钟至2小时（含处理/汇总余量）；实际用latency_seconds与进度日志更新，不把中断诊断时间混入吞吐。resume记录从12条恢复，模型加载2.78秒。
- 自动串行：剩余L0完整412→L0候选报告→R2优先再L0配对/候选报告；脚本不再等待旧训练/下载PID，明确瞬时错误最多额外两次，确定性错误停止。完成marker跳过已完成smoke。报告路径results/first_batch/l0_candidate_report、r2_l0_candidate_report，已存在则时间戳后缀，看stage_status。
- 本地证据results/rematch_20260922/first_batch/l0_smoke、locate_resume_status.txt、locate_repetition_probe_stdout.txt和locate_backend_probe_stdout.txt。诊断脚本曾复用JSON输出名，最后仅保留sdpa短探针原始JSON；重复惩罚试验和la_flash证据用独立日志/捕获，不能把最后JSON当1.1试验。
- 本阶段仅Locate冻结评估和离线覆盖；不训练、不重新启动七组干预。30分钟跟进已恢复，每次报告数量/ETA/异常。412结束后同步完整结果、核对唯一ID/78组、原始浮点GT、R2=288纠正退化与图组区间；交付后暂停。详细docs/research/2026-09-24-locateanything-execution.md。


### 诊断边界

短后端检查只比较固定前缀，不能据此断言长生成完全正确。重复惩罚诊断不以GT选参数，正式评估配置未改变；全量中断前完成12条保留，未完成的一条由按ID恢复重新执行。最终报告应统计重复框量与实际输出长度，区分首框成绩和去重候选上限，不把重复生成的候选数当目标数。

## 2026-09-24 02:16 定时检查：L0正常推进83/412

- 实查runner4982与评估5000存活，完成83个唯一ID（20.1%）；最新结果距检查13秒，GPU55%，显存11598MiB，无进程失败或OOM，系统/数据盘仍余13/22GiB。
- 82条有效框；1条模型原文为 `<ref>The blue sign on the far left of the bridge.</ref><box>None</box><|im_end|>`，属于模型无框输出，非已发现的解析器错误；按失败计入412分母，不跳过、不改答案。
- 最近20条平均17.28秒；余329条纯生成约94.7分钟，含处理和汇总预计约1小时40分钟，约03:55（UTC+8）附近。未完成全量，不以83条中间结果判断模型价值。
- 正式配置未改，继续L0→自身候选报告→R2+L0报告。证据results/rematch_20260922/locate_status_0215.txt。无需重启、重试或新增GPU任务。


## 2026-09-24 02:46 定时检查：L0正常推进194/412

- runner4982/评估5000持续存活，194个唯一ID（47.1%），最新结果距检查3.5秒；186条有效框，8条原文均为box None并正常结束，作为模型无框失败保留分母，不改解析或跳样本。
- 最近20条平均14.79秒；余218条纯生成约53.8分钟。部分无框样本提前结束使局部均值降低，按近期及约17秒长输出估计含处理/报告还需55–65分钟，约03:40–03:50（UTC+8）。
- GPU51%，显存11698MiB，峰值分配10.77GiB，磁盘13/22GiB余量；无OOM、崩溃或重试。配置/运行脚本不变，自动报告仍待全量完成。证据results/rematch_20260922/locate_status_0245.txt。


## 2026-09-24 03:16 定时检查：L0正常推进299/412

- runner4982/评估5000持续存活，299个唯一ID（72.6%），最新结果距检查8.2秒；289条有效框，10条原文均为box None并正常结束，全部保留评分分母。
- 最近20条平均16.40秒，余113条纯生成约30.9分钟；含处理和两报告预计尚需约35分钟，约03:50（UTC+8）。全量未完成，不报告最终成绩。
- GPU53%，显存13158MiB，峰值分配12.06GiB，系统/数据盘余13/22GiB；无OOM、崩溃或重试。保持原队列自动完成L0和两报告，不抢GPU或更改参数。证据results/rematch_20260922/locate_status_0315.txt。


## 2026-09-24 03:26 本机关机前云端自动推进确认

- 用户要求关机后云端自动完成。实查runner4982父进程1、stdin=/dev/null、日志写云端，已脱离本地SSH；335/412（81.3%），324条有效框，11条无框计失败，无OOM。
- 自动序列已存在并核验：剩余412评估→L0候选报告→R2+L0成对/候选报告→队列退出。远端bash语法、报告工具导入和R2文件检查通过；不重复启动或覆盖运行脚本。预计尚需约25分钟。
- 明确瞬时失败最多额外重试两次，按ID恢复；确定性错误保留日志停止。结果和日志均在云端，本机断电不影响队列。桌面每30分钟跟进仍ACTIVE，但本机关机期间不能保证本地自动检查/通知；恢复后读取完成结果并下载交付，不重跑。
- 此授权沿用Locate范围，报告生成后等待讨论，不自动追加训练或比赛提交。证据results/rematch_20260922/locate_shutdown_autorun_check.txt。

## 2026-09-24 22:15 Locate完整交付准备完毕，复赛测试数据上传中

- 无卡开机后实查云端03:47:50已完成L0 412，03:47:55两报告完成，队列退出；没有GPU补跑。完整预测、配置、runtime、summary及日志已下载。
- L0 266/412（64.56%）、mIoU0.568136、401有效框；R2 288/412（69.90%）、mIoU0.601721、412有效框。R2→L0纠正39/退化61；78图组bootstrap ACC差95%区间[-10.28,0.00]个百分点。11条无框均模型返回None。数据412唯一ID、78图组核对通过。
- L0候选@1/4/8/all=266/297/298/298；R2优先+L0=288/339/340/340。合并上限82.52%，非真实选择器成绩；较R2新增52题/33图组，达到296需87.06%有正确候选题选对。暂保留R2主模型、L0候选，不自动训练或提交，等待讨论。
- 报表工具已明确全量pool与pool_top8、显式cap生效、新增相对primary字段；6项测试通过。修订结果目录带_verified，原报告未覆盖；成绩不变。详细docs/research/2026-09-24-locateanything-results.md及results/rematch_20260922/locate_final_summary.csv。
- 新复赛测试数据本地data/复赛数据集-基于大模型的多模态视觉理解与推理，5690 Query、1295引用图组、6006文件12.227GiB，正在用.work/upload_rematch_test.py上传至/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924。完成以test_upload_complete.json为准；未完成不可宣称可用。当前不做官方推理。
- 云端旧初赛原始测试集此前已清，旧预测runs/official_test_stage2等保留；不得删除city训练/验证或模型。新测试集完成后data/official_test_current.json记录新路径和5690，不能沿用9555假设。

