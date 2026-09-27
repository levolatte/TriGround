# 复赛新实验执行记录

## 2026-09-23 14:25 无GPU现状核查：训练完成，三图无净增，Locate加载失败

- 云机可SSH，nvidia-smi返回No devices were found。主runner10663、夜间20977均已退出，无运行中训练/评估；不启动GPU任务，不重启旧夜间脚本。定时任务30读取时已为PAUSED，本次保留暂停状态，仅更新无GPU约束。
- M2于04:02:10完成928更新，训练8851秒（约2h28）；两轮验证04:12:16/04:22:08完成，均完整412且无解析失败。epoch1=283/412、mIoU0.596284；epoch2=288/412（69.90%）、mIoU0.601820。R2对应283/288，固定epoch2净增0，因此seed2027成对复验未触发。
- R2→M2固定epoch2纠正11条（10图）、退化11条（11图）。图组bootstrap ACC差95%区间[-1.97,+2.07]个百分点；mIoU仅+0.00010。暂保留更省算力的R2为主基线，不追加无净增的三图直接定位训练。
- Locate于04:22:56在smoke加载失败，ImportError明确缺cv2/decord/lmdb，未产生完整L0成绩。有限重试按确定性依赖错误立即停止。该失败发生于本次无GPU之前，不能归因于GPU当前不可用；先前环境验收没有覆盖完整processor依赖，是准备遗漏。
- overnight_finished仅为调度结束，不代表Locate成功。完整M2预测、指标、夜间报告JSON/CSV/Markdown、失败日志、stage_status及决策文件已同步F:/AIC/results/rematch_20260922/。
- 全部R0/R2两轮/M2两轮候选合并诊断上限313/412，兑现B+8需94.57%选择率，仍高于90%门槛；不据此直接训练选择器。下一步优先修复并完成Locate评估，视真实候选覆盖决定；实例辅助成对实验仍为后续候选。
- 数据盘余22GiB，系统盘余7.7GiB。已在Locate独立环境安装opencv-python-headless4.11.0.86、decord0.6.0、lmdb2.3.0，三项导入通过；AutoProcessor.from_pretrained(local_files_only=True)真实CPU加载成功，返回LocateAnythingProcessor。未改Qwen环境、未加载GPU模型。GPU恢复后仍须完成真实模型加载及8条检查，不能把CPU通过写成L0推理通过。证据locate_dependency_fix.txt、locate_cpu_processor.txt；结果报告results/rematch_20260922/overnight_report/report.md。

## 2026-09-23 02:44 用户查询：M2训练正常，夜间续跑等待中

- 02:43:33实查M2为432/928（46.6%）；最近411→432用201秒，约9.57秒/更新，余496步纯训练约79分钟，预计04:03左右训练结束。之后两轮412验证耗时待实测，再由夜间续跑20977自动启动Locate检查/完整评估。
- 主runner10663与夜间进程20977均存活。夜间日志为WAIT existing main pid=10663，属于按序等待，不是卡死；尚未发生重试。原Locate等待14954已按计划替换，不恢复。
- 全量m2_train.log扫描未发现Traceback、OOM、non-finite或FloatingPointError；日志暂未输出结构化loss/grad行，不能补造数值。GPU利用率75%，显存22553MiB；数据盘余23GiB、系统盘余7.7GiB。
- 当前完整成绩仍为R0 273、R2第一轮283/第二轮288（均412分母），M2尚无验证结果，不能判断新增模态收益。证据results/rematch_20260922/status_overnight_current.txt。

## 2026-09-23 01:44 用户关机：云端自主顺序执行及有限重试已启动

- 用户授权本机关机后云端继续，已新增并部署tools/continue_rematch_overnight.py，nohup PID20977，logs/overnight_runner.pid及logs/overnight_runner.log。实查进程存活、正在等待原主runner10663，现有M2不被中断；01:43实查57/928（6%）。
- 已确认原Locate等待队列14954只有sleep子进程，主动结束该等待壳，交由新续跑统一调度；不要将14954退出当故障重启，不要启动第二份队列。
- 固定顺序：等待当前M2训练+两轮完整验证/报告；若主runner可恢复失败则最多额外两次从已有完成标记/训练checkpoint恢复；之后Locate8条检查→完整412→汇总。已完成阶段不重算，推理按ID恢复。
- 若M2固定epoch2比R2净增>=4条，自动用seed2027运行RGB/三图成对两轮训练、各轮412验证，再汇总；结果不足4条则不盲目追加，记录overnight_decision.json，明天根据实例监督/候选覆盖证据决定。不自动训练未经候选门槛证明的选择器。
- 新阶段每项最多2次额外重试，间隔60/120秒；OOM、非有限、关键梯度/断言、依赖错误、解析文件损坏或空间不足会停相应支线，不降低像素、不改QLoRA、不静默跳样本。每次尝试独立日志，训练恢复保留优化器。重试仍有失败会保留原因，并尽量继续独立专家/汇总步骤。
- 输出results/first_batch/overnight_report（JSON/CSV/Markdown）、overnight_decision.json、overnight_finished.json；finished只表示队列退出，不代表全部方法成功，须检查日志与完整预测数。
- 本地语法检查、3项行为检查（重试有上限/确定性故障立即停/完成阶段跳过）通过；云端py_compile通过，实际主runner和续跑存活。定时任务30已更新为优先检查夜间续跑，存活时不抢GPU或重复恢复。
- 本机关机不影响云端nohup；报告留云端，明天本机恢复后同步。EGM维持退役，Locate沿用已完成镜像权重。

## 2026-09-23 01:38 R2完整结果已出，M2已接续

- R2于01:18:31完成928更新，两轮纯训练7574秒（约2h06m）；独立epoch1/2验证分别7m42/7m47，01:34完成并自动启动M2。
- 原始浮点GT完整412结果：R0 273/412=66.26%，mIoU0.569217，解析411/412；R2 epoch1 283/412=68.69%，mIoU0.593835，解析412/412；固定epoch2 288/412=69.90%，mIoU0.601721，解析412/412。当前最佳为epoch2，但只是一组seed2026，不宣称跨seed稳定。
- R0→R2 epoch2纠正33条（22图），退化18条（16图），净增15条/+3.64个百分点。78图组bootstrap10000次，ACC差95%区间[+0.25,+7.01]个百分点；mIoU差+0.0325，区间[0.0077,0.0566]。此为开发集内重采样，不等于新seed复验或官方提升。
- 完整逐样本、配置、trainer_state、阶段表已下载本地。重算报告及CSV/JSON：results/rematch_20260922/rgb_lora_report/。该报告同时给出R0/两个R2轮次候选并集309/412，仅作诊断；达到288+8需95.8%选择率，超过90%门槛，不据此立即训练选择器，任务模型来源候选仍须按图交叉预测。
- M2于01:34:01启动；01:37:45为20/928（2.2%），前20更新198秒，约9.9秒/更新，余纯训练约2h30。尚属启动窗口，下一检查用稳定连续20次更新修正；完整三图验证耗时待实测，不照搬RGB两次15分钟。后续自动两轮412验证→汇总→Locate8条检查及完整412。
- 实查M2显存22553MiB，主runner10663/Locate队列14954正常，数据盘约23GiB可用。EGM退役。未覆盖运行脚本或启动额外GPU任务。
- 现场证据status_20260923_0136.txt、m2_start_progress.txt。此前复用检查动作覆盖了status_20260923_0104.txt，其现内容为01:36，不可当01:05原始快照；01:05进度保留于本交接记录。

## 2026-09-23 01:05 定时检查：R2完成89%

- 01:05:01实查R2 830/928更新（89.4%），最近811→830用149秒，约7.84秒/更新，剩余98更新纯训练约13分钟；保存/加载与两次412独立验证另暂估15分钟，预计约01:33前后进入M2，须以后续实际日志为准。
- 主runner10663、Locate等待队列14954均存活；第一轮checkpoint-464保留，尚无R2独立验证预测或完整成绩。训练未被中断，未覆盖脚本。
- Locate完成标记仍存在，等待R2/M2结束后评估；EGM保持退役。GPU占用19469MiB，数据盘余23GiB、系统盘余7.7GiB。当前无需要修改队列的异常。
- 本地证据：results/rematch_20260922/status_20260923_0104.txt。下一次优先读取R2两轮独立结果与M2实际启动及吞吐。

## 2026-09-23 00:33 定时检查：R2第二轮，Locate权重已完成

- 00:32:58实查R2 594/928更新（64%），约8.2秒/更新，纯训练余约46分钟；之后epoch1/2两个412验证暂估合计15分钟（含加载），再自动M2。主runner10663与Locate队列14954仍活跃。当前无R2完整验证分数，不以loss下降代替定位收益。
- 首轮checkpoint-464已保存，trainer_state确认epoch=1.0、global_step=464，目录362.21MiB。最近440/450/460步loss约0.554/0.562/0.557，grad_norm有限；日志未发现Traceback/OOM。trainer_state已同步本地results/rematch_20260922/r2/checkpoint-464/。
- Locate国内镜像下载成功：日志明确LOCATE_MODELSCOPE_WEIGHTS_READY，.download_complete存在。下载PID16464正常结束，不是故障；不要重启下载。logs/locate_download_modelscope.log与stage_status.tsv已同步本地。L0仍等待主runner完成，尚无推理成绩和可靠耗时估计。
- GPU显存占用19469MiB；数据盘余23GiB、系统盘余7.7GiB，满足预留要求。未重启或覆盖任何运行脚本，EGM保持退役。
- 检查原始证据results/rematch_20260922/status_20260923_0032.txt。下一次继续查R2完成/独立评估，主队列结束后确认Locate正式加载与8条检查。

## 2026-09-23 00:01 用户同步：EGM正式退役，Locate改国内镜像

此状态覆盖下方旧EGM诊断建议及下载线路快照。

- 已读[EGM退出记录](F:/AIC/docs/research/2026-09-22-egm-retirement.md)。EGM在本任务当前配置下淘汰，不重新下载、不重跑、不再安排此前16条高分辨率诊断。仅保留E0历史预测、指标及清理记录供报告使用。实查云端models/EGM-8B不存在；清理记录报告释放16.48GiB。
- 用户已将LocateAnything切换为国内镜像直连。实查下载PID16464存活，最新日志logs/locate_download_modelscope.log；logs/locate_prepare.pid已指向新进程。不得恢复旧平台/备用下载，不覆盖当前脚本，不重复启动。00:01尚无.download_complete；两个safetensors仍有aria2控制文件，当前文件长度不能当作准确下载完成量。
- 主runner10663、L0等待队列14954仍存活。R2在00:01为362/928（39%），约8秒/更新，余纯训练约1h20，保存与两次412评估另计约15分钟；随后自动M2。没有新完整模型分数。
- 定时任务30已同步EGM退役与Locate国内镜像直连约束，保持ACTIVE和30分钟周期。主线R2/M2对照及后续按实测信号决策保持。

## 2026-09-22 23:40 下载状态更正

备用线路随后连8MiB请求也发生TLS握手失败，PID15446已退出；官方ModelScope同名仓库返回404。现将同一个小块续传脚本切回平台加速线路，去掉额外URL参数，重新启动PID15573，日志改为`logs/locate_download_platform_ranges.log`。下载未完成，速度与稳定性仍待实测，不能报告备用线路修复成功。L0队列14954仍等待R2/M2；下次检查须先看下载PID/日志/.rangepart增长，若失败只处理专家下载，不影响主训练。

## 2026-09-22 23:36 继续执行：主训练正常，专家下载线路处理中

- R2于23:35实查173/928更新（18.6%），约8.08秒/更新，纯训练剩余1h42；随后两次412独立评估预计约15分钟，再自动M2。runner10663与Locate等待队列14954均存活。三图正式压力已通过，不改像素或量化。
- Locate的aria2分段与curl整文件均遇TLS断流，不能误报已解决。已结束该下载树15331，改为8MiB HTTP Range分段，使用独立.rangepart文件，不拼接aria2预分配文件。新PID15446，入口code/scripts/download_locate_ranges.sh，日志logs/locate_download_ranges.log，进度须继续实查。备用线路仅因平台先前反复低速断流而启用。下载完成由原队列自动接续，训练不受影响。
- Locate输入接口补充模型目录的batch_utils/kernel_utils导入路径，并显式关闭TF32；4项本地测试和云端py_compile通过，尚未实际GPU评估，不声称L0完成。
- EGM只读审查未发现可实证的提示、坐标或解析接入错误；412条均唯一bbox_2d且无截断。当前602112像素下的低分保留为实际结果，不推广为该模型能力上限。
- E0框面积/GT中位比4.43；273错中262条面积>GT两倍，181条仍包含GT中心。按GT面积三等分，E0命中0/138、27/138、112/136，R0为55、91、127；因此不能把全部低IoU都称为选错对象。同图其他已标注目标能匹配38个错框（26个同类），只证明部分实例错误。
- 后续唯一低成本EGM诊断候选：固定16条小目标E0错/R0对，仅恢复checkpoint原生分辨率设置，测框尺度是否改善；这是条件诊断，不能当完整成绩。尚未排入GPU队列，不打断R2/M2。当前E0+R0上限277仍不足以训练选择器。
- 结果与成对报告已在F:/AIC/results/rematch_20260922/；30分钟任务id30已恢复ACTIVE。旧48小时任务不恢复。

## 授权与固定协议

用户于2026-09-22批准实施完整复赛计划。旧C4/T4/W4和48小时任务不恢复。当前仅保证4090D，后续一台L40；第二台不作为依赖。无自动代码提交、推送或比赛上传。

- 数据：新3707训练/412验证，对应681/78图像组。原始浮点bbox统一评测，SFT取整标签只用于训练。
- E0：冻结EGM-8B，8条检查后完整412，max_new_tokens4096。
- R0：原生Qwen3-VL-8B-Instruct RGB，完全一致的RGB SFT用户提示，128生成token。
- R2/M2：同原生权重语言q/k/v/o LoRA，r32/alpha64/dropout.05，LR1e-5，AdamW fused，wd0，linear/warmup0，clip1，BF16/SDPA/TF32false/matmulhighest，batch1/acc8，两轮928更新。各轮完整412独立评测，主对照固定epoch2。
- 像素：每图max602112/min200704，长度4096，不packing；4090D三图OOM就等待L40，不自动降低像素或改QLoRA。
- L0：LocateAnything-3B独立环境冻结推理，第一有效框计单模型，其余仅作真实候选。
- 信号后续：M2-R2>=8且覆盖至少6图优先seed2027；4–7边界信号仅复验；方向冲突最多第三seed。实例辅助R-CONT/R-ID各3707样本464更新，后者927条替换为ID任务，LR3e-6，同R2初始化。真实候选cap8、IoU.9去重，先测oracle；增量>=20且至少8图、达B+8所需选择率<=90%才训练选择器。
- GT候选只作条件诊断；实际候选不按GT排序。多正例选择损失，无正例仍保留评测分母。涉及任务模型训练候选须用按图交叉预测。

## 执行入口

- 云端：`connect.westb.seetacloud.com:15483`，工作根`/root/autodl-tmp/rematch_20260922`，Python`/root/miniconda3/bin/python`。密码仅交互输入，用户消息中可查。
- 本地SSH：`C:/Users/ROG/AppData/Local/Programs/Python/Python312/python.exe .work/remote_rematch.py ACTION.json`；系统默认Python311未安装Paramiko。
- 原生权重：`/root/rematch_models/Qwen3-VL-8B-Instruct`；EGM在运行根`models/EGM-8B`。
- 计划运行入口：`code/scripts/run_rematch_first_batch.sh`；结果`results/first_batch/`，阶段表`logs/stage_status.tsv`，每阶段独立日志。
- 恢复只重跑同一入口已失败/中断的阶段；先核对实际PID，禁止重复启动。成功marker跳过已完成阶段，推理按ID恢复，训练使用原优化器状态恢复。
- 平台网络加速优先，备用线路仅在平台失败/显著低速后使用。

## 当前实测状态

22:06现场检查：4090D GPU空闲，EGM权重完成，EGM尚无结果；系统盘约30GiB空闲，数据盘约16GiB空闲。

原生8B下载已启动；新工具正在补齐原始GT、推理续跑与数值设置。尚未开始正式8B LoRA，不能将入口测试当作实验成绩。

30分钟跟进自动化已创建（id `30`），每次必须报告真实阶段、完成数、实测ETA和下一阶段。固定流水线在云端自主衔接，跟进负责检查结果、同步本地及后续分支决策。

## 首轮验收与交付

保存真实配置、逐样本输出、固定epoch2/最佳checkpoint两张表、成对变化和78图组bootstrap。候选上限与实际选择分开。正式推理依据复赛当前清单，不默认旧9555条；用户每日最多两次上传，生成ZIP后由用户提交。

进度、测试、首次真实EGM结果及8B容量检查将在下方追加。

### 22:22真实启动

EGM压力8条完成：2命中、8解析、0截断，26.355秒纯生成，峰值allocated17,741,934,080字节。完整412评估22:19启动，runner10663。原生平台下载发生连接中断，约8分钟只传112MiB，转备用+aria2分段；不影响正在跑的EGM。现有评估11项本地测试通过，云端无pytest，已py_compile与真实GPU检查。

## 用户临时关机：保留云端后台任务，暂停聊天跟进

用户要求快速停止对话，等回来再检查。EGM全评与first_batch runner10663继续云端nohup运行，后续自动R0→原生压力→R2/M2；已部署正式训练与报告脚本。最新原生下载改为官方ModelScope完整独立副本，脚本`/root/autodl-tmp/rematch_20260922/scripts/download_rematch_native_modelscope.sh`，日志`logs/native_download_modelscope.log`，PID查`logs/native_download.pid`。原HF/aria片段保留在模型目录旁`.hf_partial`，不混用来源；ModelScope探针8MiB/0.92秒、config一致。下载完成需实查marker。Locate独立环境已就绪(4.57.1/flash-attn2.8.3)，CPU准备进程11328会等待原生下载完成再下Locate权重。L0评估工具和后续排队脚本已在本地，但尚未上传/启动；回来补齐。EGM跑完后再同步最新版eval工具(多了limit和JSONL换行边界)，不覆盖运行中的EGM入口。13个eval测试、此前23个组合定向测试通过。自动化id30按用户临时暂停，回来明确继续时再恢复。

## 2026-09-22 23:28 恢复跟进，正式RGB LoRA训练中

- 用户已回来并要求继续。定时任务id30恢复ACTIVE，每30分钟中文进度。
- 已完成新412验证：E0 EGM139/412=33.74%，mIoU0.346712，解析412/412；R0原生RGB273/412=66.26%，mIoU0.569217，解析411/412。均原始浮点GT，原始JSONL/summary已下载至`F:/AIC/results/rematch_20260922/`。
- 成对R0→E0错变对4、对变错138；并集最多277/412，仅多4条，当前不满足训练选择器门槛。暂不为EGM追加训练，正在做一次有限只读接入核查。
- 原生8B官方ModelScope权重23:01完成。4090D三图压力2更新通过：16微批、22.54秒训练，288/288 LoRA张量有梯度，loss0.9621有限；两次重载固定2条预测一致。R2/M2保留原602112配置，无需等L40。
- runner10663于23:11进入R2训练，23:22为79/928，约8.09秒/更新，纯训练剩余约1h55，之后两次412验证约15分钟，再自动M2。准确进度看`logs/r2_train.log`，勿重复启动或覆盖训练脚本。
- Locate原平台HF下载约20分钟仅142MiB且反复断流，已停止下载进程12921并改备用分段`code/scripts/download_locate_segmented.sh`，新下载PID14953，日志`logs/locate_download_segmented.log`。隔离环境此前已就绪；待权重完成。新GPU队列PID14954/`logs/locate_runner.pid`，入口`code/scripts/run_locate_after_first_batch.sh`，等待第一批runner结束后自动smoke8→L0完整412→冻结专家汇总。勿与R2/M2争GPU。
- 最新eval工具已在E0/R0完成后同步（limit/config与JSONL换行恢复补充），正式训练脚本未覆盖。汇总`results/rematch_20260922/frozen_initial_report/`；候选上限不等于实际得分。



## 2026-09-22 侧对话：用户要求退出 EGM，已清理

EGM 快速接口核对未发现明显提示/坐标/解析错误；139/412、对 R0 仅补 4 条。按用户要求已删除云端 EGM-8B 权重及专用下载残留，释放 16.48 GiB，数据盘剩余 30.56 GiB。不要重新下载或重跑 EGM。保留 E0 预测、完成标记及共享代码，当前 R2/M2 runner 不变。分辨率和推理后端未完全复现官方，不能称为彻底否定模型。详情：F:/AIC/docs/research/2026-09-22-egm-retirement.md。

## 2026-09-22 侧对话：LocateAnything 下载加速

用户要求检查并加速下载。原平台 ranges runner PID15573 约11分钟仅落盘56MiB，反复180秒超时。实测六个代理节点真实8MiB权重分片，香港IEPL01最快约1.28MiB/s；台湾0.91、新加坡0.57、美国0.99、德国0.95MiB/s，日本握手失败。
已新建专用 Mihomo 实例（17900代理/17901本机控制），未更改原共享代理选择。停止原下载器及其子进程，保留 .rangepart 续传；新脚本 code/scripts/download_locate_fast.sh，PID16213，logs/locate_prepare.pid 已更新，日志 logs/locate_download_fast.log。下载仍双文件并发，分片URL加入offset/end参数避免范围缓存混淆。原GPU队列和训练未改。不要重复启动旧平台下载器。
节点测速：云端 logs/locate_node_speed.json；本地 results/rematch_20260922/locate_speed_tests.txt。

加速后23:54核实：两个权重的有效 .rangepart 合计152MiB（原先56MiB），新增96MiB已实际落盘；代理连接确认走香港IEPL01。持续速度仍有波动，25秒窗口可能没有完整8MiB分片完成，因此暂不承诺稳定ETA。


## 2026-09-22 23:58 侧对话：LocateAnything 改用国内魔搭（取代上条代理方案）

用户明确要求换国内源。已停止代理下载 PID16213 及其子进程，旧未完成目录保存在 models/LocateAnything-3B.hf_partial_1790092678；没有混用未完成分片。新脚本 code/scripts/download_locate_modelscope.sh 从 nv-community/LocateAnything-3B 下载完整副本，进程内取消全部代理变量。PID16464，logs/locate_prepare.pid 已更新；日志 logs/locate_download_modelscope.log。完成后检查文件大小/权重头并写原路径 .download_complete，原 L0 队列无需改变。不要重新启动海外下载。
启动约28秒实测双分片总速度约11MiB/s，较大分片ETA约13分钟，小分片约8分钟；这是初期估计。结果与来源保存在模型目录 download_origin.json / download_files.json。现有 Qwen 训练未改。
