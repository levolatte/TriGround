# TriGround 视觉 Agent 交接

## 无卡模式下的数据与决策分析（2026-09-30）

用户明确是云机无卡模式开机，当前先剖析数据/根因，不启动模型。新分支对话已读取200题全事件、实际2250行训练内容、4题真实图像与当前源码，结果在F:/AIC/results/visual_agent/capability_rebuild_20260929/analysis_20260930/ANALYSIS.md及JSON。

关键新证据：第一步193/200为inspect；219次inspect中212仅RGB、7涉及IR。26次Depth中15无任何Depth图、20编码未确认（重叠口径）；14次search全RGB，11次旧框context，0新增覆盖。初始/最终覆盖175，55错分25缺池+30池内误选；30里23已定向看过正确框ID，7仅初始atlas可见。100次预算转换属于99题；不能将被迫finish说成模型主动认为证据足够。

1500基础行=900静态finish+300明确用户指令inspect+300观察后finish，另750BBox复习；无自主工具选择/Depth/search监督，1200终局候选行只给finish schema。150取证后替代标签中109指向未inspect的框ID（不代表完全没看到同物体，仍有全局图/atlas）；不能称已监督证据带来的纠正。200题106ID/129图组与训练重合，用于恢复采集，非独立评价，无同接口未训练200对照，净-7不能直接归为SFT效应。

既有未训练C新接口412也有334题预算转换、361首步inspect，说明不是新SFT才产生的问题；两批题不同不能因果比较。新学生460事件说明均空，旧参考818/830说明非空。

下一步应围绕Query条件与对象绑定、盲于GT的真实选工具、IR/Depth有效条件与观察后恢复补教学；先分析再改，不拿同样固定inspect→finish数据继续堆。阶段1/学生200均complete，正式412与阶段2未跑，总账12.22925588/20h。原参照与全部输出不改。


## 分支对话实际核查（2026-09-30）

27135云端stage1_state.json已complete：282/282训练完成，Adapter保存在B/stage1_model/adapter；随后学生200/200亦complete，主管已正常退出，无运行中的训练进程。第二阶段未启动。云端nvidia-smi当前返回No devices were found，与本机无卡无关；后续GPU任务需云机恢复显卡，当前可继续CPU数据与恢复示范。

已有本地student200_raw/score/summary.json真实离线结果：学生145/200（72.5%），mIoU0.622806；冻结直接C152/200（76%），8纠正/15破坏。259工具调用、14搜索、4非法终局（3输入超限+1工具错误）。200题属于训练来源行为采集，不是开发集成绩，也不能与412分数混比。训练后正式412尚未执行，决策修复尚未证明。

统一云账实际已用12.22925588/20h，余7.77074412h。首批教师20题离线接受45个决策（inspect19/finish17/depth7/search2）；第二批及学生真实失败接续仍需完成整理。此次仅状态检查和文档更新，没有启动新模型或修改训练代码。最新状态优先于下方历史“学生200运行中”。


## 第一阶段已完成，学生200运行中（2026-09-29 21:36）

单次云端检查确认：Stage1已完成282/282步，一遍训练成功，train_loss0.2561；GPU训练1.184小时，CPU Processor预检670秒，峰值分配显存20.67GB。Adapter已保存至B/stage1_model/adapter。主管PID150136自动进入stage1_student200，正在用新Adapter跑冻结的200训练来源题，非训练后正式定位成绩；当前GPU0%、显存19.1GB，可能处于模型加载/间隔阶段。尚未读到学生任务结果和主管终态，禁止重复启动。


最新单次云检查：第一阶段推进到178/282步（63%），PID150136仍running；GPU74%、21605MiB。第20步测速15.17秒/步，近期每步约14–16秒。尚无训练完成/定位成绩；模型loss仅作训练过程信息。batch01/02两批50个真实教师episode均完成，review共109个步骤（batch01 48，batch02 61），均为真实图像/工具观察，待GT离线筛掉不支持步骤后导出。


## 已确认实际参数训练

最新单次云检查：stage1已完成46/282优化步（约16%），不再是Processor预处理。前20步303.472秒，均15.174秒/步，按当前速度尚需约1小时；GPU86%、21605MiB。第40步记录loss0.2378，不能据此推断定位收益。PID150136仍running；不重复启动。首批20真实教师episode全部完成，两位分别19/29条步骤review，共48条待离线筛选；第二批30题三位Luna继续。


## 第一阶段训练队列已启动（2026-09-29 20:12）

云端 B/run_stage1.py 已启动，PID150136；root实际读取stage1_state.json确认running/stage1_training，目前在训练入口Processor预处理，尚无20步吞吐或已更新权重的证据。不要重复启动。第一阶段从C独立续训一遍普通CE，随后队列自动生成学生200真实轨迹，供第二阶段恢复示范。

实际训练集2250行：1500基础能力行+750原框复习，1640不同起点/1483图组。基础部分1200起点：静态900（C正确720/错误180）；300真实固定取证起点各两步（观察后C正确150/错误150）。实际工具图RGB600、IR450。固定首步是操作预热，不算自主规划；未达到原3000基础目标，如实使用已有数据先训练。canonical数据为data/capability_foundation_20260929_v2/foundation_decisions.jsonl；prebalance旧稿不训练。

最后已结算预算10.63471214/20h，当前训练另计B/ledger.jsonl。参考已完成225/412，训练后成绩尚无。教师batch01后10已真实完成19动作；前10仍执行。下一批30题按0..9、10..19、20..29分别由step_supervision、teacher_ready、foundation_ready接续，输出teachers/batch02及各自review，禁止读GT。foundation已结束GPU启动职责，仅CPU教师；唯一模型队列仍PID150136。

以下为历史状态，涉及“尚未启动”均已被本节取代。


## GPU启动所有权已交foundation_ready（优先）

root已明确授权foundation_ready在合格修正版真实CPU导出完成后，直接合并foundation+750复习到B/data/stage1_train.jsonl并独立启动B/run_stage1.py。它是当前唯一GPU启动负责人，root/下次心跳不可重复启动，先读其消息及stage1_state.json。预算起点10.63471214/20h；其他两位Luna仅CPU真实教师。启动后才报告PID/实际行数/预处理或梯度训练阶段，不把模板当已运行。

## 参照完成，立即准备开训（2026-09-29 19:36）

B/run_reference.py已complete，GPU空闲。412结果225/412、mIoU0.485008，相对C292纠正12破坏79；750复习Processor也全通过，最长1886。已结算10.63471214/20h，余9.36528786h。不得再跑新参照。

基础数据进一步要求：固定取证首步在输入明确列出待查看IDs/模态（分数未在模型摘要暴露，不能让模型猜score）；终局使用finish-only schema。静态与观察后两组必须各分配C错样本，不能把全部330个C错吃进静态导致取证后全学选C。修正版CPU输出目录B/data/capability_foundation_20260929_v2，进程148416曾启动；以子Agent最后消息为准，root尚未合并/训练。

逐步教师导出器root已补：合理非最终review不受终局IoU影响，但finish仅有离线GT命中才监督；不读GT的教师继续独立看图。新collector已部署training_code。

foundation_ready初稿是“最左ID→照抄ID”的非Query练习，root已明确拒绝纳入训练；初次导出还因KEEP内部alias失败，实际尚无合格基础训练文件。现按真实Query修成GT-free固定选竞争对象、实际RGB/IR取证及离线目标ID监督（C正确优先），加900静态真实选择；不是自主规划示范，不编解释。数据量不足不凑数，约300真取证起点先可用也可开训，不等700全量。工具返回必须真入模。

teacher_ready两文件与step_supervision导出器已部署B/training_code，原参照B/code保持冻结。两位Luna现在分别盲执行teacher_jobs_batch01后10/前10，产物B/teachers/batch01及本地对应；review分别reviews_last10/first10.jsonl。禁止读取train_labels。root待foundation实际行就合并750，启动已上传run_stage1.py；该队列仍未启动。切勿把模板或预处理称训练。

## 用户最新排程：参照结束后训练（2026-09-29）

用户明确“等参照跑出来就开始训练”，不追加参照/提示变体。当前三位Luna/max：teacher_ready完成object_teacher/remote；foundation_ready在B/training_code执行真实基础能力数据构造；step_supervision完成collect_object_teachers逐步监督。先list_agents复用。参照最近371/412，仍独占GPU。

已从冻结B/code复制B/training_code，新增准备/教师/训练代码只放此副本；运行中参照代码不动。最终修复云路径的train/holdout清单及候选已上传B/data。第一阶段训练队列模板B/run_stage1.py本地已写：实际stage1_train.jsonl准备后才启动，从C续训1遍普通CE，之后采200真实学生轨迹作为恢复起点。200题已按seed2032从训练图组冻结（City80/RGBT80/Robo40），未执行。模板尚未启动，不能称正在训练。

第一阶段实际数据尚在生成，750复习已完成；不将全finish或待办jobs算作能力集。下一步直接合并真实能力行+750复习并启动训练，完成前20步读吞吐。必要数据缺口如实交接，禁止为了“训练已开始”先拿无关数据空跑。

## 最近一次状态检查（2026-09-29 19:06）

仅检查一次：smoke8已完整结束，GPU计108.229秒；412参照已写289/412，主管PID145398仍在运行，GPU71%/19041MiB，无重启或新模型任务。已结算总9.83176842/20h，当前412尚未结算，不能当剩余预算最终值。继续等待云端有限队列。

数据子Agent已完成本地train/holdout manifest与candidate cache云路径映射，残留F:为0，数量未变；修正后的本地清单尚需下一次使用前上传（勿覆盖当前412的运行代码/输入）。教师统一输入、remote入口、逐步review导出仍未完成；无新的能力训练数据或Adapter。

## 云端独立运行中（2026-09-29 18:30）

用户要暂时关闭本机。已实际启动专用27135的 `B/run_reference.py`，PID **145398**（B=R/results/visual_agent/capability_rebuild_20260929，R=/root/autodl-tmp/rematch_20260922）。`reference_state.json`/`reference.log`为主管；`stages/c_smoke8/state.json`/`output.log`为当前子任务。最近一次检查：前4题真实模型均FINISHED，尚未评分，不能称能力改善。不要重复启动或改正在运行的B/code。

有限顺序：C权重新接口8题（600秒上限）→同接口完整412（6000秒上限）→离线完整评分→750原BBox复习Processor检查（CPU），不自动训练。若smoke出现输入超限/程序错误/无合法终局则停止；失败不能补KEEP。任务已脱离本机启动，关机会话断开不影响云进程；Luna教师接续依赖本机回来。

工具及控制器核心已部署：薄框/图外标签、框假设与来源坐标、IR线索及RGB确认入口、Depth pair ID/原因、重复观察缓存、最多3对象配对证据、当次搜索所有新图可见、一次明确错误恢复。工具19项CPU检查通过，controller语法通过；真实搜索与决策效果仍待本次模型轨迹。教师端仅部分同步，remote_object_teacher与逐步导出仍未完成，**暂勿用旧教师入口生产新训练数据**。

新数据已冻结：train2871题/2210图组，真实候选2810；61缺候选（主18+专项3上游解析失败、RGBDT40待真实模型/检测）。新增留出120图组共163题，另有holdout_eval120_*每组1题。教师首批20条GT-free jobs已落盘，尚未执行。750原框复习已真实导出并上传；云端所有图片存在，8行实测处理器最长1860 tokens。RGBDT40题160张图已上传B。注意build_capability_data仍需修正部分本地F:/AIC/data路径；root已将750文件中266图路径映射为R/data并验证存在，其他清单由rgbt_teacher_clean收尾，部署前读其最终消息。

当前没有新训练Adapter或新成绩；3000基础能力决策和约1800交互决策均未完成。返回后先读有限队列结果、完成教师同输入接口，再批量真实教学。GPU总授权20h，旧账已用9.80170479h，新实耗B/ledger.jsonl；已合并剩余额度。自动任务已更新每30分钟只查一次。

## 最新授权与实施：能力训练（2026-09-29）

用户已明确Implement the proposed plan，解除旧审计暂停。自动接续已更新ACTIVE/每30分钟；不得再按下方历史暂停文字阻塞，也不得重启旧队列。

新根B=results/visual_agent/capability_rebuild_20260929，旧N=decision_rebuild_20260929仅复用资产。已检查27135为4090空闲、磁盘余82GB。旧预算共用9.80170479h，剩10.19829521h已获准合并；新阶段按B/ledger.jsonl实际累计。GPU模型任务尚未启动。

当前并行：reference_behavior改object_tools/vision_tools；candidate_gpu_completion改object_controller/object_teacher/remote_object_teacher；rgbt_teacher_clean改能力数据与逐步采集入口并冻结120图组留出；root改run_objects/train配置、集成和部署。全部Luna/max。不要重复派工或同时改同文件。先list_agents。

新训练默认C适配器初始化并独立续训；目标基础3000+750复习、交互约1800+450复习，各一遍。纯finish文件不能当能力集启动。实际能力数据未完成，尚未启动新SFT；本轮定位成绩尚无。详见当前PLAN。


更新2026-09-29。最新指示：继续真实工具选择/恢复示范与实验；长任务每30分钟检查，不持续轮询。自动接续任务ID `triground-agent`，仅在完成、失败、显著发现或需用户行动时通知。历史见RESULTS.md、本地结果HANDOFF_history_20260929.md。

## 当前指令：暂停扩实验，彻查根因（2026-09-29 16:40）

**收尾已完成**：所有模型任务结束、GPU空闲。候选主2782/2800、专项191/194，余18/3上游解析非法；T4.546932933/8、Z5.254771857/12。原CPU894行全部City，CUDA399.673秒补1888+4合法行，未再导出foundation、未启动训练。

用户明确要求审计反思。新增教师step、训练和Qwen实验暂停；triground-agent自动任务已PAUSED，旧心跳不再作为续跑授权。已运行candidate_gpu_completion只按原900秒上限收尾并结算，不再导出新的纯finish基础集。reference_behavior与rgbt_teacher_clean转只读行为/数据审计。见[AUDIT_ROOT_CAUSES.md](AUDIT_ROOT_CAUSES.md)，报告区分已证实缺陷和待验证因果。

根Agent已逐条读取真实412轨迹，确认97破坏=80合法错框+17非法；输入图像确实入模。初始atlas截页在412没有发生，不能归因。实际164题旧工具图被淘汰，70题同ID换RGB/IR时旧模态局部图不同时可见；初始候选没有同物体框假设关联。射灯例证实同一灯多个框被当两物体比左右，并将IR亮区误解为亮灯。新两阶段训练尚未开始，基础导出仍全finish，不能将新未训练212成绩称新SFT失败。审计未修改控制器、训练代码或旧预测。

## 当前实施：决策与工具重构（优先于后文）

用户已批准新PLAN，全部开发/教师子Agent只用gpt-6-luna/max。重写的object_tools/object_controller/run_objects、真实教师object_teacher/remote_object_teacher、混合模态prepare_object_assets和基础离线export_object_foundation已落盘。新结果N=/root/autodl-tmp/rematch_20260922/results/visual_agent/decision_rebuild_20260929，本地同名results目录。全部旧队列禁止重启。

**后续数据进展（本次接续）**：recovery01 的11个真实恢复均完成，离线仅3/11命中（直接C同为3/11，0纠正/0破坏），17原始合法决策仅4行进入终局正确集合（1 inspect、3 finish）；6题初始池无覆盖，一次真实search为EMPTY。不能称恢复能力改善。RGBT19均已init而尚无教师step；原接手者退出，已交干净上下文的rgbt_teacher_clean继续实际看图/工具，不重做init。candidate_gpu_completion仍独占有限GPU任务，reference_behavior在完成412行为拆解。RGBDT batch03仅9条盲解已落盘，其中7通过；新增data/rgbdt_accepted_batch03_partial.jsonl，累计40条通过，尚非200；batch08..10未落盘，历史“正在构题140..199”不代表已完成。

**当前状态（2026-09-29 16:05检查）**：run_prepare.py与run_reference.py均complete，旧主管不得重启。主CPU候选894/2800、专项187/194，旧foundation仅752行；RGBT20候选19合法/1上游解析失败。新未训练78及完整412均已完成：48/78（C60）、212/412（C292），17纠正97破坏、覆盖339不变；详见RESULTS顶部。GPU检查为空，无Qwen。

**当前团队与资源所有权**：旧子Agent不在live列表，已按落盘状态恢复三位Luna/max任务。candidate_gpu_completion独占下一GPU阶段：部署本地显式detector-device参数，约10题CUDA小检查后补主/专项缺失候选，总GPU墙时上限900秒并写T账本，不自动训练、不覆盖现有行。reference_behavior只读完整412真实轨迹（含复用78的trace_path）分类定位/工具缺陷，写独立短报告；完成后应接专项20题真实教师。teacher_recovery_continue只读截断可见状态恢复11题recovery01，完成后继续RGBT19题，不读GT/评分。根目录已准备reviewed/teacher_modal20_manifest.jsonl（10 depth_relation、10 ir_complement，GT-free且候选已存在），供下一空闲教师，不保证模态必有效。禁止与candidate_gpu_completion并发启动模型。

**真实结果**：首smoke4为2合法/2输入超限，77.48秒；修过接口后的debug32为C17/32、Agent11/32（27合法/5超限），相对C纠正1题、破坏7题，没有成绩收益。已出现自主search补候选和多对象Depth，但仍重复观察、把相机Depth错用于二维关系。完整输出已保存。debug32之后本地又修了“局部证据替换初始图册”和“输入限额后显式finish机会”，已在首批20教师全部结束后部署，尚未Qwen重测，不混入32成绩；当前数据准备不受影响。教师和基础样本最后一条指令已统一为控制器相同文本；达到6次取证时教师也仅保留finish schema，现已一起部署。工具region描述明确grid:row:col与context:P编号语法。

**当前数据**：data/cloud_manifest.jsonl已有2800不同Query（City1200/597图组，RGBT1000/1000，Robo600/600），云端全部路径存在。RGBDT200个真实训练帧已定点解压；annotation_jobs没有Query。Luna完成batch01..07共140草稿（7题构题歧义），正做140..199，必须交另一位未见GT教师盲解后才加入训练。mainline232专项中27已在基表、11留出冲突，额外194已写本地reviewed/；已验证194项全部图像和原始深度可复用专用云端现有资产，N/reviewed/cloud_manifest.jsonl已上传，无需340MB重复图片包（中断临时包已移除）；最终训练按来源优先纳入，不能覆盖正在用的2800基表。

**真实教师并行**：teacher_batch_first20.json取已冻结512训练池前20题，不读GT。object_controller_impl负责索引0..9，object_tools_impl负责10..19，已完成多条真实inspect/finish/export；检查collaboration.list_agents避免重复。各episode在N/teachers/batch01/<id>，本地镜像同名；候选导出标记candidate_teacher_supervision，离线筛选后才能统计训练有效数。object_data_impl负责RGBDT构题（该阶段允许原标注GT），不能由同一个已见GT教师做盲验。不要等云端模型轮询才做这些独立CPU/教师工作。

**教师后续已分配**：前10完成后object_controller_impl盲解data/rgbdt_blind_batch01.jsonl（20题、只看Query与原图，不读目标标注或构题basis）；后10完成后object_tools_impl接续teacher_recovery_batch.json的11个真实学生前缀（9 UNKNOWN、2协议错误），输出teachers/recovery01。清单仅按事件状态选择、没有GT筛题；init用旧V/manifests/train32.jsonl及V/train32/candidates.jsonl、--resume-prefix与--prefix-events。这些32题本来就是训练来源行为调试，今后进入恢复训练，不再作为独立评价。collect_object_teachers.py离线保留全部结果，仅导出终局正确轨迹中合法的教师动作，不改写失败为成功；工具因果贡献仍须实际行为验证。

**RGBDT盲解结果**：首20均看过真实图，19被盲解教师认为唯一，最终仅15与原目标框IoU≥0.5，已写data/rgbdt_accepted_batch01.jsonl。另5题已由构题教师对照原标注复核：3题因标注边界/群体歧义暂排除，2题重写描述待再次盲解，未改GT或盲解答案。object_controller_impl正盲解batch02/03共38题，不读GT/basis；data/rgbdt_blind_revised_batch01.jsonl是后续2题修订版盲解入口。object_data_impl继续140..199。首15的60个真实图像文件已全部上传专用云端，data/rgbdt_cloud_batch01.jsonl已写入并上传；此清单不含GT。batch02另19条盲解已离线匹配18通过（rgbdt500_303_00000152未通过保留），data/rgbdt_accepted_batch02.jsonl是新增18条GT-free输入，尚未上传图片；累计33条通过，不等于200。

**资源**：27135唯一GPU；46057仅用户新授权只读数据，禁止运行或修改主线。最近账本Z5.25477186/12、T4.43591276/8，2800准备模型阶段已结算；reference_queue已全部结算，新的CUDA候选任务另计；权威账本仍旧V/z_budget.jsonl和t_budget.jsonl，V=R/results/visual_agent/implementation_20260928，R=/root/autodl-tmp/rematch_20260922，PY=R/.venvs/aux_selection/bin/python。当前原生与C模型路径不变；无新Adapter训练完成。

**上一阶段吞吐瓶颈（已交GPU补全，勿重复启动）**：本次单次检查prepare候选228（约1815 CPU秒）、专项候选140（约1181 CPU秒），两主管仍running，GPU在等待专项CPU阶段结束。RGBT20小候选已结束，19完成/1上游解析失败，已让object_tools_impl在恢复批后接这19题。以当前CPU速度，2800大队列在7200秒上限内可能只有部分；保留已算缓存。已为prepare_object_assets.py增加显式--detector-device cuda:0（默认仍cpu、FP32不变），仅本地语法检查，尚未部署或实测。正式参照队列结束或停止、确认GPU无Qwen后，才用小批实测CUDA检测吞吐并按T预算续补缺失候选；不能与自动将要启动的Qwen抢GPU，也不重跑已完成行。

**下一步**：78/412未训练参照已完成，先汇总真实错误与补齐训练资产；共享冻结初始框，模型过程仍不读GT。读准备队列实际完整度，处理真实失败项，优先专项194及RGBDT自然Query和600—1000真实教师轨迹；第一阶段3000基础决策不能只用1000离线finish标签冒充。基础训练一遍后采学生真实失败前缀交Luna接续，再续训交互一遍。train.py已支持--init-adapter与前20步吞吐。所有训练/推理共用新消息构建器。最后固定78与完整412，仅直接C/新未训练/新训练比较，主线296/412单列参照。

**操作入口**：从N/code运行 `PY -m experiments.visual_agent.run_objects --manifest ... --candidate-cache ... --model /root/rematch_models/Qwen3-VL-8B-Instruct --output-dir ... --dino-model R/models/grounding-dino-tiny --sam-model R/models/sam2.1-hiera-tiny`；完整已执行参数在N/prepare_queue/commands.jsonl。本地教师客户端用Python312（有paramiko），cwd F:/AIC/code，进程env TRIGROUND_SSH_PASSWORD来自用户凭据；`-m experiments.visual_agent.remote_object_teacher init|step|export --episode-dir N/teachers/... --local-dir F:/AIC/results/visual_agent/decision_rebuild_20260929/teachers/...`，init另给manifest/cache/sample-id，step给action-file。必须view_image读取实际返回后自行选动作，不用脚本伪造教师。密码不写文件。

长任务30分钟接续已更新；仅完成/失败/重要发现通知，状态不变保持安静。

## 重构前最终状态（历史参考）

**扩样已全部完成，不再续跑**：V/resume_t_expansion_native.py（旧PID87294）状态complete，512/512唯一预测、离线评分和行为审计完整。直接C434/512、原生D353/512，纠正0破坏81（52非法终局、24输入超限、5合法误选）；候选覆盖465未变。全部状态为合法413、非法66、输入超限33。510模型Query解析+2显式Query文本审阅来源保留，2条C/N均正确。见RESULTS顶部。

**用户最新要求：遇瓶颈先充分查论文/GitHub实际源码再改进。** 已完成针对PixelReasoner、Agent-FLAN、Look Less Reason More、LangGraph的定向阅读，直接源码/数据链接和采用限制写入PLAN顶部。不再给同170行加权，不立即SFT，不扩大T4到412。真实轨迹显示“格式/预算失败”与“证据决策不足”必须分开验证；UNKNOWN不能统一标为必须换工具，IR入Processor不代表有收益。

账本已核实Z3.96345162/12小时、T3.21666969/8小时（含UNKNOWN32、final128和首次观察32完整诊断）。当前GPU空闲，没有模型任务运行；本次三个有限诊断累计仍在原0.5小时预留内。512起点不等于512条可验收轨迹；候选池内仅31条C错可纠正，在KEEP≤50%前提下平衡新起点上限62，不能重复行凑200—500。

**CPU恢复任务已完成**：V/run_prefix_recovery_v2.py，状态t_prefix_recovery_queue_v2/state.json为complete，输出t_prefix_recovery_v2/。512真实自主前缀中60条满足首次多ID inspect ERROR恢复条件，452 SKIP；60条实际执行合法观察，240新图全部进入CPU Processor，60/60通过记录的上下文/视觉/终局预留/次数检查。工具27.32秒，Processor约9秒。未读GT、未调用Qwen、未写训练标签，不是自主恢复。首版t_prefix_recovery_queue失败于system.content字符串遍历；已修复并加入实际schema测试，4测试通过，原失败输出保留。本地prefix_recovery_inventory.json和prefix_recovery_processor_check.json已下载。

**UNKNOWN32诊断已全部结束**：V/run_unknown_probe.py（旧PID127550），state complete/inference_complete=true。96分支，direct22/32、IR21/32、RGB21/32（C28/32）；70实际生成全KEEP，26个LIMIT全是输入+512终局预留超过4096，非模型非法输出。IR/RGB各31/32重复已看过的同ID/模态，但每图分辨率可能更高；不能说完全同图或IR无效。两种观察相对direct少1题仅因新增长度超限。数据、来源和窗口限制见RESULTS顶部。本地unknown_probe32_records.jsonl完整96条，score/selection/behavior已下载或生成；不重跑原队列。

**final128已完整结束**：V/run_unknown_final128.py（旧PID128044）complete，96分支93 FINISHED/3 context LIMIT。三臂均28/32、mIoU .740832，与C相同；93个合法全部KEEP，原70条输出逐字不变。剩余city_000162_001三臂仍超限，C IoU也0；没有继续降预算。只解除部分执行阻塞，没有证据利用/规划改善。本地unknown_final128_records.jsonl完整96行与score已保存。

**首次局部观察32已完成**：V/run_first_observation_probe.py（旧PID128480），状态t_first_observation_queue/state.json为complete，输出t_first_observation32/。seed2031从60条首次多ID inspect ERROR/56图组中固定32不同图组，此前无局部图；ID只取原错误请求中的前两个合法ID，无GT选题/选对象。96/96 FINISHED、全部KEEP、0超限/非法；C/direct/IR/RGB均24/32，mIoU .6530919851。IR/RGB新增128张图确实进入模型；程序109.17秒，计账约111.22秒。完整96记录、评分与来源已下载本地first_observation32_*；3个针对性CPU测试通过。旧队列不得重启。

离线审查8道C错误：2道候选池无覆盖，5道池内有正确候选但原请求前两ID未包含它，1道正确候选已观察仍KEEP。该1道city_002579_005是桥上右数第二路灯：实际RGB局部清楚，IR局部无法辨出灯轮廓，全局小图的KEEP/1/2标签重叠遮挡灯。图片保存在本地review_first_observation/；标注遮挡是可见呈现问题，其对选择错误的因果影响尚未实测，不能归为已确定根因。不能由该例推广IR无用。

**下一步（尚未执行）**：审查上述6道覆盖但未纠正的训练题，依据完整Query和实际全局/局部图列出可识别的观察对象、参照与模态条件；先修复有证据的呈现问题并做最小对照，再构造真实对象选择/恢复示范。该6题由GT离线诊断筛出，后续任何定向结果须标记训练失败审查，不能当独立评估或自主选支成功。不得用GT命中ID直接替代模型请求或直接生成首步工具正标签；若当前输入不足以确定应看谁，应保留不确定性与真实搜索需求。暂不自动SFT、不重复UNKNOWN→IR模板。

运行中只检查一次并安静，不轮询、不重复启动；完成后读selection/inventory/score/summary与96实际records，判断首次真实辅助图是否改变正确选择、是否只是KEEP、是否仍有预算失效；脚本合法化和终局改善都不能称自主恢复或自主规划。不得把所有UNKNOWN强制IR，不再对原32 UNKNOWN题叠图或试更小输出上限。

三个诊断主管均已停止，没有当前新主管。MISSING/LIMIT必须保留在完整分母；模型长任务运行中只30分钟一次检查，不重复启动。仅当真实观察与当时可见条件支持时才形成后续工具选择监督。

本次实际执行：在V目录，用PY运行`run_prefix_recovery_v2.py`（已完成）和`run_unknown_probe.py`（已完成）；另启动`run_unknown_final128.py`有限对照；各队列commands.jsonl保存完整模块参数与上限。控制器入口`python -m experiments.visual_agent.probe_unknown_evidence --traces V/t_expansion512/native/traces --manifest V/t_expansion512/manifest.jsonl --candidate-cache V/t_expansion512/candidates.jsonl --model /root/rematch_models/Qwen3-VL-8B-Instruct --output-dir V/t_unknown_probe32 --limit 32 --seed 2030 --max-run-seconds 780`，工作目录CODE。V/PY/CODE为下方列出的绝对路径变量，不能原样把V当路径运行。

本地expansion512_review.json为完整512紧凑轨迹（约34MB，不要整文件输出），expansion512_failure_inventory.json为摘要；云端原图及完整trace仍在t_expansion512/native/traces。

## 历史阶段记录（已结束队列禁止重启）


专用服务器 root@connect.westc.seetacloud.com:27135，RTX4090 24GB；禁止使用主线46057。凭据见本对话用户授权，不写入代码。当前Python REPL会话44132及44532含Paramiko连接c、SFTP s、sh/rp；旧29128传输曾阻塞，不再使用。连接常在空闲后失效，操作前明确重连；SSH channel_timeout=20、SFTP timeout=30。超过1MB的JSON曾传输超时，先远端gzip后下载并解压/检查JSON行数，避免把部分文件当完整结果。

R=/root/autodl-tmp/rematch_20260922；V=R/results/visual_agent/implementation_20260928；PY=R/.venvs/aux_selection/bin/python；CODE=V/code_t_demo_v2。

**T2队列已全部完成**：V/run_t_recovery.py主管18597结束，state.status=complete，44步/2遍完整，训练循环665.97秒、峰值20,467,894,784字节。不要重跑。留出37/50、开发54/78，相对T1均净−3；比直接C更差。完整记录已下载本地t_recovery_records.tgz。

**诊断已完成**：V/run_t_diagnostics.py主管19979全部完成。N/T2在相同170训练状态均完整作答；N参数匹配28/170，T2 58/170；T2 finish46/69、inspect12/69，但Depth动作名0/11、Search动作名0/21。搜索后终局9/9正确匹配，协议ERROR后正确inspect2/20。自主fit46为C20/T2 23，纠正5破坏2、无搜索、最终覆盖40/46未变。先有工具选择欠拟合，不能只解释为自主分布偏移。

**T3已完成并按预设门槛停止**：V/run_t_balance.py，主管21318结束，state.stage=minority_actions_still_unlearned、status=complete，不是执行失败。两遍44步、629.28秒、loss .755007；探针参数44/170、finish32/69，Search0/21、Depth0/11。未运行留出/开发，不再重复此加权路线。原始170行probe已完整下载，t3_probe_decisions.jsonl为1,688,562字节/170合法JSON行。T3数据仍170行/46起点，finish46/inspect69/depth22/search33。

**真实模型审计已完成**：V/run_learning_audit.py主管23193完成48状态，68.72秒。全部动作值token受监督；T2/T3的LoRA A/B均非零，禁用/切换Adapter实际改变概率；首个真实Qwen样本full-vs-sparse logits与CE差均0。T3 Search动作值CE6.15483、其余监督词元CE.268695，正确首词概率.006724；Depth动作值CE3.41951、其余.369872，首词概率.002852。动作值token仅占Search监督约8.06%、Depth14.29%。不是Adapter未生效或稀疏投影错位；提高动作损失权重是待验证假设。完整summary/48行已下载本地t_learning_audit_{summary.json,rows.jsonl}。

**T4已全部完成**：V/run_t_action_loss.py主管23627结束，44步/2遍、623.32秒，峰值20,454,139,392字节。原170状态参数35/170、动作名107/170、finish20/69；Search1/21、Depth0/11。唯一Search参数也匹配，但训练终局明显退步。留出41/50（mIoU .709374，合法48/50，KEEP47）、开发60/78（.660901，合法76/78，KEEP73）；相对直接C为纠正0/破坏1、纠正1/破坏1；相对T2净+4/+6。自主搜索仍0，覆盖44/50和67/78未变。更多KEEP恢复正确初始框，不是已学会搜索或辅助证据推理；不推进T4完整412，不再堆这170行的权重版本。

T4真实失败：city_000943_003五ID inspect ERROR后五ID depth再ERROR；city_000013_055_00000241_003重复三ID inspect；city_000014_052_suppl_00000164_002在Depth UNKNOWN后声称测量支持更远。唯一新增定位纠正city_000696_001跨RGB/IR观察后选1，IoU .0785→.6173，但证据说明也声称深度支持前景，未执行深度测量，不能据此归因模态。留出1/开发3次Depth请求含协议ERROR；开发2次UNKNOWN均直接finish。实际IR入模41题78张、60题117张。

**当前下一阶段：新起点GT-free采集**。V/run_t_expansion.py有限队列，状态V/t_expansion512/queue/state.json。冻结512题/385图组（来自expansion_seed2029_v1），已验证与T250/train32/开发所有图组无重叠，2048图片路径均存在。依次C1500初始框（1800秒上限）、Query解析（1200）、CPU DINO初始候选（7200 CPU秒）、原生D/sft/latest自主观察（2400 GPU秒）；模型阶段全部记T，任一步不完整/非法解析明确停止保留，不偷偷删题或补框。暂不自动训练、GT选支或旧脚本示范。原主管25082已结束失败：C框512/512完整、Query解析512条中510合法、2非法。两条relation_type输出on/near，不属于固定枚举；源文件及失败队列保留。单次协议反馈修复已执行，C1500仍重复on/part，主管26535在首条停止、未尝试第二条；原query_repair/尝试与预算保留，不重复模型重试。

现使用显式Query文本审阅的两条训练预处理修订，独立文件query_reviewed/query_info.jsonl与reviews.json：木栏红警示牌为sign/railing/other/single；水边最右灰鹤为crane/water/other/single。没有读取图片或GT，没有改共享解析器；512中510模型解析+2人工审阅，不能算512自动解析成功。后续自主控制器指标须保留512完整分母并标明两条上游审阅来源，必要时单列510/2。

当前主管改为V/continue_t_expansion_reviewed.py，状态t_expansion512/queue_reviewed/state.json；仅继续CPU候选→原生D/sft/latest自主轨迹，不重跑基线/解析。原queue及queue_continue失败状态保留不覆盖。后续只每30分钟检查。

采集后先区分真实失败与可识别监督条件：无候选/序数不足/图像可见遗漏才支持搜索理由；不能仅按事后GT成功半区监督首步搜索。UNKNOWN不能变成确定近远，IR图进入不等于利用。扩样可用轨迹仍待实际生成和筛选，512不是512条训练成功轨迹。默认保留原token-mean，T4权重不升格默认。长任务每30分钟只查一次，不重启。

T4完整原始JSON归档已下载并解压：本地t4_records.tgz为4,006,730字节；t4_records/probe/decisions.jsonl完整170行，trained_holdout50/traces完整50份、trained_dev78/traces完整78份，均可解析。首次SFTP部分下载已从实际偏移补齐，不留作完整结果。大文件先gzip，发生部分下载时先核实远端大小及本地偏移，完整后才验证/提取。

监督审查另发现：6个新增search首步实际观察历史为空，仅全局三图和2—8个候选；成功区域经训练GT事后选支，尚未验证当前图像是否支持搜索必要性。3 EMPTY回放有真实负观察。T2 170状态互不相同；T3 147不同状态/23重复组，组内标签一致，无同状态冲突。不要把可识别性风险说成在线GT泄漏。

队列：训练→在原始T2的170行上跑同一探针→若Depth/Search动作名仍合计0，则status=complete、stage=minority_actions_still_unlearned，停止不跑留出；若有改善，则同留出50/开发78，与C/N/T1/T2匹配评分。不能把条件停止当实验失败，也不能把合计大于零当定位改善。每30分钟查一次，不持续轮询、不重复启动。

后续扩样清单已在本地F:/AIC/results/visual_agent/expansion_seed2029_v1准备：512题/385不同图组，排除T250/train32/开发图组及已知重用/主线诊断，完全无GT选题。文字代理标签camera105/ordinal118/identity332可重叠，未证实模态有效性。只生成清单，尚未云端预测/候选采集/训练；不要冒称512条真实轨迹。T4已完成，扩样准备队列见上。

探针按全170行统计目标动作名、去除合法可选evidence_note后的参数匹配、原始全字段匹配、finish ID；按action/origin/label_source分组。它只诊断训练状态模仿能力，不是泛化或自主Agent成绩。fit46与原始C比较，GT只用于离线评分。

T1主管16266已完成，旧OOM error字段已移至prior_errors。旧主管12362/13872/15373、CPU采集17583已停止；四个CPU搜索分片全部完成。不要重启旧队列。

## 已测结果

最好完整412仍直接C1500：292/412，mIoU0.613628；尚无新版Agent/T完整412成绩。

v5.1快速十组78题完成，A/B/S/C/D：原生61/57/60/54/54，C-LoRA62/62/60/57/59；直接C60。原生D没有实际搜索。

T1：40不同起点103决策，两遍26步，训练循环398.96秒，峰值20,402,885,632字节。Adapter V/t_aligned/train/adapter。相同D/sft/latest输入：

| 完整分母 | C | 原生N | T1 |
|---|---:|---:|---:|
| 留出50命中 | 42 | 38 | 40 |
| 留出mIoU | .724620 | .655237 | .696248 |
| 留出合法终局 | 50 | 43 | 48 |
| 开发78命中 | 60 | 50 | 57 |
| 开发mIoU | .663965 | .549359 | .623788 |
| 开发合法终局 | 78 | 62 | 73 |

T2留出37/50、mIoU.633889、合法49/50，对C纠正0破坏5；开发54/78、mIoU.605237、合法75/78，对C纠正3破坏9。无主动搜索，候选覆盖44/50和67/78未变。相邻inspect→depth留出0/开发2，不能把消失当学会相关性。动作序列inspect→finish在留出46/50、开发70/78；策略转向固定观察后结束。IR真实入Processor留出44题81张、开发66题121张；唯一UNKNOWN仍直接finish。

T2新增纠正例：city_001897_013后排最左玩具车、city_hehe_178_000002_025_00000079_009最右杆红横幅；新增破坏例：city_000014_052_suppl_00000164_001最左撑伞人、city_001897_002左数第二顶灯。均是RGB/IR cross后换ID，不能仅凭动作认定IR有贡献。完整评分及behavior JSON已下载本地t_recovery前缀。暂不推进T2完整412，不继续堆训练轮数或改提示。

留出T1对N纠正3破坏1，三个纠正均是非法输出恢复合法KEEP；开发纠正8破坏1。inspect→depth留出21→4、开发36→6，但仍有无关Depth/非法动作、无主动搜索。训练fit40诊断C20/T1 21，也未学好纠错样本。不能称彻底修复。

评分V/t_aligned/{score_holdout50,score_dev78}/summary.json；行为recovery_{holdout50,dev78}.json；training_fit40_summary.json。均已下载本地t_aligned前缀。开发T1推理曾与CPU搜索采集重叠，耗时不算干净加速比较。

## T2真实数据

原始200训练起点脚本轨迹V/t_demonstrations/observations_v2，工具无GT。真实逐候选RGB/IR观察882分支branches_v1；GT仅离线选择实际执行分支。旧man→mouse错误搜索正标签已剔除，真实失败历史保留。T1对齐数据V/t_aligned/export/train.jsonl。

30条初始不覆盖GT的训练题，离线生成GT-free清单V/manifests/t_train_uncovered30.jsonl。真实RGB/IR四半图搜索完成：26可搜、4非single跳过；208次（149 OK/59 EMPTY），308个新目标观察分支。合并V/t_demonstrations/search_branches_v2，图片仍在search_parallel/part0..3或search_branches_v1，不能删除。

仅6/30补到正确框，全部来自RGB搜索；不能编造IR优势。类别经审查与Query一致（两斑马、狐獴、白色动物、顶灯、灰鸟）；GT选支是离线专家示范，不是自主成功。记录search_coverage_offline.json。

协议探针protocol_feedback_v2：200题，138次真实三ID跨模态请求ERROR、62候选不足SKIP，错误动作不监督。v1真实记录完整但汇总SKIP空event时报错，保留；v2修复后已重跑。

T2数据V/t_recovery/export/{train.jsonl,inventory.json}：46不同起点170决策，20 KEEP；20协议错误恢复、6新搜索成功、3 EMPTY恢复回放。动作69 inspect、69 finish、11 depth、21 search。错误/EMPTY仅作真实历史输入；回放显式标注，不能冒称原模型连续成功。监督JSON统一去掉可选evidence_note，原始记录保留；这也是数据变化，不能把收益全归因搜索。46有效起点尚未达到200—500有效轨迹目标。

相关导出/采集11项测试通过；实际导出0拒绝。最新代码collect_protocol_feedback.py、collect_search_branches.py、export_recovery_demonstrations.py均在CODE。本地同名目录维护。

## 资产与预算

基座/root/rematch_models/Qwen3-VL-8B-Instruct；C1500=R/results/next_stage_20260925/c_phase2/checkpoint-500；DINO/SAM=R/models/grounding-dino-tiny、sam2.1-hiera-tiny，显式CPU。T新建语言q/k/v/o LoRA r32/alpha64、LR1e-5、micro1/acc8、2遍、BF16基座/FP32 LoRA与优化器、SDPA、梯度检查点4096。

train.py仅在监督动作预测位置做词表投影，Transformer输入完整；loss/梯度等价、step1保存恢复step2已实测，不重复。首次全序列投影OOM保留且计入预算。

T训练/推理sft/latest：全局3图各200704像素，最近工具图总602112，6次工具、3次搜索、4096上下文。N匹配对照同配置，不与Z fast混算SFT收益。

账本V/z_budget.jsonl上限12h、t_budget.jsonl上限8h，累计elapsed_seconds/3600。扩样C框与Query解析完成后Z3.96345h、T2.02133h（含失败的单次协议修复），以实时账本为准；含失败、加载/CPU预检的保守进程墙钟，不能重复加队列总耗时。

V/manifests：train32、dev16groups(78题/16组)、city412、t_train200、t_holdout50、t250，图组隔离。T候选V/t250/candidates.jsonl，离线GT V/scoring/t250_gt.json。开发候选R/results/aux_selection_refine_20260927/city412/candidates.jsonl，GT在该实验manifests/scoring/city412_gt.json。在线无GT；非法终局IoU0，不补KEEP。

## 下次定时接续

1. 一次检查queue状态、进程与日志尾；仍运行则安静退出，等下次30分钟唤醒，不持续轮询。
2. 失败查根因/预算，必要修复，不盲重启、不重算已完成阶段。
3. 先核对t_expansion512/queue_native_resume队列，不重复启动；完成后读完整原生512逐题轨迹，离线评分/分类，并审查可识别动作条件。GT只能用于离线标签和评分，不能把分支回放写成自主成功。
4. 46个有效训练起点仍不足200—500目标。新512采集结束才决定最小示范构造/新Adapter训练；先修复有证据的数据条件问题，不再新增同170行的权重版本。T4未改善直接C，不扩完整412。
5. 更新RESULTS/HANDOFF；完成本轮或预算耗尽后暂停triground-agent。通知仅限实质变化。

本地代码F:/AIC/code/experiments/visual_agent；结果F:/AIC/results/visual_agent/implementation_20260928。不修改主线A/B/V、不切共享分支、不提交比赛。
