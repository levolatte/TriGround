# 联合三模态对象证据与任务生产执行记录

## 最新范围

用户批准的新方案取代尚未启动的 D/T 数据配方：先标对象在各模态中的证据和可选框，再派生定位、候选关系、跨模态对应任务。新训练分支名为 G/U。旧 C 已完成，不再重跑。旧 248 条候选、219 条自动暂通过及人工 CSV 全部保留；人工验收仍为 0。

当前云机无卡，仅执行本地 CPU 数据工作。没有启动新训练，也没有改动已交付的 C/M2 提交包。用户报告两包官方分数均为 0.7144；本地 C=292/412 的小幅增益未转化为官方涨分。

## 已实现的入口

在 `F:/AIC/code` 使用 `.venv/Scripts/python.exe`：

1. `tools.prepare_multimodal_evidence`：对象记录校验、无 GT 盲审任务、按题号合并独立框、通过人工/批次验收记录后导出原生训练来源记录。
2. `tools.render_object_evidence_review`：离线三路原图/提议框对照页面、空白人工决定 CSV、逐项计数。新增目录输出，避免覆盖已有人工意见。框核对项与独立自然 Query 分开统计。
3. `tools.prepare_next_stage_data`：保留旧 C/D/T 兼容，增加 G/U 的固定配额、跨模态坐标输出、打乱编号后的候选答案、RGB 增强与实际份额报告。

截至接口检查：`tests/test_multimodal_evidence.py` 与 `tests/test_prepare_next_stage_data.py` 共 **18 项通过**。包含一次真实 G/U 各 8000 次呈现的 CPU 格式转换对照。测试中的图像是构造输入，不代表真实标签或 GPU 压力检查完成。

未标注 IR/Depth 框不会从 RGB 框复制。输出 IR 框时提示明确坐标属于 IR。外部复查记录不能导出训练。预览记录保留 provisional，不能绕过现有 RGBDT 人工验收入口。

## 首两批生产

输出根：`F:/AIC/results/multimodal_data_20260925`。

- batch001：原 `pilot_screened_annotation_manifest.jsonl` 已筛选训练池的首 10 个 ID，由 `annotation_queue_luna` 制作。
- batch002：`supplement_pilot_v1/eligible_pilot_supplement_v1_manifest.jsonl` 已筛选训练池首 10 个 ID，由 `rgbdt_luna_annotate013` 制作；使用 supplement_pilot_v1 数据根。
- 这 20 组首先验收对象工作流；尚不能声称已经完成 200 组按互补方向的选样，更不能把普通可辨认样本自动归为“自然模态增益”。后续依照证据产出安排 80/60/40/20 选样方向。
- 两名 LUNA-MAX 制作；SOL-MEDIUM `rgbdt_blind009_resume` 独立盲审。每 2 图组落盘并交接。同题三路输出共用一次看图过程，各坐标分别保存；普通题不额外做多轮 RGB-only 对照。
- 生成者只将 `review_partNN/blind/tasks.json` 交给复核者。`private_index.jsonl`、对象记录、红框预览均不得交给盲审者。复核不确定就返回 null/歧义，不能猜框。

## 对象格式与审核

每个 JSONL 行含 id、source、split、scene_id、images、depth_policy、objects、queries、relations、revision。split 为 train 或 external_review。images 保存实际 RGB/IR/原始 Depth 路径；数值深度另外提供仅供查看的 depth_visual。

objects 中保存 object_id、role、semantic_label、boxes（rgb/infrared/depth；未知为 null）、extent（whole/part/group）、confirmed_modalities、evidence、uncertainties。confirmed_modalities 仅表示制作代理做过目视核对，不等于人工验收。

每对象最多两条实质不同的自然 Query，保留 origin_query_id。只有可确认的对象结构才能形成 Depth 对应任务；未知数值方向不能形成近远标签。关系记录含候选对象 ID、答案对象 ID 和依据；近远关系还要求 depth_order_verified。

导出训练的人工审批格式为以图组 ID 索引的 JSON：status=human_accepted 或 approved_batch，record 指向人工记录，tasks 列出已验收的 bbox/cross_bbox/relation/augmentation。自然题还需独立盲审通过或逐题人工接受。关系还需逐关系接受状态。制作代理不得自行填写人工接受字段。

```powershell
.venv/Scripts/python.exe -m tools.prepare_multimodal_evidence prepare-review --evidence <证据JSONL> --output-dir <新复核目录>
.venv/Scripts/python.exe -m tools.prepare_multimodal_evidence merge-review --evidence <证据JSONL> --private-index <私有索引JSONL> --reviews <盲审JSONL> --output-dir <合并目录>
.venv/Scripts/python.exe -m tools.render_object_evidence_review --evidence <合并证据JSONL> --output-dir <新人工包目录>
.venv/Scripts/python.exe -m tools.prepare_multimodal_evidence export --evidence <合并证据JSONL> --approvals <人工批次审批JSON> --output-dir <训练来源目录>
```

人工预算按 100 对象对应、100 自然题随机验收、100 困难/增强、100 外部复查、100 扩量随机抽检安排。暂不受旧四类各 25 条限制；自然题 95/100 正确唯一、明确错对象/多解不超过 2/100，系统性错误仅暂停相应任务类型。随机抽样与争议仲裁不可混算。

## G/U 转换与后续训练约定

第一阶段 8000 次呈现：City 2400、RGBDT 3200、RGBT 1600、Robo 800。第二阶段 4000 次：City 3000、RGBDT 1000，全部定位。G 为直接定位，U 第一阶段最多 25% 候选关系、15% 跨模态框，可靠辅助题不足回到定位并报告实际份额。两组相同图组、原 Query 来源和呈现顺序，辅助题由相同对象场景派生。

增强仅改变 RGB，默认最大 20%，只在有人工接受的辅助依据且答案仍成立的记录中采用。brightness=.35/.55/.75 或 blur=1/2，单次只用一种；每条被增强来源保留至少一次原图呈现。G/U 共享抽样和参数。首版排除两组中对应辅助题呈现的增强，因为普通定位题的增强验收不能自动批准另一道关系题。没有已验收增强记录时实际增强为 0，不为比例凑数。

City 原始 Query/答案保持不变，新增辅助记录附着到原原生记录的 task_pool；不能把 City 原生对话误作裸 Query 再重写。外部来源沿用真实两图或三图与各自 Depth policy。RGBDT 未知数值编码用 sensor_linear_20000；Robo 使用已上传的固定 8bit 视觉清单，不把原 raw16 直接转 RGB，不编米制距离。

人工验收和真实小批次样本就绪、GPU 恢复后才执行 16 微批/2 更新、梯度/截断/保存重载检查。随后 G/U 各从 M2 初始化新优化器：1000 更新 LR5e-6，继承适配器但重置优化器再500更新 LR3e-6；每500独立评估，固定1500主对照。LoRA rank32/alpha64/dropout.05，累积8，BF16/SDPA/TF32关闭，602112 像素不变。

现有 C 是历史普通续训参照。训练 shell 尚不启动；真实混合数据/GPU 检查完成时衔接现有训练包装和每500评估，无需新框架。首训目标约800图组/2500–4000任务，不把20组反复采样冒充这一规模。

## 后续交付与尚未完成

先交首两批可视人工包与实际合格任务/小时，再推进200组试产、按任务类型抽样验收、首训规模。当前还没有新人工接受记录、完整500条人工结果、G/U模型或新官方包，不能宣称完成。

后续评分完整City412、隔离人工外部100题、固定自然诊断子集四种真实模态输入；相对同机M2/C较强者净增≥8且覆盖≥6图组、外部下降≤3题才优先第二seed。净增20是目标非承诺。官方无GT，不虚构分数，不自动提交。

## 补充实现与候选池

City 已支持 `export --city-native <原生训练JSON>` 按 origin_query_id 附着已验收 task_pool，保留原对话/图像路径；不存在或重复原ID会报错。接受候选任务后，可从自然Query和至少两个已确认RGB对象直接派生候选编号任务，不必重写一句描述。

新增 `scripts/run_multimodal_gu.sh` 已通过 Bash 语法检查，未运行GPU；需提供验收清单、真实GPU压力完成记录和外部复查输入。顺序G三次City评估/最终外部评估→U同流程→报告，延续有限瞬时重试和检查点恢复。正式运行前仍需针对真实数据做GPU验收。

已整理 screened_training_pool.jsonl：97组/97序列，首20分配给两制作批。亮度统计只作排队线索，不证明IR增益。正在从本地ZIP剩余未占用训练序列抽160个代表GT帧，目录 supplement_trial_v2；与原及追加外部保留序列隔离，筛查后才可加入制作池，尚未产生新Query。此步骤不是全盘解压或重下。

## 22:59 本地候选池补齐与阶段缓存修复

补充提取已完成176个新训练序列代表GT帧，108组通过原有缩略图相似候选筛查、68组隔离待原图核对（不能称68个确切重复）。外部保留序列交集0。与原97组合并后为205组/205序列，足够200组试产选图；只是候选池，不是205组已标注或已证明IR/Depth贡献。清单：results/multimodal_data_20260925/screened_training_pool.jsonl。无重新下载，原ZIP和原划分保留。

修复prepare_next_stage_data阶段资产命名碰撞：深度/增强图缓存现在放在各分支各phase独立目录，防止准备phase2覆盖phase1引用的图像。18项测试仍全通过，新增断言检查生成下一阶段后前阶段图像字节不变。这是新混合数据转换入口问题；C仅使用原生City图像，未使用该外部深度缓存，不以此推断旧模型成绩原因。

## 首批生产效率调整

两制作代理首轮实际看图后在目标范围判断上停留较久，尚未按预期及时输出JSONL，不能按已派20组计作完成。已将当前小任务明确收缩为已看过的2组先保存→导出安全任务→交SOL→结束小任务，后8组由后续短任务接续。batch001曾主动中断长推理后按原已看ID恢复写盘，未重做图像、未生成虚假辅助框。框范围不确定保留null/理由，复核阶段解决。此时尚无两完整10组的可靠吞吐，不能给全量ETA。

## 2026-09-25 新对象证据首6组已落盘，可视人工包可打开

已实际保存：batch001两组、batch002两组、root_probe两组，共6组/10对象/6自然Query、10个RGB框、5个IR提议框、1个Depth提议框，另2条关系记录；均未人工接受。快照 evidence_snapshot_first6.jsonl，HTML results/multimodal_data_20260925/human_preview_first6_v1/review.html，含三路原图/框、24个核对项，可页面填写导出CSV。这不是随机100题验收样本，也不把24项说成24条新Query。root_probe（068/392）是主Agent制作的两组真实流程样例，已交SOL独立盲审，共4个RGB/IR输出case，reviews路径 root_probe/review_part01/blind/reviews.jsonl，未完不报通过。两LUNA也按各safe目录交审，查实际文件按ID合并。batch002已被要求为三路均有独立框的参照对象cleaner补一条Query（target_object_id=distractor1），保留原工具GT；不能把新增人框称原GT。待完成小任务后按未做ID短任务接续首20组，不重复root_probe ID。

首轮制作过久已收缩为每2组完成即交接，GT不清留null。后续优先使用205候选池中真实辅助线索，不只给原GT目标出题；已确认参照对象也能出实质不同Query。所有制作框仍是provisional，保持独立复核和500人工分项计划。当前无卡不训练。


## 2026-09-25 23:12 定时核对：6组真实记录，9个盲审输出case已排队

实际master文件仍为batch001=2组2Query、batch002=2组2Query、root_probe=2组2Query；合计6组10对象6Query，人工0。无已落盘reviews.jsonl，故不报自动通过数。已补batch002不可变快照evidence_for_review_2310.jsonl及review_2310/blind/tasks.json（2个RGB case），交SOL接在root_probe4和batch0013之后，合计9个分模态输出case，非9条自然Query。SOL每图保存，各safe目录reviews.jsonl；未完成组不合并作已通过。

已请两LUNA只收尾当前已开始图组/cleaner新Query后结束，再用短上下文同模型任务按未做ID接续，减少累积历史负担。结束前不并发重复ID、不强行启动第四子Agent。batch002新增cleaner Query应单独增量复核，旧快照不覆盖。205候选池无须再下载/提取，旧248数据保留。当前云端无卡、无训练推理待跑；制作与审核耗时仍未形成稳定两批10组数据，整轮ETA未知。下一步优先收独立复核→合并逐模态结果→更新可视包，不继续无关接口开发。


## 2026-09-25 23:42 短上下文制作已接续，首4项盲审合并通过

master实际6组/10对象/7自然Query（batch002新增cleaner q02已保存），RGB框10、IR提议框5、Depth提议框1；人工0。制作Agent口述“3对象”不准确，batch002文件实际4对象，以文件为准。root_probe两Query的RGB/IR共4输出case已独立复核且IoU全部>=0.5、无歧义，合并输出root_probe/merged_review_part01，4/4通过；这是2条Query，不是4条新Query，不表示自然IR收益或模型成绩。其余batch001 3case、batch002原2case、cleaner新3case由SOL顺序接续，共12case中4完成8待复核（有新落盘再更新）。可视包human_preview_first6_v2，6组7题25核对项，旧v1和人工CSV未覆盖。

旧两LUNA已完成并收尾。成功新建fork-none短上下文制作代理 /root/object_short003 与 /root/object_short004，模型均gpt-6-luna/max；分别batch003处理016/020、batch004处理017/018，每组即保存、两组后prepare-review并只发safe任务路径给SOL，再结束。不恢复旧annotation_queue_luna/rgbdt_luna_annotate013并发处理同ID。SOL仍 /root/rgbdt_blind009_resume，最多两制作一盲审，当前三者在运行。输入为各batch generation_inputs.json，含明确绝对路径及原GT；已记录时间要求用于产能统计。

205候选池不重下，不追加无关开发。下一步按新任务实际完成时间衡量换短上下文是否有效；合并增量盲审、更新可视包。GPU无卡不训练，整轮ETA尚未知，人工验收0。


## 用户查询时最新盘点：10组/14对象/11Query，下一批按辅助证据筛选

batch003/004各2组均完成，新增4组未确认IR/Depth框。总master记录10图组、14对象、11自然Query；包含辅助框的仅4图组，IR框5、Depth框1。不能把10组全算有效多模态题。首200组试产按已检查/制作记录尚余190组；人工验收0。

当次读取盲审9/16个分模态输出case，6通过3需复查，7待审；非9条Query。013组合工具被仅以“两个物体”判歧义，已要求SOL仅复核语义标志并保存独立correction文件，原审核不覆盖、GT/IoU未透露；计划允许明确组合目标，不能仅因并列物体拒绝。其他边界差异保留。

短任务batch004有完整计时，两组3分15秒；batch003报告41秒只涵盖保存阶段不能当全制作耗时。整体ETA受有效线索比例、审核及人工验收制约，尚不外推。旧248/219保留，新自然题部分复用不能直接相加。

已继续两LUNA（object_short003/004）分别制作batch005(217/304)、batch006(114/026)，输入各generation_inputs.json。改为先检查辅助证据，亮度排序仅线索；无任何确认辅助对象则保存证据与no_confirmed_auxiliary_evidence、queries=[]，不再普通RGB-only句子凑数。有可确认参照对象可最多补1对象/Query，未知深度不做近远。零case不送审。SOL原队列继续，两生成一盲审，无GPU任务。


## 2026-09-26 用户反馈过慢：连续领取10组、每2组交审

实查当前14图组、19对象、14自然Query，只有5图组有独立辅助框；人工0。首200组检查/记录还余186，不能称14组均为有效三模态训练数据。batch006两组完整制作7分22秒，先前batch004为3分15秒；有效标签产率尚不稳定，不以单批推全量ETA。

定位到流程问题：两组即结束增加主代理重新派工空档；低线索图进入精标；batch005可辨辅助对象的文字与辅助框全null需区分可见但不可定位和漏标。两生成代理已改为每人连续领取10组（object_short003=batch007，object_short004=batch008），每2组落盘、安全盲审交接后直接继续，10组后结束。SOL连续消费安全任务。仍最多两LUNA-MAX加一SOL-MEDIUM，不改模型等级、不放宽标签可信度。

先辅助证据筛选再精标；清楚轮廓独立估框，不追求像素级边界；全图及最多一次必要裁剪仍不清则记录null/原因，不研究不明类别、不写RGB题凑数。batch005补框如需修订须独立revision及增量盲审，原记录保留。当前云端无卡，不训练。


## 2026-09-26 00:14 盲审合并与连续批次

实际14图组、19对象、14自然Query；batch005独立revision补3个辅助框后，有辅助框图组由5增至7。这是补齐标注，不是新增图组或模型收益。24个分模态输出case中16已审、14通过、2需人工复核、8待审；组合工具语义correction只更正歧义标志，保留原框及历史。人工0。

已合并到results/multimodal_data_20260925/merged_20260926_0014，人工预览human_preview_14groups_v3/review.html包含46核对项（非46条新Query），旧CSV未覆盖。两LUNA实际运行batch007/008各10组，每2组交审后继续，无需等下一心跳。SOL原队列连续消费，新增batch005仅3个辅助case，避免重审RGB。尚无完整连续10组产率，全200组及有效任务ETA不能可靠估计；按已记录图组仍余186。云端无卡无训练任务。


## 2026-09-26 00:44 连续制作已提速，30组记录及盲审积压

最新快照30图组/37对象/24自然Query，17组有辅助框；49个分模态输出case中29已审、24通过、5需复核、20待审，人工0。此为制作和独立标注复核，不是训练结果，不表示辅助模态相对RGB有增益。merged_20260926_0044及human_preview_30groups_v4已生成；可视包89核对项非89Query，旧CSV不覆盖。首200组按检查记录还余170组。

batch007完整10组15分36秒、6Query/13case（4组无辅助证据Query空），object_short003已接续batch009独立10组输入；batch008截至快照6/10组，object_short004继续余4；SOL连续审核。两制作一盲审。最近半小时新增16组，生产速度改善，但审核积压20case成为新增瓶颈，不以制作完成冒充可训。以近期16组/30分钟粗算余170组制作约5.3小时，仅作条件估计；样本难度变化、审核及人工验收未包含，不能承诺整批交付时间。未知深度不近远、辅助不清保留null、单对象首轮1Query避免复核积压膨胀。无卡不训练。


## 2026-09-26 01:14 40组记录，深度框复核通过率需关注

快照merged_20260926_0114：40图组47对象29自然Query，22组有辅助提议框；60分模态case中37已审30通过7待复核23待审，人工0。RGB已审20/17通过、IR10/9通过、Depth7/4通过；仅小样本标注复核，不能当模态收益。Depth三条未通过保留，继续独立审核，不用同位置代替轮廓确认。

batch008已完成10组6Query，用时48分20秒；与batch007的15分36差异大，当前生产速度尚波动。object_short004已接续batch010独立10组，object_short003继续batch009（快照6组），SOL持续审核，旧ID不重派。当前首200组还有160待检查记录。最近约1小时新增26组，若维持此产率制作剩余约6小时；审核和人工另计，不承诺整批可训时间，有效任务数不能由处理图组数代替。云端无卡，无GPU训练。人工预览仍human_preview_30groups_v4；本次只增量合并，避免每心跳重复渲染全部图像。


## 2026-09-26 01:44 已记录54组，盲审积压下降

merged_20260926_0144：54图组61对象39自然Query，32组有辅助提议框；86分模态case中73已审、57通过、16待复核、13待审，人工0。RGB29过5待复核5待审；IR17/5/5；Depth11/6/3。上一轮40组至今新增14，待审23降至13。首200组仍余146待检查，按最近约26组/小时，制作粗估5–6小时，独立复核和人工验收另计，不承诺可训时间或600–1000任务规模。

batch009完成10组5Query，46分27秒；batch010完成10组8Query，24分51秒。两LUNA已分别接续batch011/012各10组，仍每2组落盘交审直接继续。batch010 part01/02曾先合并4组再拆分，现已各2组，后续明确安全任务首次就独立2组，禁止覆盖在审任务。复核发现305把坐姿写成站立，相关三个模态case未通过，保留记录未入训练；已普遍提醒制作前核对姿态/左右等描述，不把审阅答案泄露给生成者反复凑通过。

新增human_preview_54groups_v5/review.html含150核对项，非150自然题也非随机验收100题。旧CSV均保留。增量快照脚本.work/snapshot_object_evidence.py复用既有merge_review，仅汇总有效master/revision与独立reviews，减少主代理重复手工统计；无训练逻辑改动。云端无卡，不训练，不重复旧实验。


## 2026-09-26 02:14 74组，现有119输出全部独立复核

merged_20260926_0214：74图组81对象54自然Query，47组有辅助提议框；119分模态输出case全部已审，93通过26待复核，人工0。RGB46过8待复核，IR30/10，Depth17/8。按自然Query去重：46条RGB框通过，38条RGB及至少一路辅助框均通过，9条三路均通过。只是独立标注对应暂通过，不表示必须靠辅助模态解题或模型获得收益。

batch011完整10组用时12分52秒、10Query；batch012完整10组14分17秒、5Query，口述2025年份是笔误，按实际文件/当前时钟2026记录。两LUNA/盲审一度都结束，出现等心跳派工空档；已预分配各连续两批：object_short003=013→015，object_short004=014→016，每批独立10组，每2组保存交安全任务后继续，第一批完成直接接下一批，20组结束。SOL此次范围四批，到达即审，暂空等新安全路径，四批完成且审空再结束。仍两制作一盲审，不加并发，不让审核接触GT/解释。

首200组剩126待检查。最近约1小时新增34组，制作余量条件估计约4小时，审核/人工另计，不承诺600–1000任务或可训交付时间。人工预览仍human_preview_54groups_v5，旧CSV保留；下次较大增量再渲染。当前无卡不训练。


## 2026-09-26 02:44 首批过半：114组，57条RGB加辅助对应暂通过

merged_20260926_0244快照114图组121对象85自然Query，78组有辅助提议框；192输出case中179已审139通过40需复核13待审，人工0。RGB67过13需复核5待审，IR46/17/5，Depth26/10/3。自然Query去重57条RGB+至少一路辅助框均通过、15条三路均通过；非模态增益证明，不当正式训练数据。

两代理013→015和014→016各20组均完成，实际约26分40和22分09，预分配减少等待，最近半小时新增40组。已接续object_short003=017→019、object_short004=018→020，各批10组每2组落盘交审后继续，20组结束；SOL原队列审完直接消费新四批，不读GT。首200剩86组，按最近速度制作约1–2小时，但样本难度波动，此估计不含全部复核与人工验收，不承诺600–1000任务。

人工可视包human_preview_114groups_v6已生成成功，114组85自然Query共316核对项，非316训练题；旧v5/CSV保留。原proposal在框范围/歧义不通过则保留复查，不能以数量增加视为已验收。来源已超过100候选组且有57条对应暂通过，未触发“100组不足20证据包”停止来源条件；仍未证明自然互补与模型收益。云端无卡，不训练、不重复旧包或旧实验。


## 2026-09-26 03:17 154组/当前全部复核，最后46组已分配

merged_20260926_0317：154图组161对象111自然Query，104组有辅助提议框；253输出case全已审，199通过54待复核，人工0。RGB96/15、IR69/23、Depth34/16（通过/待复核）。本轮新增40组，制作与审核均继续推进，任何未通过不作可训标签。

最后首200检查记录所需46组已按全局未完成ID排除后预分配：object_short004=021→023→025（10+10+6），object_short003=022→024（10+10）；两者实查running，SOL在原队列清空后继续这五批，五批制作及复核结束不自行扩到800。final_trial_assignments.json保存分配，205池余5备用。batch017/019两批已完成并合并，不受迟到完成消息影响重跑。

若近期40组/30分钟维持，剩46组制作粗估40–60分钟，独立复核及人工验收另计。200是检查/证据记录目标，不等于600–1000合格训练任务；收尾要按可派生真实任务统计缺口，不同义改写/重复呈现凑数。达到100条自然题自动合格后固定随机抽100用于人工验收，争议处理单列。当前RGB96条过尚不足100，不提前称正式自然验收包已齐。人工预览仍114groups_v6，旧CSV不覆盖；200完成后生成试产质量及正式分项抽样包。云端无卡不训练。


## 2026-09-26 03:45 首200制作与319输出复核全部完成，准备人工抽样

最终merged_20260926_trial200_final：200组209对象138Query，137组有Query；319输出已审252过67待复核，人工0。RGB121/17、IR92/25、Depth39/25（通过/待复核）。自然Query去重106条RGB+至少一路辅助过、25条三路过。trial200_capacity.json：121直接定位+131跨模态框候选=252任务，候选选择素材仅8条/多对象9图组，远未达到600–1000；不得重复/同义改写/增强计新增凑数。详报docs/research/2026-09-26-object-evidence-trial200.md。

停止自动扩800组。object_short004已转制作正式natural100人工抽样包（最终121池，seed2026）；object_short003制作correspondence100（131独立对应池，seed2026）。两任务输出human_acceptance对应目录，未生成核验前不称就绪；人工0，旧CSV不覆盖。只前200人工预算，困难增强/外部/扩量另300尚未齐。SOL已收尾本轮盲审，不再派GT生产。下一步检查两包并交付，之后按质量缺口补竞争对象/关系；云端仍无卡不训练，G/U未开始。


## 2026-09-26 用户授权继续新增，人工明天集中做：已开始City多对象扩充

用户明确“继续新增数据集，人工标注明天一起标”。覆盖此前等待人工才继续制作的限制：继续CPU候选/对象证据和盲审，人审明天统一，未验收仍不得训练。新重点City多对象、已有RGB-T与RoboRefIt辅助线索，不仅扩大RGBDT单目标。

已从用户新标签原ZIP读取3707训练清单（不使用旧class_name/negative判负），选40个至少3独立原框训练图组，20个深度有效区域多对象候选+20个低RGB亮度IR候选；这些筛选分数不证明模态增益。原40图组397Query/397框均保留，不计新增。用无卡云机SFTP实际下载120原图至data/city_object_extension并PIL核对，按原City映射生成40深度预览；没有GPU操作。候选/就绪清单与sync_summary在results/multimodal_data_20260925/next_sources/city。

City制作4批各10组：object_short003完成correspondence100后已唤醒接batch001→003；object_short004 natural100完成后接batch002→004（已发具体任务）。每图实际确认2–4个竞争/参照对象，RGB框/Query原样并留origin_query_id，辅助框独立估计，不明null；不将已有其他框自动当负例，不新编数值近远。每2组交safe盲任务，待SOL独立复核，来源city和旧RGBDT批目录严格分开。

new_source_screening（SOL-MEDIUM fork-none）正在整理RGBT/Robo现有文件可用性和各10候选，缺图返回精确同步列表，不重复下载、不让关键词筛选当真实证据；已要求尽快落盘结束让第三槽回独立盲审。盲审仍复用rgbdt_blind009_resume，仅读safe/原图不看原标签。

人工对应100包已完成：human_acceptance/correspondence100_seed2026，100唯一项/IR69 Depth31、400预览图、空白CSV，源131池seed2026，人工0。自然100包仍制作中不可伪称齐全。旧CSV保留，明天一起交人工，不反复催用户。完整200试产报告仍有效，未扩量训练，G/U未开始。


## 2026-09-26 04:19 City40制作已落盘，RGBT/Robo各10接续

City next_sources/city/merged_20260926_0419快照40图组77对象73原Query记录，36组有辅助框，155输出待独立盲审（RGB73/IR56/Depth26），未见reviews落盘不报通过，人工0。代理口头总数与文件可能不同，使用快照为准。第三槽源筛查已结束，rgbdt_blind009_resume实查running，已发明确City首批safe任务路径要求逐2组落盘；只读原图Query不见GT。

自然100及对应100人工包均实际完成并核验：natural100_seed2026源最终121抽100，correspondence100_seed2026源131抽100（IR69 Depth31）；CSV全空，均非全部500，按用户明天集中人工。旧CSV不覆盖。

RGBT10候选20图已由云端精确SFTP同步并PIL校验，cloud_train6000同步核对其中4组在原6000、6组为同官方train新补，未重新下载图包。各图原训练标注一并整理以支持多对象，输入next_sources/rgbt/batch001/generation_inputs.json。Robo10图本地RGB/固定深度/raw齐，共47原训练标注可供挑选对象，输入next_sources/robo/batch001/generation_inputs.json，单位unknown不近远。原Query/框复用不得算新增描述。

object_short003已接Robo10、object_short004完成City后接RGBT10，均每图确认2–4竞争/参照对象、每对象1原Query带origin_query_id，独立aux框不明null；缺模态不补假图。每2组safe交SOL后继续，标注暂存待验收，GPU未启。来源候选/同步完整不等于质量已通过。新多对象任务耗时尚不稳定，ETA未知；继续CPU工作不等人工。


## 2026-09-26 04:44 新来源首批全部盲审，继续Robo并诊断低通过来源

真实快照各在 results/multimodal_data_20260925/next_sources/{city,robo,rgbt}/merged_20260926_0444。
- City：40组77对象73原Query，155输出全部审核，107暂通过48需复核；RGB62/73、IR31/56、Depth14/26。
- Robo：10组33对象33原Query，49输出全审，43暂通过6需复核；RGB31/33、Depth12/16。
- RGBT：10组26对象26原Query，46输出全审，26暂通过20需复核；RGB13/26、IR13/20。
这些是标注审核，不是模型成绩；原Query复用不是新增文本，人审均0。RGBDT首200的252过67复核保留，不能与旧248重复相加。

object_short003（LUNA-MAX）已接Robo batch002→003共20新图组，从本地完整训练标注与已下载2000图子集选样，排除batch001，场景均衡，每2组保存并送safe任务；rgbdt_blind009_resume（SOL-MEDIUM）等safe路径逐批独立审，不接触GT。object_short004（LUNA-MAX）暂做RGBT/City失败原因诊断，最多6代表图，输出next_sources/source_quality_audit_20260926.md及分类JSON；不修改GT、不重审到通过，完成再恢复制作。三槽已运行，无重复ID派发。

Robo首批10组制作15分26秒，仅供下一20组约30–45分钟制作粗估，审核及人工另计；总体完成时间未知。用户明天集中人工，不催今晚验收，继续独立CPU工作。已完成natural100与correspondence100空白人工包保留，未宣称500项全齐。云机无卡，未启动训练。

## 2026-09-26 05:13 数据进度与辅助增量纠偏

Robo实际快照 next_sources/robo/merged_20260926_0513：30图组90对象90原Query，118输出，113已审100暂通过13需复核5待审，人审0。RGB78通过7复核5待审，Depth22通过6复核。仅9组有辅助框，首批10组5组，新20组仅4组有辅助框；不能把纯RGB新增量当模态能力数据。已通知object_short003完成本批后不盲目扩大，汇报辅助产出，后续先筛Depth边界及竞争对象证据。

新增来源总计80图组（City40/Robo30/RGBT10），189条原Query记录，319输出中314已审233暂通过81需复核5待审，人工0；与RGBDT200的旧结果分列，不把复用原Query称新自然文本。

object_short004完成source_quality_audit_20260926.md/.json，六例提示源Query/框疑似错位、辅助提议框偏移及模态轮廓不清混合存在，未改GT。已接下一有界任务：先核实RGBT官方pth框格式/尺寸/模态坐标，排除当前xywh转换假设错误，不凭变量名断言源标签错；随后已有City40中未制作原对象最多10图组辅助证据补充，city/aux_extension001独立保存，仅真实清楚IR/Depth有增量才制作，不重做旧失败Query。SOL继续安全盲审。三代理实际运行，旧文件保留。

Robo20组制作已落盘，剩审核5输出（快照时），预计数分钟；接下来的源格式核实及辅助证据筛选ETA未知，不用先前纯制作速度承诺。人工明天统一，未启动GPU或训练，继续CPU。

## 2026-09-26 05:17 Robo盲审收尾
Robo merged_20260926_0517 已118/118全部审核，105暂通过13需复核，RGB83/90、Depth22/28，人审0。新来源80组189原Query共319输出全审，238暂通过81复核，无待审。SOL等待City aux_extension001的safe任务。

## 2026-09-26 05:45 继续辅助证据优先制作

RGBT格式已由官方loader核实xywh，本地归一化正确，当前10图对RGB/IR尺寸与metadata一致，六低IoU例Query和框均对应源记录；未发现坐标转换根因，不能直接宣布源GT错误。报告next_sources/rgbt_annotation_format_audit_20260926.md/.json。
City aux_extension001/merged_20260926_0545：已落盘4组5对象5原Query（属于已有City40，非4新增图像），11输出，4已审3暂过7待审，人审0。object_short004继续最多10组未制对象辅助增量，仅独立IR/Depth清楚者制作，旧失败不重生成到通过。
object_short003已新接Robo depth_screen004：排除已做30组，从本地2000筛20新组，先看Depth至少2对象边界对应，合格才制作，不合格只screening原因。每2合格组safe发SOL，检查20后结束评估产率，不盲目增加纯RGB对象。SOL交替审City和Robo新safe，仅原图Query，不GT/私有索引。
完整旧新来源80组189原Query319输出238暂过81复核保持；RGBDT200分列，不与旧248重复相加。人工仍0，明天统一，已有两个100项包保留。制作筛选ETA未知，云端无卡未训练；继续CPU。合并City扩展必须按原图ID去重，不能将扩展记录当新场景。

## 2026-09-26 06:15 辅助证据收尾与候选任务派生

aux_extension001/merged_20260926_0615：City已有4组5Query11输出全审8暂过3复核；robo/depth_screen004/merged_20260926_0615：筛20新组留4组12对象12原Query24输出全审23暂过1复核，制作05:45:12–05:54:44。新来源累计84个有制作记录图组（City扩展不重复加组）、206原Query记录、354输出全审269暂过85复核，人工0；这是标注产物不是模型成绩，也不是84独立场景。RGBDT200仍分列，旧248不重复加。
object_short003继续depth_screen005→006各20新图筛选，排除已检查50图，强调不同布局、近邻单记不当独立场景，每2合格组safe交SOL，不明skip、不近远。按004实测筛20约10分钟，下一40粗估20–40分钟制作筛选，审核人审另计，合格率不保证。
object_short004接candidate_choice_pilot，City已审多对象/旧RGBDT多对象中最多20真实候选选择题，至少目标有辅助可靠框，2–6确认对象，原Query原框、唯一答案核查、编号seed2026打乱同步答案。proposals/screening独立文件，pending_human，不把作者自查当独立/人工、不直接训练。困难关系人工包材料不强凑100。SOL继续005/006 safe盲审，旧City/Robo004已结束不重复审。
无卡不训练，人工明天统一，继续CPU。现有natural100/correspondence100 CSV保留。下一合并要纳入扩展目录并按图/Query去重，不能仅glob batch*漏掉screen/extension；原文件不覆盖。

## 2026-09-26 07:45 恢复认证并核实新增结果

SOL此前auth.openai.com/oauth/token报错，已单次恢复成功；按safe ID确认00512/12与00620/20完整，未重复审、无已完成结果丢失。快照分别depth_screen005/006/merged_20260926_0745：3组6Query12输出全过；5组10Query20输出18过2复核。40筛选图组留8组16Query，约4种布局且近邻单列，不算8新独立场景。新来源累计92个制作图组222原Query386输出全审299暂过87复核，人审0；RGBDT200分列不与旧248重复加。
候选选择试点完成5题pending_human，不凑20，candidate_choice_pilot/proposals.jsonl与私有screening.jsonl。object_short004正在做human_acceptance/candidate_choice5_v1 HTML+空白CSV并盘点旧external100包真实可用性，不改旧CSV、不训练。
object_short003继续depth_screen007筛20新图，排除已检查90（含skip），优先不同物体/布局，近邻不当新场景；只有至少2对象Depth明确者建记录，每2合格组safe交SOL。SOL已恢复等待新safe。当前制作/筛选按近期40组21分粗估20组10–20分钟，场景多样性要求可能延长；人审/完整数据ETA未知。人工明天集中，不催；云端无卡不训练，继续CPU。

## 2026-09-26 08:15 新增007与外部盘点范围更正

Robo depth_screen007/merged_20260926_0815：筛20留3组6原Query，12输出全审11暂过1复核，人工0；制作07:42:26–07:56:47。新来源累计95制作图组228原Query398输出310暂过88复核，非独立场景数，RGBDT200另列。候选选择5题human_acceptance/candidate_choice5_v1/review.html与decisions_blank.csv已齐，人审空白。
发现object_short004盘点external误将RGBT/Robo训练候选当作外部池，其external_readiness报告的52组144Query/124过不得作为外部验证可用数。已明确要求在旧报告首段标误读范围保留历史，重新盘点RGBDT500 annotations/generated_external_batch001..008及reviewed_external001..008/预留序列；足50组100题才正式包，否则partial和缺口，严禁训练池补外部或重划split。新包human_acceptance/external_rgbdt_review_v1。当前没有把这些训练候选导出外部评测。
object_short003继续depth_screen008筛20，排除已查110含skip，优先新布局，20完评估近邻与辅助产率，不无限扩量。SOL接新safe。筛20按007约14分钟，粗估15–25分钟，审核/人审另计；总体ETA未知。继续CPU无卡不训练，人工仍0，旧CSV不覆盖。

## 2026-09-26 08:45 停止Robo随机扩量，补齐人工材料

Robo008 merged_20260926_0845：筛20留5组12Query24输出全审11暂过13复核；制作08:12:22–08:29:59。新来源累计100制作图组240原Query422输出321暂过101复核，人审0，非独立场景数；RGBDT200分列。最近80组制作筛选合格16/80（20%）不是盲审通过率，50/80有近邻。暂停广泛随机Robo扩量，不能凭筛选数量继续堆同场景。
object_short003转controlled_augmentation_pilot最多20原Query各1受控RGB增强（亮度0.35/0.55/0.75或模糊1/2一次一种），仅原RGB和aux审核通过且无必要文字颜色被破坏者，实际看可答性；原图/IR/Depth/框保留，pending_human，单计增强不当新Query，未来<=20%。输出human_acceptance/augmentation_partial_v1 HTML空白CSV，最多20不凑，不训练。
外部RGBDT人工包external_rgbdt_review_v1已完成50图组100题，源52组104Query/24序列，94旧暂过10复核；包含10复核，全部人工0。属于Train.zip内部预留不是官方Test。候选5题包已齐。object_short004继续补查13原external组历史筛查覆盖（先找new_arrival各报告，不因为不在旧external9就称未筛），核对与当前训练池序列隔离，写external_screening_coverage_addendum.md；若需更换出v2不覆盖v1/CSV。旧错误external_readiness已标明误读训练候选池。
SOL当前已交任务审空，无需重复审；待真正新safe任务再唤醒。当前两LUNA运行CPU制作/整理，增强/覆盖核查耗时未知。natural100+correspondence100+external100+candidate5共305待人工项（不同检查类型不是305新训练Query），不称500齐，人审0。云端无卡未训练，继续授权数据工作。

## 2026-09-26 09:15 人工材料320项，接续新City候选

controlled_augmentation_pilot完成15项（RGBDT13/City1/Robo1），亮度0.55或0.75共10、模糊radius1共5，5候选跳过；原Query场景不新增，IR/Depth/几何未动，pending_human，人审0。human_acceptance/augmentation_partial_v1/index.html与decisions.csv已齐。外部13组历史筛查覆盖已找到new_arrival_localzip_screening（13/13 eligible），外部与当前205训练池序列/ID/路径交集0；external_rgbdt_review_v1仍可用，无需v2，限制见external_screening_coverage_addendum.md，不当全语料语义去重证明。
当前对应100+自然100+外部100+候选5+增强15=320人工检查项，非320不同Query，人审0，未称500齐。object_short004已接统一human_acceptance/index.html/README及去重production_inventory，按source+图+origin_query_id并覆盖batch/aux/depth_screen，原CSV不改、未验收不导训练。
object_short003已接City expansion20_v1新20训练图组候选准备，排除旧40，优先较大多对象/可区分关系，弱光/深度只是筛选线索，原3707标签不改；产精确download_request缺图清单供主Agent同步，文件齐后才实际筛图，不能把候选当合格。不碰412/官方测试。SOL无新safe暂未唤醒，勿编运行；新safe出现再恢复。
已完训练候选计数仍新来源100制作图组240原Query422输出321暂过101复核，RGBDT200另列，待统一去重盘点核对。无卡不训练，继续CPU。新City选样/传图ETA未知，人工由用户集中进行。

## 2026-09-26 09:45 City新20组同步完成并恢复两制作一盲审

expansion20_v1/download_request.json实际SFTP60文件完成exit0（session87444），PIL核验60张，原City render_depth生成20预览。sync_summary.json记录20组、60原图、20预览、视觉筛查0；73原Query候选不是已标注。batch001/002/generation_inputs.json各10含raw depth与depth_visual，原记录保留。
两LUNA已分别接003=001、004=002，先aux证据再2–4原对象，至少一aux可辨否则skip，原Query框不变不猜近远，逐2合格组safe送SOL。SOL已恢复只读safe原图Query，3代理真实派发。新增合格数未知，预计20组制作约20–40分钟作粗估，审核另计。
统一入口human_acceptance/index.html和README已完成，18链接核验，320检查项全人审0、非320唯一Query。production_inventory_20260926.md/json：RGBDT200原图209对象138去重Query；新来源104记录对应100原图243对象240去重Query（City扩展重复图已去），各选定快照无重复Query键冲突；外部52组104Query24序列另列，增强15绑定另列。不与旧248重复加。原CSV未改，不导未验收训练。云机此次仅传图无训练。

## 2026-09-26 10:15 City首20扩展完成，接续不同图组与候选题

expansion20_v1/merged_20260926_1015实际13组30对象30原Query、73输出全审56暂过17复核、人审0；RGB28/30 IR19/28 Depth9/15。22Query RGB+至少一aux均过，6Query三路均过，只是标注不是模型收益。检查20留13 skip7。两制作用时约11分钟/15分34，审核已结束。新来源累计113制作原图270原Query495输出377暂过118复核，外部/增强/RGBDT200仍分列，待库存增量去重核对。
object_short003继续City expansion20_v2仅候选20/缺图清单准备，排除旧40+v1全部20包括skip，train限定原3707，不改标签、不碰412/官方；候选不是已标注，待主Agent同步。object_short004从新13组合格对象派生最多15候选选择题，目标aux通过且候选RGB通过，语义唯一/原Query/seed2026选项绑定，pending_human，新包与旧5去重，不凑；产人工页面CSV加入统一入口，旧CSV不改；同时增量库存覆盖嵌套expansion目录。SOL当前队列空，下一新safe再唤醒。
人工入口现仍320项、人审0，未开训；新增候选题未落盘不预报数量。下一候选准备/同步ETA未知，真实制作批可参考约15–30分钟但合格数量不保证。保持两生成/一盲审按真实任务调度，无卡仅CPU/SFTP。

## 2026-09-26 10:45 City v2同步完成，人工入口335项

candidate_choice_city_expansion_v1已15题pending_human，与旧5去重，2–3候选RGB通过目标aux通过；对应HTML与空白CSV已加入统一入口，共335人工检查项，人审0非335不同Query。库存production_inventory_20260926_1037确认新来源117记录/113原图/273对象/270Query，无身份键冲突，旧CSV库存保留。
City expansion20_v2 20候选46原Query，排除全部60已查图，8车8人4动物。主Agent本轮用download_request SFTP60原图完成exit0(session93431)，PIL核验60并原City映射预览20。sync_summary.json明确视觉筛查0，新合格数未知。batch001/002输入已各10，003接001、004接002、SOL新safe盲审已恢复。每2合格组交接、不复制aux不猜近远、不变GT/Query，rawDepth保留预览看图，skip如实记录，旧ID不重做。
已完成制作统计仍上一批113原图270Query495输出377暂过118复核，RGBDT200另列，新v2不预计。按v1实测制作两批约11–16分钟，可粗估15–30分钟加审核，质量不保证；人审ETA由用户决定。云端仅传图无训练。

## 2026-09-26 13:30 v2完成核实，接续v3候选与选择题

expansion20_v2/merged_20260926_1330：检查20留16组31对象31原Query70输出全审60暂过10复核，人审0（RGB31/31 IR25/31 Depth4/8）；26Query RGB+aux过、3三路过。制作时间00110:43:43–10:55:30、00210:43:06–11:01:09；不是一直制作到本次心跳。无未审积压，SOL已告知本批结束，不等不存在的batch001后续part。
新来源累计129制作原图301原Query565输出437暂过128复核；RGBDT200另列，旧248不重复加。人工入口目前335项、人审0，v2未派生题不提前计数。004接candidate_choice_city_expansion_v2最多20题及页面/CSV/增量库存，需语义唯一、目标aux过、候选RGB过，与旧20题去重，pending_human不训练；额外统计v1/v2同图>=2 RGB+aux过对象池。
003接City expansion20_v3新20候选/缺图请求准备，排除已查80含skip，只train3707、不碰412官方，优先不同场景多对象IR可辨潜力，近邻单列，原Query/GT不改，选样不当证据。待同步才视觉制作。当前两LUNA运行、SOL无任务结束，真正新safe再唤醒。准备/传图ETA未知，制作参考前批11–18分钟每10组，质量不保证；无卡只CPU无训练。

## 2026-09-26 用户新指令：新增RGBDT辅助模态证据批次

用户明确“再增加一批rgbdt，目标教会Depth/IR信息利用”，优先级转为本地Train.zip新40图组候选，非RGBDT已全量完成。新目录results/multimodal_data_20260926_rgbdt_aux_batch/PLAN.md。003当前已收到转向：保留City v3进度停止扩City，提取/筛查新40真实GT图，排除已制200帧/全部external预留/既有隔离，优先新序列，复用本地ZIP和现有工具不联网。004收到转向：v2候选选择尚只有诊断脚本未落proposals不计完成，保存后写新annotation_brief；不抢003ID。新池齐后主Agent分互斥批给两LUNA，SOL只safe盲审。
这批区分互补线索任务、对应任务、无证据跳过；记录RGB不够的信息及IR/Depth实际提供何线索、帮助选哪个竞争对象。单纯辅助框可对应不当互补收益，不以普通RGB描述凑数；Depth未知方向单位不近远，允许可见结构区域对应，IR不编温度。原GT语义保留独立aux框，不复制。每2组交safe，人工仍0，不训练。首40是候选目标不是已合格数，ETA待真实首批产率。
City v3/已有候选成果保留，不重跑训练和官方；原335人工入口不动。004补充诊断v1/v2各8/10组有>=2 RGB+aux过对象，对应17/21对象，但其新v2选择包未完成不能加人审数。
## 2026-09-26 最新优先级：当天首批小试验，复用已完成数据

用户明确要求简化并加速，今天出第一批试验数据。本段覆盖此前首训必须先300–500核心任务及1–2工作日排期：300–500是后续建设目标，不是首轮门槛；新增RGBDT40并行补充，不阻塞已有数据小试验。不要重标、重盲审已完成200图组，不再等6000–10000或全部500人工预算。
实际首包results/same_day_multimodal_pilot_20260926/pending_candidates.jsonl已落盘252任务：RGB121、IR92、Depth39，对应121独立原Query；原200图组共138Query。252是输出任务，不是252互补问题。人工0、训练放行0，复用natural100_seed2026与correspondence100_seed2026抽验，不覆盖CSV、不伪造审批；按任务类型验收，不等所有其他包完成。外部100用于效果复查。
当前真实三子代理：LUNA-MAX rgbdt_aux_prepare_resume提取筛新40（每10组即落盘）；LUNA-MAX complement_inventory对既有证据分类，视觉校准最多20组，不因互补数不足阻塞首包；SOL-MEDIUM same_day_pilot_pack组装待验收任务及最小转换检查命令。新独立盲审只对新增安全任务，后续释放SOL槽继续。训练仍需GPU恢复和16微批2更新/重载检查，当前无卡不启动。优先小规模成对试验验证后再决定完整1500更新；不自动重跑旧C/M2。时间分别报告数据打包、人工等待、GPU检查和训练，不把候选落盘说成可立即训练。

## 2026-09-26 13:57 首包完成与新40提取完成

首轮包README/转换脚本已完成：results/same_day_multimodal_pilot_20260926，共252待验收输出（RGB121/IR92/Depth39），121独立Query/120图组；空白人工决策拒绝导出通过，人工0训练0。自然100/对应100包复用；未覆盖21自然项及31对应项只作为补充，可延期，不让全量逐项人工变成新的首训门槛。SOL same_day_pilot_pack正收尾文档明确可先放行已验收子集、统计现有包覆盖子集，并保护已存在人工CSV不被build覆写；不能假造批次验收。
新RGBDT40本地提取完成，29eligible/11相似候选隔离（不认定重复），24新序列+16已用合格序列的新GT帧。eligible_manifest位于results/multimodal_data_20260926_rgbdt_aux_batch，尚非标注完成。LUNA rgbdt_aux_prepare_resume接index0:10制作batch001；LUNA complement_inventory先收尾旧138Query证据分类/最多20视觉校准，再接index10:20制作batch002。每2组保存safe任务。剩余index20:29暂未分配。新SOL独立上下文待安全任务真实到达后启用，避免看GT者盲审；目前第三槽SOL收尾首包。新增不阻塞首训。候选提取剩余0，新增标注ETA待实际产率；人工等待不可估计，无卡不训练。

## 2026-09-26 14:27 首包186覆盖子集与新增制作继续

首包252候选/121Query/120图组不变，已有自然100及对应100同时覆盖186输出（RGB100 IR61 Depth25），人工0训练0。52补充检查项可延期；转换仅放行已验收项，不把全量人审追加为首训门槛。build保留已存在人工CSV，source_id改为图组+Query避免q01跨图碰撞，空白拒绝导出已验证。
新RGBDT29eligible：batch001 index0:10已10组7Query/16输出待独立审（3互补候选4对应3无用）；batch002 index10:20已筛查1互补3对应6无用，但尚无完整aux框/安全任务，不能计标注完成。已接续LUNA complement_inventory补齐batch002对象证据；LUNA rgbdt_aux_prepare_resume继续剩index20:29制作batch003。新fork-none SOL-MEDIUM aux_blind_fresh仅看batch001/review_all/blind/tasks.json原图做16case，写同blind/reviews.jsonl，不见GT/制作证据。最多两制作一盲审。
旧库存分类已完成results/existing_complement_screen_pilot：trial200 138条=5互补候选101仅对应32缺可靠双路依据，结合20图视觉例去重9候选（含City4），全pending_human，不作模型互补收益证明，也不从首包删除可靠对应训练任务。提取已结束，新增完整已制作10组、另10仅筛查、9待制作；剩制作审核ETA暂无稳定值。既有首包验收不等新增。无卡不训练，人工入口保留旧CSV。

## 2026-09-26 14:57 新29组检查结束，接续11输出盲审

新批29eligible全部检查完：batch001 10组7Query16输出、batch002 10检查/4制作Query8输出（6skip另存）、batch003 9组1Query3输出。合计12不同Query27输出；制作方5互补候选7仅对应，不能当已验证互补收益。首包252仍独立保留，不将未审新增算放行。人工0训练0。
SOL aux_blind_fresh完成batch00116独立审，097斑马2case标歧义，最终IoU统计待合并；已接batch002两safe各4及batch003/all3共11，不重审旧。LUNA rgbdt_aux_prepare_resume正合并batch001到merged_20260926_1456并库存去重；LUNA complement_inventory做首包index.html链接既有两个人审页面及真实CSV操作，不新建表/不覆盖CSV。所有新制作已结束，不继续无止境扩池，先收齐审核/人审和试验。制作剩余0，独立盲审剩11case，耗时暂未可靠估计；人审时间由用户安排，无卡未训练。

## 2026-09-26 15:28 新增批次独立审核全部收齐

新29组检查/12Query/27输出已全审并主线程合并至results/multimodal_data_20260926_rgbdt_aux_batch/merged_20260926_1527：22暂过5需复核，人工0。分批001=12/16、002=7/8、003=3/3。23条master记录加batch002六skip=29检查，part不重复计数；batch002旧candidate_queries单行是早期草案，以完整object_evidence四Query为准，不据旧草案少算。没有改失败Query或GT来提高通过率。
首包results/same_day_multimodal_pilot_20260926/index.html已完成，链接自然100与对应100既有页面，附准确CSV导出说明：两页均下载human_decisions.csv，需各自另存不同名称，浏览器不会写回项目。252候选、186现有包覆盖、人工0未训练；52额外项可延期。新增22机器通过不能直接加作人审已完成或训练放行。当前制作和机器审核剩余0，三子代理已结束，避免空转或重复派工。下一步用户提交人工CSV后导出验收子集；GPU仍按无卡状态不启动，恢复GPU并容量检查后小试验。人工等待ETA未知，不能承诺训练完成时间。首次临时合并调用误用通用id读取review引发KeyError，未写输出，改用query_id格式读取后合并成功；源结果不受影响。

## 2026-09-26 人工验收页面改版与中文翻译

用户要求更简洁友好前端及Query中文。已更新首包index和natural100/correspondence100两个核心review页：逐题、大图点击放大、中文优先/英文折叠、直接判定按钮、前后跳转/未完成筛选/进度、浏览器localStorage草稿恢复、CSV导入及按类型不同文件名导出。判定枚举/源英文/ID/图片/GT保留，7份原CSV经内容比较完全未改。备份原HTML一次，重建工具code/tools/build_review_ui_20260926.py及同名CSS/JS；翻译results/review_ui_20260926/query_translations.json共293唯一文本，两核心页200展示项译文全覆盖，其余翻译已备但其他包页面未改版。翻译仅页面，不改训练Query。
浏览器QA用?qa=1独立草稿，确认判定进度、刷新恢复、未完成筛选、放大及对应页中文标签。正式页仍0人工；测试不计人审。CSV导出按钮触发成功提示，但下载事件监听超时，未核验实际落盘下载文件；导入尚未端到端实测。静态字段/ID/800图路径/脚本语法已检查。首包入口results/same_day_multimodal_pilot_20260926/index.html，截图results/review_ui_20260926/*_review_preview.png。本地预览server 127.0.0.1:8765 根results，session84061。页面草稿不等于已提交人审CSV，心跳不得仅凭源CSV空白推断用户未在浏览器填写；收到导出文件后处理。用户仍需导出保存交付，浏览器草稿不自动写回源目录。


## 2026-09-26 用户人工验收已导入，转深度层级与错例分析

用户提交human_acceptance/human_results两份正式CSV，各100行，ID/源字段核对无误。自然98 correct_unique/1 uncertain/1 wrong；对应97 accept/1 uncertain/2 reject。不要再称人工0或等待这两包。229鹦鹉误写人连带RGB/IR排除。首包released/native_sft.jsonl已181输出（RGB98/IR58/Depth25），98不同Query/98图组，5拒绝66未具决策不导出；不等66补齐才首试、不新设全量人审门槛。报告results/human_acceptance_audit_20260926。原人工CSV不动，真实训练仍0，GPU上次无卡，本轮仅SFTP64图无GPU调用。

用户要求补中近景真实深度层级。已查本地80组City训练图516原目标，6组15目标入depth_layer_pilot_20260926/candidates.jsonl，3组三层3组双目标，仍是候选非已核准新题。三熊/重叠鹿暂缓：数值顺序可能来自地面/背景，不能框内中位数当主体距离。City官方毫米小值近依据已查，RGBDT/Robo未知方向不近远。后续制作方先做主体小块及邻接背景核查，可靠关系派生相机距离比较/排序/最终RGB定位，分清对应任务与互补任务，不追加让用户全图分割的工作。不能把验证图反哺训练。

同机R2/M2/C完整412：286/285/292，共同错110共同对273，M2对R2纠正11退化12，C对M2纠正9退化2。failure_analysis_20260926已16例人工目视三模态图册cases.html、README、完整inventory；16组64原图及112可视图在本地。发现相邻实例混淆、框范围/阈值边界、Query参照歧义；16精选例中7目标原Depth无效，不能外推总体或宣称Depth有害。原Query/GT/成绩全保留。总报告docs/research/2026-09-26-human-review-depth-and-errors.md。

首批181可进入GPU恢复后的16微批2更新/重载检查，再做已批准小规模G/U成对试验，不自动1500長跑、不换像素。尚无训练提分。两核心人工200检查已完成，不等完整500预算。自动化30实查当前PAUSED，保持暂停状态仅更新过时指示，不擅自恢复定时。未提交git或比赛，未恢复退役专家/旧C/M2包。

