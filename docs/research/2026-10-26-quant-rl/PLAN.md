# 27B 低显存量化与在线强化学习：实施约定

## 2026-10-28 执行修订

用户批准先补共同warmup50及C29的City378、外部88全量评价，直接复用A59/B59/C550已有输出。统一HF/NF4、短答案贪心、单答、并发2、种子2026及原分辨率，保留严格与标点后处理两套评分。C29的232次呈现与A/B的236次明确为近似同曝光。

本轮取消按GPU小时分配额度及内外层自动停止，旧72小时及各阶段额度仅保留为原计划历史；加载、运行、失败和留机等待仍如实计账。补评不等待旧D队列，GPU任务顺序执行，旧自动续行保持暂停，实例电源不变。

D修复限定于重复思考结束符、受约束采样与训练概率一致性、真实学习信号。固定既有训练诊断中8答合法且命中2—6答的四题，首批实际训练G8同时验证信号与一步参数更新，原奖励不改。以同初始化、同题、同种子对照256/64与1024/64的思考／答案预算，按有效性和定位质量选择；新进程加载检查点检查12道未参与该步更新的题。

通过后正式D重新从warmup50出发，按A/B相同顺序完成前236次呈现；先检查首12次呈现的定位信号。正式D与同解码配置的D0分别完成两套开发集，不将格式或解码收益算作RL收益。出现计算错误、实际非零优势却无真实更新，或输出协议系统失效即定位，不循环补抽或放宽门槛。

以下为原实施约定；与上述修订冲突的时间额度、队列先后及D生成长度以本修订为准。

2026-10-26。用户批准实施下一阶段计划，总上限72 GPU小时。高质量补标在另一对话进行；本轮冻结已有数据，不加入Agent、候选检测器或新实例关系结构。公开Qwen3.6-27B是唯一默认初始化，不继承旧Arm2适配器。

## 数据与评价

执行时的排除修正：City400中的4个官方测试重叠帧共22道有效题剔除，开发口径改为City378；外部94中的RGBDT297序列含官方测试图，整组6题剔除，口径改为外部88。四臂及复核均使用相同净集合，原文件名保留但按实际分母计分。

2026-10-27按用户要求增加推理后处理：评价与提交允许显式选择`--format-policy punctuation`，仅修复唯一`bbox_2d`对象中明确的坐标对括号分隔错误，保持四个数字及顺序。所有实验组、共同初始化、匹配曝光和后续复现使用同一规则，保留严格原始成绩及后处理成绩。训练奖励继续使用原严格规则；后处理增益单独归因，不计为强化学习定位增益。细节见[格式修复记录](format-repair.md)。

- City原始标签：`F:/AIC/docs/research/2026-10-01-structure-pro-audit/package/reference/original_city_labels.zip` 内 `qwen_generation_train_100.json`，3707条原浮点框/681图；媒体需从已有云端恢复。必须排除官方测试重叠，历史508是混合呈现次数，不是唯一Query删除数。
- RGBDT既有目标：`F:/AIC/data/grounding_aux_round1_20261025/inventory/rgbdt_train.jsonl`，98帧/序列；与专项合并后约176个文本目标。当前复核修订优先于旧框。
- 专项：`F:/AIC/data/grounding_aux_round1_20261025/bundle/annotations/pairs.jsonl`，226个接受训练对（RGBDT148、Robo78），合并同题、去掉候选换序重复，将复核负框留在隐藏奖励字段。
- Robo：`F:/AIC/data/external/RoboRefIt/subset/converted/train_depth_visual_fixed_p01_p99.jsonl`，2000图，只有RGB和真实Depth；不伪造IR。
- 训练标准字段：`id, source, group_id, images, messages, solution, negative_boxes, negative_kind, label_source`。路径相对数据根；GT、负框、审核理由不能进入可见messages。保留实际模态和深度渲染策略。
- 同来源/图组/已知序列固定seed2026封存10%，整组排除；City400/外部94是开发诊断。封存集只在最终选定模型后评价；未知物理场景不伪称完全隔离。
- 采样：常规50%、错实例25%、范围25%；常规内部City80%、RGBDT10%、Robo10%；池内先均匀抽组。四臂相同；缺池必须明确报错，不静默重配正式实验比例。

## 训练与奖励

- NF4、double quant、BF16计算。视觉/merger/embed/head保留原精度，GDN小门控in_proj_a/b不4位化；检查真实类型与dtype，防止意外全FP32。
- 语言LoRA rank32/alpha64/dropout0，冻结视觉与aligner。原公开基座默认无adapter/ref_adapter；参考通过同一PEFT模型关闭adapter获得，无第二份27B。
- ms-swift4.5.3训练核心，初始配套Transformers5.16.1、PEFT0.20.0、Torch2.12.1+cu130；bitsandbytes0.50.2、flash-linear-attention0.6.0为已核对发行候选，实际兼容性需验证并锁定。独立推理环境运行vLLM。
- MAX_PIXELS1204224（每图）、MAX_LENGTH8192；G8、每次更新4 Query/32回答，训练microbatch1/accum32，generation_batch_size8。
- HF rollout必须明确分批：`SwiftRLHF.train(trainer)`中设`trainer.engine.max_batch_size=1`，实测后允许2；G不随并发变。默认HF单进程同精度生成/评分。
- DAPO式token loss，epsilon .2/.28，scale_rewards=none，beta=.01，num_iterations1，dynamic_sample=false；lr2e-6恒定、warmup0；temperature1/top_p1/top_k=-1。必须保留全同奖励、无效、截断样本计数。
- 短答案不思考、最多64token；D臂原生thinking、总回答384token，按原生思考结束标记解析唯一最终JSON。输出`{"bbox_2d":[x1,y1,x2,y2]}`为0–1000坐标，最终提交除1000。只接受唯一合法JSON，不自动夹紧或捞最后一个框。
- 对合法框：u=IoU(pred,target)，无复核负框d=0，否则d=clip((u-max IoU(pred,negative))/.2,-1,1)。主奖励 `1[u>=.5]+.2*min(u,.5)+.1*d`；非法输出-.2。B臂去掉`.1*d`。
- 128训练题×G8先记录未更新采样，再真实反传、保存重载。仅格式合法率<95%时允许共享热启动，最多100步且不超过一个遍历，所有臂同一起点。
- A完整短框RL；B无负框差值RL；C同数据普通SFT；D有限推理RL。四臂完整进行，不用A/C涨分作为D准入条件。

## 算力与预算

48GB优先（L40S/RTX6000Ada/独立可用L40），主像素预计28–38GiB峰值；32GB须完整流程峰值<=29GiB才选；原生2073600预计32–44GiB仍须测。24GB不作主力。现有l40在规划时总46068MiB、占41003MiB/100%利用，不动其任务。

预算：链路/量化/选卡6h；A/B/C/D各8h共32h；最佳RL及最近对照seed2027各8h共16h；扩展各6h共12h；最终封存/部署/修复6h。合计72h上限，计卡数×占用时间，独立记录卡型和费用。预留2/4/8h可恢复检查点，统计Query曝光、唯一组、回答和token、有效优势组、峰值和分阶段耗时。

开发选模City400 ACC优先，同分外部94、mIoU、成本。RL在两开发集均不退且至少一套净救回、收益不只是格式修复，才扩大；复核独立种子。扩大分辨率需峰值余>=3GiB，两对照同步调整。无重复收益不强行花完预算。NaN、图像对应错、GT泄漏、适配器漏载、参考错误立即修根因。

FP8仅作为部署对照。最终适配器分别在NF4/HF、BF16/HF、FP8/vLLM重评；FP8另有同后端BF16参照。成绩优先、同分成本优先，不把量化模型路径替换误当LoRA已生效。

## 检查与记录

奖励阈值/非法/多答案/负框性质；元信息不泄漏；分组不重叠；G8并发1的完整归属；量化范围/精度与共享参考；真实三图/非零有限LoRA梯度/冻结不变；选中回答logits；保存重载与恢复；独立全Query计分；提交仅改bbox；优化kernel与数据/生成/参考/反传计时。既有28通过1跳过不代表新GPU链路通过。

阶段README详细记录事实和结果；HANDOFF只写新增、修改、修复及解决的问题。新代码不背旧8B/Agent兼容，也不删除历史基线、不提交Git。
