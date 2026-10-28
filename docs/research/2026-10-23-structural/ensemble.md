# 多模型集成研究与本地可执行实验设计

2026-10-23。依据本地 `AGENTS.md`、`contest/` 官方材料、接手审计、当前推理源码及下列一手论文/代码。采用 experimental-design 技能。本轮完成研究、独立集成工具与 CPU 实验；未启动 GPU、未下载模型权重。

## 结论与新的主实验

**把集成提升为主线：现役三模态27B + EGM-8B专用指代定位 + Rex-Omni专用检测/定位，顺序推理、统一缓存、多个决策器复用同一候选池。** 它比再训练三个相近LoRA更可能带来可用互补。先完成本地推理接口、融合与评分；96GB云机由用户之后启动，各专家不用同时驻留。Qwen3.8-27B作为独立新基座臂，之后判断是替换主模型还是贡献额外票。Rex许可为非商业研究条件；EGM权重为Apache-2.0，适合优先接入。

原方案“不同实例不平均”只是几何注意事项，不能作为集成方案的终点。应当同时实现**选原框、簇内生成新框、学习选择、视觉裁判**四类决策，并比较净多答对多少题。WBF（加权框融合）可能把几个均未过线的框融成正确框，因此“候选最佳命中率”只是选择器上限，并非所有几何融合的绝对上限。

本地官方主PDF第7页已明确列出多模型集成推理作为优化思路；计分为单Query单目标、RGB坐标系归一化xyxy、IoU≥0.5。竞赛允许公开外部数据，禁止将官方测试集用于训练及商业闭源在线API生成测试答案。各专家仍须算法自主输出，VLM裁判使用本地开放模型。

## 1. 一手证据：确实存在可迁移的集成机制

| 工作 | 实际做法与结果 | 对本赛的具体借鉴 |
|---|---|---|
| [CVPR2025 MeViS冠军方案](https://arxiv.org/html/2504.05178v1) | Sa2VA不同大小/取帧配置与传统RVOS专家组合，像素二值投票；test J&F=61.98，冠军。作者明确观察到较大模型、更多帧不总更好，传统专家可补LMM失败。 | 采用“强VLM+专用定位器+视图差异”，不把集成限定成同基座换检查点。视频分割指标不同，61.98不能映射本赛ACC；二值mask投票也不能原封不动变成四坐标均值。 |
| [2022 Ref-Youtube-VOS冠军报告](https://youtube-vos.org/assets/challenge/2022/reports/RVOS_1_Hu.pdf) | ReferFormer、AOT及模型/表达聚合；表内Model Ensemble增加1.4 J&F。 | 多样性可来自模型和表述，但本赛不能把不同Query当同义表达合并；只对同一Query的专家输出聚合。 |
| [GeoNLI，2026](https://arxiv.org/html/2609.28741v1) | 多个VLM、SAM及混合管线按预测间重叠打分，选最高共识原预测；单框条件简化成IoU-medoid。其VHR表中三模型Acc@.5=.7870，七模型=.7293。 | 直接实现选原框共识；同时做专家子集比较。更多模型可能稀释好专家，不按模型数宣传收益。文中“64.94% grounding accuracy”与表中mIoU=.6494对应，不能误报为Acc@.5。 |
| [POPEN，CVPR2025](https://arxiv.org/html/2504.00640v1) | 先偏好优化，再对K=3响应的token与分割embedding做偏好加权融合，并训练可学习提示；不是简单“让模型再想一次”。 | 支持训练过的融合/裁判有价值；不能据其token似然结论直接给本项目未校准的bbox置信度。其分割特征、损失和注意力改造并非现有Qwen短JSON接口的即插即用组件。 |
| [WBF原论文](https://arxiv.org/abs/1910.13302)与[官方实现](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | 检测框按类别、IoU聚簇，利用分数和模型权重融合坐标，优于只删框的若干检测组合。 | 单Query只设一种标签0，先选目标簇，再比较簇内原框/加权框。检测mAP上的成功不保证本赛ACC，但足以成为必须实际跑的完整实验臂。 |

检索没有找到可核实的“本届AIC高分队公开完整方案”。群内高分队多模型信息是路线线索，不补写不存在的模型名单或成绩。以上冠军证据来自相邻指代分割任务，GeoNLI则更接近bbox选择；适用范围明确区分。

## 2. 首批专家与实际接口

| 专家 | 初始输入与输出 | 为什么加入 | 实现和资源 |
|---|---|---|---|
| 现役Qwen3.6-27B已训适配器 | 原生RGB/IR/深度三图、完整Query→单框 | 保留三模态、比赛域适配的强锚点 | 当前 `tools/predict_submission_vllm.py`，最高历史记录Arm2 ck500 k2.0。新合法留出集重新采集预测。 |
| **EGM-8B** | RGB、完整Query→思考后的单框 | 现成指代表达定位专家；与当前27B的训练目标、推理方式不同 | [官方库](https://github.com/NVlabs/EGM)、[8B模型卡](https://huggingface.co/nvidia/EGM-8B)、[4B权重](https://huggingface.co/nvidia/EGM-4B)。8B底座Qwen3-VL-8B-Thinking；SFT后GRPO，Apache-2.0。先推理成品权重，无需支付自训小专家成本。 |
| Rex-Omni | RGB、完整Query作为检测短语→零/一/多框；另独立比较主体短语→候选池 | 专门学定位、检测与细粒度几何，可能补通用VLM的视觉错误 | [官方库](https://github.com/IDEA-Research/Rex-Omni)，`RexOmniWrapper.inference(images, task='detection', categories=[query])`，有vLLM/Transformers。代码与权重按IDEA非商业研究许可；不可称Apache。 |
| Qwen3.8-27B | 与主模型相同三图/预算→单框 | 新基座独立检验，比只换旧LoRA更可能改变语义错误 | [官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)，Apache-2.0。先零样本，不把3.6旧LoRA强套过去解释为升级。 |
| GroundingDINO | RGB、主体类别/短语→候选框和词匹配分 | 低成本检测几何专家，对常见类候选丰富 | 已有本地代码和历史候选缓存；完整关系Query的top1不是强项，先作为候选源。 |

EGM不是泛用“再加8B”：官方8个RefCOCO系列拆分报告均值91.4，但其说明把准确率称作IoU，不能当本赛mIoU或预期成绩。模型卡与仓库推理有两个不同参数例：模型卡`.6/.95/8192`；[官方vLLM评价脚本](https://github.com/NVlabs/EGM/blob/main/verl/scripts/vllm_infer.py)是temperature=0、top_p=1，shell给4096 tokens。首轮跟评价脚本，随机采样作为单独一臂。完整SFT数据未公开，仅有1K示例，不能写“下载全量575K直接复训”。

### EGM接线必须保留的语义

- 官方评价提示为 `Locate {sent}, output its bbox coordinates using JSON format`；生成坐标0–1000，接入直接除1000，避免先四舍五入成像素再反归一化。
- 保留thinking能力与足够的最大输出预算。现有27B的64-token、`stop=[']']`、关闭thinking是另一路协议，不能照搬；第一处`]`可能只是思考中的候选框。
- 若后端分离`reasoning_content`与最终`content`，解析后者；若混在文本，先分离结束思考标记之后的最终答案，再找bbox。无最终合法框记失败，不能用“全文最后四个数字”凑框。
- 官方vLLM shell默认8路数据并行；单96GB运行应改DP=1、TP=1并根据实际视觉token调并发。它的8GPU配置不意味着8B推理必须8GPU。
- 复用本项目评分函数，统一`>=.5`；官方脚本是`>.5`。这是本赛对接，不复制其哈希、泛化数字fallback或整套训练环境。

### Rex接口核对

已阅读本地 `tmp/semifinal_research_20261022/perception/Rex-Omni/rex_omni/{wrapper,parser}.py`。结构结果是按类别分组的`type/coords`，默认没有每框检测概率或可直接使用的confidence。wrapper有`raw_output`，bbox通常已映射回原图像素；转换为本项目0–1时应按原图宽高除，不能把已映射坐标再除1000。多个框不是多个独立专家投票：每个family对同一候选簇只提供受限票权；全候选交给选择/裁判。

## 3. 六类集成实验设计

本轮已实现无训练几何聚合、多尺度/采样候选接口与两模型顺序推理。下文E4学习选择器、E5视觉裁判、E6检测器联合结构是后续设计，尚未实现，也没有GPU收益记录。

所有臂消费相同、已缓存的专家预测。输入至少保存`id, model_id, family, view_id, prediction, parsed, raw_text`；可选`logprob_features`和检测分数另存，不改变官方输出格式。每题最终只输出一个框。以下初值用于首轮，不应在同一412集反复挑最优后当独立提升。

### E1：异构专家IoU-medoid，完整零训练基线

每家先有一个直接框，初始等权。选

\[
i^*=\arg\max_i\sum_jw_j\operatorname{IoU}(b_i,b_j),\qquad\hat b=b_{i^*}.
\]

包含自重是加权medoid的标准目标，等权时只对每个候选同加常数。同时运行阈值版本：主分为`Σ w_j 1[IoU(b_i,b_j)≥τ]`，次分为上述连续IoU和，最终平局按预先固定主模型优先。两者分别对应“几何接近”和“过线邻居数”，不假定等价。两个模型等权时medoid必然平局，不能宣传其具有真实决策力；两家分歧需要先验、第三专家或选择器。

同一模型的尺度/随机样本不能直接当多家：给family总权重1；在family内K个输出各1/K，或者先家内选代表再家间等票。后一方式更简单，前者保留多峰信息。报告哪种实现，不能一边说三专家一边让某家占六票。

**要推翻的假设：** 即使oracle明显增加，共识选框仍持续净退化，说明正确答案常是少数派；转E4/E5，而不是无限追加同源票。

### E2：目标簇选择＋簇内WBF/medoid

对同一Query的全部框按固定来源优先级排序，用IoU阈值`.55`先贪心聚簇；另试`.7`，不把图连通分量当对象簇，以免A接B、B接C却A不接C的链式合并。每加入框更新代表；簇得分按**不同family支持**累计，不能由某个检测器吐出十个近似框垄断。

选最高支持簇后分别输出：①簇内medoid原框；②簇内坐标加权平均；③直接官方WBF产生的最高分簇。这样能区分“选对目标”和“合成边界”的贡献。所有原框并列加入候选池供E4比较；同时保留无条件四坐标均值/中位数作为实际基线，不能只凭反例拒绝计算。

[ensemble-boxes源码](https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py)可直接复用：`weighted_boxes_fusion(boxes_list,scores_list,labels_list,weights,iou_thr,skip_box_thr,conf_type)`；坐标已是0–1xyxy，本题label全0。生成式框暂设score=1，模型先验放weights。默认`.55`匹配用严格`>`；`get_weighted_box`用score×weight加权坐标。`box_and_model_avg`计不同model来源，仍不等于自动识别family；家内需先聚合或明确分摊权重。官方实现会修正越界/倒序框，本项目输入边界先判定合法，避免靠融合库悄悄修坏输入。WBF输出分数只是融合支持，不是本赛过线概率。

### E3：多尺度与有限随机采样，增加候选而不重训

先比较三模态27B的`k1、k2、RGB高/IR-D低`三种视图，以及EGM的RGB原图和高分辨率。所有视图坐标都回同一RGB框。多尺度全部保留全局图；首批不采用水平翻转，因为left/right、序数及图中物体自身左右都可能变义，不能只翻框而不处理语义。

再给一个选定专家增加3次采样（例如EGM`.6/.95`），同一个family分摊总票权；与“再加一个独立专家”在相近总GPU秒下比较。采样价值用**候选oracle新增、最终ACC新增、每百题推理秒**三项衡量，不能只看框数量。采样全落同一错误实例，增加K不会解决目标识别。

节省云费的可执行版本：所有题先跑主模型和EGM；仅在两者IoU<.5或解析失败时增加第三专家/额外尺度。该路由规则在推理时可计算；成本必须按全体题平均。首批三家完整小集用于衡量路由漏掉的可救题，随后固定路由再跑大集。

### E4：直接学习“哪个框能过线”

候选包含各专家原框、最高簇WBF框、中位框。用合法有GT的训练/调参数据生成标签`z_ik=1[IoU(b_ik,g_i)≥.5]`，训练一个简单逻辑回归或浅层梯度提升树，输出每候选的过线概率，选最高者。比模型类别多分类更合理：同一题可能多个框都对或全部都错，不能人为只设一个“正确模型”。

首轮特征仅用已有CPU信息：模型来源、框中心/面积/长宽比、边界贴边、与其他专家的全部IoU/中心差/面积比、支持family数、尺度稳定度、Query长度及空间/序数/遮挡词。没有模型置信度也能训练。可以补RGB亮度、候选内有效深度比例/分位数等，但应单独比较；不要把GT IoU、旧hit字段、按真框大小路由放进特征。

必须按场景/序列整体分组产生OOF（折外）预测；同题全部候选只能进同一折。先固定3或5折、一个小模型，报告逐题折外选框的ACC，而不是训练后回代。每个基模型的全局可靠性权重也只能由训练折算，不能用待评分整集个体ACC给每折偷偷加先验。对于反复使用City412，折外只能检验当前选择器是否记住其训练行，不能消除历史基模型选模偏差。

### E5：一次视觉裁判，不恢复旧多步Agent

输入完整Query、RGB全图、原生IR/深度及最多3–5个候选；候选展示为清晰编号框，并附同尺度裁剪，保留全局关系。隐藏专家身份，避免“信任大模型”的来源偏见。裁判只输出`candidate_id`；允许`none`作为诊断标签，但最终若none，按预先固定主模型结果输出，计入全分母，不能把拒答题删掉。

裁判首选现有27B独立单步提示或专门选择LoRA；相同候选以随机顺序训练，包含“主模型错、少数专家对”“全部错”“多个都过线”。用合法GT自动标记候选过线标签，避免追加人工测试答案。可以用逐候选yes/no二分类，或一次A/B/C列表选择；后者便宜，前者易校准，分别是实验臂。保留裁判看到原图、不看候选答案的直接框对照，判断收益来自额外计算还是候选帮助。

训练裁判目标应是候选过线而非复述专家理由；全错题继续训练为none，避免学会无条件改答案。推理先在专家分歧题触发，另对小集跑全量裁判估算路由遗漏。统计救回、破坏、未覆盖三个数；旧8B未训练候选Agent退化不能代替本次异构候选＋单步训练裁判实验。

### E6：语义候选与几何候选分工

27B或EGM负责完整Query定位；Rex/GroundingDINO用主体短语产生同类候选。将VLM原框与检测框组成集合：直接保留VLM框、与其最高IoU检测框、受约束WBF框，然后E4/E5选最终框。这样专用检测器可以修边界，也允许它的远处候选在裁判中纠正对象，不把“最近VLM初猜”的框强制当唯一目标。

这是一条完整异构结构，必须与“只跑VLM”“只用grounder完整Query”“VLM+grounder但不裁判”逐项比较。SAM掩码转框可后加一个几何候选，但遮挡物体可见mask不一定对应GT的完整bbox；不能默认细分割一定改善ACC。

## 4. 生成式模型confidence到底怎么处理

当前 `predict_submission_vllm.py:292–369` 使用确定性短框生成，只保存文本/token数等，**未请求logprobs，也未保存cumulative_logprob**；现在无法从已有JSON补出模型真实置信度。Rex默认结构结果也没有检测score。不能给它们统一填`.9`假装校准概率，更不能把模型自己写的“90% confident”与GroundingDINO词匹配分相加。

分三层实现：

1. **立即可用：无置信度集成。** 所有有效生成框score=1；每家等权，或在独立训练折学模型先验。无效框不投票，失败仍记在分母。WBF权重1是“等票”，不是声称100%正确。
2. **有新推理时记录概率特征。** vLLM申请生成token logprobs；保存输出token及其logprob，按最终bbox数字token求均值/最低值等。JSON标点、固定模板、思考前缀不计入“坐标置信度”；不要用完整输出概率相乘让长思考天然吃亏。不同tokenizer、不同模板、不同模型的原始值不直接可比，进入按模型校准器/选择器。坐标高似然通常只说明格式/数字生成容易，不能独立保证目标身份。
3. **面向任务的学习分。** 用候选+图像+Query的判别模型输出yes/no logits，或E4的过线概率。在独立数据上看可靠性曲线/Brier分数，并以最终选择ACC决定是否使用。多候选logits的softmax是相对选择分，所有候选都错时仍会分到总和1，因此没有none时不能称绝对正确概率。

GroundingDINO的文本匹配分、SAM的掩码质量分、VLM token似然与“正确目标且IoU≥.5”是不同量；它们可作为输入特征，经同一任务标签学习，不应直接当同一置信度。采样一致率也只是稳定度，某个模型可以稳定地认错对象。

## 5. 已找到并实际核对的共同GT输出

### 立即可用的四臂

统一目录：`F:/AIC/code/docs/audit-20260929/evidence/evaluation/{C0,A,B,V}/city412/predictions.jsonl`。

| 名称 | 实际模型身份 | CPU重算命中 |
|---|---|---:|
| C0 | Qwen3-VL-8B-Instruct，历史C适配器`next_stage_20260925/c_phase2/checkpoint-500`，不继续训练 | 292/412 |
| A | 同C初始化，原City定位题语言LoRA，seed2026，600步 | 296/412 |
| B | 同C初始化，City＋专项题语言LoRA，600步 | 289/412 |
| V | 同B数据/顺序，增加视觉投影LoRA，600步 | 289/412 |

文件含`id,image,prompt,target,prediction,parsed,raw_text`。本次CPU逐行确认四份ID集合及target完全一致，从坐标重新计算，不使用文件内hit。统一GT还有 `F:/AIC/code/docs/audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json`。

| 两臂 | 两者都对 | 前者独对 | 后者独对 | 至少一者正确 |
|---|---:|---:|---:|---:|
| C0 / A | 288 | 4 | 8 | 300 |
| C0 / B | 283 | 9 | 6 | 298 |
| C0 / V | 285 | 7 | 4 | 296 |
| A / B | 286 | 10 | 3 | 299 |
| A / V | 286 | 10 | 3 | 299 |
| B / V | 285 | 4 | 4 | 293 |

四臂原框oracle共302/412，比最好A只多6题（1.456个百分点）。**这证明当前同源LoRA组互补空间有限，不能推出真正异构专家集成无效。** 它仍足够在本地完成算法、评分、折外选择器的第一轮真实实验。按ID中地点字段可得46组，是可用场景代理；更完整序列分组应以manifest为准。

### 其他可复用文件

- `F:/AIC/code/docs/audit-20260927/evidence/gu_diagnosis_20260927/cloud_snapshot/{b_city412,gstar_city412,ustar_city412}/predictions.jsonl`：旧8B等预算B/G*/U*，同City412，增加训练臂多样性；身份以各目录run_config为准，不能混成27B。
- `.../aux_selection_refine_20260927/manifests/c_report_predictions.jsonl`：历史C；`.../city412/candidates.jsonl`：原C＋RGB/IR检测候选；`.../city412/trimodal/{selected_predictions,final_predictions}.jsonl`：旧候选选择/细化。用于E6离线分析，不能当独立训练过的专家。
- `F:/AIC/results/submission_{ck500_arm2_20261004,k20_20261004,arm4_ck600_k20_20261005,arm5_ck700_k20_20261005}/predictions.json`：当前27B不同训练/倍率输出，每组5690题，**没有官方逐题GT**。可计算分歧、产出融合文件，但不能离线证明集成涨分。当前Arm2k1/k2有531题框间IoU<.5；Arm2k2/Arm4k2有313题，仅表示待决策分歧。

当前27B的共同合法GT新缓存是未来云机第一次应产生的资产；之后所有CPU算法、候选重排与多数融合都不必重跑GPU。

## 6. 本地实现到云端的一次接续

**本地现在：** 统一读JSON/JSONL预测；固定完整ID集合；实现E1/E2以及均值/中位数；按GT重算个体、oracle、融合ACC、救回/破坏；实现按scene分组的E4折外训练；保留每题选中来源和候选用于错误查看。只需普通CPU依赖，不需要拉起云机。

**云机之后：** 同一小批合法有GT数据，依次跑当前27B、EGM8B、Rex；每家落完整缓存后卸载，单96GB顺序即可。先跑小批确认最终框协议与真实吞吐，再按剩余预算选完整集大小；不根据参数量虚构固定推理时长。异构三家缓存就绪即做全部CPU融合，不先训新模型。

**第一轮要回答的问题：** 新专家独有命中是否明显多于旧同源组的6题；这些增益主要是目标身份、几何边界还是小目标；E1/E2是否已经兑现，或E4/E5才能挑出少数正确答案。若oracle低，换专家/视图；若oracle高但选择差，训练判别而非继续扩候选；若几何融合超过原框oracle，说明边界组合有真实贡献。

省费不意味着只实现微小改动。完整候选生成与决策先做成，昂贵GPU只负责提供不可替代的专家预测；同一份缓存支撑多个结构实验。

## 7. 已完成的本地实现与真实CPU结果

实现为 `code/tools/ensemble_grounding.py`，覆盖 mean/median/medoid/consensus/cluster_medoid/wbf 六种算法。输入读取旧JSONL、官方字典JSON、新canonical候选JSONL；聚合函数没有GT参数。每个family的weight定义为总票，在配置成员间均分，再在有效候选间均分，失败成员不投票。正式JSON输出从原queries复制，只更新bbox；全体专家失败时明确空预测并保留评分分母，拒绝打包该正式输出。未新增学习选择器。

WBF复用ensemble-boxes1.0.9，每个候选独立输入、score=1、weight=已经分摊的票权，默认avg聚合，因此簇得分等价于簇的票权质量占比。这里没有伪造生成式置信度。安装限于`code/.venv`，保持原NumPy1.26.4，配套numba0.61.2、pandas2.3.3。

固定配置：A/C0/B/V四训练臂各1票、A优先处理平局、所有阈值方法τ=.55。这些参数未根据本轮分数回调。它们共享基座，训练臂在此次被当成四个family；未来同一checkpoint的尺度/采样必须共享family。

| 方法 | 命中/412 | 相对A救回 | 相对A破坏 | 净变化 |
|---|---:|---:|---:|---:|
| A单臂 | 296 | — | — | — |
| 均值 | 290 | 1 | 7 | −6 |
| 中位数 | 294 | 3 | 5 | −2 |
| IoU-medoid | 290 | 2 | 8 | −6 |
| 阈值共识 | 290 | 2 | 8 | −6 |
| 贪心簇medoid | 291 | 2 | 7 | −5 |
| WBF | 295 | 2 | 3 | −1 |

原框oracle=302。此次各几何融合均未产生原框oracle之外的新命中。**结果不支持把这四个同源训练臂直接投票替换A；它支持立即采集EGM/Rex等训练目标不同的专家预测，而不是继续堆相似LoRA。** 不能按这组旧8B结果断言当前27B异构集成也无效。

`code/docs/research/2026-10-23-structural/ensemble_cpu.json`保存配置、个体/融合分数、翻转、成对互补表与46个地点组的2000次配对重采样区间。配置路径已相对报告目录，GT/输出说明相对仓库根。City412长期用于选择，重采样不恢复独立泛化。逐题融合框在`tmp/structural_20261023/ensemble_cpu/{method}/predictions.jsonl`。20项集成测试覆盖family复制、候选数量票权、GT字段不影响聚合、全失败分母、融合新框越过原框oracle、正式JSON字段保持；连同10项原有merge测试共30项通过。

在`code/`运行可复现主结果（报告自身也是合法members配置）：

```powershell
python tools/ensemble_grounding.py --config docs/research/2026-10-23-structural/ensemble_cpu.json --compare-all --iou-threshold 0.55 --gt docs/audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json --baseline A --out-dir ../tmp/structural_20261023/ensemble_cpu_rerun
```

含场景重采样的复现入口为`python ../tmp/structural_20261023/run_ensemble_cpu.py`。加入`--queries <原queries.json>`可生成正式字典JSON；若任意题全失败会明确终止该打包，不用整图掩盖。

## 8. 新补充专家：GroundingPI

[GroundingPI官方库](https://github.com/groundingpi/GroundingPI)已提供4B权重和指代表达bbox接口。视觉塔为MoonViT-V2（Kimi K3），语言主干Qwen3-4B，较EGM/Qwen有更强视觉结构差异，值得作为第二批专家。默认部署需要Linux/Python3.12、vLLM0.18与项目自带Transformers5.7分支，不能直接往当前旧环境安装就假设可跑。代码原始贡献Apache-2.0，模型还保留Kimi派生条款，不统称“纯Apache”。

[官方客户端](https://github.com/groundingpi/GroundingPI/blob/main/grounding_pi/client.py)是独立HTTP接口：`GroundingPi(...).predict(image, query, task='bbox')`。输出GAM空间token、0–999网格；自定义请求要`skip_special_tokens=false`并保留相邻坐标token，不能用Qwen的0–1000 JSON解析。完整响应有`valid/parse_error/raw_output`；失败不当有效候选。采用现成专家的价值是新增视觉与训练多样性，部署成本高于EGM，当前先把EGM与27B接口跑通，再在独立服务环境接GroundingPI即可。
