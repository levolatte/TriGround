# 2026-09-27 GroundingDINO＋C1500 联合判断失败：源码与逐题取证

日期：2026-10-25。范围只包括已执行的 `aux_selection_refine_20260927` 版本；没有重跑模型。赛题官方文件 `F:/AIC/contest/基于大模型的多模态视觉理解与推理.pdf` 第 2–4、6 页要求：英文 Query 与空间对齐的 RGB／IR／Depth，输出 RGB 上归一化 `xyxy` 框，以 IoU≥0.5 的全题 ACC@0.5 计分。下文所有“命中”依此口径。

## 版本和可比结果

- 真正跑完 412 的是 [`code_final/aux_selection_evidence.py`](../../audit-20260927/evidence/aux_selection_refine_20260927/code_final/aux_selection_evidence.py) 与 [`code_final/predict_aux_selection.py`](../../audit-20260927/evidence/aux_selection_refine_20260927/code_final/predict_aux_selection.py)，配置见 [`configuration_frozen.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/configuration_frozen.json) 和 [`city412/trimodal/run_config.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/run_config.json)。C 权重是 Qwen3-VL-8B-Instruct 的 C checkpoint-500，BF16、SDPA、贪心生成 128 token；本次未训练新权重。
- `code_snapshot` 是首次冻结源码；正式阶段开始后，DINO 安装版的零检测返回空标签 `['']`，在 `code_final/aux_selection_evidence.py:386-393` 修了空检测接口；首次评分前在 `code_final/predict_aux_selection.py:150-167` 接受纯整数 ID，恢复原输出 3 题。两项是接口修复，不是调阈值。`code_final` 与 `code_snapshot` 的实际差异分别就是这两处。`code_corrections_not_evaluated/predict_aux_selection.py:32,258-277,1003` 的居中截边和裁剪版本字段是事后修正，**没有 412 GPU 成绩**；工作树 `tools/predict_aux_selection.py` 亦为事后版。证据见 [`SOURCE_VERSIONS.md`](../../audit-20260927/evidence/aux_selection_refine_20260927/SOURCE_VERSIONS.md) 和这三份源码差异。
- 我直接按 [`city412_gt.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json) 对 [`c_baseline.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/manifests/c_baseline.jsonl)、[`selected_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/selected_predictions.jsonl)、[`final_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/final_predictions.jsonl) 重算：C `292/412`（70.87%），选择 `273/412`（66.26%），细化 `221/412`（53.64%）；C→选择改对 10、改错 29，选择→细化改对 1、改错 53。与存档的 [`report412/summary.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/report412/summary.json) 一致。最终相对 C 净损 71 题。它是多阶段流程的负结果，不能笼统称作“DINO 检测器使准确率下降 71 题”。

## 每一处信息瓶颈

| 阶段 | 实际代码与输入 | 信息如何被限制；证据强度 |
|---|---|---|
| Query 解析 | `code_final/predict_aux_selection.py:634-658,694-755` 用 C 模型**只读文字**产出 `target_category`、至多两个异类参照类别、关系类型和 `scope`。:635-640 明令去掉颜色、方向、序数、尺寸等修饰；`:139-146` 删除与目标同名的参照类别。 | 检测用的类别词失去唯一指代条件。例 `city_000004_012_00000001_002` 原句是粉衣人“beside the plaid jacket person”，[`query_info.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/query/query_info.jsonl) 原始输出含 `reference_categories:["person"]`，结构解析把它删成 `[]`。这是**检测／结构参照路径**的信息损失；下游选择提示在 `:432` 仍包含原 Query，不能说颜色和关系完全未给选择器。365 题为 `single`，43 `part`、4 `group`；后三类在 `aux_selection_evidence.py:249-251` 和 `predict_aux_selection.py:840-854` 直接保 C，未试 DINO。 |
| DINO 提议 | `code_final/aux_selection_evidence.py:337-383,447-477` 对 RGB、IR 各跑同一个 `grounding-dino-tiny`，caption 为类别词用 ` . ` 连接，box/text 阈值均 .25。IR 图按 RGB 三通道送入；框按各自宽高归一。 | 检测并非完整 Query grounding。分数是类别检测信号，不能等同实例／关系概率。阈值前和截断前的全部框未单独归档，所以**不能确定** 73 个池外目标中各有多少是 DINO 未检测、类别解析错误、阈值过滤或候选截断导致。不能把它们全归因于 DINO 漏检。 |
| 合并与截断 | `aux_selection_evidence.py:138-202,215-290` 以同角色 IoU≥.9 合并，C 原框永久保留；每模态先保目标 top3、参照 top2，再按检测分数补到目标最多 8、参照最多 4；按 seed2026 随机 ID 排序。IR 坐标在 `:170-172` 仅作 `normalized_shared_frame` 投影，去重时代表框可能替换为最高检测分数框。 | IoU 去重与 top K 可以把不同边界／小目标竞争信息折叠或丢掉；这是**代码结构风险**，历史文件缺少截断前提议，不能计量其真实影响。归一坐标并不验证 RGB/IR 局部物体边界精确重合。用保存的 [`evidence.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/evidence.jsonl) 重算：最终**保留池**的目标框覆盖 `339/412`，其中 C 错而池有命中框 47；池外 73 题由于 C 也在池中，都是 C 错题。保留 RGB 检测来源框覆盖 238，IR 96；IR 独有覆盖仅 1 题且 C 已正确，47 个可新纠正的 C 错题都有保留 RGB 框。这里只能评价**已保留提议**，不能推断原始 IR 模态无用。 |
| SAM＋Depth | `aux_selection_evidence.py:661-703,763-796` 用每个候选框在 **RGB** 上提示 SAM2，三个掩码只取预测质量最大者；`480-575` 在原始 City `uint16` 毫米深度上排除 300–19999 范围外像素，掩码内缩 3px，质量≥.8、核心有效≥32、比例≥.7、整／核中位数稳定，才标“reliable”。`578-648` 对成对深度再要求四分位区间分离和绝对／相对距离差。 | SAM 的预测质量和深度有效率只表明几何统计合规，**没有验证掩码就是 Query 指向的完整对象**。IR 提议也转去 RGB 分割，若投影偏移，深度可能来自邻物。`city_000004_018_00000341_001` 中 C 警示牌 IoU `.951`，DINO 另给竖向长框 ID5（`y2=.829`，GT `y2=.552`）；ID5 深度被标 `reliable`，选择它后 IoU `.334`。这证实“可靠深度”可以伴随错误边界，不能证明深度促成此次选择。 |
| 三图选 ID | `predict_aux_selection.py:407-456,482-491,1063-1071` 把同一候选数字框画在 RGB、IR、深度可视图，列出四舍五入到 0–1000 的框和原 Query；仅 `camera_near/far` 且 trimodal 时给数值 Depth 表。来源、C 身份、mask 不入提示（`:1004-1008`）。Qwen 返回目标 ID 或 KEEP；`:856-884` 直接用选中框，不做几何校正。 | 候选覆盖不能变成选择能力：`339` 池内题仍有 `66` 题选择未命中。保存结果里 255 题至少两目标候选 IoU≥.5，列表可能混合同一物体的不同边界和不同实例；该数量并不证明各对都是同一物体。真实异实例错例：`city_000004_018_00000341_005` 原 Query 为桥中心偏右路灯，GT `x=.585–.603`；C IoU `.710`，RGB 候选 ID10 也在 GT 附近，模型却选 ID7 `x=.780–.804`，选择 IoU `0`。也有同实例／边界型：上述警示牌选长框；`city_000021_004_00000065_005` 路灯 C `.687`，选 IR 偏窄框后 `.216`。不能把 29 次退化都算成错实例。 |
| 局部细化 | `predict_aux_selection.py:247-290,459-467,888-972` 只要换候选或框短边<.06 就裁剪；实测 v1 扩大 2 倍、至少原图 .12，并在边缘**平移保持窗口大小**。模型先见“全图带数字标框”，再见“无标记局部 RGB”，被要求返回第二图 0–1000 框；所有数字框一律按实际像素裁剪变换回全图。 | 接口把“输出语法正确”误当“模型知道坐标属于第二图”。270 题触发细化：166 KEEP、104 数字框；104 数字框里 53 使原正确题失败，仅 1 次纠错。以原始数字作**事后 GT 诊断**，53 次退化中 25 次若将数字错当全图坐标会命中；这支持部分回答仍沿用全图坐标，但其余 28 次另有目标／区域／边界错误。映射实现按保存的像素窗口重算最大误差 0；不是 x/y 交换或二次除以 1000。事后居中裁剪修正只影响 42 个已跑裁剪窗口（41 边界、1 取整），涉及 11 次细化退化；另外 42 次退化不在差异窗口内。修正版没有模型推理结果，不能拿它解释或抹去 221/412。 |

### 两个可核对的细化灾难

以下数值直接来自 [`final_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/final_predictions.jsonl) 的模型原文、窗口和实际结果，并按上述 GT 独立核 IoU；[`output_audit/refinements.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/output_audit/refinements.jsonl) 也保存同一诊断：

- `city_000004_012_00000001_002` 粉衣行人：选择框 IoU `.816`；原文 `{"bbox_2d":[511,611,574,919]}`。若事后误按全图理解，IoU `.529`；按提示要求的第二图映射，框成 `[.554975,.787131,.562588,.955675]`，IoU `.088`。Qwen 输出语法合法而图像坐标系不符，是明确失效模式。
- `city_000007_011_00000501_003` 白卡车上方路灯：C `.127`，选 DINO 框 `.774`，原文 `[407,116,500,604]` 作全图坐标 `.819`；正确局部映射仅 `.131`。这个例子先展示 DINO 候选确实能纠正 C，再展示局部细化把收益销毁。**按全图解读是使用 GT 的回顾性诊断，不是已验证的推理修法或可报告的新方法成绩。**

## 能成立的根因与不能成立的说法

1. **已证实的端到端失败**：固定 412 原始浮点 GT 上，选择净 `−19`，细化再净 `−52`；每一阶段代码与原始输出都可复算。绝大部分净损发生在选择／细化，不在候选覆盖统计上。直接模型 C 的框被纳入候选，但选 ID 和改框没有守住 C 的正确题。
2. **已证实的接口／监督错配**：选择器用的是冻结的直接框 C 权重，提示要求它做离散 ID 选择与新局部坐标系回归，未有本次适配训练。`train16` 放行只检查 16/16 格式有效；事后用训练答案取整框算出 C14→选择13→细化9，5 次数值细化全部错，4 次把选择正确改错，见 [`train16_posthoc_rounded_gt.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/output_audit/train16_posthoc_rounded_gt.json)。训练取整答案不是正式浮点 GT，但错误幅度已足以暴露预检漏检。不能把“没有训练”单独定为唯一因果；实际行为缺陷已由输出证明。
3. **已证实的信息压缩**：类别提议丢修饰词，同类参照删除；候选最多 8+4 且聚合；选择器看不到检测来源／C 身份／SAM 掩码，只在 43 条显式相机近远 Query 中可能看到数值深度。43 中 35 条有可靠数值表，19 条有通过分离规则的成对比较；该子集 C32→选择33→完整27。它不是随机消融，不能称 Depth 提升 1 题。96 题条件对照因正式结果无收益未触发。
4. **尚未证实的因果**：IR 投影误差、DINO 对弱光／小物体漏检、top K 截掉真物体、SAM 掩码串物、重复标框遮挡视觉、深度数字误导模型，均有机制可能性；但没有原始全提议、逐模态隔离推理、同一实例人工标注或随机消融来量化其贡献。不能因为 IR 对本池没有独有 C 错覆盖就宣称 IR 无用，也不能把 25 次“全图坐标可命中”推广到所有 104 数值细化。

## 对下一方案的直接约束

对任何新的 DINO＋主模型组合，先在同一冻结数据上分别报告：**检测原始提议召回 → 去重／截断后召回 → 选择前后配对纠正／破坏 → 改框前后配对纠正／破坏**；记录每一步的 Query、源图、原框、提示、模型原文和坐标变换。训练集小样本门槛必须检查目标身份、第二图坐标、IoU，不可用“输出 JSON 合法”替代。如果新架构保留连续可回归框或原 C 框，应明示并单独评估保留机制；这份历史结果不能直接否定经过专门训练的检测器或对象级融合，也不能支持未经验证的冻结候选＋冻结 C 选 ID 再局部细化。
