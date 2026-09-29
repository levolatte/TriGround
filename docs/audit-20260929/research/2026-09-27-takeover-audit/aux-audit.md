# 候选、Depth 与细化链条：接手复核

日期：2026-09-27；审计起点 `audit/gpt-pro-20260927` / `29a7a692219cdf09015d8f8dcdb67f330ea4e2e9`。本报告属于接手阶段的独立阅读和 CPU 复算，不是新 GPU 实验，也尚未包含用户将提供的 GPT Pro 审计意见。

## 阅读范围与版本

完整阅读 `source-inventory.json` 中 owner=main 的 11 个文件、3396 行，覆盖 7 个 tools 文件及 4 个测试文件。另检查审计包内 18 个 aux Python 快照：与当前已读文件逐行比较，完全相同者记录相等，差异者完整阅读差异；额外的 `repair_aux_numeric_ids.py` 全文阅读。记录见 `main-coverage.json` 和 `aux-version-coverage.json`。

工作树 `tools/predict_aux_selection.py:32,252-279,1006` 为 `centered-clip-v2-not-gpu-evaluated`。它与 `code_corrections_not_evaluated` 中的源码、测试相同；相对实际 `code_final`，差异限于居中截边规则及版本字段。实测 221 的来源仍是 v1，不是当前修正版。

## 从输入到评分

1. `tools/prepare_aux_selection.py:23-32,58-100` 从原生 SFT 清单取 Query 和三图路径，构造原始 Depth 路径，输出仅 `id/query/images/depth_encoding`；浮点 GT 另存 `scoring`。C 框是已有模型预测。
2. `tools/predict_aux_selection.py:636-660,696-760` 使用冻结 C 仅从文本解析类别、参照类别、关系及 single/group/part/unclear 范围。`scope != single` 直接保留 C，当前 47 题。此47是范围策略的统计口径，与“已知地点重用47”不同，不能混作同一分组。
3. `tools/aux_selection_evidence.py:138-291,337-477` 分别在 RGB、IR 上真实运行 GroundingDINO，合并近重合提议，保留 C 原框，目标最多 8、参照最多 4，随机编号。IR 坐标采用共享归一坐标投影；没有验证每个投影框都是 RGB 的准确边界。
4. `tools/aux_selection_evidence.py:480-648,661-800` 用预测候选框在 RGB 上提示 SAM2，取最高自评分掩码，读取 City 原始 uint16 深度。有效区间 300–19999 mm；检查核心区域像素数、有效率和完整/核心中位数稳定性。`reliable` 是规则通过，不能等同于人审确认主体正确。
5. `tools/predict_aux_selection.py:409-458,1054-1074` 给模型三张带编号框的图、Query、目标/参照候选列表；相机近远类才加入通过规则的数字 Depth 表。43 题中实际 35 题有数值、19 题有支持的前后对。
6. `tools/predict_aux_selection.py:195-241,809-887` 通过候选 ID 查框；参照编号不可作为输出。非法输出为 `None`，本轮 1 题，按完整分母计失败。
7. `tools/predict_aux_selection.py:248-292,890-974` 在换候选或短边小于 0.06 时裁剪。输入顺序确为全局 RGB 标框图、局部 RGB 无标记图；输出要求第二图 0–1000 坐标，再按保存的实际像素裁剪范围映回 RGB。
8. `tools/report_aux_selection.py:53-86` 从独立原始浮点 GT 和完整 412 ID 评分；候选覆盖只在评分端计算。主报告调用 `report_rematch_experiment`，不采用预测文件自报的 IoU。

推理清单字段、C 框来源、每个候选选择与 ID 对应框、ID 集合及分母均重新验证。未发现这条保存下来的链条把 GT 直接送入选择或裁剪的证据；这是所查路径的结论，不是对整个项目所有历史过程的绝对保证。

## 独立复算

运行 `F:/AIC/code/.venv/Scripts/python.exe F:/AIC/docs/research/2026-09-27-takeover-audit/verify_aux.py`。脚本独立实现 IoU，不调用原项目评分函数，不改写旧结果。输出为 `aux-recheck-results.json`、`aux-recheck-per-query.jsonl`。

| 阶段 | ACC@0.5命中 | ACC@0.7命中 | mIoU | 可解析 |
|---|---:|---:|---:|---:|
| C | 292/412 | 224/412 | 0.6136284094 | 412/412 |
| 选择 | 273/412 | 205/412 | 0.5774749263 | 411/412 |
| 完整 v1 | 221/412 | 167/412 | 0.4710951895 | 411/412 |

- 选择：10 次纠正、29 次退化；细化：1 次纠正、53 次退化。
- 目标候选覆盖 339，其中新增覆盖 C 错题 47；已有覆盖但选择失败 66。
- 保留候选内 RGB 原检测框覆盖 238，IR 原检测框覆盖 96；IR 相对 RGB 独有 1 题。新增覆盖 C 错题的 47 题均也有保留 RGB 原检测框支持。这不能外推为 IR 无用。
- 104 次数字细化的仿射映射逐条重算最大误差 0。53 次细化退化中 25 次原始数字事后按全图解释可命中，仅作为诊断，未回写或择优评分。
- 原始整数 ID 修复影响 3 题：`city_000011_027_00000001_001`、`city_001685_006`、`city_002940_001`；其余 409 题字典完全相同。修复脚本不读 GT，也不重新生成选择答案；历史执行时间只由保存记录支持，本轮未重新连接云端验证。
- 保存的 env8 原文、框、图像网格、提示与预期全部相同。本轮只是比对保存文件，没有新跑 C。
- train16 采用原有千分位训练答案复算为 14→13→9；它是训练内行为检查，不能代替浮点验证成绩。
- 独立计算居中截边与实际 v1 像素裁剪差异，共 42 条，其中 41 条碰边、1 条舍入差异。详见 `aux-extra-checks.json`；没有以此推算修正版分数。

## 需要保留的确定缺口

**预检放行条件缺少行为正确性。** `tools/run_aux_selection_queue.py:16-24,108-110,119-132` 只验证 ID、`parsed`、无截断；没有比较同一对象、坐标所属图像和细化前后定位。`configuration_frozen.json` 的 `preflight_parse=16/16` 和 `preflight_checked.json` 的图数/token检查，与 14→13→9 不矛盾，因为前者只管格式。格式与算术通过不能据以放行视觉行为。

**同一对象约束依赖语言提示，没有验收证明。** `build_refine_prompt`（当前 461-469）指示标记区域、原 Query、第二图坐标，但全图仍画所有候选（928-932），局部图未标示选中对象，也没有独立验证模型遵守。现有证据足以确认链条不可靠；不能仅由提示措辞断言全部 53 次的唯一原因。

**KEEP 在选择阶段的指向不公开。** 当前 446 行允许保留“existing prediction”，但选择图仅匿名候选号，`is_baseline` 隐藏（1009），1068 调用也没有传 `keep_box`；模型不知道哪个编号是 C。该匿名化有明确设计来源，C 仍作为可选候选存在，因此不是 ID 取错框的软件问题。它使 KEEP 难以表示“我识别并同意原框”，是待审的协议语义问题，尚无单因素实验说明它造成多少损失。

**Depth 可靠性仍是代理规则。** SAM 自评分、内部统计稳定无法排除掩码稳定地选中了背景或邻接目标；框间重合也不能确认同一实例。不能把 339 的候选覆盖、35 个数值表或 19 个前后对称为已经实现互补推理。

## 本轮实际目视范围

本轮只追加复看以下两题既有 atlas 的 RGB、IR、Depth、selection、full 五张图片，各题均在真实本地文件上查看。没有把旧 12 题审核自动算成本轮目视，也没有新完成人工验收。

1. `city_000004_012_00000001_002`，粉衣行人。选择 ID6，来源含 RGB 与 IR 原提议，选框 `[0.5319928,0.6435832,0.5753778,0.9169935]`，IoU 0.816142。数字回答 `[511,611,574,919]`；v1 裁剪 `[947,489,1179,1080]` 对应 transform `[0.4932292,0.4527778,0.6140625,1]`，映射后 `[0.554975,0.7871306,0.5625875,0.955675]`，IoU 0.087622。目视是腿部窄框。该题 relation=other，所以虽保存可靠 Depth 统计，实际选择提示不含米制表。此例不能支持“Depth 排序帮助选择”。
2. `city_000021_004_00000065_004`，停车排左端白车。选择 ID6 是 RGB 来源框，C IoU 0.040189→选择/完整 0.774346；细化返回 KEEP。目视与左侧车辆对应。另从本地 uint16 原始 Depth 及原始 mask 重算：1080×1920，mask 2330 像素，有效深度 0，与保存字段一致；SAM 自评分约 0.8124 仍被判为 unreliable。此例证实了一个已有筛除规则生效，不是辅助模态因果增益。

图片位于 `F:/AIC/results/aux_selection_refine_20260927/atlas/assets/`，文件名前缀是上述 ID。第二题原始 Depth 位于 `F:/AIC/results/failure_analysis_20260926/source_images/depth/000021_004_00000065.png`，mask 为 `city412/masks/000085_6.png`。

## 本轮 CPU 测试与边界

在 `F:/AIC/code` 执行：

```text
.venv/Scripts/python.exe -m pytest tests/test_aux_selection_evidence.py tests/test_predict_aux_selection.py tests/test_prepare_aux_selection.py tests/test_report_aux_selection.py -q
28 passed in 0.51s
```

这些是现有接口测试，包含修正后的裁剪边界测试，不是模型行为测试。没有加载模型、重新计算 SAM、运行 GPU、改变提示、标签、预测、历史报告或阈值。

GPT Pro 审计到达后先对照本报告各项的版本与证据。下一实验可以考虑训练样本上的对象/坐标协议验收，但尚未设计为获准执行的任务；不能因旧文档写有建议便恢复队列。
