# 框JSON标点修复与全量复算

2026-10-27。按用户要求修复B外部开发集的一次格式失分。模型原始输出存在语法错误，项目此前直接用严格JSON解析计分；这不能等同于目标框预测错误，也不是官方要求模型原始回复必须是JSON。官方要求最终提交JSON中的`bbox`为四个归一化坐标，见[赛题PDF](../../../../contest/基于大模型的多模态视觉理解与推理.pdf)第6页及[官网规则快照](../../../../contest/官网赛题规则_20261022.html)第228行附近。

## 具体错误

题目`rgbdt500_025_00000051_q02`，B原始回答：

```text
{"bbox_2d":[593,362),(670,416]}
```

正常结束、未截断。确定性修复后：

```json
{"bbox_2d":[593,362,670,416]}
```

四个数字及其顺序不变，归一化为`[0.593,0.362,0.670,0.416]`。与冻结标签的IoU为**0.9036156316248893**，达到ACC@0.5。修复函数不接收标签，标签只用于修复后的独立计分。

## 统一规则

评价和提交导出增加`--format-policy punctuation`；默认`strict`保留原实验行为。后处理只识别整个最终答案中的唯一`bbox_2d`对象和四个JSON数字，接受三种明确的坐标对标点形式：`[(x1,y1),(x2,y2)]`、`[[x1,y1],[x2,y2]]`、`[x1,y1),(x2,y2]`。它保留数字的原始文本、顺序及外部思考/终止标记，并将修复结果再次交给严格解析器验证。

合法原文不变。额外字段、多个对象、答案前后解释、缺失坐标、越界、倒置框、错误思考边界均不会被修复；截断仍计失败。思考模式只处理唯一`</think>`后的最终答案，推理正文不参与提取。不能从多个框中选择，也不能根据标签修改坐标。

评价结果保留原始`text`，另记`normalized_text`与`format_repaired`；提交导出从原始回答应用同一函数。训练奖励继续使用原严格规则，当前训练没有改变。所有臂、共同初始化、匹配曝光及后续复现都采用同一后处理口径，同时保留严格指标；格式收益不能算作强化学习定位收益。

## 全部A/B结果

对A、B各自City378及外部88共932个回答统一复算。先核对原始文件与评分文件ID、唯一单答及原始文本，再重新计算并逐项复现932个严格评分，最后应用标点策略。没有重新生成回答。只有B外部的一题触发修复，其余931个回答及原预测坐标不变。

| 集合 | A严格 | B严格 | A后处理 | B后处理 | 后处理后B救回／退步 |
|---|---:|---:|---:|---:|---:|
| City378 | 295/378，78.04% | 300/378，79.37% | 295/378，78.04% | 300/378，79.37% | 5／0 |
| 外部88 | 85/88，96.59% | 84/88，95.45% | 85/88，96.59% | 85/88，96.59% | 0／0 |

后处理后，两组466个回答均合法且无截断。B相对A在City多5题、外部持平；这仍是一个种子的对照。共同初始化的完整开发成绩尚待补评，不能据此判断相对初始模型的强化学习增益。

证据：[A City](evidence/punctuation/a-final-city378/summary.json)、[A外部](evidence/punctuation/a-final-external88/summary.json)、[B City](evidence/punctuation/b-final-city378/summary.json)、[B外部及修复原文](evidence/punctuation/b-final-external88/summary.json)、[后处理逐题差异](evidence/punctuation/ab-paired-development.json)。各目录中的`predictions.jsonl`保存全部逐题输出。原始回答、原严格评分和[原严格配对报告](evidence/ab-paired-development.md)均保留。

后处理、评价、严格奖励共63项检查通过；提交导出9项通过。覆盖实际错误、思考边界、原文保留、与标签无关、额外字段/多框/越界/截断拒绝，以及只更新官方`bbox`字段。

## 运行方式

2026-10-27 08:18，C最终466答也已按相同入口复算，全部原严格字段一致、标点修复0条。严格与后处理均为City303/378、外部85/88；证据见[C City](evidence/punctuation/c-final-city378/summary.json)、[C外部](evidence/punctuation/c-final-external88/summary.json)及[A/B/C逐题配对](evidence/punctuation/abc-final-paired-development.md)。

有冻结开发集时，可直接使用评价入口：

```bash
python tools/eval_quant_grounding.py --dataset DEV.jsonl --predictions ORIGINAL_RAW.jsonl --format-policy punctuation --answer-mode short --output-dir NEW_OUTPUT
python tools/export_quant_submission.py --queries OFFICIAL_QUERIES.json --predictions ORIGINAL_RAW.jsonl --format-policy punctuation --answer-mode short --output-dir SUBMISSION_OUTPUT
```

D组使用`--answer-mode thinking`，B组评价使用`--reward-mode acc05`。最终提交仍须遵循官方字段格式，不执行比赛上传。

仅有完整历史证据时，使用[CPU复算脚本](run-scripts/rescore_saved_development.py)：

```bash
python docs/research/2026-10-26-quant-rl/run-scripts/rescore_saved_development.py --scored ORIGINAL_SCORED.jsonl --summary ORIGINAL_SUMMARY.json --raw ORIGINAL_RAW.jsonl --expected-queries 88 --output-dir NEW_OUTPUT
```

此入口只用于本轮没有负框奖励的完整开发集；参数`--expected-queries`按集合指定378或88。它要求重算严格评分与旧记录完全相同，并创建新目录，不覆盖原证据。后续C、D、同曝光状态和warmup50完成后，也先保留原严格结果，再做相同CPU复算。

30分钟续行任务已加入此步骤。本次代码与CPU复算均在本地完成；当前云端训练、奖励和既有严格评价队列保持原配置，后续通过下载完整输出在本地补充同口径结果。
