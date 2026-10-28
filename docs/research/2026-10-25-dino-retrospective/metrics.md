# 2026-09 GroundingDINO + Qwen 旧实验逐题复算

日期：2026-10-25。此记录复原 **2026-09 旧 Qwen3-VL-8B-Instruct + checkpoint-500 adapter、旧 City412 GT** 的冻结 C + RGB/IR GroundingDINO 候选选择与局部细化 v1。不是当前 27B 运行，也不是修订后的 City400。没有运行模型、下载权重或人工改标签。

## 复算口径与证据

- GT 使用存档浮点框，逐题按连续坐标 `xyxy` 计算 IoU；ACC@0.5 分母始终是 412，缺失/非法预测按 IoU 0。C 来源 `c_report_predictions.jsonl` 与 `c_baseline.jsonl` 逐题精确相同。数值候选 ID 修复后的选择/最终输出作为实测 v1 结果；修复前另作对照。
- 候选 oracle（理想上限）按保存候选框自身与 GT 的 IoU≥0.5 判断。RGB/IR proposal coverage 按 evidence 中每个 `sources[].bbox` 分别判断。未用 `source == DINO` 充当正确性或错误原因。
- 原始输入/输出：GT [`city412_gt.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/manifests/scoring/city412_gt.json)，基线 [`c_report_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/manifests/c_report_predictions.jsonl) / [`c_baseline.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/manifests/c_baseline.jsonl)，候选 [`candidates.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/candidates.jsonl) 和模型实际消费的增强证据 [`evidence.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/evidence.jsonl)，选择 [`selected_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/selected_predictions.jsonl)，最终 [`final_predictions.jsonl`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/final_predictions.jsonl)。逐题 CPU 复算脚本和完整 JSON 在 [`tmp/dino_retrospective_20261025`](../../../../tmp/dino_retrospective_20261025/)：`recompute_dino_retrospective.py`、`metrics.json`、`per_query.jsonl`。
- 主结论和冻结配置见 [`CONCLUSIONS.md`](../../audit-20260927/evidence/aux_selection_refine_20260927/CONCLUSIONS.md)、[`configuration_frozen.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/configuration_frozen.json)、[`run_config.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/city412/trimodal/run_config.json)。本轮选择/细化模型是旧 Qwen3-VL-8B-Instruct + checkpoint-500 adapter，本轮未训练新权重。
- `city412/candidates.jsonl` 是经候选优先级与上限裁剪后的保存候选，不是 GroundingDINO 未裁剪的全部原始检测输出。目标最多保留 8 个、参照最多 4 个；所以 239/412 是**保存的 RGB/IR source 框覆盖**，不是检测器总体能力上限。

## 完整 412 复算

| 阶段 | ACC@0.5 | mIoU | ACC@0.7 | 可解析 | 相对前阶段修复 / 损害 | 净命中变化 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 C 基线 | 292/412 (70.87%) | 0.613628 | 224/412 | 412/412 | — | — |
| Qwen 候选选择 | 273/412 (66.26%) | 0.577475 | 205/412 | 411/412 | 10 / 29 | −19 |
| 选择 + 局部细化 v1 | 221/412 (53.64%) | 0.471095 | 167/412 | 411/412 | 1 / 53 | −52 |

从 C 到最终输出是 4 题修复、75 题损害，净 −71。以上独立 CPU 结果与存档报告一致（浮点 mIoU 仅末位舍入不同）。

整数候选 ID 在评分前按原规则修复了 3 条，未重新生成选择、未读取 GT：修复前选择为 271/412，修复后 273/412；最终为 220/412→221/412。修复只涉及 [`numeric_id_interface_repair.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/numeric_id_interface_repair.json) 列出的三条；其中一条修复后细化仍失败。

## 候选覆盖、选择失误与模态来源

| 子集 | 分母 | C 命中 | 选择命中 | 最终命中 | 候选池 oracle | oracle 命中但选择未命中 |
|---|---:|---:|---:|---:|---:|---:|
| 全体 | 412 | 292 | 273 | 221 | 339 | 66 |
| C 错题 | 120 | 0 | 10 | 4 | 47 | 37 |
| C 正确题 | 292 | 292 | 263 | 217 | 292 | 29 |
| 单目标且实际触发 refine 阶段 | 270 | 170 | 152 | 100 | 212 | 60 |
| 其中实际输出数字细化框 | 104 | 70 | 60 | 8 | 83 | 23 |

- C 错的 120 题中，保留候选池新增覆盖 47 题（39.2%）；另 73 题（60.8%）没有任何**已保存目标候选框**达到 IoU 0.5。这是对本次截断后候选池可救范围的上限/缺口，不代表未保存的原始检测输出必然没有好框。
- 47 个被候选池新增覆盖的 C 错题，选择器只选对 10 个（21.3%）；另外 37 个虽有正确候选仍未命中。其中 27 题输出 `KEEP`，保留了错误的 C 框；10 题输出了另一个目标候选。
- 全部 66 个“正确候选已存在但选择未命中”由 37 个 C 错题机会和 29 个 C 原本正确题组成。后 29 个中，28 个选择了其他候选而丢失 C 的正确命中，1 个是非法选择输出。候选池始终包含原 C 框，因此选择阶段能够回退 C，却没有可靠保住正确 C。
- 保存的 RGB/IR source 框覆盖并集为 239/412：RGB 238，IR 96；其中 RGB 独有 143，IR 独有 1，两者都有 95，均未覆盖 173。C 错子集中 RGB 覆盖 47、IR 覆盖 8；这 8 个 IR 命中全部也有 RGB 命中，故 IR 独有 C 错覆盖为 0。唯一 IR 独有题在全体里 C 本来已正确。
- 10 个选择阶段修复的 C 错题中，9 个被选候选只有 RGB source，1 个候选同时有 RGB 与 IR source。该字段只说明来源记录；没有单模态对照，本次不能把成功归因于 RGB/IR 或把 IR 的非独有覆盖解释为无效。

## 局部细化退化

单目标 365 题中，270 题触发细化模型：166 次明确 `KEEP`、104 次输出数字框；94 题因候选框足够大跳过，47 题因 group/part 范围策略保留 C，1 题选择无效。104 次数字细化中，53 次把选择阶段正确框改错，仅 1 次把错误框改对，ACC 净减 52。C 错子集里 10 个选择修复有 6 个被细化再破坏，最终只留 4 个。

对细化原始数字按局部裁剪框映射回全图重算，最大坐标算术误差为 0；因此现有证据不支持软件映射算术出错。53 个退化里，25 个原始数字若事后当作全局坐标可过 IoU 0.5，支持不少回答混淆了坐标所属图像；这是看过 GT 后的诊断，不能作为新方法或可部署改法。剩余 28 个不能由此解释。

270 次已触发裁剪中，v1 的实际窗口与计划居中截边策略有 42 次不同；其中 41 次跨边界，1 次相差 1 像素。该 42 条审计记录包含 `explicit_keep`，不能按 104 个数字细化框作分母。只有 11/53 个细化退化与这批窗口差异重合。修正版没有 GPU 评分，不能将 221/412 说成修正版成绩。证据见 [`output_audit/summary.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/output_audit/summary.json) 和 [`border_crop_policy_audit.json`](../../audit-20260927/evidence/aux_selection_refine_20260927/output_audit/border_crop_policy_audit.json)。

## 按未决状态筛选的标签修订静态复算

另按 [2026-10-24 标签修订](../../../../docs/research/2026-10-24-label-fix/README.md)，只取 `corrections.json` 中状态为 `keep` 或 `corrected`、`query_changed=false` 且新旧 query 与历史逐字一致的 53 题（28 keep、25 corrected，其中 25 个框改变）。排除 query 已改、`deferred`、`partial`；没有给无修订状态的 321 条未复核项补状态。只用这些新框重评原预测，不重新推理：

| 同一批 53 题 | 旧 GT | 修订 GT | 修订框引起命中分类翻转（新 GT 单独增 / 减） |
|---|---:|---:|---:|
| C 基线 | 12/53，mIoU 0.255076 | 20/53，mIoU 0.351014 | +11 / −3 |
| 选择 | 12/53，mIoU 0.264757 | 18/53，mIoU 0.322551 | +10 / −4 |
| 选择 + 细化 | 10/53，mIoU 0.214678 | 15/53，mIoU 0.258848 | +8 / −3 |

在该定向抽取的 53 题上按修订 GT 计，选择相对 C 是 0 修复 / 2 损害，细化相对选择是 0 修复 / 3 损害。它只显示旧标签对逐题判定有实质影响，不是修订后 City412/City400 分数，也不是独立验证；全量 412 的主复算仍按原历史 GT 报告。

## 结论边界

本次旧设置的可见问题有三层：保留候选池对不少 C 错题没有可用框；有正确候选时 Qwen 仍经常保留错误 C 或选错实例/边界；局部细化又大量损害已正确框。保存字段不能给出各模态的因果贡献，也不能证明全部错误来自全局/局部坐标混淆。GT 修订使 53 条严格可比静态子集的命中数上升，但不能倒推全量新分数。逐题 IoU、来源类别、转移和分母见 [`per_query.jsonl`](../../../../tmp/dino_retrospective_20261025/per_query.jsonl) 与 [`metrics.json`](../../../../tmp/dino_retrospective_20261025/metrics.json)。
