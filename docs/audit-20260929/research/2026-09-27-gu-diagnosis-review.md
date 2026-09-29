# B/G*/U* 三臂诊断独立审查

日期：2026-09-27。范围为本地人工验收状态、根因审计、三臂诊断的 CPU 清单及运行入口。未使用 GPU、SSH，不改比赛标签或其他 Agent 的代码。正式远端 Linux 清单与原图可读性仍需启动前检查。

## 人工验收状态

- `results/multimodal_data_20260925/human_acceptance/human_results/` 保留此前两份正式人工 CSV，各 100 行；它们已用于 181 条放行任务，不应再次当作待提交。
- `results/depth_layer_subjects_20260926/human_review8_v1/` 目前仅有 `decisions_blank.csv`：8 行 `review_decision`、`near_to_far_candidate_letters` 全空。实际下载目录 `F:/Downloads` 中，2026-09-26 起近期 CSV 仅有旧 natural100 QA 文件，没有这 8 组的新人工提交。因此 8 组仍为 `pending_human`，不能生成关系训练标签。

## 清单与评分核查

审查对象为 `.work/gu_diagnosis_root_check/` 的真实 CPU 构造结果、旧 `results/gu_pilot_20260926/final_cloud_snapshot/manifests/seed2026/`，以及原始 City 和放行候选。`python tools/check_gu_diagnosis.py manifests` 通过。

- B、G*、U* 各 1600 行；原有 1200 个 City 位置在三臂中逐字段等于旧 G 清单，旧 G/U 的 City 行本来相同。其余 400 槽沿用旧 ID、位置和顺序。
- B 将 98 个新来源 Query 一对一映射到旧 City 1200 中未出现的 98 个 City Query，City 样本 ID 也未出现过；98 个不同 City 图组，400 次呈现的 Query 重复谱为 `2:26, 3:2, 4:33, 5:22, 6:7, 7:6, 8:2`，与原 98 个新来源 Query 完全一致。
- G*/U* 的 400 行三张图路径逐行相同；两者提示逐行只在“输出 RGB/infrared/depth 图坐标”指令不同。RGB 行提示相同；答案分别用原始浮点 RGB 框或已放行目标模态框量化为 0–1000 整数。所有 181 个放行 ID 的原始浮点框均能与 `pending_candidates.jsonl` 精确对应，训练整数答案也逐项复现。原 181 条中 103 条的原浮点框与千分位 SFT 答案有非零舍入差，评分侧车使用原浮点框。
- `fit_legacy181_gt.json` 为 181 条原浮点 GT；`fit_canonical_aux83_gt.json` 是其中 83 个已放行辅助任务的相同 GT 子集。两者的三张原图路径与云端放行清单逐 ID 一致，按物理三图组合分别为 98 图组。
- City96 的四种输入组合具有同一 96 ID、同一原始浮点 GT；每行 `<image>` 占位数等于图像数。共享评分侧车保留三张完整原图路径，报告按 78 个物理图组聚合，不会因输入组合减少图像而改变重采样单位。City412 源 GT 的 412 个框也逐 ID 等于历史 M2 的目标。

## 运行入口与已修阻断

运行队列固定 8B、BF16、SDPA、关闭 TF32、最大像素 602112；B/G*/U* 均以 M2 适配器初始化并各训练 200 次更新，100 步完整保存恢复，之后才做完整 412 条评估。`gpu_start_epoch` 固定 8 小时预算，完成标记与 checkpoint 防止重复启动；无自动第二 seed、关系训练或扩大训练。旧 G/U 只用于诊断推理。

审查期间发现并由对应 Agent 修正的阻断：旧 G/U 远端预测路径误加本地 `final_cloud_snapshot` 层；报告以文件路径运行导致 `ModuleNotFoundError`；数据初版误要求 98 Query 各有三模态标签；fit/City96 GT 缺少报告按图组重采样所需的三张图路径；`--released-candidates` 的文件名与解析 schema 一度不一致。当前运行入口显式使用已同步的 `source/pending_candidates.jsonl`，本地 181 个放行项可解析；`python -m tools.report_gu_diagnosis --help` 及上述清单检查通过。

结论限于本地 CPU 预检：**现有真实清单未见数据或评分契约阻断**。正式运行前仍需远端生成清单、检查 Linux 路径和图片可读性，并先通过 8 题旧环境复现及 1→2 步恢复预检。City96 模态组合的结果应作为描述性诊断，不能单凭输入组合差异宣称模态因果贡献。
