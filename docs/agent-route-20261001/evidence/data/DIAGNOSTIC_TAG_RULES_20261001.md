# 诊断切片标签规则（2026-10-01）

本规则只根据英文 Query 文本和清单里的模态可用性生成启发式标签，不读取 GT、预测框或成绩。标签是词面代理（proxy），不等于人工语义标注；多标签允许重叠。所有诊断清单均独立写入本目录，冻结原清单未修改。

## 标签定义

- `sample:all`：所有记录，用于查看完整分母。
- `query:appearance_attribute`：出现外观、颜色、大小/形状、材质或穿戴词，例如 black/white/red/blue/grey、small/tall/round/striped、wooden/glass、wearing/shirt/jacket/hat。
- `query:posture_action`：出现姿态或动作词，例如 standing/sitting/seated/walking/running/lying/riding/crouching/kneeling/leaning/holding/carrying。
- `query:left_right`：出现 left/right/leftmost/rightmost。
- `query:ordinal`：出现 first–tenth、last 或 leftmost/rightmost/topmost/bottommost。
- `query:target_reference`：出现明确关系短语，例如 left/right of、next to、beside、behind、in front of、between、under/below、above/over、near、inside/outside、across from、at the edge/top/bottom of。
- `query:camera_distance`：只匹配明确相机远近词（如 closer/nearer/farther/closest to the camera、distance from camera）或 foreground/background；普通的“near the car”不单独算相机距离。
- `modality:rgb_available`、`modality:ir_available`、`modality:depth_available`：来自 `available_modalities`，City412 则由 `images` 字段名映射；`modality:rgb_ir_depth_available` 表示三类模态都列在输入中。
- `modality:metric_depth_known`：清单 `depth_encoding=city_mm`。这只标记已知编码元数据，不表示深度有效性或目标区可用性。

## 清单和完整 ID 核查

`dev40_manifest.jsonl`、`final80_manifest.jsonl`、`holdout120_manifest.jsonl` 本来都有非空 `capability_slices`，但旧标签只有较宽的 `text_proxy:*` / `metadata:*` 类别，没有明确分出外观属性、姿态、左右和相机距离。本轮因此额外生成新的显式诊断清单，原清单保持原样。

- `diagnostic_dev40.jsonl`：40/40 个唯一 ID，与 `dev40_manifest.jsonl` ID 和顺序完全一致。
- `diagnostic_final80.jsonl`：80/80 个唯一 ID，与 `final80_manifest.jsonl` ID 和顺序完全一致；生成只读取清单，不读取该 split 的 GT、预测或评分文件。
- `diagnostic_holdout120.jsonl`：120/120 个唯一 ID，与 `holdout120_manifest.jsonl` ID 和顺序完全一致。
- `diagnostic_city412.jsonl`：412/412 个唯一 ID，与本地 `preparation_seed2026_v2/city412.jsonl` 一致。另两份本地 `preparation_seed2026` 和 `preparation_seed2026_stratified` 清单在 ID、Query、图像字段名和 depth_encoding 上也逐行一致；它们均无 `capability_slices`。

实际 GPU 评估主机的来源清单为 `/root/autodl-tmp/rematch_20260922/results/visual_agent/capability_rebuild_20260929/data/city412.jsonl`；本轮取回副本为 `city412_eval_manifest.jsonl`。它与 `diagnostic_city412.jsonl` 逐行核对后，412/412 个 ID 唯一且顺序一致，Query 逐 ID 完全一致；RGB/IR/Depth 可用性及 `depth_encoding` 也全部一致。诊断标签仍沿用先前从三份本地 City412 副本生成的 `diagnostic_city412.jsonl`，本轮未替换或改写它。由此确认该诊断清单可对齐到 `capability_rebuild` 的实际 City412 评价输入。

输出只含 `id`、`source`、`split`、`query`、`available_modalities`、`depth_encoding`、`capability_slices`；不含 bbox、GT 或预测字段。每个切片 ID 集与对应输入 manifest 全等，重复 ID 为0。

## 评分完成后的诊断命令

`diagnose_capabilities.py` 要求 `--capability-manifest` 的 ID 集与 evaluation summary 的完整评分 manifest 完全相同。以下只准备命令，本轮未执行诊断，也未启动 final80 评分：

```powershell
Set-Location F:\AIC\code
python -m experiments.evidence_decision.diagnose_capabilities `
  --evaluation <dev40完整评分目录>\summary.json `
  --capability-manifest ..\results\visual_agent\evidence_decision_20260930\data\diagnostic_dev40.jsonl `
  --output-dir ..\results\visual_agent\evidence_decision_20260930\capability_dev40
```

final80 完整评分结束后，把 `--evaluation` 指向该次完整评分的 `summary.json`，并使用：

```powershell
Set-Location F:\AIC\code
python -m experiments.evidence_decision.diagnose_capabilities `
  --evaluation <final80完整评分目录>\summary.json `
  --capability-manifest ..\results\visual_agent\evidence_decision_20260930\data\diagnostic_final80.jsonl `
  --output-dir ..\results\visual_agent\evidence_decision_20260930\capability_final80
```

City412 评价完成后，`--evaluation` 应指向其完整评分 `summary.json`，且该 summary 的 `manifest` 字段应指向上述实际 GPU 主机清单 `capability_rebuild_20260929/data/city412.jsonl`。原始远端清单没有 `capability_slices`，因此 `--capability-manifest` 使用已经逐 ID/Query 核对的诊断副本：

```powershell
Set-Location F:\AIC\code
python -m experiments.evidence_decision.diagnose_capabilities `
  --evaluation <City412完整评分目录>\summary.json `
  --capability-manifest ..\results\visual_agent\evidence_decision_20260930\data\diagnostic_city412.jsonl `
  --output-dir ..\results\visual_agent\evidence_decision_20260930\capability_city412
```

此命令仅作后续评分完成后的示例；本轮只核对输入 manifest，没有运行诊断或评分，也未读取 GT、预测或成绩。
