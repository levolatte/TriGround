# Query 位置编码 A/B：执行记录与云端操作手册

## 1. 本轮问题与实验边界

本轮只检验一个因果问题：辅助 Query Encoder 在投影后、Transformer Encoder 前加入无参数正弦位置编码，是否能提高比赛 `ACC@0.5`。

固定比较：

| 组别 | 配置 | 唯一处理差异 |
| --- | --- | --- |
| Treatment | `configs/stage2_joint_fusion_v3_positional.yaml` | `query_position_encoding: sinusoidal` |
| Control | `configs/stage2_joint_fusion_v3_control.yaml` | `query_position_encoding: none` |

两组均为 2 epoch、seed 2026、单末层 Joint Fusion、冻结 IR/Depth adaptor、关闭 Direct BBox Head 和 Vision LoRA、`fusion_lr_a=1e-5`、`modality_dropout=0.2`。固定运行顺序为 Treatment → Control，并分别启动进程。

## 2. 已实现内容

- Query Encoder 支持 `none` 和动态 `sinusoidal`；位置编码无参数、无最大长度表，因此旧 checkpoint 可直接加载。
- 训练、Early Probe、早停、checkpoint 选择及候选排序统一使用：`ACC@0.5 → mIoU → ACC@0.7 → parse_rate`。
- Phase A 保存 `best_phase_a.pt`、`best_mean_iou_phase_a.pt`、`last_phase_a.pt`；Phase B 对应保存 `best.pt`、`best_mean_iou.pt`、`last.pt`。
- checkpoint 的 `score` 是 `ACC@0.5`，同时保留完整 `metrics` 和 `selection_order`；旧 checkpoint 加载逻辑未改变。
- Joint Fusion 推理支持非负的 `ir_fusion_scale`、`depth_fusion_scale`。`0` 完全跳过该模态，`1` 保持原行为，中间值缩放投影特征、Query 分量和一致性特征。
- `evaluate.py --rows-output` 输出逐样本 JSONL；内容包括预测框、真值框、IoU、解析状态、场景分组、目标面积/尺寸分组、已有类别字段及已有 Query 类型字段。
- `tools/compare_grounding_runs.py` 完成同样本 A/B 指标、正确性翻转、IoU 差值、按 scene/sequence 聚类的 2,000 次 bootstrap 95% 区间和分层指标。
- `combined284` 排名脚本只保留诊断用途，禁止通过它复制主选模型。
- 提交工具可强制检查 9555 个 Query，并记录解析失败和整图 fallback 数量。

## 3. 云端预检：任何一步失败都停止正式训练

在仓库根目录执行。先安装与 CUDA 匹配的 PyTorch，再安装固定实验依赖：

```bash
python -m pip install -r requirements-experiment.txt
python -m pip install -e .
pytest -q
```

解析两组共同初始化点。脚本优先使用 `runs/stage2_clean_after_weak1024/best_phase_a.pt`；不存在时下载公开的 `triground-parallel-a-v1.pt`。来源、epoch、score、metrics 和 checkpoint 配置写入 `runs/stage2_query_position_ab/shared_init.source.json`。若 `source_kind` 为 `pre_joint_retrain`，实验解释为“重新训练最终联合融合阶段”；若为 `release_continuation`，解释为“在现有冠军模型上继续训练时的位置编码增益”。

```bash
python tools/prepare_query_position_ab.py
python tools/verify_query_position_ab.py \
  --output runs/stage2_query_position_ab/config_equivalence.json
```

审计当前训练集、弱监督祖先训练集、两个 adaptor 训练集与 119 条验证集的 ID、图像路径及 scene/sequence 重合：

```bash
python tools/audit_manifest_overlap.py \
  ../city_detection_prepared/train/reviewed_split/train_100.json \
  ../city_detection_prepared/train/reviewed_split/val.json \
  --additional-train ../city_detection_prepared/train/weak_subsets/weak_1024_raw.json \
  --additional-train data/RGBT-Ground-Dataset/extracted/subsets/train_50.jsonl \
  --additional-train data/RoboRefIt/manifests/formal_subsets/train_50.jsonl \
  --output runs/stage2_query_position_ab/validation_overlap_audit.json \
  --fail-on-overlap
```

只有三个 overlap count 全为 0 时，才能把这 119 条称作独立主验证集。否则停止，不删除样本，也不继续训练。

分别执行真实 Qwen 单批次前向/反向。由于 Joint Fusion 的恢复投影采用零初始化，预检内部做两次反向并在中间做一次仅驻留内存的优化步，以确认梯度最终能到 Query Encoder 和 Joint Fusion，同时确认冻结 adaptor 无梯度：

```bash
python tools/preflight.py \
  --config configs/stage2_joint_fusion_v3_positional.yaml \
  --expected-val-samples 119 --backward
python tools/preflight.py \
  --config configs/stage2_joint_fusion_v3_control.yaml \
  --expected-val-samples 119 --backward
```

若显存不足，只能同时修改两份配置中的 batch、分辨率或梯度累积，并重新执行配置一致性检查和两组全部流程。

## 4. 正式训练

严格按 Treatment → Control 顺序，各自使用全新进程：

```bash
mkdir -p runs/stage2_joint_fusion_v3_positional \
  runs/stage2_joint_fusion_v3_control
python train.py --config configs/stage2_joint_fusion_v3_positional.yaml \
  2>&1 | tee runs/stage2_joint_fusion_v3_positional/train.log
python train.py --config configs/stage2_joint_fusion_v3_control.yaml \
  2>&1 | tee runs/stage2_joint_fusion_v3_control/train.log
```

不要因为其中一组在 119 条验证集上暂时较差而取消另一组。

## 5. 三类 checkpoint 的完整离线评估

对每组的三个 checkpoint 分别运行一次 `evaluate.py`。Joint 配置的一次调用会同时得到 Triple、RGB-only、RGB+IR、RGB+Depth 四种模式，并产生逐样本证据。

```bash
for run_name in stage2_joint_fusion_v3_positional stage2_joint_fusion_v3_control; do
  for checkpoint_name in best_phase_a best_mean_iou_phase_a last_phase_a; do
    python evaluate.py \
      --config configs/${run_name}.yaml \
      --checkpoint runs/${run_name}/${checkpoint_name}.pt \
      --output runs/${run_name}/eval_${checkpoint_name}.json \
      --rows-output runs/${run_name}/eval_${checkpoint_name}.jsonl
  done
done
```

只对两组的 `best_phase_a.pt` 运行修复后的缩放与错配诊断：

```bash
python tools/diagnose_modality_interventions.py \
  --config configs/stage2_joint_fusion_v3_positional.yaml \
  --checkpoint runs/stage2_joint_fusion_v3_positional/best_phase_a.pt \
  --output runs/stage2_joint_fusion_v3_positional/modality_diagnostics.json
python tools/diagnose_modality_interventions.py \
  --config configs/stage2_joint_fusion_v3_control.yaml \
  --checkpoint runs/stage2_joint_fusion_v3_control/best_phase_a.pt \
  --output runs/stage2_joint_fusion_v3_control/modality_diagnostics.json
```

比较两组最优模型的 Triple 结果：

```bash
python tools/compare_grounding_runs.py \
  --control runs/stage2_joint_fusion_v3_control/eval_best_phase_a.jsonl \
  --treatment runs/stage2_joint_fusion_v3_positional/eval_best_phase_a.jsonl \
  --mode rgb_ir_depth \
  --bootstrap-replicates 2000 \
  --seed 2026 \
  --output runs/stage2_query_position_ab/paired_comparison.json
```

## 6. 官方提交文件

使用两个全新的 progress 文件，避免旧格式断点记录掩盖 fallback 统计：

```bash
python tools/predict_competition_submission.py \
  --config configs/stage2_joint_fusion_v3_control.yaml \
  --checkpoint runs/stage2_joint_fusion_v3_control/best_phase_a.pt \
  --queries /path/to/official_queries.json \
  --data-root /path/to/official_dataset \
  --expected-query-count 9555 \
  --progress outputs/qpos-control.progress.jsonl \
  --output-json outputs/triground-qpos-control-bestacc.json \
  --output-zip outputs/triground-qpos-control-bestacc.zip

python tools/predict_competition_submission.py \
  --config configs/stage2_joint_fusion_v3_positional.yaml \
  --checkpoint runs/stage2_joint_fusion_v3_positional/best_phase_a.pt \
  --queries /path/to/official_queries.json \
  --data-root /path/to/official_dataset \
  --expected-query-count 9555 \
  --progress outputs/qpos-sinusoidal.progress.jsonl \
  --output-json outputs/triground-qpos-sinusoidal-bestacc.json \
  --output-zip outputs/triground-qpos-sinusoidal-bestacc.zip
```

工具会检查 ID 数量与顺序、仅替换 `bbox`、有限归一化坐标及 `x1<x2, y1<y2`。官方数据只用于生成这两份提交，不能回流选模、训练或调参。

## 7. 判定规则

- Treatment 官方 `ACC@0.5` 严格高于 Control：本轮可判断位置编码有效。
- Treatment 官方成绩还严格高于 0.6785：才升级为新的比赛主模型。
- 官方同分时依次比较本地 `ACC@0.5`、mIoU、`ACC@0.7`、解析率；仍同分时选位置编码版本。
- 本地提高但线上下降：保留实现和配置，不替换主模型，记录为域偏移或单种子不稳定。
- 本轮只有 seed 2026，只能得出一次受控工程结论，不能声称跨种子稳定。

## 8. 待研究计划（本轮不实现）

1. Direct BBox Head + SmoothL1/GIoU，并比较 generated/direct 输出；这是下下一步首个单变量实验。
2. Fusion Stage 1→2，有效后才试 4。
3. adaptor 全冻结与仅解冻最后 1～2 层。
4. Joint Fusion legacy warm start 和延长训练。
5. IR/Depth 专属输入 Stem、质量估计和配准处理。
6. 同步增强、模态退化训练和新弱 Query。
7. 多尺度、滑窗与模型集成。
8. GroundingDINO 候选路线，先验证 recall@K。
9. 更大的 Qwen 主干。

以上均只是待实验假设。必须等待本轮两个线上分数返回后，再正式设计 Direct BBox Head 实验。
