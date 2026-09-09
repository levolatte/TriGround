# Query Fusion 因果诊断

该诊断用于回答：在主 Qwen 始终接收正确文本查询的前提下，额外把 Query 注入三模态 Fusion 是否确实改善定位结果。

诊断只执行推理，不更新任何模型参数。它包含三个模式：

| 模式 | 主 Qwen Query | Fusion Query | 作用 |
| --- | --- | --- | --- |
| `query_correct` | 正确 | 正确 | 当前完整模型基线 |
| `query_zero` | 正确 | 关闭 | 移除 Fusion 中的 Query 信息 |
| `query_shuffled` | 正确 | 其他场景、其他文本的 Query | 判断 Fusion 是否读取了 Query 语义 |

`query_shuffled` 只替换独立的 `query_input_ids`，不会改变 Qwen 生成提示中的原始 Query。donor 优先选择不同场景且文本不同的样本，并由 seed 固定。

## 运行

使用当前方案二 checkpoint 和与其匹配的配置：

```bash
python tools/diagnose_modality_interventions.py \
  --family query \
  --config configs/stage2_joint_fusion_v2.yaml \
  --checkpoint /path/to/triground-parallel-a-v1.pt \
  --output runs/query_fusion_causal/diagnostics.json
```

快速检查可以增加 `--samples 32`；正式诊断不设置 `--samples`，使用配置中的完整验证集。

输出包含三种模式各自的 `ACC@0.5`、mIoU、`ACC@0.7`、解析率和逐样本结果，并在 `paired_vs_query_correct` 中记录：

- `acc_0.5_delta`；
- `mean_iou_delta`；
- 错误转正确与正确转错误的样本数；
- IoU 改善、退化和不变的样本数。

逐样本记录同时保存 `main_query` 和实际送入 Fusion 的 `fusion_query`，便于确认错配只发生在辅助 Query 路径。

## 解释

```text
query_correct > query_zero > query_shuffled
```

说明 Fusion 利用了正确 Query，而且错误 Query 会损害结果。

```text
query_correct ≈ query_zero ≈ query_shuffled
```

说明当前 Fusion 基本没有使用 Query；继续研究 Query 位置编码的优先级较低。

```text
query_zero > query_correct
```

说明当前 Query 注入总体有害，值得重新训练 Visual-only Fusion。

```text
query_correct > query_shuffled，但 query_zero > query_correct
```

说明 Fusion 确实读取了 Query，但当前注入方式带来的净收益为负。

该诊断是对现有 checkpoint 的推理干预。即使 `query_zero` 更好，也只能证明关闭当前 Query 路径有即时收益；要证明 Visual-only 架构更好，仍需从相同初始化重新训练有 Query 和无 Query 两组模型。
