# TriGround A/B/V 执行说明

本轮计划经用户批准：修复训练 FP32 精度与恢复，比较 City 继续训练 A、专项混合任务 B、加视觉出口 LoRA 的 V。三组从历史 C 开新优化器；推理继续 BF16。GPU 总预算 12 小时，准备数据与人工等待不计入。历史结果、旧队列和官方提交均不改动。

## 当前实施边界

代码、CPU 测试与候选审核包先完成。候选经过机器预筛仍是未批准数据；只有真实人审达到既定门槛后，构建器才生成 `release.json`。没有放行清单就不能运行 A/B/V 队列。

- 当前重构包：`F:/AIC/results/triground_abv_depthv2_20260928/review/index.html`。旧包 `triground_abv_20260927` 继续保留历史题目和记录。
- 审核台可直接填写、自动保存浏览器草稿并导出 CSV；将下载结果另存为同目录 `decisions.csv`。也可继续填写 `decisions_blank.csv` 的副本。
- 阶段报告：`F:/AIC/docs/research/2026-09-27-abv-implementation.md`。
- 旧候选选择器和裁剪修正的分数不作为本轮成绩；旧 C=292/412 与新 C0 同环境重测必须分开。

## 训练与推理约定

`scripts/run_qwen3vl_native_lora.sh` 增加：

| 参数 | 含义 |
|---|---|
| `LORA_SCOPE=language` | 36 层语言 q/k/v/o，r32/alpha64，共288张量 |
| `LORA_SCOPE=language_merger` | 上述语言层＋主merger及3条DeepStack的fc1/fc2，视觉r8/alpha16，共304张量 |
| `VISUAL_LORA_LR=2e-5` | 视觉参数组学习率；语言由 `LEARNING_RATE=5e-6` 指定 |
| `INIT_ONLY=1` | 通过与训练相同的初始化路径导出零步适配器，随后退出 |
| `PYTHON_EXECUTABLE` | 明确训练解释器；队列与评估用同一环境 |

冻结基座 BF16，LoRA 与 Adam 一二阶状态 FP32。历史 C 以 `INIT_ADAPTER` 加载；新阶段恢复以 `RESUME_FROM_CHECKPOINT` 加载，二者互斥。复合适配器的视觉 B 初始为零，语言288张量必须逐值继承。训练模块只审计已知参数集合，不解冻视觉编码器。

队列显式固定有效batch=8、默认600更新步、min/max pixels=200704/602112、max length=4096、线性衰减、无预热、梯度裁剪1、seed2026。新增视觉学习率是预定假设，不能据验证分数试调。

## 数据、人审与冻结

默认600步共4800次呈现：旧City2880、Depth关系512、普通竞争448、IR互补480、可靠性480。专项目标64/56/60/60题，最低48/42/40/45；不足目标时缺额补旧City，不能用增加重复次数补题。专项每题最多8次，原City每Query最多2次。B/V完全同序，A/B/V的共同旧City位置一致。

2026-09-28用户明确改为快速看图审核：同时显示Query、RGB框和辅助图，直接保留/剔除/跳过；不再要求姓名、盲审顺序或逐项填写。保留表示题框可用，未填写的模态证据保持未知，不能据此声称已完成盲审或证明互补性。未知Depth单位仍不标米制。新题/诊断GT读取原始浮点标注；训练输出统一RGB框0–1000，评分使用原始[0,1]框。

同场景/序列派生题保持同一分组。City诊断只是本轮留出，不能称历史C未见；官方5690和City412不进入训练。RGBT始终实际两图。缺失辅助使用同尺寸空白并明确提示，正常City提示保持原样。

审核台默认中英对照，支持搜索、放大、自动保存及快捷键1/2/3。备注与英文修改可选，辅助模态判断不强制；点击保留/剔除后自动看下一道未决定题。图片是最大1920×1440的JPEG预览。

CSV增加review_mode和modality_judgment两列，快速审核无需姓名、旧式逐项证据或备注即可被放行工具读取；旧结构化CSV仍支持。快速保留单独标记human_quick_approved，未填字段不伪造yes，冻结统计记录审核模式与实际模态判断。浏览器草稿沿用原键，不清旧决定；CSV用于后续放行，页面不自动训练。仅刷新页面用 `python -m tools.prepare_triground_abv_data review`。

数据工具的 `prepare` / `revise-depth` 从旧包继承未改题，用 `configs/depth_relations_v2.json` 的逐图规格替换 Depth 草稿，默认写入独立的新包；已有输出目录不覆盖。`review` / `release` 仍默认旧路径，对新版须明确 `--output F:/AIC/results/triground_abv_depthv2_20260928`。`release` 接受真实决策表。用各工具 `--help` 查看路径参数，不要手写 `status=ready` 绕过人审。

新版深度题不再从部分标注推断全场景最近/最远，也不按最大深度差选题。规格写明比较范围、目标/竞争对象/参照物、原始框内的主体采样区域及中英题意；仅从原始标注读取浮点 GT。支持限定集合比较、明确参照物和中间层单题。深度统计检查零值、超出本轮可用范围、原始顺序和可视化灰度；统计通过仍不等于已证明模型需要 Depth。

新版只有 `pair_required=true` 的正反题要求整对保留，其他单题可以独立放行。按场景轮转、关系家族均衡选题；保持强制对相同呈现次数、单题最多8次、缺额补旧City、B/V同序。旧版两题组继续使用原约束。

建议通过 `http://127.0.0.1:8766/depth-v2/index.html` 打开新审核页，与旧页保持同一浏览器和域名端口，自动复制未改题的记录。旧深度决定和备注另外保留，不转为新题批准；也可导入旧 JSON 备份。原页面及其浏览器记录不删除。

600步、seed2026的冻结文件位于 `data/release_600_seed2026/release.json`；400步与seed2027各自生成新目录，已有目录直接报错。冻结目录同时保存真实审核行、场景曝光统计及历史清单对照。诊断每类不足16题只作个案描述，不额外阻断满足训练门槛的数据发布。当前外部场景的历史独立性仍未被证明。

## 云端部署与串行执行

本地先用 `tools.export_triground_abv` 将已放行清单导出到全新deployment目录。它仅复制专项实际需要的绝对路径图片，并重写为部署路径；普通City相对路径继续由云端City根解析。RGB/IR/Depth文件按原字节复制，不改变原编码。

运行配置包含 `python`、`repo`、`qwen_finetune_dir`、`data_root`、`model`、`initial_adapter`、`city_manifest`、`city_gt`、`budget_dir`。已核云环境示例保存在结果根的 `execution_config.cloud.json`。代码部署到本轮 `code_snapshot`，不覆盖旧云端代码或实验目录。City412 manifest/GT另放本轮 `inputs`。

云端调用示意（同一套参数，只改phase）：

```bash
python -m tools.run_triground_abv \
  --config /ABS/execution_config.cloud.json \
  --release /ABS/deployment/release.json \
  --output-dir /ABS/runs/seed2026 \
  --phase initialize
```

默认只显示命令；加 `--execute` 执行。顺序为 `initialize → preflight → train → evaluate`。`--resume` 仅跳过本轮已完成的GPU阶段，不自动重试失败训练或覆盖部分预测。CPU报告和图册每次重新生成，避免将之前的partial报告误当最终报告。

- initialize：导出三组初值，并以同一8题核对C0/A/B/V提示、网格、原始文本和框。
- preflight：每组连续32步，与16步保存恢复至32步对照；始终保持600/400正式调度周期。比较LoRA、Adam、调度、消费序列及Python/NumPy/CPU/CUDA随机状态，必须通过才能正式训练。
- train：从主路径checkpoint-32继续至既定终点。保存间隔由16改为200并记录，保留上限5个本轮检查点。
- evaluate：C0/A/B/V各完整412及专项正常/缺失条件，随后CPU重算报告。
- report：独立CPU报告和翻转图册阶段；GPU失败或预算到时也可生成pending/partial报告，不把局部分母当正式成绩。全部翻转案例的原因默认为未知，图册支持人工填写并导出JSON。

每个GPU子进程（含模型加载）按wall time计入共享 `budget_dir/gpu_budget.json`，失败耗时也计入，人工等待与CPU报告不计。种子2027复跑使用新output目录及 `--arms B`（或实际入围臂），但必须继续使用同一个budget_dir，不能重置12小时。

train前以预检实测速度及8题推理耗时估算剩余训练与全量评估，保留10%余量。若600步不容纳，停止；用相同人审清单重新生成400步release，在新output目录重做对应调度的预检，仍共用预算。不能修改已运行的MAX_STEPS。400也不容纳时不启动正式训练。

入围后可运行一次 `tools.prepare_triground_abv_shuffle --normal <云端normal.json> --scene-map <云端scene_map.json> --output-dir <新目录>`。它只按固定种子选择异场景、同模态且同尺寸供体，保持RGB、Query和提示不变；缺少合格供体的ID明示跳过。每个模态必须在输出的 `eligible_recipient_ids` 上重算normal对照，不能拿子集置换成绩与全量normal比较。此诊断不在自动三臂队列中，禁止用置换结果挑框。

## 成绩与保留规则

2026-09-28已完成14题措辞配对推理：历史C在当前完整版命中11/14，简洁版8/14，新增3次对象选择错误、无救回。保持当前完整Query；新题优先写清限定对象、参照物与关系，再删重复套话，不机械追求短句。该结果仅来自5个已知train场景，不证明泛化或Depth收益。详见[研究记录](../../docs/research/2026-09-28-query-wording/README.md)。本次GPU含加载48.1223秒已记入原共享预算，后续不得重新从12小时起算。

报告原浮点IoU、ACC@0.5/mIoU/ACC@0.7、47/365、按场景bootstrap、救回/损害及翻转ID。非法预测按0计完整分母；缺行报告partial且不排序。专项每类至少16题才作类级描述，场景本身不受此数量门槛限制。

先看A→B与B→V的正常输入对照，再看同一检查点正常与删模态的净收益。B→V不是“模态效用”本身。净增至少4题且救回跨3图组才优先复跑；1–3题是弱信号。未第二种子复现不称稳定提升，未改善mIoU/ACC@0.7不称框精度提高。所有初选门槛均不是统计显著性保证。

没有收益时保留C与已验证的工程修复，报告负结果，不自动扩数据、恢复旧候选细化、启动强化训练或提交比赛。
