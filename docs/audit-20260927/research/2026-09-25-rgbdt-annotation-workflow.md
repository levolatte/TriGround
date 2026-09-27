# RGBDT500 试标与独立复核工作流

本工作流只处理 RGBDT500 官方 Train 划分中**已下载且有原始跟踪框**的帧。当前源清单由 `code/tools/acquire_rgbdt500.py` 生成，图组字段含 `id`、`split`、`sequence`、`rgb`、`infrared`、`depth`、`bbox_xyxy_normalized`、`depth_units`；`query` 初始为 `null`。试标100图组、外部复查50图组来自不同序列。不能把未标框视频帧当作已标框训练图组。

## 一、生成者任务

入口：`code/tools/prepare_rgbdt_annotation_jobs.py prepare-generator`。每个 JSON 文件最多10图组。生成者可看到三路**原始图像**、额外的固定深度灰度预览，以及一个指定目标框，用于理解待描述对象。四种 `requested_focus` 依序分配为外观/普通位置、实例或序数、深度线索、红外证据；这是试题方向，不要求每图强行写该类题。当前深度单位为 `unverified`，不允许声称相机距离或制作精确近远数值题。无足够证据时输出 `generation_status=unsupported`，不要补造关系或温度。

候选 Query 采用英文自然描述，每图最多1–3条，须在当前单帧中唯一指向原目标。不能写“红框中的物体”或依赖前后视频帧才能定位的描述。组合目标可保持组合框，但不转成单对象实例选择辅助题。同一候选最多重写一次。

生成结果逐行保存 JSONL：

```json
{"query_id":"rgbdt500_010_00000231_q01","image_group_id":"rgbdt500_010_00000231","query":"the ...","focus":"appearance","generation_status":"proposed","rewrite_count":0,"evidence":{"rgb":"...","infrared":"...","depth":"..."}}
```

`focus` 的标准名为 `appearance`、`instance_ordinal`、`depth_relation`、`infrared_evidence`。实际生成文件也可写 `instance`、`depth`、`infrared`，工具会对应转换。`generation_status=generated` 等同尚待复核的 `proposed`，`skipped` 等同 `unsupported`；都**不表示人工批准**。`evidence` 可为非空文字或分模态记录，仅供制作与人工检查，不进入模型提示。Query 生成须真实看图；工具本身不会生成任何 Query。含固定深度预览的首批真实任务位于 `F:/AIC/results/rgbdt500_annotation_20260925/generator_jobs_depthpreview/generator_batch_000.json`，10图组。

## 二、盲复核任务

生成者输出保存后运行：

```powershell
F:/AIC/code/.venv/Scripts/python.exe -m tools.prepare_rgbdt_annotation_jobs prepare-reviewer --manifest F:/AIC/data/external/RGBDT500/pilot_manifest.jsonl --data-root F:/AIC/data/external/RGBDT500 --generated <生成结果.jsonl> --output-dir <独立复核目录>
```

盲复核 JSON 每批最多10题，**只含** `query_id`、三路未标记原图路径（及同一深度的固定灰度预览）和 Query。独立复核 Agent 不能读取生成任务、原目标框、生成者推理或“正确候选”标记；它重新预测 RGB 归一化 `xyxy` 框，判断是否歧义，以及表达依据是否在图中可见。复核输出逐行：

```json
{"query_id":"rgbdt500_010_00000231_q01","predicted_bbox_xyxy_normalized":[0.1,0.2,0.3,0.4],"ambiguous":false,"evidence_confirmed":true}
```

无法定位时 `predicted_bbox_xyxy_normalized` 可为 `null`。模型与原框不一致是待查信号，不能自动判原标注错误。

## 三、合并与人工验收

`merge-reviews` 在程序内部把复核预测与原始框算 IoU。只有 IoU≥0.5、`ambiguous=false`、`evidence_confirmed=true` 同时满足才标为 `provisional`；其他为 `needs_review`，缺失复核在汇总中明示。输出 `joined_reviews.jsonl` 和 `provisional_train_candidates.jsonl`。后者的 `review_status=provisional`，**不能直接投入 D/T 训练**；`prepare_next_stage_data.py` 只接收 `human_accepted` 或试标人工门槛通过后明确赋予的 `approved_batch`。模型一致不等于人工批准。

试标人工检查从自动初选中按四类各固定抽25条；外部复查集须有50图组各2条真实 Query，100条全部人工检查。两部分凑齐才运行 `prepare-human-pack`，生成 `human_review_200.csv` 与 `human_review_200.html`；未凑齐会报数量缺口，不用复制相似帧或虚构描述填满。HTML 展示原三图并在 RGB 上覆盖原目标框，CSV 留空 `human_decision`/`human_note` 供人工填写。人工验收门槛按批准计划：正确且唯一≥95/100，明显错目标或多解≤2/100，深度和红外题需可展示实际证据。失败类型先修订规则，不扩大该类型；独立100条复查不进入训练。

## 当前状态

- 首批10个试标图组图像已在本地，生成任务已写出；首批生成文件正在逐行增加，盲复核任务应记录生成文件快照的实际题数。工具生成任务时不自造Query。
- RGBDT500 原深度为16位且单位仍标 `unverified`。补标任务同时给出原图与固定灰度预览：零值黑色，正值按0–19999原始数值固定反向映射，越小越亮。原始16位文件保留；这只是防止普通图像处理器把高位深数据压成近乎白图的视觉编码，不宣称这些数值是毫米或可直接用来标“距相机更近”。合并结果使用 `depth_policy=sensor_linear_20000`，数值近远辅助任务继续禁止。
- 后续100图组生成、独立盲复核和200条人工包按真实下载与实际生成结果推进，不把未完成阶段写成通过。
