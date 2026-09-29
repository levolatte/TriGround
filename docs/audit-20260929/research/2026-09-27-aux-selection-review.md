# C 候选—辅助模态选择：输入与评分接口审查

2026-09-27；本文件只记录既有资产和后续实现的审查要点，不改变训练、预测、GT 或人审结果。

## 冻结基线和数据

- C 的完整 412 条预测：`results/next_stage_20260925/c_step1500/predictions.jsonl`，同目录 `summary.json`。后者明确写明适配器是云端 `/root/autodl-tmp/rematch_20260922/results/next_stage_20260925/c_phase2/checkpoint-500`，命中 292/412、解析 412/412。`c_step1500` 是预测结果目录名，并非另一个 C 权重。完整 City 412 云端输入：`/root/autodl-tmp/rematch_20260922/data/city/train/target_v2/qwen3vl_native_sft/trimodal_val.json`，浮点 GT：同级 `qwen_generation_val.json`。
- 本地原始浮点 GT 快照：`results/gu_diagnosis_20260927/source_snapshot/city_gt.json`，键为 ID，值含 `bbox`、`query`、`visible`、`infrared`、`depth`。`city_train.json` 和 `city_val.json` 分别为完整训练、验证的三图 SFT 清单；`image` 顺序为 RGB、IR、深度可视图，深度可视图相对路径为 `target_v2/qwen3vl_native_sft/depth_rgb/...`。这两个训练/验证清单里的模型答案为整数千分位框，离线评分必须用 `city_gt.json` 的原始浮点框。
- 已有本地图像 `results/failure_analysis_20260926/source_images/{visible,infrared,depth,depth_rgb}` 是错例图册所需的部分图组，不能视为完整 412 图或 3707 训练图。云端 City 原图根为 `/root/autodl-tmp/rematch_20260922/data/city/train`。`depth` 是原始深度，`depth_rgb` 是模型曾读取的深度可视图；两者不能混用或把像素亮度直接当毫米值。
- 固定 96 ID、浮点 GT：`results/gu_diagnosis_20260927/cloud_snapshot/manifests/city96_gt.json`；同目录 `city96_{rgb,rgb_ir,rgb_depth,trimodal}.json` 是四种既有输入，选样种子 2026。C 的相应真实预测在 `cloud_snapshot/c_city96_*/predictions.jsonl`，三图为 73/96。这 96 条只作固定诊断，不替代完整 412。
- 已知重用场景 47 ID：`results/gu_diagnosis_20260927/cloud_snapshot/report_city412_diagnostic/summary.json` 的 `city412.shared47_ids`，其余 `other365_ids`。原报告说明这是七种已目视确认的同地点序列样式，帧不同，不是逐像素重复；不能将 365 称为全部独立新场景。

## 最小评分与审查组合

`code/tools/report_rematch_experiment.py` 可直接导入 `load_manifest(path)`、`load_run(name,path,set(gt),allow_partial=False)`、`score_run(run,gt)`、`compare_runs(C,new,gt)`。其 `box_iou` 使用归一化 `xyxy`，阈值是 **IoU ≥ 0.5**；`compare_runs` 输出纠正/退化 ID、图像组三元组（visible/infrared/depth）及默认 10000 次图组配对重采样。新预测 JSONL 只需 `id` 与归一化 `prediction`，供该评分器读取；它会忽略行内自报的 `target/iou/acc_0.5` 并从浮点 GT 重算。`load_run` 会拒绝缺失 ID、额外 ID、重复 ID、非法框；正式评分不要启用 `allow_partial`。

`code/tools/compare_grounding_runs.py` 的 `compare(control,treatment,bootstrap_replicates,seed)` 接受已带 `iou/parsed/target/scene_group` 的行；它相信行内 IoU，因此对新 C 比较应优先用上一评分器。若需类别/Query 分层，再先用原始 GT 重建行内 IoU 后调用。

`code/tools/build_gu_error_atlas.py` 的 `build(gt_path,c_path,m2_path,specs,old_dir,image_root,out)` 可用于纠正/退化图册。`gt_path` 指向上述 `city_gt.json`；`old_dir` 为 `results/failure_analysis_20260926`，`image_root` 为其 `source_images`。此工具只为离线展示读取 GT，缺图会列在 `missing_images.jsonl`，不能把它的 GT 叠框或旧 16 例审核信息输回模型。默认 C 路径已经是 `c_step1500`。

新一轮应先冻结 C 412 框，从无 GT 输入表生成真实候选，再以 IR/Depth 证据选择，最后只对选中框在 RGB 局部精修。`input.jsonl` 应只有 `id,query,images:{rgb,ir,depth_visual,depth_raw},depth_encoding:'city_mm'`；基线单独为 `id,bbox`，查询分析和证据单独文件。候选生产和模型公开提示均不得读取/携带 GT `bbox`、旧评估 `target/iou/hit/acc_0.5`、已知 47/96 成员标签或能暗示正解的 source 标签。评分代码是唯一可读取 GT 的路径。局部裁剪坐标需从裁剪空间明确反算到原 RGB 归一化 `xyxy`；对解析错误要显式记失败，不能静默回退成 C 而将解析率虚报 100%。RGB-only、RGB+IR、RGB+Depth、三图消融须保持候选集和样本 ID 完全一致，并报告是否复用 C 框。

## 后续新文件审查状态

`aux_selection_evidence.py`、`prepare_aux_selection.py`、`report_aux_selection.py` 已进行首轮静态审查。`predict_aux_selection.py` 待落盘；另有并发修复正在进行，以下结论以本次读取版本为准。

- **已明确配额，修复进行中。** 参考物设计为 RGB/IR 各 top2，target 为 C+RGB/IR 各 top3。首读版本的 `select_candidate_cap` 对 reference cap4 仍传默认 top3，导致 RGB 优先占3、IR只占1；需调用时显式 top2。主 Agent 已发并发修复。
- **设计局限，非阻断。** `assess_depth_mask` 以 eroded core ≥32有效像素、有效率≥70%及 full/core 中位数稳定性判可靠，full 有效率只记录不设硬门槛。这是本轮确定的主体内部门槛；外围大面积无效仍可通过，需要在报告中保留这个限制，勿称整块掩码深度可靠。
- **非阻断：RGB/IR 来源覆盖率的归因稍宽。** 合并候选在 IoU≥0.9 时可能把 RGB 与 IR 来源放在同一条；该条最终 bbox 可取分数较高来源，但 `report_aux_selection.candidate_summary` 会把它算作两路检测器各自覆盖。候选整体覆盖和端到端得分不受此影响；单路检测器覆盖若用于结论，应按各自 `source.bbox` 单独重算。
- **非阻断：同类目标与参照物。** `_label_role` 用类别字符串判 role；target/reference 都是 `person` 之类同类时，相同标签优先归 target，单独的 reference 池为空。若关系推理需要同类实例，可让后续选择器在必要时比较 target-target；否则需要特别报告该类样本无法形成 Depth pair。
- 已核对 `prepare_aux_selection` 公开输入字段仅 `id/query/images/depth_encoding`；C 基线另存 `id/bbox`，原始 GT 在 `scoring/` 下。`city_train.json` 3707、`city_val.json` 412、原浮点 GT 412。`report_aux_selection` 使用 `load_run` 完整 ID 校验与原 GT 重算，主分数路径未信任预测行里已有 IoU。SAM `qualities.argmax()` 对已实测的三掩码/三质量形状可选择同索引最高质量掩码。

`predict_aux_selection.py` 首读版：`load_manifest` 拒绝 `bbox/gt/target_bbox/class_name`，只给推理层 `id/query/images/depth_encoding`；选择提示公开候选编号与坐标，但不公开 `sources/is_baseline/mask_path`。Depth 数值只在三模态且显式 camera near/far 问题中公开可靠候选的 full/core 统计。解析失败生成 `prediction:null` 且 `parsed:false` 并打印 `INVALID_MODEL_OUTPUT`，没有静默记为 C 命中。局部裁剪用 floor/ceil 后**实际像素边界**形成 `transform`，`map_crop_bbox_to_original` 按此变换反算，静态坐标逻辑正确。

主 Agent 已指出三处并发修复：选择 ID 只能是 target 候选，`needs_refine` 需改为「框相对 C 发生变化或短边<0.06」，大图编号可读性。除这三处之外，首读选择器没有发现新的阻断性 GT 泄漏或坐标错误。待修复落盘后需复核实际代码；当前结论不是运行验收。

## 队列与报告二审（静态）

`run_aux_selection_queue.py` 将 smoke 与 full 分开。env8精确复现、train16基线、Query解析、候选、Depth和选框精修逐项完成后才写 `smoke_completed.json`；full 另外要求 `configuration_frozen.json`，不会因 smoke 结束自动启动。每阶段完成标志在子进程退出零值及完整ID验证后才写。`report_aux_selection.py` 的完整412主报告调用既有 `load_run(...allow_partial=False)`，从原浮点GT重算，图组配对重采样10000次；净增才安排固定96条件诊断。部分预测无法通过阶段完整性检查，也无法写 `gpu_queue_complete.json`。

推理命令均只传公开 manifest、C 框、query-info 和候选证据；`scoring/city412_gt.json`、`scoring/city96_gt.json` 仅在离线 report 命令中传入。`prepare_aux_selection.py` 公开 manifest 仅含 `id/query/images/depth_encoding`。评分报告的 `candidate_summary` 最新版按每路 `source.bbox` 算单模态检测覆盖，已修正先前合并候选框引起的归因误差。

**剩余时间边界风险：** 队列用 `execution_window.json.deadline_epoch` 作为绝对截止并向每个子进程传 timeout；但入口未验证该截止与窗口起点相差不超过8小时。若窗口文件被误配为过晚，代码会允许超出批准预算。启动前需审核窗口文件或在入口加跨度断言。子进程 timeout 抛异常时 `stage_status.tsv` 可能只留下 start 行，但不会写阶段完成标志或完整结果；这属于状态解释问题，不会误报全412完成。

`aux_selection_evidence.py`、`predict_aux_selection.py` 的多项并发修复在二审时已见代码：reference RGB/IR各 top2、SAM每候选最高质量掩码索引、选框仅 target、精修条件按相对C变化或小框。仍以主线程实际运行和最终版本验收为准。

## 完整412后精修与固定12例目视审查

真实 C→selection→full 为292→273→221；固定96条件诊断未触发。`results/aux_selection_refine_20260927/failure_review.jsonl` 与 `failure_review.md` 记录了评分后seed2026固定抽出的4纠正、8退化逐例目视：每例均实际打开编号RGB、IR、Depth可视图和selection/full原GT叠框图。没有把12例的类别频次外推全412，也没有从有Depth数值推断模态因果。

代码审查确认精修模型的两张图实际顺序是全景标框RGB、无标记RGB局部crop，提示也要求在第二张图用0–1000整数坐标；现有v1坐标反算依照实际像素crop边界，独立104次数字输出复算最大误差0。固定退化样本中3例的`refine_raw`若**事后**当作全图千分位框，原GT IoU为0.736、0.529、0.527；这支持这3例模型没有按提示遵守crop坐标，不构成可上线后处理，也不解释其他全部退化。其他固定例中有精修框过小与一个选择阶段换错杆、精修KEEP。

v1另有确定性计划偏差：靠图边时原代码平移裁剪窗口保尺寸，而计划要求居中扩张后截到图内。最终CPU审计为41例边界策略差异加1例取整差异，共42，含11例精修退化；其余42/53精修退化无法由这点解释。工作树已改成居中截边规则并通过针对性CPU测试，但历史v1推理代码与预测均保留，修正版本未做GPU评测，不能声称修正后成绩。43个相机近远Query只有35个提示附数值表、19个可靠pair，成绩C32→selection33→full27；没有96消融，不能判断Depth/IR是否提分。

主线程补记：队列现已校验 `0 < deadline_epoch-start_epoch <= 8*3600`，并在超时写入 `budget_expired`；以上时间边界待修项已落实。首次真实train16选择出现1条reference编号被拒绝，队列确实阻断了full；具体原因和一次通用提示修正记录见执行报告，不将该失败隐藏为C回退。
