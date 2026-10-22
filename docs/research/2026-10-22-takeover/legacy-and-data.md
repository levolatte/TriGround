# 传统模型、数据与评估代码接手审阅（2026-10-22）

## 审阅边界和覆盖

仓库为 `F:\AIC\code`，审阅基于 `sprint/qwen36-27b-native-resolution` 的 `60b6db5`。本报告只根据源码说明行为；比赛规则以 `F:\AIC\contest` 原件为准，模型实际成绩须用原始 GT 和完整预测重新核算。未改动业务代码、配置、数据或现有结果。

已逐文件阅读全文：`src/mm_grounding/` 的全部 10 个文件、`train.py`、`evaluate.py`，`configs/` 的 22 个 YAML 与 `depth_relations_v2.json`；`tools/` 范围内 121 个 Python 文件与 7 个 HTML/CSS/JS 界面文件。余下的 `tools/review_ui/queries_zh.json` 是 417 条翻译映射，已解析并核查结构，未逐条审定译文。`tests/` 的 90 个文件中，已阅读全文 86 个，另 4 个原生/vLLM 实现对应测试由并行审阅者负责。工具目录共 135 个受 Git 跟踪的 Python 文件；另外 14 个原生/vLLM 等文件由并行审阅者负责。本报告末尾列出逐文件清单。仅执行过此前的 8 个直接相关 CPU 测试文件；后续审阅未继续跑测试。被排除的 14 个工具和脚本级原生训练实现不在本报告结论范围。

## 主路径图谱

1. **较早的自定义多模态融合**：`train.py` 加载配置、处理器与 `MultiModalGrounder`，`GroundingDataset` 从 RGB、红外、深度及查询构造样本。模型有 `legacy_patch`、`safe_post_embed`、`rdt_deep`、`parallel_backbone` 四种融合形态；后者将 RGB/IR/Depth 通过共享视觉主干，配合独立适配器与稀疏融合。见 `src/mm_grounding/model.py:296`、`:335-410`、`:512-579`，`src/mm_grounding/data.py:32-115`。两阶段训练入口在 `src/mm_grounding/engine.py:721-797`，当前 22 个 YAML 均设置 `phase_b_epochs: 0` 和 `vision_lora_enabled: false`；不能将这批配置说成视觉 LoRA 训练。旧融合技术路线可以作消融或低成本基线，不能直接代表目前原生 LoRA 主线。
2. **原生 Qwen3-VL LoRA 数据路线**：`tools/prepare_qwen3vl_native_sft.py:75-145` 将查询写成 RGB 单图或多图 `<image>` 提示，把深度毫米图按固定映射变为可视 RGB（`:28-68`）；`tools/prepare_next_stage_data.py:29-39` 给 C/D/T/G/U 不同阶段设置配额，审查 RGBDT 与增广的人工状态（`:92-104`、`:251-313`），按源组合训练展示。原生训练器实现由另一位审阅者负责，此处只确认数据生产链。
3. **能力切片补强**：`tools/build_capability_slices.py` 修正两图输入应有两个 `<image>`（`:112-142`），限定 RGBT 超长查询（`:69-79`、`:180-229`），预渲染 RGBDT/City 深度避免原始 `I;16` 被错误解释（`:339-400`、`:423-468`），默认排除跨域 RoboRefIt（`:531-566`、`:580-587`），提供帧级测试重叠剔除（`:502-508`、`:542-544`）。这些是代码所载的设计与注释，不等于本次独立复验了其分布数字。
4. **候选与重排**：`tools/aux_selection_evidence.py:1-5` 声明只用查询、C 基线和图像建候选；通过 RGB/IR GroundingDINO 检测、IoU 去重、目标/参照数量上限，附 SAM2 与深度证据（`:138-290`、`:325-363`）。`tools/predict_aux_selection.py:73-101` 阻止把目标框或类别标签作为推理输入，先解析查询，再从匿名候选中选 KEEP/目标候选，必要时局部裁剪细化（`:115-248`、`:305-461`）。`tools/prepare_instance_candidates.py:1-10` 的同图同类框则来自真值标注，输出文件明示仅供 oracle 实例选择诊断，不是端到端 ACC（`:383-392`）。这两类候选必须严格分开报告。
5. **结构化修正**：`tools/grounding_revision.py:1-59` 用冻结 A 的建议框让第二次生成 KEEP 或 REPLACE，解析失败记失败。`tools/structured_grounder.py:66-148` 用视觉/语言 hook 插入模态证据及可选对象读出，`training_forward` 用 GT 形成修正监督（`:159-169`），`predict_two_calls` 在推理时直接运行两次调用，不读外部建议框缓存或 GT（`:222-224`）。`tools/prepare_grounding_revision.py`、`prepare_triground_structure.py`、`train_grounding_structure.py`、`evaluate_grounding_structure.py` 实现训练与 A/R/S/S+G 等结构化对比。此线与候选重排各自增加推理成本；部署前要用同一完整集核对净收益和时延。

## 评估口径与可信边界

`src/mm_grounding/data.py:122-174` 监督模型输出 0–1000 的 `bbox_2d`，数据框是归一化 xyxy；`src/mm_grounding/engine.py:174-280` 解析生成值，解析失败作为无效预测计入分母，输出 ACC@0.5、mIoU、解析率。`tools/report_rematch_experiment.py:74-221` 重新从原始 GT 算 IoU，`:280-337` 做同 ID 成对翻转和图像组 bootstrap，`:339-403` 的 Recall@1/4/8/全部候选是“候选中是否存在真值命中”的**上界覆盖率**，不是选择器最终命中率。`tools/report_triground_abv.py:37-93` 对完整预测要求 ID、单文件与元数据一致，`:214-340` 将不完整文件降为进度预览、分开已知复用 47 与其余 365、排序 C0/A/B/V 并做同权重模态缺失干预。候选门槛（净增 ≥4 命中且 ≥3 图像组）只是探索性门槛，不是因果证明。`tools/report_gu_diagnosis.py:1-20` 的 G/U 诊断另有固定序列分组，`fit_legacy181` 的 181/98 分母硬约束在 `:43-94`，City96 同输入模态差分在 `:117-168`。跨报表必须保持原始样本 ID、GT 与模态条件一致。

## 查到的可操作问题

- **低优先级潜在续训错误**：`src/mm_grounding/engine.py:777-797` 在 Phase B 续训前总是读取 `best_phase_a.pt`，随后把 `phase_b_done+1` 传进 `_run_phase`，却未传 Phase B 的训练 checkpoint；`_run_phase` 只有收到 `resume_checkpoint` 才恢复优化器、调度器和全局步数（`:417-458`）。如果未来启用 Phase B 且 `resume_epoch` 已跨入 B，模型会从 A 最优权重以较晚 epoch 继续，而非恢复 B。当前全部 YAML 的 B=0，所以不是当前运行结果的已证根因；启用前补充 B checkpoint 参数和小规模中断/续训等价测试。
- **历史转换器不应直接喂原生训练**：`tools/convert_rgbt_groundbench.py:132-141` 生成三模态提示，却仅写两张 RGB/IR 图，注释称训练侧补黑深度。训练脚本对应校验见 `scripts/run_qwen3vl_native_lora.sh:777`，而新的 `tools/build_capability_slices.py:112-130` 明确修为“图像数=占位符数”。旧转换器应标历史/迁移到两图协议，避免独立调用时错位；不应据此指控已修正的切片混合清单。
- **训练数据证据等级**：`tools/prepare_triground_abv_data.py:1-4` 默认只生审阅候选，release 才写训练数据，`:492-559` 的 quick 审阅只表示人工保留/接受，不等同细粒度模态必要性证据；`tools/run_triground_abv.py:24-44` 强制 ready 状态、固定步数与 B/V 同清单，`tools/run_triground_tm.py:35-72` 对 T/M release 与预算进一步约束。报告三模态“因果贡献”应优先依据同 checkpoint 的输入干预及成对翻转，不能由样本来源标签或 quick 审阅直接推出。
- **序列复用与泛化**：`tools/prepare_target_v2_data.py:52-125` 将人工复核样本、弱标注和序列键分离；`tools/report_triground_abv.py:239-243` 对 City412 强制识别已知复用 47/其余 365，但源码自身说明这只是地点/序列划分，不能证明场景独立。选模型时先看 365 子集、类别/尺度切片，再看总分；小样本类别结论受 `MIN_CLASS_SAMPLES=16` 限制（`tools/report_triground_abv.py:21`、`:101-120`）。
- **续评估配置遗漏**：`tools/evaluate_pretrained_grounder.py:796-797` 的 `--model-max-length` 会在 `:716-717` 改变 tokenizer 设置，且调用处 `:938-941` 确实传入；然而 `build_run_config`（`:343-385`）与调用（`:862-875`）均未记录该参数。`verify_run_config`（`:436-449`）因此不能识别更改了长度上限的续跑，汇总元数据也缺此值。建议将其加入配置及续跑比对，默认值 4096 可兼容历史记录。这是可复现性风险，未证明现有结果受损。

## 接手优先顺序

1. 用原始 GT、完整预测和 `run_config.json`/`summary.json` 复算目前候选模型的同 ID ACC@0.5、mIoU、解析率，并分开已知复用/其他样本与目标尺度。报告中的离线候选覆盖率另列上界，不与端到端指标混用。
2. 锁定当前线上推理实际采用的模型、权重、图像/深度预处理、提示词、量化坐标和最终提交脚本；对照训练端尤其深度渲染与双/三图占位符。不要因旧版融合配置和已过时转换器误判当前实现。
3. 在同一完整验证集上逐层比较 C/原生 LoRA/候选选择/结构化修正的净增命中、退化样本、图像组置信区间和每条推理开销，再选继续优化路线。先专注小目标与空间关系的错误簇，避免同时改数据、融合与解码而无法归因。
4. 如果重新启动 Phase B，先修上述续训路径；如果继续用 `prepare_instance_candidates.py`，输出必须保持 oracle 标识。旧 RGB-T 转换器宜弃用或修为两图协议。

本次 CPU 回归：`.venv\Scripts\python.exe -m pytest -q` 对数据、配置、适配器、target-v2、下一阶段数据、能力切片、rematch 报表和 ABV 报表共 **83 passed**。没有跑 GPU 推理/训练，未独立复验历史实验分数。

## 逐文件覆盖清单

传统模型源码：

```text
acquire_rgbdt500.py
acquire_roborefit_subset.py
analyze_gu_outcome_20260926.py
analyze_modality_inputs.py
analyze_modality_target_regions.py
apply_candidate_reviews.py
apply_grounding_reviews.py
audit_aux_selection_outputs.py
audit_multimodal_runs.py
audit_structure_cpu.py
auto_assign_candidate_destinations.py
aux_selection_evidence.py
build_abv_audit_bundle.py
build_capability_slices.py
build_grouped_subsets.py
build_gu_error_atlas.py
build_gu_low_agreement_review.py
build_review_ui_20260926.py
cache_grounding_proposals.py
check_external_image_overlap.py
check_gu_diagnosis.py
check_length_budget.py
check_stage1b_epoch3_ready.py
check_submission_format.py
compare_before_after_reports.py
compare_grounding_runs.py
compare_joint_checkpoint_deltas.py
compare_lora_resume.py
compare_structure_resume.py
continue_rematch_overnight.py
convert_rgbt_groundbench.py
convert_visual_genome.py
depth_relation_tasks.py
diagnose_8b_fusion.py
diagnose_modality_interventions.py
diagnose_parallel_checkpoint_deltas.py
diagnose_safe_v2_epoch1.py
download_model.py
evaluate_aux_reading.py
evaluate_grounding_structure.py
evaluate_locateanything.py
evaluate_pretrained_grounder.py
export_inference_checkpoint.py
export_triground_abv.py
gate_gu_pilot.py
grounding_revision.py
grounding_structure.py
inspect_fusion_checkpoints.py
merge_triground_reviews.py
parallel_range_download.py
predict_aux_selection.py
predict_competition_submission.py
preflight.py
preflight_extension.py
preflight_from_a3.py
preflight_new_base.py
preflight_phase_b.py
preflight_safe_v2.py
preflight_safe_v3_bbox.py
preflight_structure_gpu.py
prepare_8b_experiment.py
prepare_aux_selection.py
prepare_city_stage4.py
prepare_depth_reading_pairs.py
prepare_grounding_revision.py
prepare_gu_diagnosis.py
prepare_gu_same_day_pilot.py
prepare_instance_candidates.py
prepare_modality_interventions.py
prepare_multimodal_evidence.py
prepare_next_stage_data.py
prepare_query_position_ab.py
prepare_qwen3vl_native_sft.py
prepare_refcoco.py
prepare_rematch_run.py
prepare_review_repairs.py
prepare_rgb_ir_controls.py
prepare_rgbdt_annotation_jobs.py
prepare_rgbt_groundbench.py
prepare_roborefit.py
prepare_same_day_pilot.py
prepare_slurm_run.py
prepare_structure_evaluation.py
prepare_target_v2_data.py
prepare_triground_abv_data.py
prepare_triground_abv_shuffle.py
prepare_triground_structure.py
prepare_triground_tm_data.py
probe_retained_a_precision.py
rank_combined_results.py
render_aux_selection_atlas.py
render_city_audit_samples.py
render_leakage_pairs.py
render_object_evidence_review.py
render_triground_abv_atlas.py
report_8b_experiment.py
report_aux_selection.py
report_failure_modes.py
report_gu_diagnosis.py
report_modality_interventions.py
report_query_wording.py
report_rematch_experiment.py
report_structure_results.py
report_triground_abv.py
review_grounding.py
run_aux_selection_queue.py
run_grounding_structure.py
run_triground_abv.py
run_triground_tm.py
run_triground_tm_ir_two_pass.py
select_fusion_checkpoint.py
select_scene_coverage_candidates.py
select_weak_subset.py
stress_8b_training.py
structured_grounder.py
train_grounding_structure.py
validate_grounding_manifest.py
verify_a3_checkpoint.py
verify_acc05_dev.py
verify_mixture.py
verify_query_position_ab.py
```

界面源码（全文）：tools/review_candidate_split.html、tools/review_grounding.html、tools/review_ui/review.css、tools/review_ui/review.js、tools/review_ui/shell.html、tools/review_ui_20260926.css、tools/review_ui_20260926.js。
翻译映射（仅解析结构）：tools/review_ui/queries_zh.json（417 项）。

测试源码（全文）：

```text
test_8b_diagnostics.py
test_8b_experiment.py
test_8b_report.py
test_8b_runner.py
test_acquire_rgbdt500.py
test_adapters.py
test_apply_candidate_reviews.py
test_apply_grounding_reviews.py
test_auto_assign_candidate_destinations.py
test_aux_selection_evidence.py
test_auxiliary_training.py
test_bbox_stopping.py
test_build_grouped_subsets.py
test_capability_slices.py
test_check_submission_format.py
test_compact_checkpoint.py
test_compare_grounding_runs.py
test_compare_lora_resume.py
test_config.py
test_convert_manifest_to_swift.py
test_data.py
test_depth_reading_pairs.py
test_depth_relation_sampling.py
test_depth_relation_tasks.py
test_depth_review_revision.py
test_enrich_city_queries.py
test_evaluate_locateanything.py
test_evaluate_pretrained_grounder.py
test_evidence_decision_data.py
test_export_triground_abv.py
test_frozen_adapter_roles.py
test_grounding_revision.py
test_grounding_structure.py
test_gu_pilot_gate.py
test_lora.py
test_merge_by_predicted_size.py
test_merge_triground_reviews.py
test_modality_interventions.py
test_model.py
test_multimodal_evidence.py
test_predict_aux_selection.py
test_preflight.py
test_preflight_runtime.py
test_prepare_aux_selection.py
test_prepare_gu_diagnosis.py
test_prepare_gu_same_day_pilot.py
test_prepare_instance_candidates.py
test_prepare_next_stage_data.py
test_prepare_qwen3vl_native_sft.py
test_prepare_refcoco.py
test_prepare_review_repairs.py
test_prepare_rgbdt_annotation_jobs.py
test_prepare_rgbt_groundbench.py
test_prepare_roborefit.py
test_prepare_slurm_run.py
test_prepare_target_v2_data.py
test_prepare_triground_abv_data.py
test_prepare_triground_abv_shuffle.py
test_prepare_triground_tm_data.py
test_prioritize_blind_jobs.py
test_render_triground_abv_atlas.py
test_report_aux_selection.py
test_report_modality_interventions.py
test_report_query_wording.py
test_report_rematch_experiment.py
test_report_triground_abv.py
test_resume_learning_rate.py
test_review_grounding.py
test_rgb_ir_controls.py
test_rgbt_converter.py
test_run_grounding_structure.py
test_run_triground_abv.py
test_run_triground_tm.py
test_run_triground_tm_ir_two_pass.py
test_select_weak_subset.py
test_selection.py
test_slurm_scripts.py
test_sparse_checkpoint.py
test_structure_cache.py
test_structure_data.py
test_structure_report.py
test_structure_resume_comparison.py
test_structure_training.py
test_tm_evaluation.py
test_training_output_lifetime.py
test_visual_genome_converter.py
```

未深入阅读并由并行审阅者负责的路径：`tools/` 下 predict_submission_vllm.py、native_model_loading.py、native_lora_training.py、predict_native_submission.py、eval_qwen_native_grounding.py、eval_qwen_rgb_grounding.py、convert_manifest_to_swift.py、probe_native_resolution.py、enrich_city_queries.py、merge_by_predicted_size.py、audit_test_overlap.py、audit_manifest_overlap.py、filter_sequence_overlap.py、verify_vllm_token_parity.py；`tests/` 下 test_native_lora_runner.py、test_native_lora_training.py、test_predict_native_submission.py、test_predict_submission_vllm.py。
