# 27B结构路线可实施性审查

2026-10-24。依据[91条复核](../../../../docs/research/2026-10-24-user-review/README.md)、[历史实验](../2026-10-23-reassessment/history.md)与[定点文献](../2026-10-23-reassessment/literature.md)；本轮只读，未训练、推理或改模型。91条是旧标签下筛出的失败题，不能估计总体收益；用户本轮只修这些题，不要求再审原321条命中。

**首轮保持27B原生三图、一次直接出框、冻结视觉侧。** `tools/prepare_qwen3vl_native_sft.py` 现把三张图与Query直接映射到单个`bbox_2d`；`tools/convert_manifest_to_swift.py` 保留该输入/答案；`scripts/train_qwen36_27b_swift.sh` 以ms-swift做语言LoRA。新增监督应当是经核实的“同一目标完整可见范围”和“相似邻居为何不满足Query”，以及确能对应到主体区域的IR/Depth关系，不能把全框非零深度或一段模型自述当证据。可将少量同图A/B竞争框与短原因作为**训练辅助行**，保留等量直接框行；部署提示与输出仍沿用`tools/predict_submission_vllm.py`。对照臂使用同批可靠场景、同初始权重、同直接框呈现次数与优化预算，比较无候选单次框的配对救回/损失；只会回答A/B而不改善直接框，即否定迁移假设。修订91条只供机制诊断，效果须在独立场景判断。

若主体IR信号或主体有效深度确实存在、配准可信，且给定正确区域仍读不出目标属性或前后关系，再动视觉侧。**最小真实结构变更是仅开放现有对齐层LoRA**：`scripts/train_qwen36_27b_swift.sh` 已有`FREEZE_VIT=true`、`FREEZE_ALIGNER=false`开关，三图输入、语言骨干与最终单框均可保持；ms-swift[官方参数说明](https://swift.readthedocs.io/en/v4.1/Megatron-SWIFT/Command-line-parameters.html)确认`all-linear`下冻结标志决定哪些模块加入LoRA。与视觉全冻结臂必须从同一基座和随机种子启动，用同一清单、深度渲染与评估流程；现有脚本的续训入口要求完整训练状态，不应假设能直接给Arm2适配器增添视觉模块。不能拿同时改变数据/视觉侧的Arm4/5作因果对照。若对齐层仍不能读出IR域内轮廓，再**单独**试视觉LoRA；每次核对实际适配器张量归属。当前vLLM入口`tools/predict_submission_vllm.py`只启用普通LoRA，未开`enable_tower_connector_lora`；[vLLM文档](https://docs.vllm.ai/en/stable/features/lora/)称该支持为实验性，训练可行不等于当前5690题推理入口可部署，需先验证目标版本和Qwen3.6-27B的加载与输出一致性。

RGB二维patch作锚的局部IR交叉注意力与原始Depth数值/validmask投影残差，**目前只是设计候选**。`src/mm_grounding/model.py`和`adapters.py`是旧8B Qwen3-VL改前向实现，不能移植为27B模块：当前27B模型类型为`qwen3_5`，其[视觉实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modular_qwen3_5.py)与旧代码依赖的DeepStack接口不同。新模块需逐图解析`image_grid_thw`及merger后二维位置，处理三模态网格大小、相机错位、零深度与遮挡，并改训练加载和vLLM多模态前向；稠密跨图注意力还会抬高二倍分辨率下显存。若与原生三图**共存**，零残差的起点是原生三图；若**替换**IR/Depth图token，零残差的起点是RGB单图，绝不能称“回退到现役三图”。只有匹配数据下视觉侧小改仍失败、且可靠专项题显示跨模态对应而非对象选择是瓶颈，才值得支付这项开发与部署成本。

一次“全图＋局部观察”仅在小目标/完整范围错误有可靠触发、裁剪含目标及关键邻居或参照物时试；固定原图坐标与裁剪变换，保留全图，和等计算量的直接二次采样或提高分辨率比较。旧Agent的坐标混淆、裁剪退化不能被换名绕过。Rex只作为待测候选源：先量出它相对**当前27B**在可靠错题上新增的正确实例覆盖，再量真正选择后的净收益；旧8B候选oracle及Rex论文检测分数均不可外推。没有新增覆盖或选择净收益，就继续单模型直接框路线。
