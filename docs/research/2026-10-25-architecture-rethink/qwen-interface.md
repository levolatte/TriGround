# Qwen3.6-27B 接入连续定位头的实际接口

2026-10-25。以下来自本地模型配置、当前项目入口，以及官方 Transformers 中 `qwen3_5` 的实际源码；未加载27B权重、未做运行时前向。

## 已确认的模型与版本

本地留存配置 `F:/AIC/tmp/semifinal_research_20261022/perception/qwen36_config.json` 指定 `Qwen3_5ForConditionalGeneration`：语言宽度5120、64层；视觉宽度1152、27层、patch16、空间merge2，合并后宽度5120；`deepstack_visual_indexes=[]`。因此不能按旧Qwen3-VL的DeepStack注入代码设计本模型。

项目接手报告记录历史训练使用 Transformers 5.16.1、推理使用5.18.0。本次分别读取对应官方源码：
- [训练版本 v5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [推理版本 v5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

另检查了main和v5.3.0的接口变迁，不将其当作现有云环境版本。新实验尚未确定安装版本，不能声称当前本机已具备历史训练环境。

## 语义分支可以直接读 hidden

`Qwen3_5ForConditionalGeneration` 内部包含 `model=Qwen3_5Model` 和 `lm_head`。底层 `model.forward` 完成图像编码、占位token替换、M-RoPE位置计算和语言前向，返回 `last_hidden_state`。本方案调用这一层，取完整Query位置的状态；无需先生成坐标文本，也无需计算248320词表上每个位置的logits。

```python
# 架构接口示意；尚非已运行实现。qwen为未再包外层的原生HF模型。
outputs = qwen.model(**processor_inputs, use_cache=False, return_dict=True)
query_hidden = outputs.last_hidden_state[query_token_mask]  # 按batch恢复 B×Lq×5120
semantic_memory = bridge(query_hidden)                     # B×Lq×256
boxes, target_logits = grounder(spatial_memory, semantic_memory, bert_tokens)
loss = box_loss(boxes, gt_boxes) + target_loss(target_logits, matched_target)
loss.backward()
```

实际实现需要保留batch与padding mask，不能直接把不同长度Query混成一个序列；PEFT包装后的模块路径也以实例结构为准。图片放在Query之前，保留整句Query的token边界及原生处理器生成的 `image_grid_thw`、`mm_token_type_ids`、attention mask。训练前向使用 `use_cache=False`；不用生成调用产生的旧RoPE/cache状态。

只冻结Qwen主权重、开放语言LoRA时，不能用 `torch.no_grad()` 包裹其前向，否则框损失无法训练LoRA。冻结语义分支的预热阶段可以缓存hidden；联合阶段须在线计算。桥接层末端若严格零初始化，第一个更新步只到桥的输出层，随后才能向上游传播；采用很小的非零残差系数可以从开始就保留弱梯度。检查梯度时需据此区分正常初始化与断图。

## 视觉空间表示的实际位置

两个历史版本的视觉模型都返回：

- `last_hidden_state`：merge之前的patch表示，宽度1152；
- `pooler_output`：2×2合并后的视觉token，宽度5120。

`get_image_features` 根据 `image_grid_thw.prod(-1)/4` 切分各图合并token，随后按图像占位符写入语言输入。merge之前的顺序按2×2局部组重排，并非可任意 `reshape(H,W)` 的普通光栅顺序；位置工具的顺序必须一起复用。v5.18.0新增 `mm_encoder_outputs` 输入，v5.16.1的底层forward没有这个参数。

**主方案暂不替换这些视觉token，也不改语言模型内部层。** Qwen仍走原生RGB处理，额外的高分辨率RGB/IR/Depth由专用空间定位支路编码。这样语义接口只依赖末层Query hidden；不需要为新三模态memory伪造Qwen的图像网格、位置编码或缓存。

## 与仓库现有入口的关系

当前 [训练脚本](../../../scripts/train_qwen36_27b_swift.sh) 默认语言LoRA、冻结视觉塔与aligner；[HF推理](../../../tools/predict_native_submission.py) 和 [vLLM推理](../../../tools/predict_submission_vllm.py) 都是三图输入后 `generate`，再解析bbox。自定义定位decoder不属于普通vLLM LoRA请求所能表达的模块。

实现应新增一条自定义PyTorch/HF训练和推理入口，复用现有数据读取、官方输出格式与坐标检查。它仍是一张联合计算图：框损失训练定位decoder、三模态融合、语义桥和语言LoRA。原生生成式入口保留为现役对照。

本次实际审读的是 `Qwen3_5VisionModel`、patch embed/merger、`get_image_features`、图像占位mask、位置计算、`Qwen3_5Model.forward`、`Qwen3_5ForConditionalGeneration.forward`；没有逐行审计DeltaNet/FlashAttention等底层算子，亦未声称完成运行时兼容验证。

