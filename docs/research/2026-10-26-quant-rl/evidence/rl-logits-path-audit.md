# GRPO 短 logits 路径审计（Swift 4.5.3）

现有 `probe-a-runtime-audit.json` 中 `selected_logits_forward` 记录 2158 次调用，`first/minimum/maximum` 均为 1。这个旧 hook 安在 `trainer.model.get_base_model()`。本地 PEFT 0.20.0 的 `BaseTuner.forward` 直接调用 `self.model.forward(...)`（`peft/tuners/tuners_utils.py:332-333`），绕过底层模型的 `nn.Module.__call__` 和前置 hook。因此该计数只证实了生成调用，不能证明训练或参考分布前向的 logits 形状。D 臂当前运行仍使用旧 hook，继续完成原实验。

Swift 源码路径：

1. `swift/rlhf_trainers/grpo_trainer.py:759` 在训练 microbatch 调用 `collate_to_grpo_micro_batch(..., use_logits_to_keep=True)`；`swift/rlhf_trainers/utils.py:2168-2170` 从非 `-100` 的回答 labels 算出批次回答区宽度 `logits_to_keep`。
2. `grpo_trainer.py:777-793` 的旧策略和参考分布，以及 `:948-954` 的当前策略损失，都调用 `_get_per_token_logps_and_entropies`。
3. 本实验无 sequence parallel，该函数进入 `_get_logps_via_local_forward`；`grpo_trainer.py:1564-1568` 对支持该参数的模型传 `logits_to_keep + 1` 并执行前向。随后 `:1570-1573` 截取对应区间计算回答 token 的 log probabilities。
4. Transformers 5.16.1 的 `Qwen3_5ForConditionalGeneration.forward` 在 `modeling_qwen3_5.py:1785-1787` 先选 `hidden_states[:, -logits_to_keep:, :]`，再执行 `lm_head`。故词表投影宽度是本批回答区宽度加 1，而完整输入的隐藏状态仍需计算以提供上下文。

Qwen3.5 的 Swift 前置 hook 会改写输入形态：`Qwen3_5Template._post_encode` 委托 `Qwen2VLTemplate._post_encode` 生成 `inputs_embeds`（`swift/template/templates/qwen.py:604-608,440-450`），`Template.pre_forward_hook` 随后移除 `input_ids`（`swift/template/base.py:1677-1688`）。因此审计的输出 hook 从实际传入的 `input_ids` 或 `inputs_embeds` 取 `shape[1]`。CPU 测试覆盖了外层 PEFT hook 后发生这一改写的路径。

本地训练工具已将 hook 移至 `trainer.model`（PEFT 外层），保留 `selected_logits_forward` 字段记录它观察到的所有调用，并新加 `grpo_logps_forward`，只记录 Swift 本地 logps 方法内的输入序列长、请求的 `logits_to_keep` 与输出 logits 序列长；其 `with_grad` 和 `without_grad` 分别覆盖训练及旧策略/参考计算。CPU 小型 GPT2+LoRA 实验证实：旧底层 hook 漏记，外层 hook 记录 25 位置的真实输出，方法外模拟的逐 token 调用仅进入总计数。后续正式臂可取得模型上的运行时实证，无须重启 D 臂。
