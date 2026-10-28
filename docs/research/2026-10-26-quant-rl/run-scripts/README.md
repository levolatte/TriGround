# 本次云端运行脚本快照

运行根固定为`/root/autodl-tmp/quant_rl_20261026`，训练工具位于其`code/tools`；模型和正式数据路径保留实际云端值。这里只保存可复核源码，不包含凭据、模型或数据。

- `run_optimized_arms.py`：读取A完成记录与优化决议，运行B/C/D和匹配曝光全量评价；依赖`run_initial_arms.py`中的检查点选择函数。当前队列已启动，不能重复运行。
- `evaluate_warmup50_baseline.py`：队列全部完成后，补共同初始化的City378与外部88全量贪心评价；使用既有final阶段预算，整段上限1 GPU小时。`--check-only`只核对启动条件；当前队列运行时拒绝启动评价。
- `run_initial_arms.py`与`complete_a_before_optimization.py`：记录A所用的旧队列和接续方式，旧任务已结束。
- `optimization_job.py`：通过唯一账本运行短测，计入阶段间占机并采样整卡显存。
- `deterministic_fixed_benchmark.py`与`check_causal_conv_kernel.py`：固定回答的确定性包装和独立卷积内核检查。
- `live_optimization_check.py`与`run_live_optimization_sequence.py`：合格候选的真实生成、保存重载及C/D检查；本轮没有固定回答合格候选，因此未执行。

当前状态、决议、预算与未采用原因见[优化记录](../OPTIMIZATION.md)。解释器必须保留`train_env/bin/python`或`train_env_opt/bin/python`启动路径，不能解析软链接后改用基础Python。
