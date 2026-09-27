# G/U 同队列小试验执行阶段记录（2026-09-26）

本阶段只完成本地训练入口、复验门槛和 CPU 验证；没有连接云端、启动 GPU 或声称任何模型收益。训练队列入口为 `code/scripts/run_gu_pilot.sh`，默认远端根目录 `/root/autodl-tmp/rematch_20260922`，输出 `results/gu_pilot_20260926`。默认读取该输出目录中的 `released_cloud.jsonl` 与 `released_metadata.jsonl`，City 源为 `data/city/train/target_v2/qwen3vl_native_sft/trimodal_train.json`。云端必须用 `--city-root` 重新生成配对清单；本地已生成的 City 绝对图路径不能直接用于云端。

训练沿用 M2 `results/first_batch/m2/checkpoint-928` 的 LoRA 权重，每个新阶段重新创建优化器和线性学习率调度器；学习率 `5e-6`、BF16、SDPA、TF32 关闭、像素范围 `200704..602112`。正式小试验每分支、每 seed 训练总计 200 更新，在 100 更新保存并暂停，完整评估 412 条 City，再恢复同一优化器至 200 更新并评估 412 条。seed2026 过门槛才生成 seed2027 的相同策略队列。每个 seed 都重算 M2、C、G100、G200、U100、U200 的完整报告。

`PRESERVE_MANIFEST_ORDER=1` 才使用显式 `SequentialSampler`、零 worker 和样本轨迹；默认训练行为仍维持原来的随机采样及两个 worker。轨迹写在成功返回的 `Trainer.training_step` 之后，预取的样本不会入账。断点恢复把检查点更新数乘以 8 作为已提交样本数；若异常留下额外轨迹，先备份到 `uncommitted_after_checkpoint_*.jsonl`，再截断并重放。压力预检遍历 8 个代表任务的图像网格、token、监督答案；16 个 microbatch 在第 1 更新保存暂停并恢复第 2 更新，核对优化器、调度器、随机状态和完整 16 条轨迹，随后两次加载同一检查点比较 2 条 City 预测。正式训练前要求 10 GiB 空闲，各训练段要求 3 GiB；阶段日志、完成标记和 PID 均位于输出目录。

City 门槛使用原始 GT 重算：至少一个 G/U 分支在 seed2026 的 200 更新达到 ACC@0.5 ≥ 300/412、相对 C 纠正至少 6 个图像组、mean IoU 和解析率均不低于 C，才做 seed2027。U 胜出要求两个 seed 各自通过且 ACC@0.5 都高于 G，平均净增至少 4 条；否则 G 两 seed 都通过且方向不冲突时选 G，其余停止。外部 600 更新前需 100 个唯一 ID 的隔离原 GT、同 ID 的 native 清单及逐条 `source=<EXTERNAL_SOURCE>`、`split=external_review`、`review_status=human_accepted` 的人工复核 JSONL；M2 与胜者须在同一主机、同一基座与同一评估设置下重新评估，胜者外部 ACC@0.5 相对 M2 下降不得超过 3 个百分点。缺少外部文件会记录 `WAIT_reviewed_external_100` 并正常结束，不训练 600。过门后按胜者策略、seed2026 从 M2 新阶段训练 600 更新，保存并评估 200/400/600；保存上限设为 3，保留三份检查点。

本地验证：`tests/test_gu_pilot_gate.py` 与 `tests/test_native_lora_runner.py` 共 10 项通过；两个 shell 脚本通过 `bash -n`，门槛工具通过 `py_compile`。GPU 上的压力轨迹、恢复及预测一致性尚待实际运行验证。
