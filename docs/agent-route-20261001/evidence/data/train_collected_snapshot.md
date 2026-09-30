# 采集课程扩充快照

这是当前可用原始采集轨迹的扩充中快照，不是冻结版课程。每个样本仅选一份原始会话轨迹，不拼接不同会话的前缀；完整轨迹优先，已执行且独立盲审认可的部分轨迹也会保留。有合格审核步骤的恢复轨迹优先，其余按完整状态、合格步骤数与来源顺序选择。开发集/最终集图组不会进入训练输出。

导出仅监督审核认可且协议有效的原动作；不完整轨迹上的 finish 不作为监督，训练集 GT 只在离线过滤已完成轨迹中的错误 finish 时使用。错误工具历史保留在原轨迹上下文中，合格的 search/inspect/depth 等证据动作仍可监督。输入为 raw `fresh` 及 `fresh_reviews.jsonl`；不会再次读入 `fresh_collected*` 旧导出。

- 训练行数：1064（teacher 314，原seed非teacher行逐行保留 750 条）。
- 原始轨迹：139，选中 134 条；选中完整轨迹 129 条，选中部分轨迹 5 条。
- 新teacher动作计数：`{"frozen_seed_teacher": {"depth": 6, "finish": 34, "inspect": 43, "search": 3}, "new_real_teacher_trajectory": {"depth": 43, "finish": 79, "inspect": 65, "search": 41}}`。
- 新teacher按轨迹状态计数：`{"complete": {"depth": 42, "finish": 79, "inspect": 55, "search": 39}, "partial": {"depth": 1, "inspect": 10, "search": 2}}`。
- 冻结holdout输出行：0。
- 逐轨迹选择记录：`train_collected_snapshot.selection.json`。
- 完整统计：`train_collected_snapshot.summary.json`。
