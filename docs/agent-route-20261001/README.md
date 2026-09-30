# Agent路线隔离存档与审计

分支：agent-route-20261001。基于原工作区HEAD 5e4bc3c，从独立工作区提交；只新增experiments/visual_agent、experiments/evidence_decision、两项Agent专用根测试及本目录。主开发工作区原有HANDOFF、scripts、tools及其它测试改动未纳入本提交，也未切换原工作区分支。

实验已按用户要求停止，定时任务PAUSED；正式训练未启动，最终80未评分，不得自动接续。城市完整412题：冻结C292/412，新接口未训练217/412，纠正11、破坏86、净-75。失败结果、课程缺口、无证据推断及停止复盘见experiments/evidence_decision/POSTMORTEM_20261001.md。

本目录evidence保存原始评分和最近课程统计；它们是已完成工件快照，不是新一次实验。local_execution是原F:/AIC/tmp下实际操作脚本的原样档案，保留原路径和外部认证会话假设，供审计阅读，不承诺从本目录直接执行。未包含凭据、权重、全部工具图、AGENTS.md或最终80标注。详细审计ZIP由F:/AIC/audit_exports单独交付。
