# 恢复比较器顺序误报修正

2026-10-01人工核查：父6466在完成BF16缓存和R三段预训练后停止于R_compare。冻结A已完整复现历史City412=296，全部412条原文、输入词元数和图像网格一致；训练1045/1045和评估673/673，0解析失败。

R_compare首个异常为adapter_config.json.target_modules.0。PEFT将该字段作为无序集合保存，独立Python进程导出的列表排列可不同。既有compare_lora_resume已经处理同一语义。本次仅排序adapter_config中列表形式的target_modules；正则字符串、其余配置、所有模型权重、Adam状态、调度器、随机状态、消耗样本顺序仍精确比较，没有放宽数值容差或略过状态。云端2项针对性回归测试通过，覆盖换序通过、成员/秩/正则变化失败、样本顺序和微小张量变化失败。随后真实R和S的连续4步与断点2+2全部状态比较通过。

从完成阶段继续，不重复缓存及R训练；新父11852，正式R子12440，各臂400步。预算预测12959.11秒，选择时原账本剩余36578.88秒；预估含加载和裕量，不含可选第二种子。旧错误GPU计算保留，不重置账本；原GPU累计含当前段6646.18秒。正式准确率尚无。

证据：results/triground_structure_20260930/A_historical_recheck.json、preflight/R/comparison.json、preflight/S/comparison.json、formal_horizon.json、budget_forecast.json、gpu_budget.json。进度快照与进程号须按实际账本核查，不当作持续健康证明。
