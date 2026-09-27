# 源码版本说明

- code_final：完整412实际使用版本，包含首次评分前的空检测与数字编号接口修复。此版本结果是选择273/完整221。
- code_snapshot：首次冻结的原始版本；code_patches保留评分前修复与原结果。
- code_corrections_not_evaluated：事后按原批准计划改正居中裁剪截边，增加crop_policy_version避免恢复时混用。11项选择器CPU测试通过，未重跑GPU，没有412分数。
- output_audit：仅事后CPU检查和GT诊断，任何反事实坐标解释均未回写预测。

工作树tools/predict_aux_selection.py是最后一项裁剪修正版。复现实测v1应使用code_final，不可把修正版称为已经测得221或更好的结果。
所有模型仍在云端models和原C适配器位置，未随结果重复下载。
