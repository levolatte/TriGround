# EGM 快速接口核对与退出记录（2026-09-22）

## 结论
按用户本次明确要求，退出当前 EGM 路线，删除云端专用权重、下载辅助脚本和零散日志。保留完整逐样本结果与必要配置，后续不要重新下载或自动重跑 EGM。

E0 为 139/412（33.74%），原生 Qwen R0 为 273/412（66.26%）。EGM 仅纠正 R0 的 4 条错误，同时损失 138 条；理想候选并集上限 277/412。当前配置下没有继续投入的证据。

## 本次实际核对
- 官方仓库：https://github.com/NVlabs/EGM
- 官方接口：https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.py
- 官方启动：https://github.com/NVlabs/EGM/blob/main/verl/scripts/sglang_infer.sh
- 官方 qwen3 定位提示与当前 EGM 提示一致，完整 Query 未截取或改写；412 条 Query 与原始清单一致，也都存在于 R0 输入中。
- RGB 图像路径及原始浮点 GT 在 E0/R0 全部一致。R0 的 query 元数据为空是因为 Query 位于 SFT prompt，不是输入缺失。
- 官方使用 0–1000 xyxy；当前除以 1000 的归一化方向正确。
- 官方取首框、当前取末框；离线检查所有 412 条都只有一个 bbox_2d，首末解析结果完全一致。
- 官方温度 0、生成上限 4096；当前贪心、上限 4096。全部 412 条有 </think> 思考结束标记、解析成功、没有触及生成上限。均值 107.12 个生成 token。
- 实际视觉网格全部 [1,36,64]，对应 576 个视觉 token、589824 像素。

## 结论的边界
这不是官方后端的完整复现：本地使用 Transformers/SDPA，官方示例使用 SGLang/vLLM；本次约 60 万像素限制也不同于官方未显式限制像素的示例，权重处理配置默认 longest_edge=16777216。官方还在映射至原图后做像素取整。
没有发现提示、Query、框解析或坐标缩放的明显接错，但未做高分辨率/官方后端的新一轮 GPU 复验。因此结论仅为当前预算配置和本任务数据上淘汰 EGM，不是证明其全部能力无效。按照用户要求快速结束该路线，不中断正在运行的 R2 去追加实验。

## 清理与恢复信息
已删除 /root/autodl-tmp/rematch_20260922/models/EGM-8B（含专用下载缓存）及 EGM 专用下载/旧 smoke 脚本、零散日志和下载清单。
云端 results/egm_retired_20260922 保存模型/处理/生成配置、压缩辅助脚本日志和 cleanup.json。
E0/e0_smoke 预测、指标、完成标记保持原位，确保现有主 runner 重启时不会再加载已删除模型，后续汇总仍能读取历史 E0。
共享推理入口、正在运行的 runner、Qwen/LocateAnything 权重与环境均未修改。
本地完整证据：F:/AIC/results/rematch_20260922/e0/；删除明细：F:/AIC/results/rematch_20260922/egm_cleanup.json。

实测释放 16.48 GiB；数据盘剩余 30.56 GiB（删除完成时）。
