# 纠错1600半决赛推理加速

## 当前瓶颈与选择

原入口沿用开发集 HF/NF4、三图及每批2题；半决赛实测约4.04秒/题，5691题约6.4小时。当前48GB RTX4090的已观测显存约24GiB，优先尝试提高批量，保持权重、图片像素、提示、贪心解码和输出规则一致。

复赛 vLLM 方案确实存在：老机为96GB Blackwell、BF16模型、32并发与前缀缓存。2026-10-28老Arm2对同一半决赛数据采用二倍图像与更高像素上限，实际耗时6小时26分；不能将其他历史小样本速度直接外推到当前任务。

当前云端 `/root/vllm_env` 安装vLLM0.30.0，其Qwen3.5映射包含 `in_proj_qkvz`、`in_proj_ba` 等纠错适配器所需目标。最初只查主包未发现bitsandbytes实现，漏查了官方插件迁移，不能据此判断vLLM不可用。

用户要求直接尝试适配的vLLM后，已安装官方 `vllm-bnb-plugin==0.0.3` 与 `bitsandbytes==0.50.2`。注册与GDN适配器映射已通过实机导入。现使用原BF16基座在线NF4双重量化，保留视觉、输出头及融合小门控 `in_proj_ba` 为BF16，直接加载纠错1600适配器。无需下载或保存另一份模型。来源：[官方插件文档](https://docs.vllm.ai/en/latest/features/quantization/bnb/)、[官方插件仓库](https://github.com/vllm-project/vllm-bnb-plugin)。

## 本轮最小改动

- 提交入口新增 `--resume`，复用已准备数据，按原查询顺序检查已完成ID前缀，追加剩余原文，最终打包完整5691题。
- 原配置保留，续跑批量写入 `resume_history`。模型、数据路径、像素、提示协议和生成参数不能随续跑改变。
- HF批量改为正整数，默认值保持原值；独立实测单次加载模型，比较批量2、4、8。
- 固定16道遍布官方清单的题目，记录各档原文、框、耗时和显存。候选须原文与框完全相同且至少提速5%，再反序复测确认；不通过则原批量2续跑。
- 实测由独立云端监督接管，保留已完成预测，实测最长10分钟。失败或未通过均明确记录，并继续原批量；训练仍保持用户暂停。

## 接续入口

云端根目录：`/root/autodl-tmp/quant_rl_20261026/runs/semifinal_correction1600_20261029`。

- 新监督：735874；脚本 `acc_code_20261029/scripts/run_semifinal_batch_upgrade.py`。
- `batch_benchmark/report.json`：实测各档数据与候选。
- `batch_decision.json`：实际采用决定。
- `inference_resumed.log`：继续全量推理日志。
- `state.json`：最新状态与子进程；`state_before_batch_upgrade.json`保留旧入口。
- `full/predictions_raw.jsonl`持续追加，不丢弃原预测。

实现相关36项检查通过；实际加速以云端实测结果为准。

## 批量实测结果与vLLM接续

16题：批量2为63.446秒；批量4为61.429秒；批量8为57.203秒。后两者原文与框均出现变化，批量4涉及12题，均未采用。974题原始预测保留，已从该处恢复原批量2。

vLLM新监督756966接替735874，入口仍为 `run_semifinal_batch_upgrade.py --probe vllm`。状态 `benchmarking_vllm` 是计划实测，最长15分钟；输出 `vllm_nf4_probe/report.json`、`vllm_benchmark.log`。先以并发8、8192上下文、不启用前缀缓存与编译做真实加载和16题吞吐。试跑结束先恢复HF并保留所有预测，再依据输出和实际速度决定后续部署；不得把仅插件导入成功写成GPU推理成功。训练父257429继续保持暂停。

## vLLM已跑通，完整成绩核对接续

真实NF4模型加载占用19.33GiB；首次初始化182.09秒。相同16题推理33.088秒（2.068秒/题），相对HF63.446秒约1.92倍吞吐，全部合法框。10题框不完全相同，主要为0—1000坐标上的1—6单位差异；不能据此宣称成绩相同。

CPU核对首题大图和004028_003小图深度样本：Swift原始编码与vLLM保存的完整提示词元逐项相同，分别3491和776词元。初步证据与原文保存在[实测归档](2026-10-29-inference-speed-evidence/)。

新监督769064接替756966，`--probe vllm_deploy`调用 `tools/validate_vllm_and_resume_submission.py`。单次加载模型，按相同NF4和并发8依次完成City378及外部88；命中不得低于313及84，含图片准备的开发集每题耗时须不高于3.6秒。满足后同一模型直接续写剩余官方预测，不重新推理已完成题目；否则明确记录并恢复HF。

最新状态见 `state.json` 的 `stage`，完整报告与决定见 `vllm_validation/{report,decision}.json`，新日志 `vllm_deploy_benchmark.log`。通过后 `full/run_config.json`追加vLLM运行段，最终summary按HF/vLLM分别记录。全部完成仍打包到原 `full/` 并按原计划下载。

## 完整对照结果：未采用vLLM

22:44完整结果为City308/378（81.48%）、外部84/88（95.45%）；HF纠错1600为313/378与84/88。City减少5题，外部持平。466题含请求准备耗时976.76秒，2.096秒/题，速度通过、成绩未通过，因此未将vLLM用于正式预测。

云端已从保留的1210题自动恢复HF批量2，监督769064、推理子770381，日志 `inference_resumed_vllm_deploy.log`。之后实查正式1438/5691，最近256题约4.055秒/题。完整决定见[归档](2026-10-29-inference-speed-evidence/vllm-validation-decision.json)。已报告用户，后续不重复通知该决定；继续完成半决赛原任务。

## 用户决定采用vLLM

23:20用户查看逐题差异后明确采用vLLM。此次6题退步、1题救回中，多数为框边界跨过0.5，不能据此断言后端普遍退化。保留原始完整成绩和拒绝记录，另存`vllm_user_adoption/decision.json`标明用户采用决定，不修改旧成绩或伪装命中门槛通过。

新增`--adopt-existing-validation`复用完整对照，检查模型、适配器及运行选项一致后直接续写，不重复评价。监督以`--probe vllm_adopt`接管，失败明确停止。新监督824809、worker824812已加载模型并从HF保留的1720题后接续剩余3971题，完成后沿用原严格打包入口。训练继续暂停。新日志为`vllm_adopt_benchmark.log`。
