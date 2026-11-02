# 纠错1600：官方半决赛预测

用户要求暂停训练，先使用纠错1600预测官方半决赛数据并取得结果。

## 数据与模型

- 原始数据：`F:/AIC/data/半决赛数据集-基于大模型的多模态视觉理解与推理`，5691题、1290组三模态图片，共3870张引用图片。
- 深度格式：1193张1920×1080的I;16毫米深度、97张640×360的RGB可视化深度。前者沿用训练映射，后者保留原始像素，不当作毫米。
- 云机：`connect.weste.seetacloud.com:17413`，当前训练机，不启动已关闭的老Arm2机器。
- 基座：`/root/autodl-tmp/rematch_models/Qwen3.6-27B`。
- 适配器：`/root/autodl-tmp/quant_rl_20261026/runs/acc_correction_20261029/correction/train/checkpoint-1600`，完整保存点存在。
- 推理：原评价HF/NF4、原三图提示、贪心单答、64词元、8192上下文、每图200704～1204224像素、并发2、seed2026、既有punctuation标点规范化。

## 训练暂停

主父257429收到SIGSTOP，SFT训练子628269收到SIGTERM并释放GPU。SFT最新完整点2800、纠错最新完整点3200均保留；当前推理指定1600。云端`runs/acc_correction_20261029/paused_for_semifinal.json`记录用户暂停原因。

训练不会自动恢复。原父仍T是用户暂停，不是故障；未来恢复须从完整检查点处理被中断的段，不直接误恢复旧队列。此任务不关闭实例、不提交Git。

## 自动流程与路径

云端独立监督630175，入口`acc_code_20261029/scripts/run_correction1600_semifinal.py`。输出根：

`/root/autodl-tmp/quant_rl_20261026/runs/semifinal_correction1600_20261029`

流程为8题实测→等待113个上传分片完整→解压数据→全量推理→严格打包。任何真实错误保留日志并停止，不生成替代框。全量结束后比较前8题与小批原文一致。

- `state.json`：阶段、子进程及错误信息。
- `supervisor.log`、`smoke.log`、`inference.log`：原始日志。
- `smoke8/`：8题实测，不能当正式提交结果。
- `full/predictions_raw.jsonl`：逐题原始生成，行数表示已完成进度。
- `full/predictions.json`、`full/submission.zip`：完整5691题预测和ZIP；只改变bbox。
- `full/run_config.json`、`full/summary.json`：正式配置及耗时统计。

数据传输为113个64MiB分片，全部7,563,548,251字节；本地进度`.work/quant_rl/semifinal_transfer.json`。云端分片`/root/autodl-tmp/semifinal_correction_parts`，数据解压到`/root/autodl-tmp/semifinal_correction_20261029`。上传脚本通过getpass接收凭据，只保留在内存。

## 当前实测与后续

8题均合法，推理34.07秒，约4.26秒/题；模型适配器496个模块、992个张量加载覆盖一致，显存峰值约23.2GiB。全量初步按6–7小时预估，实际以稳定进度更新，不承诺精确时刻。

数据传完后云端独立完成，不依赖本地SSH。正式完成后下载JSON、ZIP、原始输出、配置、统计和日志至`F:/AIC/results/submission_semifinal_correction1600_20261029/`，核对5691条覆盖、合法框、非bbox字段不变及ZIP内容，再给用户结果。没有官方真值，不推测本次比赛得分。

每30分钟自动化已改为跟进此推理任务，完成下载后删除；不得恢复原训练。后续最新状态以`.work/quant_rl/CURRENT_ACC_RUN.md`顶部为准。

## 完整结果

2026-10-30 01:34:12（香港时间）完成5691题。前1720题为HF，后3971题按用户决定由vLLM续跑，后段含准备耗时8023.64秒。完整产物位于`F:/AIC/results/submission_semifinal_correction1600_20261029/`，含ZIP、JSON、原文、配置、统计及日志。

本地核对5691个ID及顺序、合法框、非bbox字段不变、ZIP根目录及JSON一致，并从原文重新解析确认与包完全一致。自动化已删除；训练保持暂停，实例未关闭。
