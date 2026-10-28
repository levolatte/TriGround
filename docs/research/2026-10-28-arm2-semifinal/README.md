# Arm2 原流程推理半决赛数据

2026-10-28。按用户要求，在老 AIC 推理机上使用原 Arm2 checkpoint-500 对新数据推理，生成预测包供用户自行提交。

## 数据与运行配置

- 本地数据：`F:/AIC/data/半决赛数据集-基于大模型的多模态视觉理解与推理`。
- 查询数：5691；引用1290组三模态图，共3870张图片。上传原始查询文件及其引用图片，不改Query。
- 机器：`connect.westd.seetacloud.com:41799`，RTX PRO 6000 Blackwell Server Edition，96GB。
- 代码：老机器原有 `/root/AIC_code`；解释器 `/root/vllm_env/bin/python`。
- 基座：`/root/rematch_models/Qwen3.6-27B`。
- 适配器：`/root/autodl-tmp/runs/q36_arm2_capability/archived/checkpoint-500`。
- 远端数据：`/root/autodl-tmp/semifinal_20261028`。
- 远端输出：`/root/sprint_qwen36/submit_semifinal_arm2_k20_20261028`；日志为同路径加 `.log`。

| 参数 | 值 |
|---|---|
| 输入 | RGB、红外、深度三图，原提示词和原深度预处理 |
| 图像放大 | 2.0倍，LANCZOS |
| 每图像素范围 | 200704～8294400 |
| 上下文长度 | 32768 |
| 并发序列 / 分块 | 32 / 256 |
| 精度 / LoRA rank | bfloat16 / 32 |
| 前缀缓存 / Mamba缓存模式 | 开启 / none |
| 生成 | 关闭思考，温度0；首轮64词元，解析失败才重跑128词元 |

配置沿用旧 `F:/AIC/results/submission_k20_20261004/run_config.json`；只改变输入、输出路径及查询数。使用老机原版代码，未同步本地正在开发的量化与强化学习改动。

## 续接入口

- 本地传输与启动脚本：`F:/AIC/.work/transfer_launch_semifinal_arm2.py`。
- 远端执行脚本：`/root/run_semifinal_arm2_20261028.sh`。
- SSH辅助：`F:/AIC/.work/old_arm2_ssh.py`，用 `python .work/old_arm2_ssh.py <本地命令文件>` 执行。凭据由本机用户加密保存，禁止输出凭据内容。
- 启动回执：`F:/AIC/.work/semifinal_arm2_launch.json`。
- 运行目录内 `launcher.pid`、`started_at.txt`、`exit_code.txt`、`finished_at.txt` 分别用于定位进程和确认结束状态。
- 成功后下载 `predictions.json`、`predictions.jsonl`、`submission.zip`、`run_config.json`、`summary.json` 和格式检查记录到 `F:/AIC/results/submission_semifinal_arm2_k20_20261028/`。

原入口写出完整JSON和ZIP后，执行5691条覆盖、非bbox字段不变、框合法性、ZIP根目录成员与JSON一致性检查。没有官方真值，不推测本次比赛成绩；由用户自行上传平台。

## 完成记录

- 推理时间（香港时间）：2026-10-28 02:50:04～09:16:12，约6小时26分钟。
- 全部5691条完成，首轮直接成功5691条，二轮重试0，解析失败0，兜底0。
- 本地结果目录：`F:/AIC/results/submission_semifinal_arm2_k20_20261028/`。
- 提交包：`submission.zip`（192038字节），根目录仅含 `predictions.json`；JSON共5691条，非bbox字段改动0，非法框0，ZIP内外JSON逐字节相同。
- 已保存逐题原始输出、运行配置、统计、完整推理日志及关机回执。没有官方真值，本次比赛成绩由用户提交后取得。
- 运行统计中的 `total_generation_seconds=0.0` 是旧脚本没有记录逐题耗时的产物，不代表真实推理耗时；以开始/结束时间和分块日志为准。

## 关机记录

用户明确授权下载后关闭老AIC云机。完整结果下载并通过本地检查后，于2026-10-28 09:29:19（香港时间）在 `connect.westd.seetacloud.com:41799` 执行云机内置 `/usr/bin/shutdown`；返回 `SHUTDOWN_COMMAND_EXIT=0`、SSH命令退出码0。关机前确认主机为 `autodl-container-mbdykw3825-fcf7f826`，无GPU计算进程。

09:33再次连接SSH映射端口，TCP代理仍接入但不再返回SSH握手并关闭连接，符合后端实例停止后的表现。未从云平台控制台核验计费状态，不将SSH关闭写作“已确认停止计费”。原始记录见本地结果目录 `shutdown_receipt.json`。收尾自动化在本轮完成后删除。

## 传输问题修复

首次8路长连接上传在约1.52GB时被远端重置；改为64MiB独立分片、2路连接及已完成分片续传，113片完成后合并解压。保留旧中断文件与传输日志。后续直接使用已经下载的本地结果，无需再启动推理。
