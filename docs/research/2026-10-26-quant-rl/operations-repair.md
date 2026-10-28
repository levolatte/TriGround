# SSH续行与定时任务中文修复

2026-10-27，香港时间。两个故障均已定位并修复运行方式；云端训练持续执行，未重启训练或修改奖励。

## SSH被拒绝的原因

拒绝发生在本机Codex的`write_stdin`权限检查，命令尚未送达服务器。之前把它笼统描述为“自动审批审查拒绝SSH”不够准确：没有证据表明审查模型判断SSH内容危险，实际是旧终端与当前运行环境的权限快照不一致，随后`approval_policy=never`不允许新审批。

本机记录：

- 旧SSH终端于10月26日23:16启动，使用默认权限，没有`require_escalated`或额外权限参数。
- 启动时与10月27日01:16的`runtime_workspace_roots`均为`[]`；01:35:49改为本聊天的可视化目录。权限模式仍为完全访问，审批策略仍为`never`。
- 本机启用了`write_stdin_approval`。该检查比较旧终端启动时和当前环境的完整策略，包括工作区列表；它不会因为两边都显示“完全访问”就跳过比较。
- 旧会话空轮询成功，非空输入被拒绝；新建本地终端的输入正常。按当前默认权限新建SSH后，同一条原监控命令和SFTP下载均成功。

公开源码说明：[非空输入检查及策略比较](https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/unified_exec/stdin_approval.rs#L120-L230)、[工作区进入沙箱上下文](https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/session/turn_context.rs#L233-L259)、[Never拒绝新增审批](https://github.com/openai/codex/blob/970111f6cb62cc02fe36a82ea7d71b37a2889c97/codex-rs/core/src/tools/approvals.rs#L465-L485)。已观测的工作区变化足以解释失配；旧进程的内部快照未导出，不能断言它是唯一不同字段。

修复方式：每轮定时检查都在当前权限下新建SSH终端，不跨轮复用历史会话ID。检查结束后正常关闭该连接。保留原审批和沙箱设置，不关闭`write_stdin_approval`，不增加全局放行规则。云端长期训练由既有队列继续运行，检查连接不承担训练进程生命周期。

## 中文乱码的原因

此前更新自动化时，读取TOML的Python进程默认使用GBK标准输出，工具按UTF-8解释输出；`JSON.parse`收到的中文已经损坏，随后被写回任务。追加的新中文直接通过结构化工具参数发送，所以没有同样损坏。问题由本次维护操作引入，不能归因于定时器随机破坏文件。

无害样本实测：PowerShell 7.6.5的控制台输出和`$OutputEncoding`均为UTF-8，但所用Python的`sys.stdout.encoding=gbk`、区域编码为`cp936`、`utf8_mode=0`。中文“中文定时任务”的默认输出字节以`D6 D0 CE C4`开头；按UTF-8解释后出现`U+FFFD`替换字符。添加`-X utf8`后输出以`E4 B8 AD E6 96 87`开头，正确保留中文；`ensure_ascii=True`则输出纯ASCII的JSON转义，同样可无损解析。

修复包括：

- 通过自动化工具恢复完整中文正文，保留30分钟间隔、72 GPU小时预算及既有实验约束。
- 新增本地`F:/AIC/.work/quant_rl/read_automation.py`：按UTF-8读取TOML，以ASCII转义JSON输出，正文含替换字符时明确报错，避免再次写回坏文本。
- 自动化后续读取固定使用`python -X utf8`与上述脚本，保存后回读并逐字核对。
- `remote_session.py`的JSON输出改为ASCII转义，避免远程中文再次经过本机默认GBK输出边界。

读取示例：

```powershell
& 'F:/AIC/.work/grounding_aux_round1/cpu_preencode_env/Scripts/python.exe' -X utf8 'F:/AIC/.work/quant_rl/read_automation.py' 'F:/Tools/OpenAI/CodexHome/automations/aic-rl/automation.toml'
```

返回JSON应先解析，再通过`automation_update`更新；不要通过终端文本直接覆盖TOML。

## 修复后的实际检查

01:57:59，新SSH连接成功执行原监控命令。C完成77步、616次题目呈现，已保存用于同曝光比较的第29步状态；所有已记录梯度有限且存在非零梯度。当前GPU利用率99%，数据盘剩20.89GiB；账本已结算21.9273 GPU小时，含当前训练估计共22.9017/72小时。仅有一个运行预算任务，没有新增GPU作业。[实时快照](evidence/monitor-latest.json)。

同一连接已完成快照下载、远程中文“监控正常”往返，并正常退出。中文读取脚本完成无损往返及损坏文本拒绝检查；自动化正文保存后逐字一致，共3192字符，保持ACTIVE和30分钟间隔。后续调度仍需按新规建立连接，不把本次检查当作未来运行结果。

故障与恢复状态见[访问记录](evidence/monitor-access.json)。
