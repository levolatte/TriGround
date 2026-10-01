# SSH 自动监控修复与验收

2026-10-01 04:17香港：真正triground定时触发已成功执行exec_command调用check_structure_ssh.ps1，新建SSH连接，无write_stdin、无密码交互、无审批拦截。远端时间20:17:22 UTC，本地读取20:17:34 UTC；父11852及R_formal子12440存活，R90/400；含在途GPU7402.862秒，余35797.138秒。验收证据automatic_ssh_validation.json标记trigger=heartbeat、退出码0，并引用实际云端状态。本次确认修复后的自动监控路径有效，平台内部旧PTY拒绝机制未更改。恢复正常30分钟检查；今后监控失败立即通知，不用旧快照证明健康。下方“验收未完成/1分钟”段落为历史记录。

用户授权：2026-10-01要求彻底修复自动审批及工具问题，简化定时任务。

## 已确认的故障

自动轮次已使用approval_policy=never、sandbox=danger-full-access。本地exec_command读取成功，但write_stdin续写既有SSH PTY在0.0秒被策略拒绝；同一会话人工消息后可以写入。故拒绝发生在工具执行前，不是SSH服务器或网络错误。平台内部审批决策细节未在本地日志给出，不能断言具体App实现错误。此前安静遗漏了实验停机，应视作监控失效。

## 最小修复

- 每次执行固定PowerShell脚本，建立独立SSH连接并退出，不依赖长存PTY、stdin或上一轮会话号。
- 专用密钥正常SSH认证，不需要自动轮次交互输密码。私钥只在用户.ssh，公钥增量登记原AIC，现有密钥保留。
- check_structure_ssh.ps1读取真实父进程、当前阶段子进程、GPU账本、日志及正式步数，生成带云端时间的ssh_progress_snapshot.json。失败写独立ssh_monitor_attempt.json并非零退出，不覆盖最后成功快照、不给旧数据冒充本次检查。
- aic_ssh.py提供run/put/get；run读UTF-8命令文件，get先创建本地目录，避免此前协议字段拼错或不存在目录导致传输进程退出。旧tm_remote.py不再用于定时任务。
- 定时提示改为先运行固定脚本；监控失败立即报告。继续原R/S/S+G范围与43200秒账本，不重启正常实验。

## 验证

04:13、04:15香港，实际新建SSH成功读取云端，R64及74/400，父11852及子12440存活；新目录真实下载成功。自动任务暂1分钟进行独立验收，尚未取得真正自动轮次结果，不能宣称已全部修复。首次自动成功后记录automatic_ssh_validation.json，恢复原30分钟间隔。

官方文档说明自动任务沿用默认沙箱、通常never审批；本次未修改权限设置或绕过工具拒绝：https://learn.chatgpt.com/docs/automations?surface=app#permissions-and-security-model。
