# LocateAnything退役记录

## 2026-09-24 LocateAnything按用户要求退役

- 用户明确要求退役并删除。云端权重models/LocateAnything-3B、envs/locateanything独立环境、third_party/LocateAnything专用源码、两处Hugging Face专用缓存已删除。该权重约7.2GiB、环境约1.3GiB；并发上传期间数据盘可用空间净增加8.40GiB，删除后余22.02GiB。
- 本地及云端专用下载/准备/排队shell脚本已删除，30号跟进PAUSED；禁止重新下载/推理/训练/添加为后续候选来源，除非用户再次明确要求。
- 保留已下载的原始预测、日志、汇总和CPU报告/解析代码作为历史证据；不删除R2/M2、训练验证集或复赛测试数据。此前“保留Locate候选”建议现已被用户退役决定覆盖。
- 最终结果266/412，R2=288；完整报告docs/research/2026-09-24-locateanything-results.md。删除证据results/rematch_20260922/locate_retirement_deletion.txt及云端results/first_batch/locate_retirement.json。
- 复赛测试集上传仍继续，完成标志results/rematch_20260922/test_upload_complete.json；无需GPU。

