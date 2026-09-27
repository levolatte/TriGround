# 复赛官方测试数据上传记录

## 2026-09-24 22:25 复赛测试集上传和切换完成

- 云端目录 `/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924`；Query为该目录下 `queries/queries.json`。完整6006文件、13128736542字节（12.227GiB），5690条Query，引用1295组三模态图像。原包visible2005/infrared2000/depth2000全部上传。
- 所有文件大小逐项与本地一致，全部Query的visible/infrared/depth路径可对应上传文件。未修改官方Query、未生成标签、未做官方推理或提交。
- 云端 `data/official_test_current.json` 及数据目录 `upload_complete.json` 已写入；本地证据 `results/rematch_20260922/test_upload_complete.json`、`rematch_test_inventory.json`。以后以5690清单为准，不沿用初赛9555。
- 旧初赛原始测试数据在此云机上此前已清理，本次没有再次删除；历史预测结果保留。Locate已按用户明确指令退役并删除权重、独立环境、源码/缓存与下载排队脚本，30号跟进暂停。无卡模式下本轮工作全部完成，后续GPU实验需另行讨论。

