# RGBDT500 换源调查：公开托管平台复查

调查时间：2026-09-25 13:53–13:54（香港时间）。调查目的：寻找能补充既定 RGBDT500 固定计划缺组的公开训练数据来源。调查仅查询公开仓库目录和小型 API 响应；未读取或下载新的 ZIP/图像，没有访问 Google Drive，没有继续复测旧 HF 分卷，也没有改动清单、原图或 HANDOFF。

## 结论

本轮没有找到新的、可据以补齐现有固定计划的公开镜像。Hugging Face API 的精确数据集搜索只返回既有第三方仓库 Blythehhh/RGBDT500-mirror；Kaggle 对精确词查询返回空列表；ModelScope API 返回服务端错误，因而无法据此确认有或没有仓库。GitHub 搜索 API 连接被远端重置。没有新增可执行的替代下载路径。

已知的 HF 仓库并未通过“完整官方 Train 镜像”核验，不应与本地官方采集源等同。旧调查中仅验证了该仓库 018 分卷包含一个可解码三模态首帧及一条初始化框；这不能证明是作者 Train 划分或补足序列的多帧 GT。旧调查对 011 的尾段观察到 Test_100/031/depth/... 局部路径，但没有据此判定所有分卷损坏或全部属于 Test。其余分卷不在本轮重读或重试范围内。

## 平台查询记录

| 平台 / 查询 | 本轮实际响应 | 可得结论 |
| --- | --- | --- |
| Hugging Face Datasets API：search=RGBDT500 | HTTP 200；只返回 Blythehhh/RGBDT500-mirror，lastModified 2026-08-12T11:24:13Z，public、非 gated | 未发现新的 RGBDT500 精确命名仓库 |
| Hugging Face Datasets API：search=RGBDT-500 | HTTP 200；仍只返回同一个 Blythehhh/RGBDT500-mirror | 没有额外候选 |
| Hugging Face Datasets API：search=RDTTrack | HTTP 200；空列表 | 没有新的 RDTTrack 数据集仓库 |
| Kaggle Datasets API：search=RGBDT500&page=1 | HTTP 200；响应 [] | 没有精确命名的公开 Kaggle 数据集 |
| ModelScope Datasets API：Name=RGBDT500 | HTTP 500，Code 10020000500（“系统错误”） | 查询被平台服务错误阻断，不能由此断言不存在 |
| GitHub repository search API：q=RGBDT500 | 连接在收到搜索结果前被远端重置 | 本轮未得到新仓库结果；不重试 |

HF 搜索原始响应保存在 F:/AIC/.work/rgbdt500_sourceswitch_20260925_135310/hf_search.json；Kaggle、ModelScope 与 GitHub 响应保存在 F:/AIC/.work/rgbdt500_sourceswitch_20260925_135413/platform_search.json。这两个目录还保留了各自的单次查询脚本。Hugging Face 搜索入口可由 [Datasets API](https://huggingface.co/api/datasets?search=RGBDT500) 复核；当前已知仓库为 [Blythehhh/RGBDT500-mirror](https://huggingface.co/datasets/Blythehhh/RGBDT500-mirror)。作者列出的公开渠道仍见 [RGBDT500 官网](https://xuefeng-zhu5.github.io/RGBDT500/) 和 [官方 GitHub](https://github.com/xuefeng-zhu5/RGBDT500)。

## 对已有 HF 候选的范围说明

不重做此前的 011/018 Range 检查，引用既有报告 [2026-09-25-rgbdt-mirror-followup.md](F:/AIC/docs/research/2026-09-25-rgbdt-mirror-followup.md)：

- HF API 中只有 8 个大 ZIP 分卷（011–018），仓库没有 Dataset Card、覆盖清单或来源/再分发说明。
- 018 的中心目录只列 018 根；单行 GT 内容是四个 XYWH 数字，按首帧初始化框解释时只有 1 个已确认候选，不能把 531 张帧当成 531 个标框样本。
- 018 的 Train/Test 来源以及与官方 Train 标注格式的对应关系未证实。它不满足直接补入当前官方固定计划的条件。
- 011 仅有尾段局部证据，出现 Test_100 路径；这一观察既不能说明整个仓库只含 Test，也不能提供可核实的完整 Train 提取映射。

## 当前工作状态与后续接口

本轮没有执行数据下载、ZIP 提取或清单写入。项目负责人随后告知用户已选择百度会员客户端下载，并提示本地存在预分配的 .downloading 文件；该临时文件不等于已完成 Train.zip，本轮不检查、不改动它，也不再推进公开镜像搜索。待该下载明确完成后，应先本地验证 ZIP 目录、序列路径及 GT，再沿用既定采集计划补齐缺失图组，避免重建划分或把单帧初始化框扩写为多帧标注。

