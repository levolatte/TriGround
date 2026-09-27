# RGBDT500 数据镜像跟进调查

调查范围：核对本地数据获取状态与采集脚本；查作者项目主页、作者 GitHub、Hugging Face 与 ModelScope 是否提供可确认的 Train.zip 镜像或正式托管。未接受许可协议、未登录/绕过百度访问限制、未启动大文件下载，也未改动本地图像或图组。

时间记录：核查开始 2026-09-24 21:23:21 UTC；公开来源检索截至 21:36:55 UTC；报告落盘并核验 21:40:43 UTC（香港时间分别为 2026-09-25 05:23:21、05:36:55、05:40:43）。

## 结论

截至本次检索，没有找到可确认完整、可用且获作者认可的 Train.zip 镜像。作者官网仍只列 Google Drive 与百度网盘两个下载渠道。发现一个公开的第三方 Hugging Face 仓库，含 8 个大型 ZIP 分卷；但没有说明内容、覆盖范围、来源或授权，不能据此认定它是完整 Train 数据镜像。ModelScope 未找到正向匹配，且站内/API 查询受限，故不把“未找到”扩大为“确定不存在”。

## 作者公布的入口与所需操作

- [RGBDT500 作者主页](https://xuefeng-zhu5.github.io/RGBDT500/)当前页面列出 Baidu Disk 与 Google Drive 两个 Full Dataset 链接。页面有数据协议步骤：查看 license PDF、勾选已阅读并接受，再点“Accept & Download”。页面文字明确表示勾选是在代表团队确认接受条款；本次没有替团队勾选或接受。
- [作者 RGBDT500 GitHub 仓库](https://github.com/xuefeng-zhu5/RGBDT500)的 README 也只列 Google Drive 和 Baidu Cloud。仓库目录是 README、网页和图片资源，没有数据 ZIP；官方 GitHub Releases API 返回空列表。
- [作者 RDTTrack 仓库](https://github.com/xuefeng-zhu5/RDTTrack)的 README 说明 Train/Test 目录格式并链接回作者数据主页，未提供 Train.zip。该仓库的公开 issue [“Release RDTTrack and RGBDT500 on Hugging Face”](https://github.com/xuefeng-zhu5/RDTTrack/issues/1)由 Hugging Face 员工于 2026-03-18 发起，直接询问作者是否愿意把数据集放上 HF；检索时 issue 仍为 Open，未见作者答复。这是历史佐证，不单独作为当前不存在 HF 发布的证明。
- 采集脚本 [acquire_rgbdt500.py](F:/AIC/code/tools/acquire_rgbdt500.py) 只配置一个 `ARCHIVE_URL`，指向 Drive 的 `drive.usercontent.google.com`，并以 HTTP Range 读取；未配置百度、HF、ModelScope 或其他镜像回退。`ARCHIVE_SIZE` 固定为 26,447,029,453 字节。
- Google Drive 仍是本地状态记录中的原始目录及 Train.zip 文件 ID；[现有数据获取记录](F:/AIC/docs/research/2026-09-25-data-acquisition-status.md)已记录 Range 请求返回 Quota exceeded。本次没有再次请求 Drive，也没有重下。
- 作者公布的百度分享仍为 [Pan Baidu 入口](https://pan.baidu.com/s/1M1_HF977Hd_-5P-vUWQebg?pwd=j3ek)，提取码为 j3ek。现有记录称无会话目录查询返回 need verify；本次未尝试绕过或自动化验证。若用户选择该渠道，需要本人在百度页面输入提取码，并按页面要求完成登录、验证或客户端操作。

## Hugging Face 检索结果

精确检索 Hugging Face Datasets API（[搜索结果接口](https://huggingface.co/api/datasets?search=RGBDT500&limit=100)）返回一个名为 [Blythehhh/RGBDT500-mirror](https://huggingface.co/datasets/Blythehhh/RGBDT500-mirror) 的公开仓库。仓库元数据为 public、gated=false，lastModified 为 2026-08-12；作者字段是 Blythehhh，且没有 dataset card。仓库树只有 .gitattributes 与 011.zip–018.zip 八个 ZIP 文件，没有 README 或许可文件。

| 文件 | API 登记字节数 |
| --- | ---: |
| 011.zip | 2,866,343,413 |
| 012.zip | 2,623,836,521 |
| 013.zip | 2,224,182,683 |
| 014.zip | 2,766,146,079 |
| 015.zip | 2,800,251,421 |
| 016.zip | 2,565,957,879 |
| 017.zip | 2,674,727,588 |
| 018.zip | 2,557,501,818 |
| 合计 | 21,078,947,402（约 19.63 GiB） |

该仓库名字和公开元数据表明它是第三方上传，而不是已知作者账号的正式发布。文件名也不是 Train.zip；仓库没有清单或说明这些 ZIP 是什么、覆盖哪些序列，亦无授权/来源声明。因此既不能确认它包含完整的400条训练序列，也不能确认可按作者 research-only 协议使用。本次对 011.zip 发起了单次轻量 HEAD 验证，但本地 TLS 握手异常退出；只通过 HF API 读取了小型元数据和目录列表，没有读取任何 ZIP 内容或下载数据。故该仓库是待作者/上传者确认的公开候选入口，不是经验证可用镜像。

## ModelScope 检索结果

以 RGBDT500 精确名称检索公开网页和 ModelScope 数据集入口，没有得到能确认的匹配条目。对 ModelScope 公共数据集查询接口的轻量请求返回 Code 10020000500 / “系统错误”；站内搜索页也无法由当前浏览工具读取。因此记录为“未检出可确认托管”，而不是断言平台上绝无相关仓库。

## 建议的下一步动作

要从作者认可渠道补数据，用户需在作者主页阅读许可条款、亲自确认接受协议，并选择作者公布的 Drive 或百度链接。Drive 目前受配额阻塞，需等待配额恢复；百度侧按分享页面提示输入提取码 j3ek 并完成平台要求的交互。

第三方 HF 仓库在其 ZIP 内容、序列覆盖、数据来源和许可获作者或上传者确认前，不建议作为完整训练集镜像。若要确认该候选，最直接的外部动作是请上传者或作者给出分卷清单、完整性/序列范围说明及再分发许可；本次没有向任何人发送消息。

## Hugging Face 候选内容级验证补记（2026-09-24 UTC）

跟进调查时间：2026-09-24 21:53:35–21:59:26 UTC（香港时间 2026-09-25 05:53:35–05:59:26）。这次没有因上传者不是作者账号而直接排除候选，而是先读现有 Range 采集脚本，随后尝试读取一个 ZIP 的中心目录尾段。

- 以 API 先前登记的 011.zip 大小 2,866,343,413 字节为准，对公开入口 [`011.zip` resolve URL](https://huggingface.co/datasets/Blythehhh/RGBDT500-mirror/resolve/main/011.zip) 请求最后 4 MiB（字节偏移 2,862,149,109–2,866,343,412），`curl` 在 TLS 握手阶段以 Schannel 错误 35 退出，没有收到任何字节。因此没有读到 EOCD/中心目录，不能确认压缩包内路径、实际序列范围、Train 目录结构或 GT 内容。
- 现有报告已记录过一次 HF 011.zip HEAD 的 TLS 握手失败；加上本次 Range 握手失败，已达到最多两次网络错误的限制。没有再请求 HF 国内镜像或重试。
- 本地官方源控制样本 `pilot/011` 可读：`groundtruth.txt` 中 `00000263.png` 框为 `70,750,236,97`；`pilot_manifest.jsonl` 中相同序列/帧的 `bbox_xywh_pixels` 一致，且本地 RGB、IR、Depth 与 `depth_visual` 文件齐全。这只验证本地样本和本地标注对齐，不能证明 HF 的 `011.zip` 包含同一序列、同一 GT 或可补取缺失图组。

结论仍为：HF API 曾公开列出 011.zip–018.zip，但第三方 ZIP 的真实内容和覆盖范围尚未验证。当前不能把 HF 候选认定为可用补数镜像，也不能给出基于已核实目录的提取路径；上面的 URL 只是仓库公开入口，不代表当时成功读取或下载。当时没有再继续请求；后续 Range 验证见下节。

## HF 国内镜像受限 Range 验证补记（2026-09-24 UTC）

调查和记录时间：2026-09-24 22:25:41–22:29:54 UTC（香港时间 2026-09-25 06:25:41–06:29:54）。

经后续授权，使用 `F:\AIC\code\.venv\Scripts\python.exe` 中的 requests 2.34.2 对 [HF 国内镜像 011.zip resolve URL](https://hf-mirror.com/datasets/Blythehhh/RGBDT500-mirror/resolve/main/011.zip) 发起一次流式 Range GET；TLS 验证保持开启，没有重试。请求最后 4 MiB（字节偏移 2,862,149,109–2,866,343,412），返回 HTTP 206、Content-Range 与登记大小 2,866,343,413 字节一致；经两次 HTTP 重定向到 `us.aws.cdn.hf.co`。只读取并保存了 4 MiB 尾段到 `.work`，没有获取整包。

尾段不含标准 ZIP EOCD、Zip64 EOCD 或 central-directory header，因此标准 `zipfile` 无法将其作为完整 ZIP 解析。尾段中能解析出三个局部文件头，路径分别为 `Test_100/031/depth/00000015.png`、`Test_100/031/depth/00000016.png` 和 `Test_100/031/depth/00000017.png`；第三个条目声明的压缩数据结束位置比该文件登记末尾还晚 1,466,528 字节。这说明公开的 `011.zip` 至少在尾部表现为分卷/片段或不完整文件，现有证据不足以确定拼接方法。观察到的路径前缀是 `Test_100`，与本地官方 Train 样本采用的数字序列根目录（例如 `011/color/...`、`011/depth/...`、`011/infrared/...` 加 `groundtruth.txt`）不同。

本次只暴露了尾段中的三个 Depth 路径；因为没有中心目录，也没有 `groundtruth.txt` 文件内容，不能确认整个候选是否包含训练分卷、GT 标注或与本地框一致。当前结论是：HF 国内镜像的公开端点确实能响应 Range，但 `011.zip` 暂不能作为已验证的可用训练图组来源；不能据此提取缺失训练图组，也不能宣称任何数据已补齐。解析脚本和 4 MiB 取样保存在 `.work/hf_mirror_011_range_once_20260924.py`、`.work/hf_mirror_011_analyze_tail_20260924.py` 与 `.work/hf_mirror_011_tail_once_20260924.bin`，供本地复核。

## 018.zip 尾部分卷目录核验补记（2026-09-24 UTC）

调查和记录时间：2026-09-24 22:53:42–22:57:38 UTC（香港时间 2026-09-25 06:53:42–06:57:38）。使用既有 HF 文件树登记的精确大小 2,557,501,818 字节，对 [018.zip 国内镜像入口](https://hf-mirror.com/datasets/Blythehhh/RGBDT500-mirror/resolve/main/018.zip) 发起一次、TLS 验证开启的流式 Range 请求，范围为文件尾部 4 MiB（2,553,307,514–2,557,501,817）。返回 HTTP 206，Content-Range 与总大小吻合；只读取并保存了这 4 MiB 到 `.work`，没有重试或下载整包。

这次尾段含一个 EOCD、1,598 项 central directory 和 148,555 字节目录记录，没有 Zip64 EOCD。中心目录能由 Python `zipfile` 成功解析；所有成员根路径都是 `018`，有 `018/groundtruth.txt`，并含 531 张 `color`、531 张 `depth`、531 张 `infrared` PNG（另各有一个目录项）。`groundtruth.txt` 在目录中标记为 24 字节解压、23 字节压缩，Deflate 方法；其局部文件头偏移为 1,740,877,891。该分卷的中心目录只列 `018` 根，偏移与该 ZIP 的文件末尾对齐，故 `018.zip` 本身是结构完整的独立 ZIP，而不是列出 011–018 全部内容的单一大 ZIP 的最终目录。其结构像一条 RGB-D-T 训练序列，但 GT 内容极短，尚不能确认它是有效的框标注。

这也修正了对前一分卷的解释边界：`011.zip` 尾段局部缺少 EOCD、且局部文件头显示 `Test_100/031/depth`，不能据此推断整个 HF 仓库损坏或所有分卷都是测试数据；018 的有效中心目录显示至少有一个数字序列根及 GT 路径。反过来，018 的目录也不能证明 011–017 的完整性或目录结构。本地当前没有 sequence 018 的 RGBDT500 图组/GT 可与之逐字节或按框比对。

该次记录时仍未读取 GT 载荷或图像；后续的 GT 和首帧三模态 Range 核验见下一节。该次解析脚本和尾段保存在 `.work/hf_mirror_018_tail_once_20260924.py`、`.work/hf_mirror_018_inspect_tail_offline_20260924.py`、`.work/hf_mirror_018_tail_once_20260924.bin`。

## 018.zip 标注与首帧组三模态核验补记（2026-09-24 UTC）

调查和记录时间：2026-09-24 23:27:30–23:38:05 UTC（香港时间 2026-09-25 07:27:30–07:38:05）。按已授权的 20 MiB 上限执行 Range 抽取；TLS 校验开启，重试 0 次。累计读取量为 9,032,717 / 20,971,520 字节，其中计入了前次目录尾段 4 MiB 和本次 GT 512 字节前缀。

### GT 内容和划分判断

从局部头偏移 1,740,877,891 读取 512 字节，读出的成员名是 `018/groundtruth.txt`；本地头给出的 Deflate 载荷为 23 字节，解压后 24 字节，CRC 与中心目录相符。实际内容只有一行：`788.0,905.0,51.0,143.0`，没有文件名，也没有后续框行。四个数构成正面积且位于 1920×1080 画面范围内的 XYWH 框。因为 GT 是单行四数而不含帧名，结合序列首帧命名，按首帧 tracking 初始化框解释为 `00000001.png`；这是一项基于格式惯例的判断，不是 GT 文件的显式帧索引。可用框数仅 1，不能把该序列的 531 张图当成 531 条标注，也不对其它帧插值。

候选 `018/` 的数字根路径和三模态目录形状与官方 Train 的常见序列布局相似；但官方 Train 的 `groundtruth.txt` 格式是每条 `frame.png,x,y,w,h`，每序列十条代表帧标注。当前工作区没有官方 ZIP 中心目录缓存或本地 sequence 018 原件，因此这里只能与已知官方目录/标注格式比较，不能声称逐条核对过官方 018 中心目录。现有本地官方样本 `pilot/011/groundtruth.txt` 有十条带文件名的记录，与候选的一行无帧名 XYWH 不同。因此本候选**不符合可直接并入官方 Train 多代表帧标注的格式**。单框更像 tracking 评测/初始化格式；018 分卷没有显式 `Train` 或 `Test_100` 根名，单凭这里的结构不能证明正式划分。保守处理为：划分来源未证实、官方 Train 兼容性未通过，只保留一个首帧候选。

### 已验证的单组三模态

中心目录中确认 `018/color/00000001.png`、`018/depth/00000001.png`、`018/infrared/00000001.png` 均存在。按每个成员局部头前缀 512 字节和剩余压缩载荷分段 Range 读取；Zip CRC、Deflate 解压及 Pillow PNG 解码均成功。三张图都是 1920×1080；RGB 与 IR 为 RGB，Depth 为 `I;16`，不从该表示推断物理单位。RGB 原图目视中，框坐标落在前景人员手边的细长浅色物体上，位置与单个手持物体相符；未在图像上绘制框或改动文件。IR 中前景人员同样可见；Depth 仅做无单位原始图解码核对。

| 成员 | 局部头偏移 | 载荷偏移 | 压缩字节数 | 本次实际 Range |
| --- | ---: | ---: | ---: | --- |
| `018/color/00000001.png` | 753,988,679 | 753,988,759 | 2,194,312 | 753,988,679–753,989,190；753,989,191–756,183,070 |
| `018/depth/00000001.png` | 2,280,721,255 | 2,280,721,335 | 1,534,586 | 2,280,721,255–2,280,721,766；2,280,721,767–2,282,255,920 |
| `018/infrared/00000001.png` | 1,537,460,594 | 1,537,460,677 | 1,108,760 | 1,537,460,594–1,537,461,105；1,537,461,106–1,538,569,436 |

GT 前缀 Range 是 1,740,877,891–1,740,878,402；压缩载荷位于 1,740,877,968–1,740,877,990。已取出的三图及 GT 均仅保存在 `data/external/RGBDT500/mirror_probe/`，没有加入任何清单或训练池。逐 Range 记录在 `mirror_probe/018_triplet_transfer_log.json`，结果摘要在 `mirror_probe/018_frame00000001/verification.json`。

当前可以说 HF 的 018 序列中存在一个解码正确、与首帧 XYWH 区域视觉相符的三模态候选；它仍不是已验证的官方 Train 标注。若要判定正式 Train/Test 来源或官方 018 对齐，下一步只需获得官方 Train 的 `018/groundtruth.txt`（官方格式应含十条带帧名记录）及对应中心目录条目后作对照；当前没有为此再发起官方源请求。



