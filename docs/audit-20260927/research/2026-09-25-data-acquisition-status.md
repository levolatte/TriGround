# 下一阶段外部数据取得状态（2026-09-25）

此记录区分作者公开入口、实际抽样读到的文件、尚未取得的图像。训练只用各数据集作者划分中的 train；数据中的 Query 与框不会从比赛官方测试图生成。

## RGBDT500：已实际取得训练样本

- 作者项目与划分：[RGBDT500主页](https://xuefeng-zhu5.github.io/RGBDT500/)、[NeurIPS论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/b4962fcd5d4410a9f43ef70f528eedd8-Paper-Datasets_and_Benchmarks_Track.pdf)、[作者GitHub](https://github.com/xuefeng-zhu5/RGBDT500)。作者Google Drive公共目录为 `https://drive.google.com/drive/folders/1UAe_maNR_ukYgtBrmeqiv87WhW28yK9r`，其训练包是 `Train/Train.zip`，文件ID `1DGH4YNRpiJcNG6VnbWuzV1pP5-ZYYlwq`。
- 已用 HTTP Range 实读 ZIP 目录，未下载全包。压缩包 **26,447,029,453字节（24.63GiB）**，解压文件总量 **27,542,573,476字节（25.65GiB）**。目录内是 **400个训练序列×每序列10张已标注帧＝4000个三路图组**，每个序列均有 `groundtruth.txt` 及 `color/depth/infrared` 三种 PNG；GT行格式为 `frame.png,x,y,w,h`。作者论文§3.2说训练标框选择了 K-means 代表帧，而非所有约16万训练帧。抽样目录的一个真实例子是 `007/color/00000001.png` 和对应 `007/groundtruth.txt`。
- 数据提取脚本：[acquire_rgbdt500.py](F:/AIC/code/tools/acquire_rgbdt500.py)。直接从 ZIP 按字节段抽所需文件，本地不保留整包，支持按已有图像文件续跑。使用固定 seed 2026：20个训练序列专供外部复查，共50个图组；另外100个不重合训练序列各取1张框图，作为100图组试标；余280序列可补正式训练。`acquisition_plan.json`记录完整划分。
- 首10组已实际下载到 `F:/AIC/data/external/RGBDT500/pilot/`，图像都为 1920×1080，图像路径、GT文件名和框对齐；Depth 图经 PIL 读为 `I;16` / NumPy `uint16`，值域样本在0到19999间，非零像素比例从0.454到0.987。**物理单位尚未由作者文档或原图元数据证实**，所以当前只可制作视觉定位Query，不把像素值解释为毫米，也不生成带数值门槛的远近判断题。作者论文还指出存在少数视角不一致的模态扰动场景，试标时须逐图确认。作者官方[RDTTrack读取代码](https://github.com/xuefeng-zhu5/RDTTrack/blob/main/lib/train/dataset/rgbdt.py)把16位图读入后做逐图min-max及JET伪彩色，属于模型可视化输入，不能作为米制单位证据。直接用`PIL.convert('RGB')`会把大于255的原始数值压到255，失去深度结构。采集工具另存`depth_visual`，使用与本项目现有固定映射同一数值公式把raw1..19999映到255..1，raw0映黑；原16位图保留，明确不作米制断言。
- 首批100试标＋50外部复查的继续命令：`python code/tools/acquire_rgbdt500.py --output-dir F:/AIC/data/external/RGBDT500 --pilot-groups 100 --external-groups 50 --workers 4`。已下载文件保留并跳过。输出的两份JSONL均包含 `sequence`、`frame`、三路相对路径、RGB大小、像素 `xywh` 和归一化 `xyxy`，`query=null` 待补标，`depth_units=unverified`。新增图组应从这些清单获取，不读取原数据集测试划分。
- 估算每10图组约71MB，所以150图组约1.1GB，仅是首10组实测的线性估算；正式2000组须按实际抽样后的体积计算。云机扩容后应仍只同步需要的子集，减少等待。

### 本次获取中断与现有可用子集

扩大提取时发现序列325的所选帧`00000042.png`原GT为`[0,0,0,0]`，已明确排除。脚本现会根据作者GT为该序列改选有效框帧，并保留已取得的其他图像。随后作者Google Drive `Train.zip` 对字节范围请求返回“Quota exceeded / Too many users have viewed or downloaded this file recently”；标准Drive地址同样返回限额页面。该状态目前属于**外部下载源阻塞**，可能需要作者限额恢复，不能把剩余图组写作已完成。离线重建后的实际清单是 `pilot_manifest.jsonl` **67个完整有效图组**、`external_review_manifest.jsonl` **4个完整有效图组**；余33/46待补齐。已落地的图像约587MB，部分目录仅有不全的三路文件，清单不会包含它们。两份清单可以供候选Query准备，但人工100条试标验收和独立外部100题复查仍需要剩余图组。

作者[百度镜像链接](https://pan.baidu.com/s/1M1_HF977Hd_-5P-vUWQebg?pwd=j3ek)可打开提取码页，公开无会话API列目录返回`need verify`。目前未验证可靠的无登录分包下载，不把该入口称为可立刻自动续传。Google Drive限额恢复后沿上面的同一命令按现有图像断点补齐；脚本只在实际完整落地后写逐组清单。

## RGBT-GroundBench：三源官方训练标注可读，图像待下载

- [作者仓库](https://github.com/crazyxiaoxi/RGBT-GroundBench)和[作者HF文件树](https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/tree/main)给出独立的 FLIR、M3FD、MFAD 图像和标注 tar。固定下载前缀 `https://huggingface.co/datasets/JiawenXi/RGBT-Ground-Dataset/resolve/main/`，六个包为：

| 文件 | 字节数 | 本地状态 |
| --- | ---: | --- |
| `ann_flir.tar` | 3,307,520 | 已下载并读 train |
| `data_flir.tar` | 935,802,880 | 未由本记录下载 |
| `ann_m3fd.tar` | 4,567,040 | 已下载并读 train |
| `data_m3fd.tar` | 7,246,704,640 | 未由本记录下载 |
| `ann_mfad.tar` | 8,529,920 | 已下载并读 train |
| `data_mfad.tar` | 2,191,175,680 | 未由本记录下载 |

- 三个 `train.pth` 实读分别为 **7000、3604、16000条**，行首是 `[文件名, {height,width}, [x,y,w,h], 英文Query, ...]`，与本地现有 [prepare_rgbt_groundbench.py](F:/AIC/code/tools/prepare_rgbt_groundbench.py)预期一致。例：FLIR首行 `FLIR_04093.jpg` 640×512、目标框 `[581,225,16,42]`；M3FD首行 `03220.png` 1024×768；MFAD首行 `cali_l_0323145136-361.jpg` 1280×960。首批6000建议三源各抽2000，按图组去重与场景分层之后才定实际配额。完整图像尚未抽验，不能称其已可训练。
- 数据均只有 RGB+热红外，不要补假Depth。FLIR/M3FD/MFAD是本基准的来源，不能与母集重复计入新增图组。

## RoboRefIt：作者仓库新链接可取

- [作者项目主页](https://luyh20.github.io/RoboRefIt.github.io/)的旧Drive ID `1-kEfagTovmcbEFG6-C_xatZfFiLWQmiU` 当前返回404。[作者GitHub仓库](https://github.com/luyh20/VL-Grasp) README中的数据链接指向新文件 `https://drive.google.com/file/d/1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW/view?usp=sharing`。该 `final_dataset.rar` 的HEAD返回有效下载，大小 **6,084,791,269字节（5.67GiB）**，支持字节范围请求；尚未在本记录中下载或解包。
- 作者主页给出 train 7929图／36915句；JSON字段 `text,bbox,rgb_path,depth_path,mask_path,scene,class`，框是像素 `xyxy`，并注明 `depth_colormap`。本地已有 [prepare_roborefit.py](F:/AIC/code/tools/prepare_roborefit.py)；必须拿到新RAR实际图像、深度编码和训练JSON再验收。不可把第三方HF RGB-only test镜像算作原始 RGB-D train。

## 交接和限制

三源中的RGBDT500可立刻开展试标。RGBT和RoboRefIt的下载文件已明确，但图像及深度语义仍待落地核对。任何近远数值标签应在RGBDT与RoboRefIt原始深度单位被证实、目标区域有效比例和对象相对位置合格后生成。下载、补标和City重叠检查的结果继续写入同一实验执行报告，不把作者论文宣传总帧数换算成可用Query数。

## RGBDT500 低频恢复检查与续采结果（2026-09-25，香港时间）

- 07:53 左右对原作者 Google Drive `Train.zip` 做了一次带 TLS 验证的 1 KiB Range 请求，得到 `206`、`Content-Range: bytes 0-1023/26447029453`，总长度与此前记录一致；响应体只读 1 KiB。首次和启动前进程检查均未发现采集任务；启动前 F 盘可用空间约 271.92 GiB。
- 按原命令恢复既有 100 pilot / 50 external-review 计划，未创建新划分、未下载整包、未启动训练。首轮脚本在计划生成时触发 `KeyError: '325'`，尚未下载图像。根因是所选帧框无效时短路跳过 `used.setdefault`，候选帧计算再访问未初始化集合；已在 `code/tools/acquire_rgbdt500.py` 中对每个序列先初始化集合，`py_compile` 通过。作者 GT 显示 325/00000042.png 框为零面积，按原序列划分改用该序列的有效标注帧 00000058.png。
- 修复后第二次脚本实际运行约 3 分 35 秒（07:55:47–07:59:22），按原计划逐项续采。运行在读取 GT 的 Range 请求处终止：请求 `10841820739-10841820923`（应为 185 字节）三次均收到 2009 字节，curl 返回码为 0，内容长度不符。脚本未保留该响应正文，因此不能确认是配额页还是其他错误响应；这次异常说明持续 Range 访问不可用，未再发额外网络请求。
- 当前正式清单为 `pilot_manifest.jsonl` 85/100、`external_review_manifest.jsonl` 4/50；逐项检查清单内记录的 RGB、原始 Depth、IR 和 `depth_visual` 文件均存在。对固定计划逐组检查本地四类图像文件，pilot 有 93/100 组齐全、external-review 有 9/50 组齐全；另有 8 组 pilot 和 5 组 external-review 已齐全但中断前未写入清单，重启脚本可从本地文件补记。仍有 7 组 pilot 与 41 组 external-review 缺少至少一个文件，尚未完成。两个目录当前分别约 701.48 MiB 与 110.22 MiB；F 盘仍约有 271.70 GiB 可用。
- 本次后续 Range 的失败载荷没有被验证为配额页面，故只记录为“Range 响应长度异常，获取中断”，不将 100/50 目标标记完成。保留全部断点；下一次应在低频检查确认来源恢复后继续使用同一命令。没有改动 `mirror_probe`，没有新增划分或训练数据，也没有训练。

## RGBDT500 本地清单收尾（2026-09-25，香港时间）

- 未再访问网络。依据固定计划、原有 GT 缓存及本地文件，对 8 组 pilot 和 5 组 external-review 逐一校验后，已分别追加至原 JSONL。新增 pilot：`366/00000002`、`086/00000162`、`310/00000337`、`156/00000312`、`101/00000267`、`282/00000371`、`383/00000768`、`113/00000007`；新增 external-review：`180/00000254`、`180/00000153`、`180/00000265`、`054/00000067`、`054/00000026`。13 组均有本地目标序列 GT、完整 RGB/Depth/IR/`depth_visual`；三路图像均为 1920×1080，GT 框有效且在图像内，Depth 模式 `I;16`，IR 模式 RGB。追加后两份清单共 93/100 pilot 与 9/50 external-review，102 个 ID 唯一。
- 使用本地 `zip_index.json` 和缓存 GT 离线重算的计划与 `acquisition_plan.json` 完全一致：pilot 的100个序列、external-review的20个序列及50个帧组选取均未改变。325仍在原pilot序列中；原选帧 `00000042.png` 框为 `[0,0,0,0]`，现选 `00000058.png` 框为 `[1490,487,332,312]`。未推断缺失 GT，也未改图像。
- 新增回归测试 [test_acquire_rgbdt500.py](F:/AIC/code/tests/test_acquire_rgbdt500.py)：用本地合成的400序列索引和缓存 GT，指定序列325所选框无效，并断言计划改选同序列的有效备选帧且序列划分不变；测试将 ZIP 读取替换为直接失败，确保离线。`pytest -q tests/test_acquire_rgbdt500.py` 结果为 **1 passed**。
- 当前固定计划尚未进入清单的48组为：
  - Pilot（7组）：`072/00000064` 缺 IR、`depth_visual`；`173/00000416` 缺 RGB、Depth、IR、`depth_visual`；`334/00000042` 缺 Depth、IR、`depth_visual`；`024/00000303`、`125/00000001`、`083/00000299` 各缺 IR、`depth_visual`；`398/00000123` 缺 RGB、Depth、IR、`depth_visual`。
  - External-review缺RGB、Depth、IR、`depth_visual`（28组）：`318/00000022`、`318/00000049`、`054/00000001`、`170/00000425`、`144/00000020`、`144/00000122`、`297/00000101`、`297/00000001`、`297/00000410`、`139/00000189`、`139/00000152`、`139/00000055`、`029/00000023`、`029/00000001`、`029/00000026`、`347/00000408`、`309/00000001`、`352/00000001`、`066/00000255`、`025/00000051`、`311/00000217`、`184/00000010`、`052/00000310`、`313/00000459`、`117/00000070`、`117/00000083`、`242/00000380`、`242/00000245`。
  - External-review缺IR、`depth_visual`（5组）：`318/00000042`、`170/00000272`、`144/00000104`、`311/00000173`、`313/00000619`。
  - External-review缺Depth、IR、`depth_visual`（8组）：`170/00000420`、`347/00000627`、`347/00000079`、`309/00000037`、`352/00000860`、`306/00000128`、`306/00000008`、`184/00000029`。
- 上述未完成组都有可读的本地 GT（部分只在 `metadata/groundtruth` 缓存中），没有将其写成完整图组；目标子目录尚无 GT 文件、但 `metadata` 有 GT 的组为 pilot `173/00000416`、`398/00000123` 和 external-review `297/00000101`、`297/00000001`、`297/00000410`、`117/00000070`、`117/00000083`、`242/00000380`、`242/00000245`。清单内全部102条再次通过 ID 去重、计划成员、GT框一致性、三路文件存在性及尺寸对齐检查。剩余图组仍受前述非预期 Range 响应阻塞；本次没有启动采集脚本、没有重下、没有训练。

## RGBDT500 再次低频恢复探测（2026-09-25，香港时间，约10:26）

- 探测前确认没有独立的 Python、curl 或 RGBDT500 采集进程；F 盘可用空间约 270.93 GiB。系统 Python 3.11 和 3.12 均未安装 `requests`，故本次用标准库 `urllib` 发出一次 HTTPS Range 请求，默认 TLS 证书验证保持启用，没有重试。
- 对作者原始 `Train.zip` 请求 `Range: bytes=0-1023`。响应为 HTTP **200 OK**，没有 `Content-Range`，`Content-Length: 2009`，`Content-Type: text/html; charset=utf-8`；读取并保留了前 1024 字节后立即关闭。正文开头为 HTML，标题明确是 `Google Drive - Quota exceeded`。探测元数据和这 1 KiB 响应分别保存在 `F:/AIC/.work/rgbdt500_range_probe_20260925/response.json` 与 `response.bin`。这是已确认的配额页，不是有效 ZIP Range；没有根据长度异常猜测原因。
- 来源仍未恢复，因此没有启动 `acquire_rgbdt500.py`，没有下载或改动任何图组，也未发出额外网络请求。固定计划未改变；离线重查 `pilot_manifest.jsonl` 为 **93/100**（93 组四类图像均齐全，剩 7 组不完整），`external_review_manifest.jsonl` 为 **9/50**（9 组齐全，剩 41 组不完整）。未改写两份 manifest，也未启动训练或使用 GPU。
## RGBDT500 低频恢复探测（2026-09-25 11:29，香港时间）

- 探测前及探测后均未发现 acquire_rgbdt500.py 采集进程。仅向作者原始 Train.zip 地址发送一次带默认 TLS 验证的 Range: bytes=0-1023 请求，无重试；只读取并关闭前 1024 字节。
- 响应为 HTTP **200 OK**，无 Content-Range，Content-Length: 2009，类型 text/html; charset=utf-8；正文标题为 Google Drive - Quota exceeded，开头为 HTML，非有效 ZIP Range。证据保存在 F:/AIC/.work/rgbdt500_range_probe_20260925_112858/response.json、response.bin 和单次探测脚本 probe_once.py。因此未启动采集脚本，也没有触发其内部重试。
- 清单核实仍为 pilot **93/100**、external-review **9/50**，ID分别93/93和9/9唯一；本轮新增图组 **0**，新增ID清单为空，manifest修改时间不变。没有变更划分、下载全包或使用GPU/C推理。
- 原源仍处于已确认Quota exceeded状态。本轮停止，不再重试；后续仅在新的低频检查窗口再按同一固定计划处理未完成项。

## RGBDT500 原源再次低频恢复探测（2026-09-25 12:24，香港时间）

- 探测前后均未发现 acquire_rgbdt500.py 采集进程。复用一次性探测脚本，仅向作者原始 Train.zip 地址发送一次 TLS 验证开启的 Range: bytes=0-1023 请求，无重试；只读取并关闭前 1024 字节。
- 响应仍为 HTTP **200 OK**，Content-Length 为 2009，无 Content-Range，Content-Type 为 text/html; charset=utf-8。正文标题为 Google Drive - Quota exceeded，且不以 ZIP local-header 标记开头；判定不是有效 ZIP Range。证据（脚本、JSON元数据、受限响应体）位于 F:/AIC/.work/rgbdt500_range_probe_20260925_122446/。
- 本次不启动采集器。清单仍为 pilot **93/100**、external-review **9/50**，分别93/93和9/9 ID唯一；新增图组和新增ID均为 **0**，清单修改时间未变。未改计划、未重复下载已有图组、未下载整包，未使用GPU/C推理。
- 原Google Drive源仍明确返回Quota exceeded；本轮停止，不再重试。剩余固定计划图组继续保留断点，等待后续低频恢复检查。


## 2026-09-25 13:25 下载源仍限额，暂无新增批次

距12:24超过一小时后，13:25:08仅发起一次1KiB Range请求；仍为HTTP200、text/html、Content-Length2009、无Content-Range，正文明确Quota exceeded。未启动采集、无重试，证据.work/rgbdt500_range_probe_20260925_1325。进程检查无正在运行的采集器。

本地实查清单93pilot/9external；最新部分人工包63候选59provisional，decisions.csv无已填写人工结论。未生成重复批次，D/T不启动。C提交包已交付，训练/推理剩余0。下一次原源探测不早于14:25；数据ETA未知。


## 2026-09-25 13:55 已切换百度客户端下载，官方C/M2同分

用户报告C和M2官方成绩均0.7144，为当前并列最高。两份本地官方预测对照4583/5690框不同、1107相同，非重复提交包；无官方GT不能推断纠正/退化数量。普通续训本地285到292的收益未转化为官方显示分数提升，不据此追加普通续训。

用户使用百度会员客户端下载作者Train.zip。13:54实查F:/AIC/data/external/RGBDT500/Train.zip.baiduyun.p.downloading存在、26447029453字节；这是未完成临时文件，大小可能预分配，不能当下载进度或完成。最终Train.zip尚未出现。不操作临时文件，停止Google轮询及重复镜像下载。30号跟进改为检查本地最终ZIP，完成后沿原100/50计划优先补缺失7pilot/41external，不覆盖已齐93/9和标注，必要时给现有工具补本地ZIP入口。

标注仍63候选59自动暂通过4待人工，人工0；C训练/推理已结束，D/T不启动。下载ETA无法从临时文件大小估计，等待客户端完成。百度源是作者主页公布的备用入口：https://pan.baidu.com/s/1M1_HF977Hd_-5P-vUWQebg?pwd=j3ek 。


## 2026-09-25 14:25 百度下载仍为临时文件，准备本地提取入口

约定目录仍只有Train.zip.baiduyun.p.downloading，最终Train.zip未出现；不读取或解压临时文件，也不根据26447029453预分配字节数估计进度。F盘余约255.5GiB。无新增下载或Google探测。

已派rgbdt_pilot_generate_002最小适配acquire_rgbdt500.py的--local-zip入口及合成ZIP测试，复用固定计划/清单、拒绝下载临时文件、不触网、不覆盖标签；尚在实现，不能声称完成或已抽取。最终包到齐后补缺失7pilot/41external。数据仍93/9，候选59自动暂通过，人工0；D/T未启动。镜像调查Agent001已完成停止，报告2026-09-25-rgbdt-source-switch.md无新增可执行公开源。下载ETA未知。


## 2026-09-25 Train.zip已到，本地150图组已补齐

用户确认下载完成，实查最终Train.zip为26447029453字节。本地zipfile目录14000项、400个groundtruth.txt，各10行带帧名标注，符合原训练包结构，共4000标注帧。未将视频总帧数当标注数。local_zip_acquisition_summary.json保存统计。

已运行新增--local-zip入口，保留旧102组及所有标注，成功补齐7pilot+41external，现100pilot/50external（100/20序列），两划分序列无交集。原固定计划未变。入口3项合成测试通过，本地读取不联网，跳过已完成ID，拒绝.downloading。未全盘解压也未重新下载。

Agent001正做新48组的重复/近邻筛选，输出new_arrival_localzip_screening，先返回pilot可派标清单再处理external。筛选后继续按10组生成与独立盲审，不将新图片到齐计为Query通过。旧63候选59自动暂通过4待人工，人工0，D/T仍待验收。C/M2官方同0.7144记录不变，不重跑。


## 2026-09-25 17:34 RGBDT150云端同步完成

SFTP已正常退出，云端自动提取完成并核对756文件大小、100pilot/50external清单。结果.work/rgbdt150_upload_result.txt；云端/root/autodl-tmp/rematch_20260922/data/external/RGBDT500/upload_snapshot_status.json。仅首批图组和当时标注快照，不上传全Train.zip，无GPU训练。注意annotation绝对Windows路径仍为原制作记录，正式训练前须用转换入口生成云端路径；不能直接作为训练清单。

外部003已生成6题3组（源输出results/rgbdt500_annotation_20260925/annotations/generated_external_batch003.jsonl已复制到标准data/external/RGBDT500/annotations），与external002合计新26题，旧4题共30候选；26新题尚待盲审。独立Agent已顺序安排20+6题，仍只有一名盲审。此新增输出在打包后落盘，云端标注快照未包含本次最终版本，后续增量同步。311两帧同场景同目标不能算两个独立场景。pilot66自动暂通过、人工0，D/T未启动。


## 2026-09-25 18:22 外部补充80组完成，34组可用，开始batch004/005

补充v1已从本地ZIP提取80组40个新序列，与原120序列无交集。近邻候选按序列隔离46组23序列，剩34组17序列；加原15组共49个筛选可用图组，尚不等于50组全部完成验收。路径supplement_external_v1/eligible_external_supplement_v1_manifest.jsonl；其相对图像路径的data-root必须data/external/RGBDT500/supplement_external_v1（非原根）。原计划/标签不改，新序列全部external不训练。

已派external002_resume做新增34组前10（batch004），external003_resume做索引10:20（batch005），每组目标2题，实际看三图/GT，最多两生成；余14组后续批次，不重复派发。完成后一个独立盲审串行接续。当前已通过计数不变：pilot66/70，external26/30，人工0。新提取数据未同步云端，首批150已同步，不混称全量同步。

311核查明确原GT仅伞布，独立框包含伞顶/伞杆/底座，非认错目标或漏掉GT；四题保持needs_review供人工，不为了IoU改GT或反复改Query。证据reviewed_external003/gt_extent_audit.md及gt_vs_blind_311.png。暂无训练，标注ETA未知。

