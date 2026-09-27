# M2复赛官方测试推理

## 用户决定与范围

2026-09-24用户提供新RTX4090云机connect.westc.seetacloud.com:46057，数据已克隆。用户要求无测试答案时选已训练模型推理，随后明确指定M2。采用Qwen3-VL-8B-Instruct + M2第二轮checkpoint-928三图LoRA，不重新训练，不恢复已退役EGM/Locate。

官方测试清单5690条，字段只有visible/infrared/depth/query，0条bbox；无法本地计算官方ACC，也不能据其挑最高测试分模型。历史412验证：R0=273、R2 epoch1/2=283/288、M2 epoch1/2=283/288；M2 epoch2 mIoU0.601820，解析100%。选M2按用户偏好，并未证明超过R2。

## 环境与输入输出

新卡RTX4090 24564MiB，driver570.124.04；Torch2.8.0+cu128、Transformers5.14.1、PEFT0.20.0，BF16可用，初始GPU空闲。系统盘余13GiB、数据盘余18GiB。

基础模型/root/rematch_models/Qwen3-VL-8B-Instruct；适配器/root/autodl-tmp/rematch_20260922/results/first_batch/m2/checkpoint-928。

输入/root/autodl-tmp/rematch_20260922/data/official_test_rematch_20260924/queries/queries.json，数据根为其上级官方测试目录。输出独立submissions/m2_epoch2_4090_20260924。

固定三图RGB、IR、Depth顺序，depth_mm_to_grayscale_rgb沿用训练的绝对映射，不逐图拉伸。提示复用prepare_qwen3vl_native_sft._trimodal_prompt，按image占位符交错构造，不能用不同的通用NATIVE_PROMPT替代。BF16、SDPA、关闭TF32、每图min/max_pixels=200704/602112、贪心128token。

依据contest/基于大模型的多模态视觉理解与推理.pdf第4、6页：ACC@0.5，所有ID保留，原字段不变，仅添加归一化xyxy bbox，预测JSON压成ZIP。解析沿用验证；无有效框才使用记录在案的固定全图框兜底，独立报告数量，不假称模型预测准确。用户自行上传比赛，不自动提交。

## 验证与自动运行

先对固定8条验证样本比较旧验证入口和新提交入口的raw_text、prediction与image_grid。新入口无真实框依赖，不创建伪造标签或分数。通过后自动完整5690推理，按ID恢复，结束打包和完整ID/字段/框/ZIP校验。

实际PID、进度、ETA、失败及交付结果随运行追加。时间由新卡实测估算，不能以过往Locate时间估计。本轮不自动加训练、改像素或更换模型。

## 2026-09-24 23:31 新4090已启动M2复赛5690推理（当前任务）

- 最新SSH connect.westc.seetacloud.com:46057 root，密码仅见用户消息，不入文档。RTX4090 24GB，Torch2.8.0+cu128/Transformers5.14.1/PEFT0.20.0，BF16正常。旧weste28355不再作为当前目标。
- 用户明确指定M2第二轮checkpoint-928（不是R2），三模态RGB/IR/Depth。测试5690条无bbox，不能本地计算官方分数。历史M2第2轮288/412（与R2同分）；不重新训练、不重跑所有版本。
- 新工具tools/predict_native_submission.py复用训练_trimodal_prompt和固定depth映射、旧验证生成解析路径；22项相关CPU测试通过。23:30:40真实8条新旧入口raw_text/框/image_grid完全一致、0fallback，峰值约17GiB。旧参考8条只为一致性检查，不宣称全量分数。
- 云端根/root/autodl-tmp/rematch_20260922；输出submissions/m2_epoch2_4090_20260924。runner2481、全量子进程2805；runner.pid/runner.log/stage_status.tsv、official5690_attempt日志和full/predictions.jsonl为准。脚本scripts/run_m2_rematch_submission.sh已nohup独立执行，勿重复启动/覆盖脚本/并发GPU任务。
- 23:30:40开始完整5690。完成自动生成full/predictions.json和full/submission.zip；保留原ID与字段仅新增bbox。解析无框时明确记录并全图框兜底，不隐瞒失败。按ID断点恢复，明确瞬时故障最多额外两次，不改像素、不切模型。
- 30号跟进已更新最新机器并恢复每30分钟汇报。完成后下载至F:/AIC/results/m2_rematch_submission_20260924、核对5690与ZIP结构，交付用户自行上传，再暂停。EGM/Locate仍退役，不能恢复。
- 详细docs/research/2026-09-24-m2-rematch-submission.md；证据results/rematch_20260922/new4090_46057_inventory.txt、m2_reference8_console.txt、m2_submission_start.txt、m2_submission_status.txt。实测ETA待全量稳定吞吐确认。


### 23:32 全量已推进62/5690

- 8条入口一致性检查通过，原始回答/预测框/image_grid完全相同、无兜底；reference8和validation8的GPU峰值均约17GiB。
- 全量62条完成，0解析失败/0全图兜底；GPU100%、显存17933MiB，runner2481/子进程2805存活。最近20条纯生成约0.98秒/条；包含初次加载及图像处理，目前约1.61秒/条，剩余估计约151分钟，初期按2–2.5小时预留，后续用稳定吞吐更新。
- 定时任务30已切至46057的M2提交，每30分钟报告；本机关机不影响云端自动推理/打包。产物尚未完成，不能把validation8/submission.zip当正式5690包。


## 2026-09-25 00:55 深度格式故障与修复

00:39:06全量停在2993/5690，0解析失败/0兜底；下一条004005_001的深度是360×640×3 uint8，旧入口仅支持二维整数毫米深度。这是提交入口遗漏混合格式，不是OOM或模型生成失败。

全清单引用1198张1920×1080 I;16 PNG和97张640×360 RGB JPEG。JPEG实质为现成灰度可视化（78张三通道相同，19张有不超过6级JPEG通道差），004005可见近亮远暗。官方PDF仅说明16位毫米值，未提供JPEG反变换标定，不能将其0–255当毫米。4张JPEG全黑（004390/004397/004404/004418），按原输入保留，不跳样本。

修复tools/predict_native_submission.py：16位继续原绝对深度映射，8位RGB直接保留已可视化像素；其他格式明确报错。模型、提示、像素、生成设置不变。新增JPEG像素保留回归检查，提交工具7项测试通过。云端预检全部引用深度格式，验证已完成预测所用深度均为16位；原run_config保存在run_config_before_depth_format_fix.json，仅显式更新depth_policy，保留2993条预测，按ID续跑。证据depth_jpeg_inventory.json、m2_depth_failure.txt、m2_depth_config_migration.txt；修复后真实运行进度另记。

这是输入格式适配，不能据此宣称JPEG深度与训练的毫米映射同分布；其距离尺度和局部缺失仍是数据局限。


### 00:58 真实续跑确认

00:57:29已从2993条恢复，runner5379/推理5409。00:58:16完成3035/5690（53.3%），新增42条已越过原失败样本，0解析失败/0兜底，GPU73%、17573MiB。旧8条入口一致性再次通过。当前含加载与处理约1.13秒/条、短窗口剩余约50分钟；后续回到大图区可能变慢，暂按50–70分钟预计，完成后自动打包。97张JPEG涉及175条Query，其余5515条采用16位映射。证据m2_depth_resume_status.txt；定时任务继续，未完成不交付8条测试ZIP。

### 2026-09-25 01:20 定时检查

真实进程5379/5409正常存活，完成4149/5690（72.9%），恢复后新增1156条，0解析失败/0兜底；无新错误，最新ID014050_004。恢复后含加载和预处理平均1.19秒/条，预计余约31分钟，另留几分钟打包与下载检查，约01:50–02:00可完成推理。最近20条纯生成约0.97秒/条。继续现有队列，无重复启动或配置修改。证据results/rematch_20260922/m2_depth_resume_status.txt。


## 2026-09-25 最终完成与交付

云端stage_status记录01:52:24 official5690和submission_queue均complete，任务已退出。完整5690条，无解析失败、全图兜底或生成截断；总生成时间5518.42秒，另有预处理、加载及格式故障停机时间，不将纯生成时间当端到端耗时。峰值已分配约16.96GiB。

完整产物已下载至F:/AIC/results/m2_rematch_submission_20260924。local_validation.json记录5690唯一ID精确匹配、非bbox字段保持、有限有效归一化框、ZIP根内仅predictions.json并与外部JSON一致。submission.zip已提供用户自行上传；无GT，因此没有本地官方ACC。30号跟进已暂停。

输入局限仍保留：175条Query使用现成RGB uint8深度JPEG，5515条使用原16位绝对映射；不虚构JPEG毫米值。格式故障与断点续跑未隐藏，旧配置已保存。后续训练/新方案只在用户再决定后执行。
