# optimized_C_seed2026 配方与恢复证据

## 起点、Trainer、数据

Qwen3.6-27B基座 + `runs/shared_format_warmup/training/checkpoint-50` 的LoRA权重，通过 `adapters` 加载；不是从裸基座直接开始，也没有通过 `resume_from_checkpoint` 继承warmup的优化器与步数。C从第1步重新计数，最终550次优化更新、4400次题目呈现；共同warmup另有50次更新、400题。不能将两段合并成一条相同优化器的600步训练。

`run_optimized_arms.py` 是BCD串行调度器，实际调用旧版 `tools/train_quant_grounding.py --config …/arm_config.json`；其中 `arm=C` 选择 ms-swift 4.5.3 的 `QuantSwiftSft/SwiftSft`。不是10月29日重写后的 `--mode sft` 自定义循环。旧训练源码可从Git提交 `36db37acaab5e5177b9b5820101e71246e478f81` 读取；当前同名文件已经改变。

正式输入为 `/root/autodl-tmp/rematch_20260922/data/quant_grounding_20261026_formal/schedule.jsonl`，19200条固定呈现序列，关闭dataset与DataLoader洗牌。`train_index.jsonl` 是后续纠错流程的输入，不是本次C550。共同warmup从schedule首次出现的前800个不同ID准备数据，实际训练50×8=400条。

## 参数

| 参数 | 实际值 |
|---|---|
| 实际更新 | 550，按6.75小时限时停止，非预先指定550 |
| max_steps / save_steps | 1000000 / 1000000 |
| 学习率 / 调度 | 2e-6，constant，warmup_ratio=0 |
| batch / 累积 | 单卡micro batch 1，梯度累积8 |
| 精度 | NF4双重量化，BF16计算；视觉塔、输出头和线性注意力门控保留BF16 |
| LoRA | rank32，alpha64，dropout0，语言all-linear；视觉塔与aligner冻结 |
| 像素 / 上下文 | min_pixels200704，max_pixels1204224，max_length8192 |
| 随机种子 | 2026 |
| 其他 | SDPA；逐层梯度检查点；weight_decay0；max_grad_norm1 |
| runtime_optimization | false，deterministic_training=false |

名字里的optimized不代表采用了加速候选：当时所有候选未通过门槛，`optimization/selection.json` 明确选择原train_env与micro batch1。

## checkpoint-550保存机制

`save_steps=1000000`，额外回调在2/4/8小时、匹配曝光量及最终限时停止时要求保存。本轮实际：

- 第29步：232次呈现的匹配曝光点，另复制为 `matched-exposure-checkpoint-29`。
- 第162步：累计7217.304秒，触发2小时保存。
- 第325步：累计14432.729秒，触发4小时保存。
- 第550步：累计24300.521秒，触发6.75小时停止并保存；未到8小时。

因此不是每50步保存，也不是从多个验证成绩里挑出550。`best_model_checkpoint=null`，调度器选择最大global_step。2026-10-27清盘记录明确删除普通checkpoint-29、checkpoint-162、checkpoint-325；当时保留最终550和受保护matched29。历史保留记录不等于今天远端仍存在。

若仅有完整matched29，可以按旧Trainer恢复再跑521步；若只剩warmup50适配器，需要重新初始化C并跑550步；若warmup也丢失，先重做独立50步warmup，再跑C550。仅有适配器、没有optimizer/scheduler/RNG，不能冒称完整续训。原运行未启用确定性训练，同seed重训不保证权重逐位相同。

## 原文与重建文件

- `launch-record.original.json`：从原预算账本提取的实际启动记录（原命令参数数组与时间）。
- `launch-command.original.txt`：上述命令的shell表示。
- `arm_c.archived-original.json`：从当时runtime ZIP逐字提取的配置模板。
- `arm_config.reconstructed.json`：按模板、调度器和优化决议重建的运行配置；不是声称下载到了原云端arm_config.json。
- `effective_config.reconstructed.json`：再应用原启动命令覆盖后的有效配置。
- `reconstruction-provenance.json`：配置来源和实际时间保存边界。

原云端最终路径确认为 `/root/autodl-tmp/quant_rl_20261026/runs/optimized_C_seed2026/train/checkpoint-550`。

主要证据位于 `../2026-10-26-quant-rl/evidence/`：`shared-initialization.json`、`shared-warmup50-config.json`、`budget-20261027-c-completed.json`、`c-final-trainer-state.json`、`c-final-resource-steps.jsonl`、`c-launch-runtime-audit.json`、`newmachine-cleanup-20261027.json`。调度器原文在 `F:/AIC/.work/quant_rl/run_optimized_arms.py`。

## 权重备份与预测恢复

搜索F:/AIC本地文件名、交接/清盘记录及26份相关ZIP/TAR归档成员，未找到C550、纠错1600、初始checkpoint-0或reference_adapter的本地权重。现有权重归档为A18/A36及probe A3/C3/D1，不能当作C550。现有作品材料适配器为Arm2 ck500/ck700；相同文件大小不证明相同权重。本次没有访问网盘或其他云机，因此不能断言它们绝对没有备份，也未找到已成功备份C550的记录。

新纠错/SFT流程曾保存checkpoint-0；它由C550初始化，如云端还有该点可进一步查找，这是恢复线索，不是已找到的文件。

未找到本地同名 `pkg_corr1600.json/jsonl`；`F:/AIC/.work/quant_rl/convert_pkg.py` 将云端 `/root/autodl-tmp/astra_eval/pkg_corr1600.json` 转为逐行 `{id,bbox}`，但未找到原pkg源JSON版本的确切映射。存在向17369机器同步该候选文件的脚本，未发现足以确认传输完成的证据。

两份完整半决赛预测均在本地，各5691题，合法框与ZIP/JSON一致；两包仅186个框相同，不能混称一份。已按上述转换规则分别恢复：

- `F:/AIC/results/correction1600_candidates_recovered_20261102/pkg_corr1600_lowres.jsonl`：来自 `submission_semifinal_correction1600_20261029/predictions.json`，未2倍放大、像素上限1204224、上下文8192，HF1720+vLLM3971。
- `F:/AIC/results/correction1600_candidates_recovered_20261102/pkg_corr1600_k20_highres.jsonl`：来自 `submission_semifinal_correction1600_k20_nf4_20261030/predictions.json`，2倍、8294400、32768，vLLM。

两份恢复文件都与各自来源的5691个ID顺序和bbox逐条相同。它们能恢复对应候选输出；若要逐条复现当时最终融合结果，仍需确定原pkg使用哪一版，并具备其余候选与融合配置，不能仅凭名称认定等价。恢复记录在同目录 `recovery_report.json`。
