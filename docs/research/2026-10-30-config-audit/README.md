# 半决赛配置审计与修复（2026-10-30）

接手 `DEEPSEEK_HANDOFF_20261030.md` 后的审计结论。**结论先说：配置层有真实的结构性隐患，
其中一条会让 10-29 的掉分错误随时重演；但当前推理慢的主因不是配置，而是硬件与分辨率的固有成本。**

---

## 一、已确认的问题

### 1.1 默认值就是低配（会复发的那种）

`tools/predict_vllm_quant_submission.py` 在审计前的默认值：

| 参数 | 审计前默认 | 复赛口径 | 差距 |
|---|---|---|---|
| `--max-pixels` | 1204224 | 8294400 | 6.9× |
| `--max-length` | 8192 | 32768 | 4× |
| `--upscale` | 1.0 | 2.0 | — |

**沉默即降配。** 任何新脚本漏传这三项，就静默复现 10-29 那次错误——半决赛成绩从
0.8267（Arm2 ck500，BF16，2× 放大，8294400 像素，32768 上下文）掉到 0.8014
（Arm2，NF4，低分辨率）。交接文档第 18 行已承认这是 Codex 套用了量化开发集入口，
但**根因不是那一次手滑，而是默认值本身指错了**。

### 1.2 缺「放大倍数 ↔ 像素上限」配套校验（危害最大）

原校验只有：

```python
if args.upscale <= 0 or args.limit < 0 or args.min_pixels > args.max_pixels or args.max_tokens >= args.max_length:
    raise ValueError("inconsistent query/pixel/token limits")
```

它不检查 `max_pixels` 是否撑得住放大后的尺寸。后果是：传 `--upscale 2.0` 而 `--max-pixels`
用了默认值时，图像在 `build_requests` 里被 LANCZOS 放大到 3840×2160，
**随后被图像处理器按 `max_pixels` 缩回去**——付了放大的 CPU 代价，模型收到的仍是低分辨率图，
而日志、耗时、显存全程正常，**没有任何信号**。

**这是代码分叉丢了护栏。** 复赛用的 `tools/predict_submission_vllm.py` 有
`check_upscale_consistency` 专门拦这个坑；半决赛的量化工具分叉时没带过来。

### 1.3 一次 15 分钟探针的配置被冻结成不变量

`build_llm_options()` 原本硬编码 `max_num_batched_tokens=8192`、`enforce_eager=True`、
`enable_prefix_caching=False`、`mamba_cache_mode="none"`。

来源见 `2026-10-29-semifinal-inference-speed.md:38`——原文写的是
"先以并发8、8192上下文、不启用前缀缓存与编译**做真实加载和16题吞吐**"，
**明确是一次试跑探针**。但随后它被三处冻结：

* `tests/test_predict_vllm_quant_submission.py:52` 的测试直接叫
  `test_nf4_options_match_successful_probe`，把探针值断言成规范；
* `run_correction1600_k20_nf4_semifinal.py:27` 用 `assert build_llm_options(config,8)==expected` 比对参考运行；
* `run_arm2_nf4_semifinal.py:27` 同上。

**本意防腐化的断言，反过来成了改进的枷锁。** 想调任何一个性能开关，都会先撞上这两条断言。

### 1.4 同一仓库里两套相反的引擎设置

| 场景 | `enforce_eager` | `enable_prefix_caching` | 出处 |
|---|---|---|---|
| 训练侧 RL rollout | **False** | **True** | `2026-10-28-rl-reassessment/evidence/A-args.json:368,371` |
| 提交推理 | True | False | `build_llm_options` |

没人对过账。训练侧证明这两项在本模型上是能开的。

### 1.5 测试套件原本就是红的

`test_chunked_resume_appends_full_batches_and_preserves_completed_bytes` 里的
monkeypatch lambda 只收 3 个参数，而 `run_chunks` 自加入放大功能起就传 4 个
（`predict_vllm_quant_submission.py:94`）。**测试一直失败**，所以 1.3 里那些不变量
就算被破坏也不会有人发现。审计时一并修好。

### 1.6 防御式代码密度：低，不是屎山

统计 10-26 之后新增的 10 个脚本：

| 文件 | 行数 | try | except | retry | sleep |
|---|---|---|---|---|---|
| `run_acc_correction.py` | 424 | 2 | 2 | 0 | 1 |
| `run_acc_prefill_gate.py` | 188 | 5 | 4 | 0 | 1 |
| `run_semifinal_batch_upgrade.py` | 133 | 2 | 2 | 0 | 1 |
| `run_arm2_nf4_semifinal.py` | 78 | 1 | 1 | 0 | 0 |
| `retry_arm2_nf4_invalid.py` | 102 | 1 | 1 | 3 | 0 |
| `run_correction1600_k20_nf4_semifinal.py` | 64 | 1 | 1 | 0 | 0 |
| `predict_vllm_quant_submission.py` | 225 | 1 | 1 | 0 | 0 |
| `validate_vllm_and_resume_submission.py` | 200 | 1 | 1 | 0 | 0 |
| `probe_vllm_nf4_submission.py` | 112 | 1 | 1 | 0 | 0 |

**每个入口脚本那唯一的 try/except 是同一个模式**：捕获后写 `state.json` 的 failed
状态，然后 `raise` 原样抛出。这是记录现场而不是吞异常，符合 AGENTS.md 的 fail-fast 要求。
没有静默 fallback、没有自动重试、没有多层输入校验。

**唯一真正过度工程的是 `run_semifinal_batch_upgrade.py`**（见 1.3 与第三节），
而它不在当前关键路径上。

**所以"GPT 写了一堆防御式屎山"这个判断不成立。** 真实的问题是 1.1~1.4 那几条
**配置层的默认值与冻结**，以及 1.5 的**测试失守**。这两类问题比防御式代码危险得多，
因为它们不产生噪音，只在成绩上体现。

---

## 二、效率判断（与直觉相反，需要单独说清）

**实测**：128 题 1767 秒 = **13.8 秒/题** → 5691 题约 **21.8 小时**，预计 10-31 中午完成。

**这不是配置拖出来的。** 10-30 14:50 实查前 192 题的原始记录，把估算变成了实测：

```
prompt 长度: 中位 24237 token（min 24228 / max 24248）
finish_reason: 全部 stop，无一截断
每题秒数中位: 13.80
→ 预填充吞吐 24237 / 13.80 ≈ 1756 token/秒
```

按工作量拆：

* 2× 放大后每图 3840×2160 = 8.29 MP，视觉塔 32×32 合并 → 每图约 8080 token，三图共 **24237 token/题**（实测值，与估算吻合）；
* 解码只有 64 token，**预填充与解码的工作量比约 380:1**；
* 单张 4090 上 27B NF4 的预填充吞吐约 1756 token/秒 → 24237×5691/1756 ≈ **21.8 小时**。

所以这是**预填充受限**的必然结果。再逐个看那几个开关为什么省不出时间：

* `enforce_eager=True` 关掉的是 CUDA graph，而 CUDA graph 主要加速**解码**——这里解码只占工作量约 1/380，开了几乎无效；
* `max_num_batched_tokens=8192` 把一个 24237 token 的预填充切成 3 块，改成 32768 只是 3 步并 1 步，**总 FLOPs 不变**，省的是调度开销，量级 5~15%；
* 前缀缓存对这轮无用——5691 条查询的提示词各不相同，没有可复用的公共前缀。

**结论：那些开关合计大概只占 10~20%，压不到 2 倍。** 21.8 小时是这台 4090 跑 27B×NF4×2 倍放大的物理量级。

**而且不能靠重启提速**：`--resume` 的配置一致性校验（`predict_vllm_quant_submission.py:149-152`）
会比对 `llm_options`，改任何一项都会校验失败，只能开新目录从零重跑 22 小时。
已完成的 100 多题会作废。

**参考对照**：旧机 96 GB Blackwell 上同样 2× 放大的 BF16 跑同一批数据用 6 小时 26 分
（4.07 秒/题）。差距来自硬件（4090 48 GB vs Blackwell 96 GB）+ NF4 反量化开销，
不是配置写错。

**结论：让它跑完。** 想大幅缩短只有减工作量或换更大显存的卡两条路。

---

## 三、已修内容

| 编号 | 改动 | 文件 |
|---|---|---|
| 1.1 | 三个分辨率参数去掉默认值，改为 `required=True` | `tools/predict_vllm_quant_submission.py` |
| 1.2 | 新增 `NATIVE_PIXELS` 常量与放大配套断言，错误信息给出该倍数应配的像素值 | 同上 |
| 1.3 | `build_llm_options` 的三项性能开关改为从 config 读取，缺省值保持原样（不改变既有运行结果） | 同上 |
| 1.3 | 测试改名 `test_nf4_options_keep_only_mandatory_quantization_invariants`，只断言必须成立的事实（NF4 双重量化、skip modules、LoRA），不再断言性能开关取值；新增 `test_upscale_requires_matching_pixel_budget` | `tests/test_predict_vllm_quant_submission.py` |
| 1.5 | 修好 lambda 参数个数，测试套件转绿（4 passed） | 同上 |
| 1.1 | 补 `--upscale 1.0` 显式声明（该入口本就是低分辨率对照） | `scripts/run_arm2_nf4_semifinal.py` |
| 1.2 | 两个 vLLM 工具文件头互相记录分叉与护栏位置 | `tools/predict_submission_vllm.py`、`tools/predict_vllm_quant_submission.py` |
| 1.3/清理 | 给监督脚本打退役标记，说明它是迁移期一次性工具、不在关键路径 | `scripts/run_semifinal_batch_upgrade.py` |

**验证**（`PYTHONUTF8=1`，`.venv`）：

```
PASS: 漏传分辨率参数 → argparse 拒绝启动
PASS: --upscale 2.0 配 --max-pixels 1204224 → ValueError: upscale 2.0 needs --max-pixels >= 7879680
PASS: --upscale 2.0 配 8294400 → 通过校验，继续到读数据
tests: 4 passed
```

**对在跑任务的影响：零。** 上述改动都改在代码层，PID 850004 的进程已加载完模块，
不受影响；`build_llm_options` 的缺省行为逐字段未变，`resume` 的配置校验也不会因本次改动失配。

---

## 四、待办与需要决定的事

1. **云机密码**：交接文档写"密码另给"，目前未拿到，无法核实在跑任务、无法下载预测包。
2. **半决赛提交截止时间**：`contest/` 里只有复赛材料，本地查不到。这决定 22 小时是否可接受。
3. **`1600-2-nf4` 那个每 30 分钟监控**：在 Codex 侧，不在本会话；接管时需停掉避免双控。
4. **引擎选项基准**：计划写 `tools/benchmark_vllm_engine_options.py`，在同一批 16 题上对比
   现状 / 开 CUDA graph / 提高 `max_num_batched_tokens` 到 32768 / 两者同开。
   判据沿用 10-29 那次批量实测的标准：**原文与框完全一致**且至少提速 15%。
   **只在预测包下载完成后跑，绝不打断在跑任务。** 按第二节的分析，预期收益有限，
   但必须实测而非猜测。

---

## 五、接手时的当前状态（2026-10-30 交接）

* 在跑：PID 850004（入口 `run_correction1600_k20_nf4_semifinal.py`），适配器
  `runs/acc_correction_20261029/correction/train/checkpoint-1600`，
  运行根 `runs/semifinal_correction1600_k20_nf4_20261030`
* 固定口径：2× LANCZOS / `min_pixels` 200704 / `max_pixels` 8294400 / `max_model_len` 32768 /
  并发 8 / 分块 64 / 贪心 / `max_tokens` 64 / seed 2026
* 这个实验真正的价值：**把"NF4 掉分"和"低分辨率掉分"两个变量分开**。
  0.8267（BF16 + 2×）与 0.8014（NF4 + 低分辨率）之间同时变了两个因素；
  本次跑的是 NF4 + 2×，若落在 0.8267 附近，说明掉分全来自分辨率。
