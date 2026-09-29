# 接手阶段验证记录

日期：2026-09-27。所有检查在本地 CPU 完成，没有加载真实 GPU 模型或重新训练/推理。实际 Python 是 `F:/AIC/code/.venv/Scripts/python.exe`；未安装或升级依赖。

## 旧预测与数据独立复算

| 程序 | 输入 | 结果 |
|---|---|---|
| `verify_aux.py` | 已保存 C/选择/完整412、候选证据、train16、env8 | 292/273/221；104映射误差0；train16为14/13/9；保存env8四字段相同；数字ID修复只变3题 |
| `verify_execution.py` | 7模型完整412预测、独立浮点GT、三臂清单与消费/训练状态 | 285/292/288/283/287/284/286，与保存报告一致；每臂1600消费顺序相同；学习率记录符合总200步 |
| `recompute_data_coverage.py` | 人工CSV、252候选、181放行、City和三臂清单 | 181=98/58/25、98Query；City3707/412；共同1200/新400、217同/183异；B新98图组全部旧图 |

以上脚本只读取历史输入，将结果写在本审计目录；没有回写原预测或修改人工决定。具体字段和输入路径见脚本本身及对应JSON。

另做两项小范围CPU核查：按批准的居中扩张截边数学规则重算旧270次裁剪，42条像素窗不同（41碰边）；对一条白车样本的原始uint16 Depth和SAM mask核对有效像素为0。保存为 `aux-extra-checks.json`。官方输入JSON另检查5690条均无bbox，Depth路径后缀5515 PNG、175 JPG，保存为 `official-input-check.json`；不是全部图像解码/物理标定。

## 现有测试

运行目录均为 `F:/AIC/code`。

### 核心旧融合模块：66通过

```text
F:/AIC/code/.venv/Scripts/python.exe -m pytest -q tests/test_adapters.py tests/test_auxiliary_training.py tests/test_compact_checkpoint.py tests/test_config.py tests/test_data.py tests/test_lora.py tests/test_model.py tests/test_resume_learning_rate.py tests/test_selection.py tests/test_sparse_checkpoint.py tests/test_training_output_lifetime.py
66 passed in 8.20s
```

### 新候选链条接口：28通过

```text
F:/AIC/code/.venv/Scripts/python.exe -m pytest tests/test_aux_selection_evidence.py tests/test_predict_aux_selection.py tests/test_prepare_aux_selection.py tests/test_report_aux_selection.py -q
28 passed in 0.51s
```

### 原生入口、评分和脚本：42通过、1失败

```text
F:/AIC/code/.venv/Scripts/python.exe -m pytest -q tests/test_native_lora_runner.py tests/test_predict_native_submission.py tests/test_evaluate_pretrained_grounder.py tests/test_report_rematch_experiment.py tests/test_gu_pilot_gate.py tests/test_slurm_scripts.py
1 failed, 42 passed, 1 warning in 17.23s
```

原始输出保留 `execution-pytest.log`。系统默认 Python 缺少 pytest；上述实际测试使用已有项目虚拟环境。数据代理未在该虚拟环境运行其另外三个测试文件，以标准库逐记录复算作为本轮数据链验证。

失败项 `tests/test_slurm_scripts.py::test_slurm_environment_allows_source_archive_without_git`：Windows Git Bash子进程返回码检查通过，后台读取输出时按UTF-8解码字节0xce抛 `UnicodeDecodeError`，随后 `result.stdout=None`，检查 `GitCommit=unavailable` 时抛 `TypeError`。本轮没有改测试或shell脚本，也没有以忽略错误后重跑的方式算作通过。能确认这次失败链的直接原因是输出解码，不能据此声称已经验证该测试原想检验的输出行为。

本轮上述不同测试范围合计136通过、1失败；数据链另由标准库复算程序验证。没有声称全测试套件通过。所有单测仍只能验证合成输入和接口，不能检验真实模型的同对象绑定、跨图坐标遵从或模态因果贡献。
