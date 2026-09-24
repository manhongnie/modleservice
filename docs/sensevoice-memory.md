# SenseVoice Small 内存专项实测

2026-09-24 在本机补做。此前只有短句推理、峰值 RSS 和卸载验证，没有连续调用内存专项。此次结论：**当前 sherpa-onnx CPU int8 路径在这批请求中未出现持续增长；只关闭模型对象确实不能让 RSS 回到初始水平；项目现有的执行进程退出机制通过了实际回收检查。** 这不等于已排除长期泄漏。

## 环境与方法

- 模型：`sensevoice-small-int8@2024-07-17-c71f0ce00bec`，ONNX SHA256：`c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`。
- Linux / WSL2，Python 3.13.13，sherpa-onnx 1.13.8，随其发布的原生 ONNX Runtime 1.28.2，CPU、2线程、实例并发1。独立诊断读取原生库版本，未导入 FunASR 或 PyTorch。
- 生产配置：按需加载、空闲60秒卸载，驻留预算768MiB、单请求临时预算256MiB，最长30秒音频，JSON输入上限2MiB。
- 使用真实 `Service → 调度 → 执行子进程 → sherpa` 路径和独立临时 SQLite，不修改部署数据库。输入为已登记的真实 Matcha 合成参考句，重复/截断为1、5、15、30秒的16kHz单声道PCM16 WAV。
- 不在每次请求后执行 GC，不调用 `malloc_trim`，不修改 arena/provider 参数。每次结束检查请求名额、队列、临时预算及工作进程任务数归零，并检查同一 PID、仅加载一次。
- RSS 是操作系统看到的进程驻留内存；USS 是该进程独占的内存。报告中 `_mb` 字段均按 MiB 计算。每次请求后采样；另有20ms采样线程，但在初次验证/重新加载推理结束后才挂接，**不能把该采样最大值当成完整冷启动峰值**。

## 连续调用和卸载

[机器可读报告](sensevoice-memory-validation.json)，原始逐请求样本为 `var/sensevoice-memory/samples.jsonl`。

| 阶段 | 次数 | 音频长度 | 首次请求后 RSS → 最后请求后 RSS |
|---|---:|---|---:|
| 预热 | 40 | 1/5/15/30秒交替 | 374.88 → 483.26MiB |
| 固定短音频 | 300 | 1秒 | 483.26 → 483.45MiB |
| 配置允许的最长音频 | 100 | 30秒 | 483.45 → 485.88MiB |
| 长短交替 | 300 | 1/5/15/30秒 | 485.88 → 485.88MiB |
| 长音频后继续短音频 | 300 | 1秒 | 485.88 → 485.88MiB |
| 再次持续长音频 | 100 | 30秒 | 485.88 → 485.88MiB |

同一工作进程共完成1,140次上述识别，最后700次请求后采样 RSS 保持485.88MiB。阶段运行合计约224秒；包含取消、真实空闲等待和重新加载的整个测试约315秒。

随后额外验证：

- 10次30秒音频执行中取消：确认耗时0.53–0.58秒，确认后执行许可、临时预算和队列均归零，RSS仍为485.88MiB。ASR原生调用不能立即中断，需要等待本次调用结束；这是服务层取消测试，未重新执行HTTP断连测试。
- 取消后再次识别成功，仍复用原PID。
- 实际等待配置中的60秒空闲时间，观察到60.57秒后原PID退出，驻留与临时预算均归零。
- 5轮按需重新加载、推理、管理卸载：5个PID均确认退出，预算均归零；每次无在途请求的卸载约0.064–0.076秒。模型登记保留启用，下次可重新加载。

额外调用计入后共1,161个测试请求（包含10个取消，不包含新增模型的1次验证推理）。测试控制进程RSS从43.29升到46.16MiB；探针自身持有统计样本、报告和服务状态，该变化不能直接归因为服务内存泄漏。生产配置的768+256MiB预算覆盖本次观测的运行阶段，但本测试没有测完整冷启动瞬时峰值，也不构成所有输入的容量证明。

## 只关闭模型对象的对照

在另一个独立子进程中，重复3次真实 `load → 30秒推理 → backend.close()`，每次关闭均确认模型引用为 `None`，等待1秒后测量。`close()` 已执行 `gc.collect()`；期间不退出该进程。初始RSS为54.53MiB、USS为28.26MiB。

| 轮次 | 推理后 RSS | close + GC + 等待1秒后的 RSS | 关闭后 USS |
|---|---:|---:|---:|
| 1 | 476.23MiB | 344.99MiB | 306.95MiB |
| 2 | 430.30MiB | 354.27MiB | 316.06MiB |
| 3 | 433.40MiB | 353.32MiB | 315.06MiB |

可见对象关闭后仍比初始保留约290–300MiB RSS；GC不能保证全部归还操作系统。三轮变化没有证明持续泄漏。最后让子进程正常退出，退出码0，确认原PID身份不再运行；没有强制终止。[完整对照报告](sensevoice-close-validation.json)。

ONNX Runtime 官方说明默认arena会保留分配的区域并复用，RSS不立即下降可能来自内存池、分配器缓存或碎片。[官方内存管理说明](https://onnxruntime.ai/docs/get-started/with-c.html#features)。本轮未做原生堆分配归因，不能认定上述全部残留都来自ORT arena。

上游确实存在相关讨论：[SenseVoice #127](https://github.com/QwenAudio/SenseVoice/issues/127)涉及原项目/FunASR调用路径；[sherpa-onnx #3032](https://github.com/k2-fsa/sherpa-onnx/issues/3032)讨论CPU SenseVoice较高内存和arena设置。它们不是当前固定版本和配置必然持续泄漏的证据，不能跨后端直接套用结论。

## 当前回收机制和限制

后端的 `close()` 释放模型引用属于对象生命周期；服务的卸载还要完成执行进程生命周期。对应代码：

- `model_service/backends/sherpa_models.py`：每次识别创建本地stream，后端关闭清除模型引用并GC。
- `model_service/worker.py`：无在途任务时关闭后端，确认卸载并结束进程。
- `model_service/executor.py` 的 `unload()`：等待实际进程退出，未退出不能报告成功。
- 生命周期/模型管理器：确认停止后才释放驻留预算，空闲策略复用同一安全卸载流程。

因此业务处理完一次请求后RSS保持在约486MiB是当前保留模型复用的行为；空闲卸载或管理卸载后才退出整个工作进程。此处回收依据是PID退出及预算归零，不能将其描述为系统“空闲内存”必定增加同样的数值，操作系统仍可能保留可回收文件缓存。

连续流量会阻止空闲卸载；目前没有按请求次数或RSS阈值自动轮换工作进程。预算是准入估算，不是操作系统强制内存上限。本次只覆盖重复合成语音、当前CPU/int8版本、并发1和最大30秒输入，没有24/72小时持续测试、多种真实录音、多实例并发或其他ORT版本测试。未因本次结果替换模型、调小预算或加入每请求GC。

## 复现

在项目根目录、模型已下载的当前环境执行，无需先启动HTTP服务：

```bash
.venv/bin/python scripts/validate_sensevoice_memory.py
.venv/bin/python scripts/probe_sensevoice_close.py
```

第一条完整测试包含真实60秒空闲等待，默认覆盖上表及10次取消、5轮卸载。`--warmup`、`--short`、`--long`、`--mixed` 调整阶段次数，`--cancel` 和 `--cycles` 调整取消/卸载次数。`--report` 指定独立JSON报告，`--samples` 指定逐请求JSONL，避免覆盖已有证据。第二条直接关闭对照固定三轮，也支持 `--report`。

相关回归检查本次重新运行 **61 passed，9.58秒**：

```bash
.venv/bin/python -m pytest -q tests/test_sherpa_models.py tests/test_executor.py tests/test_lifecycle_races.py tests/test_runtime_maintenance.py
```

原始输出：`var/sensevoice-memory-regression.log`。这些自动化测试与上述真实模型内存观测分别计数，不将Mock控制测试算作真实模型验证。
