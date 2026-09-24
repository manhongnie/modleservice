> 历史阶段说明：本文的 Qwen3.5-0.8B / Whisper / VITS 实测与配置属于迁移前。当前模型、操作和验证请以 [README](../README.md)、[多模态部署](multimodal-deployment.md) 和 [部署报告](deployment-validation.json) 为准。

# 模型与插件接入

任务插件只负责校验、准备输入和整理输出；后端只负责加载、执行和释放。控制进程构造插件时不导入 NumPy、Torch、Transformers、OpenVINO 或 sherpa-onnx。分词器和处理器在执行进程第一次 `prepare` 时加载，模型权重由后端 `load` 加载。模型目录只由管理员配置，业务请求不能指定路径、URL、后端或设备。所有 Transformers 加载均使用 `local_files_only=True`，不执行远程模型代码。

## 配置示例与真实支持边界

| 模型 | 任务插件 | 后端 | 输入与导出约定 |
| --- | --- | --- | --- |
| BGE-M3 Dense | `bge_m3_dense` | `openvino` | `texts` 字符串数组；IR 输出 `last_hidden_state[batch,tokens,hidden]`，取 CLS 并做 L2 归一化。只支持 Dense，不声称支持 M3 sparse/multivector。 |
| BGE Reranker v2 M3 | `bge_reranker` | `openvino` | `query` + `documents`；成对分词，输出 `logits[batch,1]`；可配置 sigmoid，结果按分数降序并保留原索引。 |
| Chinese-CLIP | `chinese_clip` | `openvino` | 文本塔和图像塔分别登记；使用 ChineseCLIPProcessor，IR 须输出投影后的 `text_embeds` 或 `image_embeds`，然后归一化。不是任意 hidden state。图像只接收 base64，无远程取图。 |
| BM42 | `bm42_http` | `http` | `texts` → 真正 BM42 服务的 JSON 响应。模型注意力聚合、词项映射、IDF/检索语义留在远端实现；本项目没有本地 BM42 实现，也没有提供运行中的 BM42 服务。 |
| Qwen3.5-0.8B | `qwen_chat` | `transformers` | `messages` 只接受文本；使用官方 chat template、关闭 thinking、限制输入 token 数和新 token 数；CPU 贪心生成。首版不提供图像/视频输入或工具调用。 |
| Whisper tiny multilingual | `whisper_asr` | `transformers` | `audio_base64` 是 mono PCM16、16kHz WAV，默认最多 30 秒；固定中文转写，可通过管理员配置语言。 |
| VITS icefall zh aishell3 | `vits_tts` | `sherpa_tts` | 中文 `text`、可选 `speaker_id` 0..173、`speed` 0.5..2；返回 base64 PCM16 WAV。模型自带词典/FST 前端由 sherpa Runtime 执行。 |
| 明确的测试替身 | `mock` | `mock` | 覆盖上述 7 类业务能力；所有响应含 `mock:true` 和 Synthetic notice，不加载任何真实模型。 |

`examples/models.mock.json` 可用于框架测试。`examples/legacy/models.real.json` 包含用户新增的 Qwen/ASR/TTS，`examples/models.retrieval.json` 是四类检索模型的接入模板，不能直接视作已下载的模型。BGE/CLIP 需先由管理员导出符合上表的 OpenVINO IR，并将分词器/processor 文件放在 XML 同目录。缺少输出名或形状不符会明确失败，不猜测池化方式。HTTP 通用插件要求远端已经使用这里的业务 JSON 格式，现有服务格式不同时应增加任务适配器。

## 准备三个真实模型

在仓库根目录运行，下载是明确的部署准备步骤，服务请求不会自动下载：

```bash
uv pip install --python .venv/bin/python 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[generative,speech]'
python scripts/download_models.py qwen asr tts
```

Qwen、Whisper 下载脚本固定上游 commit，`source.json` 记录来源和权重 SHA256。大权重采用有文件锁的分段下载，中断后可补齐稀疏文件中尚未完成的区间，必须匹配上游总 SHA256 后才改名为正式文件。小文件中断后重新下载。TTS 使用上游命名发布包；该 URL 不等于不可变内容哈希，正式部署应把包纳入自己的制品库并校验哈希。

示例 Qwen CPU FP32 常驻预算为 5500 MiB，临时预算 512 MiB；这是保守配置值，需以实际进程 RSS 校准，并非硬内存隔离。默认服务 4096 MiB 会明确拒绝该配置，运行三个真实模型请把 `total_memory_mb` 调到宿主机可承受的值（例如 8192），并按需加载。每个真实模型示例 `concurrency=1`，避免分词/生成状态被多个线程并发使用。

管理新增、验证、启用仍走统一 API：不会因为下载完成或存在配置就自动启用。Whisper 的默认验证输入为极短静音 WAV，只验证加载、输入输出和推理路径，不证明转写质量。TTS 验证输入为“你好，世界”。Qwen 验证输入为短问候。

## 业务输入例子

以下是统一推理请求中的 `input` 内容（外层 URL、模型选择、鉴权见 README）：

```json
{"messages":[{"role":"user","content":"用一句话介绍北京"}],"max_new_tokens":48}
```

```json
{"audio_base64":"<16kHz mono PCM16 WAV 文件的 base64>"}
```

```json
{"text":"你好，欢迎使用语音服务。","speaker_id":10,"speed":1.0}
```

Qwen 的 HTTP SSE 已支持真实增量文字；ASR/TTS 仍缓冲完成后返回一块，没有音频增量输出。Qwen/Whisper 使用 generation stopping criteria 响应取消；TTS 在分段回调处响应取消。若当前算子不能即时停止，则等待实际执行结束后才返还许可，不强杀模型进程，不在部分输出后重试。

## 扩展入口

实现 `TaskPlugin.validate/prepare/finish` 后注册到 `tasks.TASKS`。`prepare` 返回 `Prepared(inputs,context)`，`context` 只在同一次请求使用。后端实现 `load/infer/close` 并在 `backends.create_backend` 注册。兼容插件增加模型只需配置；新增 Python 插件代码需重启控制服务，不提供运行时代码热更新。任务必须拒绝自己不支持的能力，后端取消结束前不得报告成功停止。

## 官方资料

- [BGE-M3 官方模型卡](https://huggingface.co/BAAI/bge-m3)
- [BGE Reranker v2 M3 官方模型卡](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Chinese-CLIP 官方仓库](https://github.com/OFA-Sys/Chinese-CLIP)
- [BM42 / FastEmbed 官方说明](https://qdrant.tech/articles/bm42/)
- [Qwen3.5-0.8B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Whisper tiny 官方模型卡](https://huggingface.co/openai/whisper-tiny)
- [sherpa-onnx VITS aishell3 模型与下载](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html)
- [sherpa-onnx Python TTS 示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/offline-tts.py)

## 本机真实验证结果

已在本机下载实际权重并通过 `ProcessExecutor` 子进程执行。原始记录见 [models-validation.json](models-validation.json)，可用以下命令重跑：

```bash
.venv/bin/python scripts/validate_real_models.py --cancel
```

| 模型 | 加载耗时 | 本次推理耗时 | 执行后进程 RSS | 结果与限制 |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 2.27 秒 | 0.84 秒 | 5453 MiB | 短提示得到“你好”；17 输入 token、4 输出 token；CPU FP32、含官方视觉权重但仅开放文本。 |
| Whisper tiny | 1.91 秒 | 0.34 秒 | 577 MiB | 极短静音样例跑通，但输出“你”，不能作为识别质量依据。 |
| VITS zh aishell3 | 0.38 秒 | 0.12 秒 | 145 MiB | 实际生成约 1.65 秒、8kHz PCM16 WAV，保存在 `var/real-tts-smoke.wav`。 |

三个模型均完成 warm 请求取消测试，返回 `cancelled` 后执行进程 `busy=0`；取消确认耗时依次约 0.104、0.182、0.022 秒。这些数值是本次小样例观察值，不是延迟保证。首次加载、不同输入和宿主机负载会改变耗时及内存。

还将真实 TTS 的“你好，世界。”重采样成 16kHz 后交给 Whisper tiny，识别为“你好事件”，存在同音字错误。见 [speech-roundtrip-validation.json](speech-roundtrip-validation.json)。链路已运行，语音质量与识别准确率尚未验收。

BGE-M3、BGE Reranker、Chinese-CLIP 没有下载实际权重或完成真实导出验证；BM42 没有可用远端服务。对应插件边界、配置和张量/JSON 合约单测不构成真实模型已验证。OpenVINO 的真实 Runtime 小图测试也不等于这些检索模型已经实测。

本次依赖为 Torch 2.14.0+cpu、Transformers 5.17.0、sherpa-onnx 1.13.8、NumPy 2.5.3。任务插件 23 项单测通过，包括控制进程不加载 ML 库、HTTP 结果形状校验、Dense CLS 池化、reranker 排序、WAV 输入输出及明确 Mock 标识。
