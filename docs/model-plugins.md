# 添加模型与具体推理过程

新增模型分为两种情况：输入输出及推理方式与已有插件兼容时，只添加配置；新架构、新预后处理或新推理 SDK 才增加代码。业务继续调用 `POST /v1/{capability}`，不需要新增一个按模型名称命名的 Web 接口。

## 已有推理代码在哪里

| 能力 | 任务：校验、预处理、后处理 | 后端：加载、执行、释放 |
|---|---|---|
| Qwen 对话、图片问答 | [ov_genai.py](../model_service/tasks/ov_genai.py)：模板、token/图片限制、结果块 | [ov_genai.py](../model_service/backends/ov_genai.py)：GenAI pipeline、原生生成、取消等待 |
| BGE、Qwen 检索、BM42 | [retrieval_ov.py](../model_service/tasks/retrieval_ov.py)、[embeddings.py](../model_service/tasks/embeddings.py)：分词、池化、排序、稀疏词项 | [openvino.py](../model_service/backends/openvino.py)：IR 编译及张量推理 |
| Chinese-CLIP / OpenAI CLIP | `DualClipTask`：文本/图片处理、归一化 | [retrieval_ov.py](../model_service/backends/retrieval_ov.py)：同实例的两个 OpenVINO 塔 |
| ASR、TTS、声纹、克隆 | [sherpa_tasks.py](../model_service/tasks/sherpa_tasks.py)：WAV、文字、参考语音、输出编码 | [sherpa_models.py](../model_service/backends/sherpa_models.py)：各类 sherpa-onnx Runtime |
| 生图、视频 | [media_generation.py](../model_service/tasks/media_generation.py)：参数、PNG/MP4 编码 | [media_generation.py](../model_service/backends/media_generation.py)：OpenVINO GenAI / Diffusers CUDA |

精确模型配置和真实验证报告见 [README](../README.md)。旧 0.8B / Whisper / VITS 的资料移至 [历史文档](model-plugins-legacy.md)，不属于当前部署。

## 只添加模型配置

1. 将完整权重放在 `model_roots` 下的新目录，固定来源版本；不要覆盖正在使用的旧版本文件。
2. 从对应 `examples/models.*.json` 复制一个对象，修改 `name`、唯一 `version`、`path` 和实际执行选项；检查任务输入/模型输出布局兼容。
3. 按本机实测设置 `resident_mb` / `request_mb`，CUDA 模型还需要 `gpu_resident_mb` / `gpu_request_mb`。驻留包括运行后保留的编译缓存，不仅是权重文件大小。
4. 设置有代表性的 `validation_input`。登记后服务会实际加载和执行，失败保持禁用。

例如保存为 `examples/my-model.json`（JSON 配置列表，即外层使用数组）后，在服务运行时执行：

```bash
.venv/bin/python scripts/register_models.py examples/my-model.json --unload-after-validation
```

脚本从 `ADMIN_API_KEY` 读取管理密钥。首次成功后用 `PUT /admin/aliases/{alias}` 选择业务默认版本；不要盲目给多种检索模型同时使用 `--defaults`，它会依次更新同能力默认别名。

同版本配置不可变。模型升级时先新增版本并验证，切换别名，再按 [移除流程](multimodal-deployment.md#调用和删除) 排空旧实例、移除旧登记；默认保留磁盘文件。

## 添加新的具体推理逻辑

稳定契约定义在 [contracts.py](../model_service/contracts.py)。代码执行顺序由 [worker.py](../model_service/worker.py) 统一维护：

```text
控制进程：task.validate(payload, capability) → 原子准入 → 加载/复用 worker
执行进程：backend.load()                         # 每实例一次
         task.prepare(payload) → Prepared(inputs, context)
         backend.infer(inputs, cancel) → outputs
         task.finish(outputs, context) → 业务 JSON
卸载时： backend.close() → 确认 worker 退出 → 释放驻留预算
```

任务负责业务输入、预后处理，不自行创建模型执行进程或操作登记表。`validate` 和构造函数只能执行轻量校验；分词器、NumPy 等在 `prepare/finish` 中延迟导入。`context` 仅属于本次请求，不把可变解码状态挂在共享单例上。

后端负责 SDK 及模型对象。只加载管理员提供的本地文件；原生调用结束之前，`infer` 不得因取消事件直接返回“已停止”。有原生取消方法则发出取消并等待；没有则等待调用结束。取消发生在后处理时，也会等后处理结束再确认，且不再发送成功结果。

真正增量生成实现 `backend.stream(inputs, cancel)` 和 `task.finish_chunk(outputs, context)`，并声明 `BackendFeatures(incremental_output=True)`。生成器关闭必须等待底层线程停止；不允许将整个成功结果先计算完再宣称增量推理，或在输出一部分后静默重试。

组装时只改以下受信任注册位置：

- [tasks/__init__.py](../model_service/tasks/__init__.py)：加入任务工厂及支持的后端 family、能力约束。
- [backends/__init__.py](../model_service/backends/__init__.py)：用 `register_backend` 登记可导入的工厂、配置校验和 `BackendFeatures`。仅换张量模型时继续复用 `openvino`，不必新建后端。
- 增加配置和测试，然后重启控制服务。管理 JSON 不接受任意 Python 模块路径，新增代码不热更新。

后端若连接外部 HTTP 服务，必须声明 `management="external"` 和 `execution_may_outlive_worker=True`。无法确定远端已停止时返回 `execution_unknown`，由核心保留隔离预算；断开 HTTP 连接不等于远端停止。

无需修改 API、请求协调器、SQLite、资源调度或模型生命周期。若新实现无法满足上述取消、资源与错误契约，应先明确扩展契约，再接入。

## 怎么验证

先运行 `pytest` 中相关插件和框架测试，覆盖输入上限、输出形状、取消、失败和重复使用；确保控制进程不导入 ML Runtime。再通过真实权重执行准备脚本和服务登记，保留输出样例、来源摘要、峰值与稳定 RSS/显存、取消停止及卸载记录。

Mock 或受控管线通过，只能说明框架和协议行为通过。实际模型结果仍需检查：如视频应有可识别的内容与变化，不能把成功编码的噪声 MP4 算作已支持。
