# Shared Model Service

**单机、进程隔离、可扩展的多模型推理服务。**

通过一套 HTTP API 为多个业务提供对话、视觉问答、检索、语音、图片和短视频生成能力。服务使用 FastAPI 接收请求、SQLite 保存模型登记，并按需启动和复用独立的模型执行进程，适合在一台机器上共享有限的 CPU、内存和 GPU 资源。

## 核心能力

- **统一调用**：业务只指定能力、模型名称或别名和输入，无需了解模型路径与推理后端。
- **按需加载与复用**：支持常驻、按需加载和空闲卸载；模型执行与 HTTP 控制进程隔离。
- **资源准入**：统一管理内存、显存、模型并发、全局执行名额和有界等待队列。
- **模型生命周期**：登记时实际加载并验证，验证通过才允许启用；支持版本、别名、依赖检查和等待在途请求结束后卸载。
- **流式输出与取消**：对话支持 SSE 增量文本；执行停止并完成输出清理后才归还请求额度。
- **可扩展插件**：任务插件负责输入输出，后端负责加载、推理和释放；兼容模型可仅通过配置接入。
- **运维接口**：独立的业务与管理密钥、健康检查、队列与进程状态、Prometheus 格式指标和 SQLite 在线备份。

## 支持的模型与能力

下表列出仓库提供的插件和配置示例。模型权重需要单独准备，克隆源码不会自动安装或登记这些模型。

| 能力 | 模型示例 | 推理后端 | 配置与说明 |
| --- | --- | --- | --- |
| 对话、看图问答 | Qwen3.5-4B AWQ、Qwen2.5-VL-7B INT4 | OpenVINO GenAI，CPU | [配置](examples/models.genai.json) · [指南](docs/genai-models.md) |
| 稠密检索、重排序 | BGE-M3、BGE Reranker、Qwen3 Embedding / Reranker | OpenVINO，CPU | [配置](examples/models.retrieval-migrated.json) · [指南](docs/retrieval-migration.md) |
| 图文检索、稀疏检索 | Chinese-CLIP、OpenAI CLIP、BM42 | OpenVINO，CPU | [配置](examples/models.retrieval-migrated.json) · [指南](docs/retrieval-migration.md) |
| 语音识别、合成、声纹、声音克隆 | SenseVoice、Matcha、ERes2Net、ZipVoice | sherpa-onnx，CPU | [配置](examples/models.sherpa.json) · [指南](docs/sherpa-models.md) |
| 图片生成 | SD-Turbo | OpenVINO GenAI，CPU | [配置](examples/models.media.json) · [指南](docs/media-models.md) |
| 短视频生成 | AnimateDiff-Lightning + epiCRealism | Diffusers / PyTorch，CUDA | [配置](examples/models.media.json) · [指南](docs/media-models.md) |

另提供 HTTP 后端用于对接外部推理服务，以及用于开发和接口联调的 Mock 后端。旧 Qwen3.5-0.8B、Whisper Tiny 和 VITS 示例保留在 [examples/legacy](examples/legacy/README.md)。

## 快速开始

### 1. 安装基础服务

需要 Python **3.11 或更高版本**；现有完整验证环境为 Linux / Python 3.13。以下命令均在下载或克隆后的仓库根目录执行。基础服务和 Mock 示例不需要模型权重或 GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

也可以使用 `uv venv .venv --python 3.13` 创建环境，再运行 `uv pip install --python .venv/bin/python -e .`。

### 2. 启动开发服务

业务和管理密钥必须非空且不同。用 Python 生成随机密钥，在当前终端保留它们：

```bash
export BUSINESS_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"
export ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(24))')"

python -m model_service --config examples/service.json --check-config
python -m model_service --config examples/service.json
```

默认监听 `127.0.0.1:8000`。相对路径以启动时的工作目录为基准，运行数据写入 `var/`，本地权重放在 `models/`。这两个目录中的运行产物不提交到 Git。

### 3. 登记 Mock 模型并调用

打开第二个终端，进入同一仓库并激活虚拟环境。设置**与服务终端相同**的密钥，不要重新生成：

```bash
source .venv/bin/activate
export ADMIN_API_KEY='<服务终端使用的管理密钥>'
export BUSINESS_API_KEY='<服务终端使用的业务密钥>'

python scripts/register_models.py examples/models.mock.json --defaults --unload-after-validation

curl --fail-with-body http://127.0.0.1:8000/ready

curl --fail-with-body http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer $BUSINESS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"texts":["多个业务共享模型"]}}'
```

响应包含 `request_id`、`model`、`output`、`mock` 和 `done`。这里的 `mock:true` 表示合成的联调结果，不是真实模型推理。`--defaults` 为每种能力设置默认别名，之后可省略请求中的 `model`。

## 使用真实模型

基础安装不包含机器学习运行时。按需要安装可选依赖，并使用显式准备脚本下载权重；模型加载与推理阶段只使用本地文件。

| 使用场景 | 可选依赖组合 | 准备脚本 |
| --- | --- | --- |
| 对话、视觉问答 | `.[openvino,genai,text]` | `scripts/prepare_genai_models.py` |
| 向量、排序、图文与稀疏检索 | `.[openvino,retrieval,generative]` | `scripts/prepare_retrieval_models.py` |
| 语音与声纹 | `.[speech]` | `scripts/prepare_sherpa_models.py` |
| 图片与视频生成 | `.[openvino,genai,text,media]` | `scripts/prepare_media_models.py` |

下载默认使用系统 `curl`；视频编码还需要 `ffmpeg`。CUDA 视频需要匹配硬件与驱动的 PyTorch CUDA 构建。准备脚本固定模型来源版本，校验文件长度及上游提供的 SHA256，并记录本地文件摘要。

CLIP 首次准备需使用 `--export-clip`，通过 PyTorch 导出 OpenVINO IR；已有 IR 的检索推理只需 `.[openvino,retrieval]`。

例如，准备对话与视觉模型：

```bash
python -m pip install -e '.[openvino,genai,text]'
python scripts/prepare_genai_models.py chat vision
```

停止开发服务后，按硬件条件调整 [多模态服务配置](examples/service.multimodal.json)。该配置启用生产模式并禁止 Mock；两组密钥均需至少 32 个字符，上文生成的密钥符合此要求。开发示例与多模态示例默认共用 `var/models.sqlite3`；若已经登记 Mock 模型，请先为多模态配置设置一个独立的 `database` 路径，再启动：

```bash
python -m model_service --config examples/service.multimodal.json
```

**一个数据库只能由一个控制进程使用**，不要启动多个 Web worker 或开启自动 reload。

服务启动后，在另一个终端登记已经准备好的模型：

```bash
python scripts/register_models.py examples/models.genai.json --defaults --unload-after-validation

curl --fail-with-body http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"请用一句话解释向量检索"}],"max_new_tokens":64}}'
```

登记会实际加载模型并执行 `validation_input`。成功后才启用，失败时保留禁用登记和错误原因；`--unload-after-validation` 在验证后释放模型进程，业务调用时再按需加载。

多模态示例采用 **12288 MiB 内存预算、10240 MiB 显存预算和 1 个执行名额**，来自一台 16 GiB RAM / 12 GiB NVIDIA GPU 的验证机器。它是部署参考，不是所有模型的统一硬件要求；请按所选模型调整预算，并为操作系统、控制进程和其他应用留出空间。[多模态部署指南](docs/multimodal-deployment.md)包含完整准备、离线登记和默认别名设置步骤。

## API 概览

业务统一使用 `POST /v1/{capability}`，并携带 `Authorization: Bearer $BUSINESS_API_KEY`：

```json
{
  "model": "模型名称、name@version 或别名",
  "input": {},
  "stream": false
}
```

`model` 可省略，此时解析 `default:{capability}`。`input` 由对应任务插件校验，具体字段见各模型指南。音频、图片和视频通过 Base64 传输，业务接口不接受本地权重路径。

将 `stream` 设为 `true`，使用 `curl -N` 接收 SSE 的 `chunk`、`result` 或 `error` 事件。GenAI 对话提供真实增量文字；其他能力目前整段计算后返回。请求在输出和底层执行均结束前持续占用许可，已输出部分内容的失败请求不会静默重试。

管理接口使用 `Authorization: Bearer $ADMIN_API_KEY`：

| 操作 | 方法及路径 |
| --- | --- |
| 健康、就绪检查，无需密钥 | `GET /health`、`GET /ready` |
| 模型列表、运行状态、指标 | `GET /admin/models`、`GET /admin/status`、`GET /admin/metrics` |
| 登记并验证模型 | `POST /admin/models`，输入 `{"config": ModelConfig, "enable": true}` |
| 验证、加载、卸载、启用、停用 | `POST /admin/models/{name@version}/{operation}`，operation 为 `validate`、`load`、`unload`、`enable`、`disable` |
| 移除模型登记 | `DELETE /admin/models/{name@version}` |
| 设置、删除别名 | `PUT /admin/aliases/{alias}`、`DELETE /admin/aliases/{alias}` |
| 设置、删除业务依赖 | `PUT /admin/dependencies/{business}`、`DELETE /admin/dependencies/{business}` |
| 确认未知远端执行已停止 | `POST /admin/requests/{request_id}/reconcile` |

卸载保留登记与启用状态，下次请求可重新加载；停用或移除前需要处理别名和已登记业务依赖。移除默认保留磁盘权重。等待在途请求结束超时会保留暂停状态，不自动强杀。完整操作、故障处理和备份方法见 [操作手册](docs/operations.md)。

## 配置与扩展

模型配置包含唯一版本、能力、路径、任务、后端、设备、加载策略、并发、内存与显存预算，以及真实验证输入 `validation_input`。模型文件必须位于服务配置的 `model_roots` 中。

同一个 `name@version` 的配置不可原地修改。升级时登记新版本、验证、切换别名，再卸载或移除旧版本；保留旧版本可用于回滚。

兼容现有任务和后端的模型只需添加 JSON 配置。新的推理实现分为两层：

- `model_service/tasks/`：实现 `validate → prepare → finish`，负责输入校验、预处理和后处理。
- `model_service/backends/`：实现 `load → infer → close`，负责模型运行时；增量推理可实现 `stream`，任务可实现 `finish_chunk`。

插件在各自的 `__init__.py` 中登记，代码更新后重启服务。设计与示例见 [架构说明](docs/architecture.md)、[插件指南](docs/model-plugins.md)和[执行语义](docs/execution.md)。

## 开发与验证

```bash
python -m pip install -e '.[test]' 'numpy>=2,<3' 'pillow>=11,<13' 'mmh3>=5,<6'
python -m pytest -q
```

上述安装覆盖基础测试的直接依赖。部分测试需要 OpenVINO、PyTorch 或系统 `ffmpeg` / `ffprobe`，缺少时会跳过；完整环境安装方式见 [部署指南](docs/multimodal-deployment.md)。[requirements-tested.txt](requirements-tested.txt) 是已验证 Linux / Python 3.13 / CUDA 环境的版本快照，不是跨平台通用安装清单。

仓库保存了 326 项自动化测试和 15 个真实模型的历史验证记录，详见 [验证说明](docs/verification.md)、[自动化结果](docs/automated-test-results.json)和[部署报告](docs/deployment-validation.json)。这些报告记录特定环境、版本与输入下的结果；Mock 和受控管线测试用于检查框架行为，真实效果与容量由独立模型报告说明。

欢迎通过 Issue 提交问题或通过 Pull Request 贡献改进。问题报告请附上环境、模型与后端版本、最小复现步骤及脱敏日志；增加插件时请补充输入输出、取消和失败场景测试，声明依赖与模型来源，并单独记录真实模型验证结果。

## 部署边界

- 服务面向单机、单控制进程，暂不提供分布式调度、高可用、多租户配额或模型级 ACL。
- 内存与显存预算用于准入估算，不是操作系统或 GPU 的硬隔离；FIFO 队列可能队头阻塞，不抢占执行中的模型。
- 原生编译、预处理或编码无法立即中断时，取消会等待底层真正停止；HTTP 后端无法确认远端停止时保留隔离额度。
- 当前图片与短视频示例有尺寸、帧数限制，音频按整段 WAV 处理；暂不支持实时音频流、长视频或插件热更新。
- 默认仅监听本机。生产部署请配置独立密钥、可信模型目录和反向代理，参考 [systemd 模板](deploy/model-service.service)与 [nginx 模板](deploy/nginx.conf.example)。

## 许可证与模型权重

项目源码采用 [Apache License 2.0](LICENSE) 开源。

模型权重不随源码分发，其许可证与使用条件由各自的上游项目规定。模型准备脚本及对应指南记录来源、版本和可获得的许可证元数据；部分量化或转换仓库没有完整许可证声明，技术验证结果不代表模型授权结论。
