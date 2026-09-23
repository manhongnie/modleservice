# 单机共享多模型推理服务

一个 FastAPI 控制进程、SQLite 登记表，模型按需启动独立执行进程并复用。已实现模型增加、验证、加载、仅卸载、启用、停用、移除；业务按能力和名称/别名访问，模型路径与后端只在管理配置中出现。

本仓库开始为空，没有可复用的现有服务。工程根据用户提供的 REQUEST / ADD / REMOVE 流程图实现；骨架、接口和分阶段说明见 [实施计划](docs/implementation-plan.md)，实际验证记录见 [验证报告](docs/verification.md)。

**当前工作区已准备好 `.venv`、三个真实模型权重和登记表。** Qwen3.5-0.8B、Whisper Tiny、中文 VITS 已实测、启用并设默认别名，验证后已卸载。直接设置两组不同的密钥并用 `examples/service.production.json` 启动即可按需调用，无需重新下载或登记。

```bash
export BUSINESS_API_KEY="$(openssl rand -hex 24)"
export ADMIN_API_KEY="$(openssl rand -hex 24)"
.venv/bin/python -m model_service --config examples/service.production.json
```

生产加固后全量 **169 项测试通过**，证据见 [验证报告](docs/verification.md)；Qwen/ASR/TTS 真实推理和取消均实测。ASR 有识别错误，BGE/CLIP/BM42 未完成真实模型验证，详细边界请见验证报告。启动、添加/删除模型和备份的完整步骤见 [操作手册](docs/operations.md)，模块依赖见 [架构说明](docs/architecture.md)。

## 快速启动（明确标记的 Mock）

Linux、Python 3.11+。所有命令在仓库根目录执行；配置中的相对路径以工作目录为基准。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
export BUSINESS_API_KEY="$(openssl rand -hex 24)"
export ADMIN_API_KEY="$(openssl rand -hex 24)"
.venv/bin/python -m model_service --config examples/service.json --host 127.0.0.1 --port 8000
```

另开终端，设置同一组环境变量（不要重新生成密钥），登记并执行 Mock 验证，再建立默认别名：

```bash
.venv/bin/python scripts/register_models.py examples/models.mock.json --defaults
curl -s http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"texts":["多个业务共享一个模型实例"]}}'
```

响应中 `mock:true` 及输出提示表示合成测试结果，不是模型真实推理。API 密钥必须非空且业务/管理集合互不重叠，无默认弱密钥。`/health` 和 `/ready` 公开，`/admin/status`、`/admin/models`、`/admin/metrics` 需管理密钥，公开 Swagger/OpenAPI 已关闭。

当前开发环境使用 `uv` 建立了 `.venv`，其中可以没有 pip；继续安装可使用 `uv pip install --python .venv/bin/python -e '.[test]'`。

`requirements-tested.txt` 保存本次 Linux / Python 3.13 的完整可选依赖版本快照；使用它重建时需要 CPU Torch 官方索引：`uv pip install --python .venv/bin/python -r requirements-tested.txt --extra-index-url https://download.pytorch.org/whl/cpu`。

## 新增的三个真实模型

提供 Qwen3.5-0.8B 文本对话、Whisper Tiny 多语种 ASR、中文 VITS Aishell3 TTS 的插件、固定来源配置及下载脚本。模型验证结果、精确版本和尚未验证项见 [模型接入说明](docs/model-plugins.md) 和 [实测数据](docs/models-validation.json)。

```bash
# CPU torch 单独安装，避免默认拉取不需要的 CUDA 包。
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[generative,speech,openvino,text]'
.venv/bin/python scripts/download_models.py qwen asr tts
# 使用较大的明确预算；先正常停止旧控制进程。
.venv/bin/python -m model_service --config examples/service.production.json
# 另一终端，密钥相同：先验证各模型，登记默认别名，然后释放验证用驻留进程。
.venv/bin/python scripts/register_models.py examples/models.real.json --defaults --unload-after-validation
```

`service.real.json` 使用 8192 MiB 估算预算，部署时按实际可用内存调整；Qwen 的配置预算超过基础 4096 MiB 示例，所以不能用基础示例直接登记它。管理注册会实际加载并运行 `validation_input`，失败就保持禁用、记录原因。所有运行时加载都使用本地文件，不自动下载，也不执行模型自定义远程代码。

```bash
curl -s http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","input":{"messages":[{"role":"user","content":"请用一句话解释向量检索"}],"max_new_tokens":64}}'

curl -s http://127.0.0.1:8000/v1/tts \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"text":"你好，这里是共享模型服务。","speaker_id":0,"speed":1.0}}' > /tmp/tts-response.json
.venv/bin/python -c 'import json,base64; r=json.load(open("/tmp/tts-response.json")); open("/tmp/speech.wav","wb").write(base64.b64decode(r["output"]["audio_base64"]))'
```

ASR 接受 16 kHz 单声道 PCM16 WAV（30 秒上限），TTS 返回 WAV 的 base64。可用以下 Python 调用已有录音：

```python
import base64, os, httpx
audio = base64.b64encode(open("input-16khz-mono.wav", "rb").read()).decode()
r = httpx.post("http://127.0.0.1:8000/v1/asr",
    headers={"Authorization": "Bearer " + os.environ["BUSINESS_API_KEY"]},
    json={"input": {"audio_base64": audio}}, timeout=180)
r.raise_for_status()
print(r.json()["output"]["text"])
```

## 业务接口

统一请求：`POST /v1/{capability}`，body 为 `{"model":"名称、名称@版本或别名，可省略", "input":{...}, "stream":false}`。省略模型时解析 `default:{capability}`；一个名称有多个版本时需指定版本或别名。输出包含 `request_id`、`model`（唯一版本）、`output`、`mock`、`done`。

| 能力 | input | output 主要字段 |
| --- | --- | --- |
| embeddings | `{"texts":["文本"]}` | `embeddings` 稠密向量 |
| rerank | `{"query":"问题","documents":["文档"]}` | `results` 下的 index、score |
| image_embeddings | `{"images":["图片base64"]}` | `embeddings` |
| sparse_embeddings | `{"texts":["text"]}` | BM42 HTTP 服务约定的稀疏向量 |
| chat | `messages`、可选 `max_new_tokens` | text、usage |
| asr | `audio_base64` | text |
| tts | text、可选 speaker_id/speed | audio_base64、sample_rate、format |

`stream:true` 使用 SSE：`chunk`、最终 `result`（`done:true`）、异常 `error`。Qwen 已提供真实增量文字块（`output.delta`、`streaming_mode:incremental`），末个内容块包含完整 `text`、`usage` 和 `finish_reason`，之后发送终止帧。ASR/TTS 仍完整计算后返回一块（`streaming_mode:buffered`），不支持逐块语音输出。慢读者超过有界缓冲会明确失败并确认停止，无静默丢块。

```bash
curl -N http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"你好"}],"max_new_tokens":16},"stream":true}'
```

无静默重试。SSE 已输出后发生错误会发送 `error`，不会重新执行。普通返回和流式返回均在传输期间持有执行许可；断连设置取消信号，等待底层停止。某些编译/推理实现无法立即中断，超时表示提出取消，资源在停止确认之前仍被占用，响应清理可能超过配置时限。

## 管理接口

所有管理调用使用 `Authorization: Bearer $ADMIN_API_KEY`。配置及变更即时持久化，无需重启来增加兼容模型配置。新增代码插件需要重启。

| 操作 | 方法及路径 | 输入/规则 |
| --- | --- | --- |
| 增加并验证 | `POST /admin/models` | `{"config":ModelConfig,"enable":true}` |
| 重新验证 | `POST /admin/models/{name@version}/validate?enable=true` | 验证失败保留登记，禁用 |
| 加载/启用 | `POST /admin/models/{id}/load` 或 `/enable` | 已验证模型可预热；加载不改变启用状态 |
| 仅卸载 | `POST /admin/models/{id}/unload` | 保留登记/启用状态，下次重新加载 |
| 停用 | `POST /admin/models/{id}/disable` | 保留登记，禁止业务调用 |
| 移除 | `DELETE /admin/models/{id}` 或 `POST .../remove` | 默认不删除磁盘文件 |
| 别名 | `PUT /admin/aliases/{alias}` | `{"model_id":"name@version"}` |
| 删除别名 | `DELETE /admin/aliases/{alias}` | 移除前需迁移/删除关联别名 |
| 登记依赖 | `PUT /admin/dependencies/{business}` | `{"model_id":"name@version"}` |
| 删除依赖 | `DELETE /admin/dependencies/{business}` | 仅处理已登记依赖 |
| 就绪与指标 | `GET /ready`、`GET /admin/metrics` | 就绪返回 200/503；指标需管理密钥 |
| 模型列表 | `GET /admin/models` | 配置、启用与加载态 |
| 状态 | `GET /admin/status` | 启用/加载态、队列、预算、PID/RSS、耗时、近期事件 |
| 远端停止确认 | `POST /admin/requests/{request_id}/reconcile` | `{"confirmed_stopped":true}`，仅在管理员已核实远端停止后调用 |

```bash
curl -s http://127.0.0.1:8000/admin/status -H "Authorization: Bearer $ADMIN_API_KEY"
curl -s -X PUT http://127.0.0.1:8000/admin/aliases/search-embedding \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"mock-embeddings@1"}'
curl -s -X POST 'http://127.0.0.1:8000/admin/models/mock-embeddings@1/unload?timeout_s=10' \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

移除/停用前检查别名和已登记业务依赖，有引用则返回 `model_referenced`，要求先迁移或删除关联。随后暂停新请求、拒绝排队请求，等待在途请求结束；`drain_timeout` 保留暂停状态，不自动强杀。可重试管理操作；若决定恢复业务，调用 `enable`。卸载已启用的 `resident` 模型后，常驻策略会在下一次维护时重新加载；长期释放请停用或创建采用按需策略的新版本。

## 后端与插件

`contracts.py` 定义稳定契约，`service.py` 是兼容入口。`coordinator.py` 协调业务请求，`management.py` 负责模型管理，`runtime.py` 负责组装与恢复；`configuration.py` 管策略，`observability.py` 管审计与指标。SQLite 只由登记表访问；调度器通过公开接口独占预算账本；重型模型 SDK 和推理只在子进程中运行。`tasks/__init__.py` 和 `backends/__init__.py` 显式登记实现，未知名称失败。

- 任务插件负责格式检查、预处理、后处理；后端负责加载、推理、释放。
- OpenVINO 是首个本地通用张量后端，通过 `Core.compile_model` 和 `InferRequest` 执行，无 OVMS 依赖。
- Transformers 后端接入 Qwen/Whisper，Sherpa ONNX 后端接入中文 VITS；这两个是已有库适配，不要求导出所有模型为同一格式。
- HTTP 适配器向 allowlist 内 `base_url + infer_path` 发送同步 JSON。任务 `http_json` 保留请求/结果对象；现有服务协议不同则新增兼容任务/后端适配器。密钥从 `auth_env` 读取，不写入模型配置。
- HTTP 模型状态明确是 `external`，加载/卸载只建立/关闭本地适配器。取消等待同步远端响应；网络故障、远端状态未知时持久化隔离占用，需人工确认停止才能释放。不能通过 HTTP 202 接受异步任务后假定推理结束，异步任务服务需实现专门的完成/取消协议。

原需求的 BGE-M3 Dense、BGE Reranker、Chinese-CLIP、BM42 配置在 [models.retrieval.json](examples/models.retrieval.json)。BM42 仅提供真实 HTTP 服务边界，未用词频/hash 模拟 BM42；缺少权重、导出或服务时明确失败。插件规范和来源参见 [模型接入说明](docs/model-plugins.md)。

## 资源、运行及恢复边界

- 原子预留：驻留预算按实例一次，临时预算按每个请求；并同时检查模型并发和全局名额。模型版本/执行配置不可变，同一实例有单加载锁。
- FIFO 有界等待队列，支持队列超时/取消。超出总预算直接拒绝；暂时不足排队。空闲按 `idle_seconds` 回收；首版不做抢占或按内存压力淘汰，必要时主动卸载空闲模型。
- RSS 是实际进程观测，MiB 预算是准入估算，不是硬内存隔离；未实现 GPU 显存统计、cgroup 限额或自适应预算。请为控制进程和操作系统预留余量。
- 请求体按实际接收字节限制；模型另有输入限制；输出按请求累计限制。业务输入不能指定文件/URL。模型目录和辅助文件受 allowlist 与符号链接检查，管理员应确保这些目录不能由不可信业务写入。
- 单数据库独占文件锁拒绝重复控制进程；不要运行 `uvicorn --workers N`。Linux 子进程在父进程异常退出时收到 parent-death signal，避免重启后重复驻留。
- 重启恢复模型配置、验证、启用、别名、依赖和未完成管理暂停；不会把旧的本地加载态恢复成已加载。远端在途意图先写 SQLite，崩溃后恢复为隔离占用。
- JSON 请求/管理日志不记录输入、输出或密钥；SQLite 保留最近 10000 条事件。`/admin/status` 同时给出加载/推理耗时、队列长度、预算、进程 RSS。
- 提供 systemd/nginx TLS 部署模板但未实际部署；无多租户配额、按模型 ACL、内置 TLS 终结、批调度、硬件设备池或分布式组件。多业务凭业务密钥共享已启用能力；需要额外隔离可在 API 鉴权边界扩展。

## 测试

```bash
.venv/bin/python -m pytest -q
```

已准备真实权重和登记表后，还可运行 `.venv/bin/python scripts/smoke_http_models.py`，它会按生产配置临时启动 CLI 服务，经真实 HTTP 调用三个模型、验证 Qwen SSE/断连/复用并卸载，最后关闭服务。明确使用 Mock 的有界过载测试为 `scripts/validate_http_load.py`；真实执行器增量测试为 `scripts/validate_real_streaming.py`。完整控制流程可用 `scripts/validate_service_models.py` 重跑，纯执行器实测使用 `scripts/validate_real_models.py`。

覆盖重复加载、队列满/超时、资源不足、流式取消与发送期间持有许可、移除/依赖/排空超时、进程故障、重启恢复、外部 HTTP 生命周期与未知执行、目录白名单、分离鉴权。实际执行结果与未验证范围见 [验证报告](docs/verification.md)；有测试代码并不等于真实模型质量已经验收。
