> 当前多模态部署使用 `examples/service.multimodal.json`。下文旧模型名称示例属于先前版本，新增模型与调用清单见 [README](../README.md)。

# 启动、调用和管理模型

这份手册适用于单机单控制进程部署。当前工作区已有 Qwen3.5-0.8B、Whisper Tiny ASR、中文 VITS TTS 的权重和持久化登记，正常启动即可按需加载。生产配置启用更严格的密钥检查并拒绝 Mock；它不等于已经完成真实机器的容量、安全或质量验收。实测范围见 [验证报告](verification.md)。

## 1. 当前目录直接启动

在仓库根目录执行；所有配置相对路径都以当前工作目录为基准。密钥只生成一次，并通过安全渠道给相应调用方；另一终端必须使用同一组值。

```bash
export BUSINESS_API_KEY="$(openssl rand -hex 24)"
export ADMIN_API_KEY="$(openssl rand -hex 24)"
.venv/bin/python -m model_service --config examples/service.production.json --check-config
.venv/bin/python -m model_service --config examples/service.production.json --host 127.0.0.1 --port 8000
```

`--check-config` 只检查配置结构和密钥约束，不启动服务，不验证真实模型权重/插件依赖，也不输出密钥。生产模式要求两组密钥互不相同、每个至少 32 字符，且 `allow_mock=false`。示例不保存静态密钥。重启时保持密钥不变，否则原调用方会收到 401。

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/ready
curl --fail-with-body http://127.0.0.1:8000/admin/models \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

`/health` 说明 HTTP 进程还活着；`/ready` 返回 200 或 503，反映控制服务是否可接受工作，不承诺每个模型都已加载或每次请求都有预算。`/admin/status` 显示完整模型、别名、登记依赖、队列、占用和事件；`/admin/metrics` 提供需要管理密钥的 Prometheus 文本指标。

新机器没有 Python 环境时，先按 [README](../README.md) 安装依赖并下载模型，随后登记：

```bash
.venv/bin/python scripts/download_models.py qwen asr tts
.venv/bin/python scripts/register_models.py examples/legacy/models.real.json --defaults --unload-after-validation
```

第二条命令要求服务已启动。它通过管理 API 实际验证模型，再设置默认别名；已有相同版本且配置一致时会重新验证；配置不同则明确拒绝并要求新版本，不能把文件存在当成可用。当前 8192 MiB 是准入估算预算，不是操作系统硬限制；Qwen 实测驻留约 5.3 GiB，机器还需给控制进程、临时输入和操作系统留余量。

## 2. 业务如何调用

业务只需要地址、业务密钥、能力、模型名称或别名。省略 `model` 时使用 `default:{capability}`，无需知道本地路径和后端。

对话：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-0.8b","input":{"messages":[{"role":"user","content":"用一句话介绍你自己"}],"max_new_tokens":32}}'
```

文字转语音，保存 WAV：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/tts \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"text":"你好，这是共享模型服务。","speaker_id":0,"speed":1.0}}' > /tmp/tts-result.json
.venv/bin/python - <<'PY'
import base64, json
from pathlib import Path
result = json.loads(Path('/tmp/tts-result.json').read_text())
Path('/tmp/speech.wav').write_bytes(base64.b64decode(result['output']['audio_base64']))
print('/tmp/speech.wav')
PY
```

语音识别接受 **16 kHz、单声道、PCM16 WAV，最长 30 秒**。使用符合要求的录音；TTS 输出采样率不一定是 16 kHz，不能不转换就当作 ASR 输入。

```python
import base64
import os
from pathlib import Path
import httpx

response = httpx.post(
    'http://127.0.0.1:8000/v1/asr',
    headers={'Authorization': 'Bearer ' + os.environ['BUSINESS_API_KEY']},
    json={'input': {'audio_base64': base64.b64encode(
        Path('input-16khz-mono.wav').read_bytes()).decode()}},
    timeout=600,
)
response.raise_for_status()
print(response.json()['output']['text'])
```

流式请求在 body 中增加 `"stream":true`，用 `curl -N` 接收 SSE。Qwen 提供真实增量文字（`output.delta`），最后内容块包含完整 `text`、`usage` 和 `finish_reason`，再发送 `done:true` 终止帧。ASR/TTS 仍完整计算后返回一块，不支持逐块语音合成。传输期间仍占执行许可；客户端断开或写入超时后，服务发出取消并等待底层停止，再归还资源。收到过部分结果后不会自动重试。

## 3. 添加模型：先备文件，再登记验证，再给业务别名

把兼容模型的权重放在 `model_roots` 允许目录中。当前为 `models/`；业务调用方不得有该目录的写权限。配置中的 `path` 指向服务器上的本地文件，不能是客户端路径或任意下载 URL。已有任务插件和后端能处理的模型，只需增加配置；新的处理协议/预后处理方式要新增受信任插件代码，并重启控制服务。

配置示例位于 `examples/legacy/models.real.json`（对话/ASR/TTS）、`examples/models.retrieval.json`（BGE-M3 Dense、Reranker、Chinese-CLIP、BM42 边界）。检索模型示例不代表真实模型已完成验收。

首次登记已有示例文件中的 Qwen 配置，可以生成单模型请求并提交：

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('examples/legacy/models.real.json').read_text())[0]
Path('/tmp/add-model.json').write_text(json.dumps({'config': config, 'enable': True}))
PY
curl --fail-with-body http://127.0.0.1:8000/admin/models \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/add-model.json
```

当前工作区已经登记该版本，重复新增会返回 `version_exists`（409）。要添加另一个模型或版本，先修改 JSON 中的 `name`、唯一 `version`、真实 `path`、`task`、`backend`、资源预算及 `validation_input`，然后提交。`name@version` 对应不可变配置，更新权重必须使用新版本并保留旧版本文件，不能直接覆盖正在使用的权重。

例如兼容 Qwen 的新权重可继续使用 `task:qwen_chat`、`backend:transformers`；Whisper 使用 `whisper_asr`/`transformers`；当前 VITS 使用 `vits_tts`/`sherpa_tts`。这并不保证任意同类模型都兼容，实际登记验证失败时会保持禁用并记录原因。

新增调用包含加载与测试，可能较慢；客户端超时应覆盖排队、加载与验证时间。`enable:false` 表示验证通过也先保持禁用。验证失败修复文件后可以重验；需要改配置则使用新版本。

```bash
# 示例使用已有 Qwen 的真实版本号；替换成你新增的 name@version。
MODEL_ID='qwen3.5-0.8b@2fc06364715b'
curl --fail-with-body -X POST "http://127.0.0.1:8000/admin/models/$MODEL_ID/validate?enable=true" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
# 验证通过后绑定业务默认别名（也可使用自定义 alias）。
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/aliases/default:chat \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"$MODEL_ID\"}"
# 登记业务依赖，之后停用/移除会检查它。
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/dependencies/customer-service \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model_id\":\"$MODEL_ID\"}"
```

依赖检查只知道已登记的引用；它无法发现写死模型名称的外部业务。升级前应通知这些调用方改用稳定别名。

## 4. 加载、卸载、停用、删除有什么区别

| 目的 | 管理请求 | 登记/文件 | 后续业务请求 |
| --- | --- | --- | --- |
| 提前加载 | `POST /admin/models/{id}/load` | 保留 | 已验证模型可预热；禁用状态不改变 |
| 仅释放驻留实例 | `POST /admin/models/{id}/unload` | 保留 | 若已启用，下次重新加载 |
| 暂停某模型 | `POST /admin/models/{id}/disable` | 保留 | 拒绝调用 |
| 恢复已验证模型 | `POST /admin/models/{id}/enable` | 保留 | 可按需加载 |
| 从服务移除 | `DELETE /admin/models/{id}` | 删除登记，保留权重文件 | 模型不存在 |

`load` 要求模型已通过验证，但允许先加载仍处于禁用状态的模型；加载不等于启用，业务仍要等待 `enable`。仅卸载可以保留默认别名/依赖；停用或移除必须先迁移或删除它们。常驻策略 `resident` 会重新加载已启用的实例，要长期释放则先停用，或为新版本配置 `on_demand`。

```bash
MODEL_ID='qwen3.5-0.8b@2fc06364715b'
# 仅释放内存；默认 chat 仍可用，下次请求重新加载。
curl --fail-with-body -X POST "http://127.0.0.1:8000/admin/models/$MODEL_ID/unload?timeout_s=60" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

**删除示例会让默认对话能力不可用，直到你绑定其他已启用模型。** 如果有新版本，优先用上节 `PUT` 把别名和依赖改到新版本，业务可以继续调用同一别名。确实不再使用对话能力时，才删除引用后移除模型：

```bash
curl --fail-with-body -X DELETE http://127.0.0.1:8000/admin/aliases/default:chat \
  -H "Authorization: Bearer $ADMIN_API_KEY"
# 仅当你登记过这个依赖才需要删除；还应处理 status 中列出的其他引用。
curl --fail-with-body -X DELETE http://127.0.0.1:8000/admin/dependencies/customer-service \
  -H "Authorization: Bearer $ADMIN_API_KEY"
curl --fail-with-body -X DELETE "http://127.0.0.1:8000/admin/models/$MODEL_ID?timeout_s=60" \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

移除先检查引用、暂停新请求、拒绝排队请求、等待在途执行和传输结束，再卸载并删除登记。返回 `drain_timeout` 时模型仍保持暂停，任务没有被强杀；等待后重试操作，或者调用 `enable` 恢复业务。移除不会删除磁盘文件；需要回收权重时，在确认无其他登记/版本使用该目录后由管理员另行处理。

## 5. 新版本切换与回滚

先以新的 `name@version` 登记并验证，用显式版本调用验证输出，然后把 `default:chat` 或业务自定义别名切换到新版本，并更新已登记依赖；旧的在途请求仍持有旧实例的使用权。旧版本排空后可以卸载并保留登记，观察稳定再移除。

回滚是把别名/依赖切回仍保留且已启用的旧版本。配置不可原位修改，数据库备份也不是业务版本回滚工具。两个 Qwen 同时驻留可能超过默认 8192 MiB 预算；若预算不足，需安排业务维护窗口并先卸载旧模型，或在物理内存确实足够后调整预算重启，不能只虚增预算。

## 6. 服务部署和停止

提供 [systemd 模板](../deploy/model-service.service) 和 [nginx 模板](../deploy/nginx.conf.example)，本次没有实际安装系统服务或配置公网 TLS。部署到 `/opt/model-service` 时，创建专用无登录用户 `model-service`，代码和模型由管理员管理、服务用户只读，`var/` 允许服务用户写入；在目标目录重新建立虚拟环境，不直接搬运旧 `.venv`。

把两组密钥写入权限为 `0600`、root 所有的 `/etc/model-service.env`，格式如下，值需使用实际生成的随机密钥：

```text
BUSINESS_API_KEY=替换为至少32字符的随机业务密钥
ADMIN_API_KEY=替换为另一组至少32字符的随机管理密钥
```

检查模板中工作目录、环境文件、解释器、模型和数据库路径，创建所需目录，按机器实际内存调预算。准备好后由部署管理员安装模板并执行 `systemd-analyze verify`、`systemctl daemon-reload`、`systemctl enable --now model-service`，再访问 `/ready` 和模型接口验收。若使用其他目录，必须同步修改 `WorkingDirectory`、`ExecStart`、`ExecStartPre`、`ReadWritePaths` 和缓存目录。

模板只发送 SIGTERM 给控制进程，控制进程等待请求与执行器清理。CLI 遇到明确的关闭/卸载超时会记录 `shutdown_waiting` 并每秒重试，保留控制锁和工作进程，直到实际停止。`TimeoutStopSec=15min` 后报告停止超时，`SendSIGKILL=no` 和 `Restart=no` 防止默默强杀并重启慢 native 推理。需要检查日志和残留任务后手工处理，不要在旧进程仍占用时再启动一套服务。参数语义已根据本机 `systemd.service(5)`、`systemd.kill(5)` 核对。

nginx 模板需要替换域名、证书和密钥文件；管理接口默认仅允许来自本机的连接，确有远端管理需求时添加具体可信网段。若 nginx 前还有代理，应先正确配置受信任真实来源地址，再使用 IP 访问控制。模板关闭请求缓冲以让应用先鉴权、关闭响应缓冲以传递 SSE，同时禁用自动上游重试。[nginx 代理模块说明](https://nginx.org/en/docs/http/ngx_http_proxy_module.html)、[访问控制说明](https://nginx.org/en/docs/http/ngx_http_access_module.html)。

CLI 固定一个 worker、默认监听 localhost、HTTP 入站数量受限、keepalive 为 5 秒。不要使用多 worker、自动 reload、外部定时强制重启或未检查在途任务的滚动替换。业务连接数量、模型执行名额和等待队列是三组独立限制：`max_http_requests` 限定进入应用的并发请求；`max_executions` 限执行；`queue_size` 限等待者。接收总时限 `body_timeout_s` 和每次输出写入时限 `write_timeout_s` 防止慢连接长期占据入口或执行许可。

## 7. 备份与故障处理

SQLite 使用 WAL，运行中**不要只复制 `models.sqlite3` 主文件**。用在线备份工具生成一致快照，目标文件必须不存在；该工具会包括已提交 WAL 数据并执行 `quick_check`，不会修改源库。

```bash
mkdir -p var/backups
.venv/bin/python scripts/backup_registry.py var/models.sqlite3 \
  "var/backups/models-$(date +%Y%m%d-%H%M%S).sqlite3"
```

备份还应包含部署配置和对应版本的权重清单；密钥单独放入受控秘密存储。恢复时停止服务并确认所有本地执行进程结束，把当前数据库及同名 `-wal`/`-shm` 文件一起保留为故障现场，再将快照放到配置数据库位置并恢复权限。不要把备份覆盖进仍运行的 SQLite，或把旧 WAL 搭配新快照。远端执行存在未知状态时，要先核实这些执行，不能通过回滚数据库来绕过隔离占用。

常见状态与处理：

| 状态或错误 | 含义与处理 |
| --- | --- |
| `http_capacity` / 503 | HTTP 并发达到上限；降低调用并发，检查慢连接/在途操作 |
| `body_timeout` / 408 | 请求体未在总时限内接收完；检查调用方上传 |
| `queue_full` / `queue_timeout` | 等待者太多或等待过久；按 status 查队列和模型并发 |
| `resource_limit` / `resource_exhausted` | 配置或请求超预算；先确认模型实际驻留/临时占用，必要时卸载空闲模型 |
| `load_failed` / `load_timeout` | 查管理 status 和日志中的模型状态、加载时间、目录/依赖；底层仍未停止时不会归还预算 |
| `worker_crashed` | 执行进程异常；当前请求明确失败，检查 RSS/系统 OOM/模型错误后决定下一次调用 |
| `model_referenced` | 还有别名或登记依赖；先迁移或删除引用 |
| `drain_timeout` | 仍有执行/输出；模型保持暂停，等待后重试；取消操作则 enable 恢复 |
| `execution_unknown` | 外部服务是否停止未知，资源保留并持久化隔离；先向外部服务核实 |

外部 HTTP 服务没有启停能力时，状态标注为外部管理。关闭本地 HTTP 适配器不能证明远端模型已卸载。只有管理员从远端服务确认该 request_id 对应执行已经结束，才调用：

```bash
REQUEST_ID='替换为已经核实停止的请求ID'
curl --fail-with-body -X POST "http://127.0.0.1:8000/admin/requests/$REQUEST_ID/reconcile" \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"confirmed_stopped":true}'
```

日志不记录业务输入/输出和密钥；以响应头 `X-Request-ID` 关联请求日志、管理 status 事件和异常。重启会恢复配置、启用状态、别名、依赖及管理暂停；本地模型重新按需加载，远端未知执行继续占用隔离预算。没有提供自动判定远端停止、GPU 显存硬隔离、多租户模型级 ACL 或高可用能力，这些不能靠增加 Web worker 解决。
