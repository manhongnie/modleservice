# 验证记录与交付边界

日期：2026-09-23。环境：Linux x86_64、Python 3.13.13、Intel Core Ultra 9 275HX；宿主机约 15 GiB RAM，实际可用内存随其他进程变化。精确依赖快照在 `requirements-tested.txt`；基本服务、OpenVINO、Torch CPU、Transformers、Sherpa ONNX 已安装到本工作区 `.venv`。

## 自动化测试证据

最终全量执行：`.venv/bin/python -m pytest -q --junitxml=var/pytest-results.xml` → **169 passed, 1 warning in 19.09s，无失败、无跳过**。包含并发关闭和 CLI 停止超时重试的回归。摘要见 [automated-test-results.json](automated-test-results.json)，原始输出在 `var/pytest-final.log`。该警告来自 OpenVINO 小图测试期间第三方库在 Python 3.13 多线程进程中使用 fork 的弃用提示；本服务的模型执行器明确使用 spawn。测试未因该警告失败。

其他已执行命令：

- `.venv/bin/python -m compileall -q model_service scripts`：通过。
- `.venv/bin/python -m model_service --help`：通过。
- `tests/test_executor.py` 中的真实 OpenVINO CPU 张量图：实际创建 IR，编译、执行、释放，`[[1,2,3]] × 2 → [[2,4,6]]`。此项不使用 Mock，但不证明任何真实 BGE/CLIP 权重已兼容。
- `tests/test_http_socket.py` 启动真正的 CLI/uvicorn 进程，经 TCP 连接注册模型、读取 SSE、断开客户端、检查资源归还；另注入控制进程崩溃，确认子进程停止及 SQLite 重启恢复。
- `.venv/bin/python scripts/smoke_http_models.py`：实际启动真实模型登记表，经 HTTP 调用 chat/asr/tts，再经管理 API 卸载，预算全部归零，应用正常关闭。记录在 `http-real-smoke.json`。Uvicorn 处理 SIGTERM 后重新发出该信号，进程返回 -15，日志确认应用关闭流程完成。

## 用户流程图到代码与验收的映射

来源是本次用户提供的 Mermaid，保留 START/REG、REQUEST、ADD、REMOVE 节点含义。下列“通过”是对应实现和测试实际执行的结果；模型结果正确性另列。

| 节点 / 分支 | 规则与实现 | 验收测试 | 状态 |
| --- | --- | --- | --- |
| START → REG | `Registry` 保存不可变版本配置，启用/验证与加载态独立；`ApplicationRuntime.start` 获取唯一控制锁 | `test_restart_restores_registry_but_not_loaded_state`、`test_second_controller_refused` | 已实现，通过 |
| A/B → ERROR | `HTTPBoundary` 限实际接收字节，业务/管理分别鉴权；`ConfigurationPolicy.validate_input` 检查能力、模型输入限制 | `test_api.py`、`test_service_api.py` | 已实现，通过 |
| C 否 / 是 | `Registry.resolve` 名称/版本/默认别名；禁用/不存在/能力不符明确错误 | `test_http_lifecycle_with_real_control_plane_and_mock_worker`、`test_tasks.py` | 已实现，通过 |
| D 超上限 → ERROR | 驻留+单次请求预算超总上限即拒绝，不进入队列 | `test_permanent_budget_limit_and_temporary_shortage` | 已实现，通过 |
| D 暂缺 → Q，Q 可用 → D | `Scheduler.acquire` FIFO 有界队列，原子检查模型并发、全局名额及预算 | `test_fifo_and_drain_timeout_do_not_release_active_lease` | 已实现，通过 |
| Q 满 / 超时 / 取消 → ERROR | 明确 queue_full、queue_timeout、cancelled，取消移除排队记录 | `test_queue_overflow_timeout_cancel_and_pause` | 已实现，通过 |
| E | 同锁建立模型使用权；驻留一次、临时逐请求；在途不允许卸载 | `test_resident_memory_reserved_once_and_temporary_released`、`test_unload_refuses_active_inference` | 已实现，通过 |
| F/G 已加载 / 单次加载 / 加载失败 | `Lifecycle.ensure_loaded` 单实例锁；`ProcessExecutor` 复用子进程 | `test_single_load_under_concurrent_requests_and_unload_reload`、`test_process_load_is_single_flight_and_reused`、`test_load_failure_confirms_process_exit` | 已实现，通过 |
| G 超时 | 尚在编译时保留驻留、临时预算及许可，等 ready/死亡后清理 | `test_load_timeout_does_not_release_lease_before_native_loading_finishes` | 已实现，通过 |
| I → OUT | worker 内 `prepare → infer → finish`；普通返回等待成功终帧 | `test_executor.py`、`test_tasks.py`，以及真实模型记录 | 已实现，边界见下文 |
| I / OUT → FAIL | 取消请求直到停止确认；流式部分输出失败显式报错，无重试 | `test_cancellation_waits_for_backend_stop`、`test_partial_stream_failure_is_reported_without_retry`、`test_api.py` | 已实现，通过 |
| OUT/FAIL → CLEAN | 发送期间保留使用权，客户端断连/ASGI 取消也等待清理 | `test_stream_pin_cancel_and_drain_timeout`、`test_real_socket_startup_stream_disconnect_and_metrics`、`test_api.py` | 已实现，通过 |
| CLEAN → END | 归还临时预算，保留驻留；空闲卸载，常驻重启按预算加载 | `test_resident_memory_reserved_once_and_temporary_released`、`test_idle_unload_and_resident_restart_policies` | 已实现，通过 |
| ADD1–ADD4 → ADD5 | 配置及路径校验后登记禁用；按预算真实运行 validation_input；失败保留验证错误 | `test_failed_validation_stays_disabled_and_is_persisted`、`test_management_rejects_model_metadata_references_outside_allowed_roots` | 已实现，通过 |
| ADD4 → ADD6 → REG | 验证通过才能启用，所有管理变化写 SQLite | `test_http_lifecycle_with_real_control_plane_and_mock_worker`、真实服务流程记录 | 已实现，通过 |
| RM1/RM2 引用检查 | `ModelManager.operate` 先检查别名/已登记业务依赖，移除/停用有引用则拒绝 | `test_remove_checks_dependencies_aliases_and_preserves_files` | 已实现，通过 |
| RM2/RM3 排队和在途 | 暂停准入，排队请求收到明确错误；等待使用权归零 | scheduler pause 用例、`test_stream_pin_cancel_and_drain_timeout` | 已实现，通过 |
| RM3 超时 → RM8 | 报 drain_timeout 并保留暂停，不自动强杀 | `test_stream_pin_cancel_and_drain_timeout` | 已实现，通过 |
| RM4 → RM5/RM6/RM7/RM9 | 进程退出确认后归还驻留；仅卸载可重载、停用禁调用、移除保留文件 | `test_single_load_under_concurrent_requests_and_unload_reload`、`test_remove_checks_dependencies_aliases_and_preserves_files` | 已实现，通过 |
| 工程支撑：故障恢复 | worker 故障当前请求失败、不重试，下次可新建；父崩溃停止子进程；配置恢复但不伪造已加载 | `test_worker_crash_is_reported_no_retry_and_next_request_recovers`、`test_controller_crash_stops_child_and_restart_recovers_registry` | 已实现，通过 |
| 工程支撑：关闭竞态 | 关闭等待业务请求和管理排队；超时保留 DB/控制锁，拒绝新的管理写入 | `test_lifecycle_races.py` | 已实现，通过 |
| HTTP 执行未知 | 外部生命周期不伪造；202/超时/断链/worker 死亡保留占用；持久化恢复及人工停止确认 | `test_executor.py`、`test_http_reconciliation.py`、`test_remote_uncertainty_survives_restart_and_requires_confirmation` | 已实现，通过 |

上述对照是可复核的工程映射，不是对无限并发组合的形式化证明。

## 本轮生产加固补充验收

- `tests/test_architecture_runtime.py`：不同名称的注入插件仍可执行、外部管理/隔离语义保持；加载与启用分离；重新验证拒绝旧队列；重复请求 ID 不覆盖取消信号；常驻加载失败退避；启动失败释放控制锁；并发关闭幂等；核心依赖与调度器封装检查。
- `tests/test_scheduler_races.py`：暂停/恢复代际、队列到期时资源刚释放、重复 ID、旧许可误释放、原子空闲回收、卸载超时状态以及不可中断加载的重复取消。
- `tests/test_api_production.py`：鉴权先于读 body、总接收超时、HTTP 请求总量限制、健康检查与管理指标、普通及 SSE 发送超时；业务和管理两套密钥隔离。
- `tests/test_cli_shutdown.py`：真实 CLI 在常驻模型 native 加载超过关闭时限时仍保留控制/执行进程，日志报告等待；加载结束后完成关闭，释放控制锁；只重试明确的关闭/卸载超时。
- `tests/test_register_models.py`：注册脚本遇到已存在版本时先比较配置，配置不同必须使用新版本，不能静默验证旧配置。
- `tests/test_runtime_maintenance.py`：native 加载中的关闭超时保留资源与控制锁，加载结束后可安全重试关闭。
- `tests/test_registry_production.py`：未知 schema 关闭连接、保留旧数据；SQLite WAL/FULL；模型名/别名冲突；未知模型及远端未确认执行的管理错误。
- `tests/test_backup_registry.py`：SQLite 在线备份包含 WAL 中已提交数据，目标权限 0600，拒绝覆盖已有备份或原数据库。
- `tests/test_streaming.py`：Qwen 解码、结束原因、生成线程取消和回压；慢消费者不会让终块挤掉内容后伪装成功。真实模型证据另列，不以这些受控单测代替。

配置与 schema 保持兼容；生产模式额外拒绝 Mock 与弱密钥。OpenVINO 编译参数限制为指定标量调优项，禁用磁盘缓存路径和嵌套设备属性。服务就绪由 `/ready` 反映，Prometheus 指标由需要管理鉴权的 `/admin/metrics` 提供；指标计数器为本次控制进程生命周期数据，持久审计事件有最近 10000 条上限。

## 真实模型证据

各份报告验证不同边界，不能互相替代。`models-validation.json` 和语音回环报告为第一阶段实测，新的完整控制流程与 HTTP 报告已在生产配置下重跑：

- [`models-validation.json`](models-validation.json)：真实权重进入隔离 worker 的加载、推理、RSS、取消停止；包含精确模型来源/revision。
- [`real-service-validation.json`](real-service-validation.json)：真实模型通过完整控制流程注册、验证、启用、默认别名选择、调度、复用、卸载、释放预算；对应 `scripts/validate_service_models.py`。本轮使用独立测试数据库，额外验证引用阻止移除、删除别名后成功移除且权重文件仍在；没有删除工作区业务登记表。
- [`speech-roundtrip-validation.json`](speech-roundtrip-validation.json)：真实 TTS 的输出经 16 kHz 重采样交给 Whisper，保留输入参考和实际识别结果。
- [`http-real-smoke.json`](http-real-smoke.json)：CLI、真实 TCP HTTP、两个独立鉴权域、默认别名与三个真实模型，以及管理卸载/应用关闭的组合烟测。本轮使用 `service.production.json`、强密钥和 `allow_mock=false`；Qwen 产生 65 个真实内容块，首块 0.187 秒、终止帧 4.341 秒，断连后约 0.070 秒确认资源回收，同一进程继续可用，load_count=1。这些时长来自已加载模型的本次观测。
- [`real-streaming-validation.json`](real-streaming-validation.json)：真实 Qwen 在执行器边界的 Unicode 安全增量拼接、终止原因、取消、复用和卸载。
- [`http-load-validation.json`](http-load-validation.json)：明确使用 Mock 的 200 请求、16 并发短时过载测试，33 成功、167 queue_full；峰值执行 4、排队 8 均等于配置上限；结束后请求/队列/临时额度归零，关闭无残留子进程。不代表真实模型吞吐或长期稳定性。

三个新增模型均通过真实权重执行和完整控制流程验证：

| 模型 | 一次 worker 烟测结果 | 观测 RSS | 取消确认 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 返回“你好”，prompt 17 / completion 4 tokens | 约 5453 MiB | 已确认停止，busy=0 |
| Whisper Tiny | 极短静音输入输出“你” | 约 577 MiB | 已确认停止，busy=0 |
| 中文 VITS | 实际生成 8 kHz、约 1.65 秒 WAV | 约 145 MiB | 已确认停止，busy=0 |

具体时长、SHA-256/revision 和运行记录见 JSON。它们是单次 smoke 观测，不是延迟/吞吐性能承诺。Qwen 的 PyTorch CPU 实现使用部分算子的通用后备路径，未安装专用加速内核，仍正常完成推理。

中文语音回环把“你好，世界。”识别为“你好事件”：证明路径可运行，同时说明不能将烟测当作识别准确率验收。VITS 质量和目标业务适用性未做主观听测或标准数据集评估；Qwen 未进行通用任务质量基准。

本工作区的 Qwen、Whisper、VITS 已登记到 `var/models.sqlite3`，均已验证并启用，分别设 `default:chat`、`default:asr`、`default:tts`；验证后已正常卸载，下一次调用按需加载。源代码分发不包含 `.venv`、`models` 权重和 `var` 运行数据；重建环境按 README 的下载、启动及注册命令执行。

## 未验证、未实现与范围差异

- **未验证真实检索模型**：BGE-M3 Dense、BGE Reranker、Chinese-CLIP 的真实权重/IR 未下载或运行；只验证插件的输入、张量后处理和配置边界。BM42 仅有真实服务的 HTTP 协议边界，本机没有 BM42 推理实现或已连接服务。
- **HTTP**：用本地受控真实 HTTP 服务器验证网络/生命周期语义，没有接入用户的已有模型服务（用户已说明没有）。只接受同步完成协议；不支持通用异步任务轮询/远端取消或远端启动/关闭。
- **流式输出**：Qwen 的真实增量文字、Unicode 拼接、HTTP SSE、断连取消和后续复用已测；ASR/TTS 仍缓冲输出，无逐块语音。慢读导致有界通道耗尽会明确报错；不无限积累或静默丢片段。
- **资源隔离**：预算按配置估算并观测实际 RSS，未设置操作系统硬内存上限、GPU 显存预算/计量或自动学习估算。不可立即取消的 native 算子需等实际结束，故清理时限可能超过请求配置时限。
- **模型文件**：配置路径、伴随软链接、已知权重索引和 tokenizer/processor 文件引用受白名单限制；模型文件必须由可信管理员部署。这不是执行恶意模型二进制文件的操作系统沙箱。版本配置不可变，但不监控管理员直接覆盖磁盘权重；升级应新建版本。
- **管理语义**：有别名/业务依赖的停用/移除要求管理员先迁移或删除登记引用；无法发现未登记的外部业务依赖。排空超时保留暂停，可重试或显式 enable 恢复。常驻模型仅卸载后可能被常驻策略再次加载。
- **调度与恢复**：单机单控制进程、全局 FIFO（可能队头阻塞），不抢占、不按内存压力自动淘汰；本地在途请求随控制进程崩溃失败，不做自动请求重放。远端未知任务需要管理员核实停止。
- **部署与质量**：提供生产配置、systemd 和 nginx TLS 模板、在线 SQLite 备份，未实际部署到系统服务或公网。systemd 模板语法已检查；本机没有 nginx，未运行 nginx -t。未做持续压测、GPU 验证、模型质量基准、多租户配额/模型 ACL、内置 TLS、分布式或插件代码热更新。

图中的普通返回或流式返回均已实现；Qwen 支持真正增量文本，语音模型当前使用缓冲结果的 SSE 传输。HTTP 的 RM4 是释放本地适配器，远端始终 external，符合用户“不伪造远端启停”的要求。
