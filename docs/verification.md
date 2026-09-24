# 当前验证记录与交付边界

本轮为2026-09-23模型迁移和多模态扩展。旧三个模型的记录保留在 [历史验证](verification-legacy.md)，不代表当前登记或当前模型质量。

2026-09-24补充 [SenseVoice内存专项](sensevoice-memory.md)：真实同进程1,140次识别、10次取消、60秒空闲卸载、5轮重新加载/卸载，以及独立的3轮对象关闭对照。另重跑61项相关回归通过。这是约5分钟的专项观测，尚未完成24/72小时持续压力验证。

## 自动化验证

### 网页测试入口增量验证（2026-09-24）

- 原有回归：`.venv/bin/python -m pytest -q --ignore=tests/test_playground.py`，326 passed，24.22 秒。
- 新增入口及目录测试：`.venv/bin/python -m pytest -q tests/test_playground.py`，21 passed，0.57 秒；覆盖静态资源访问边界、业务/管理鉴权、目录字段过滤、可用状态、只读行为和 Mock 调用闭环。两组共 347 项通过，无失败或跳过。
- Chromium 浏览器验证：真实本地 HTTP Mock 服务下检查连接失败/成功、模型选择、普通及 SSE 文本、向量/排序、WAV 上传与播放、JSON 输入、取消和清除密钥；其余能力使用明确标记的受控响应检查图片、音频、视频展示，表单提交经真实任务校验器验证。12 类能力表单均覆盖，桌面和手机布局无横向溢出，未发现 JavaScript 或 CSP 错误。
- 独立构建 wheel，确认 HTML、CSS、JavaScript 均打包；本次没有重新执行下方全部真实模型质量验证。

此前多模态全量结果：**326 passed，无失败、无跳过，23.19秒**。命令及精确时间见 [automated-test-results.json](automated-test-results.json)，原始JUnit见 `var/pytest-multimodal-results.xml`。当时依赖检查 `uv pip check --python .venv/bin/python` 和 `compileall` 通过；实际环境记录在 `requirements-tested.txt`。

这些测试覆盖控制逻辑、受控模型替身、真实TCP、OpenVINO小张量图以及真实ffmpeg编码。不能由测试数量推断模型质量。新增覆盖包括：

- 主存/显存双预算原子准入、驻留仅计一次、隔离额度重启恢复：`tests/test_gpu_budget.py`。
- GenAI输入、图片token预算、原生线程取消、流式结束：`tests/test_ov_genai.py`。
- 检索预后处理和BM42、CLIP边界：`tests/test_retrieval_migration.py`。
- sherpa音频、参数和输出边界：`tests/test_sherpa_models.py`。
- 图片/视频限制、编码、停止确认及编码器父进程死亡：`tests/test_media_generation.py`。
- 普通/缓冲/增量后处理期间取消，完成前不确认停止、取消后不发成功：`tests/test_worker_postprocessing.py`。
- 有界分块下载、续传、完整性与HTTP Range响应检查：`tests/test_artifact_download.py`。
- 所有示例任务构造和校验不在控制进程导入ML运行时：`tests/test_tasks.py`；核心不导入供应商SDK：`tests/test_architecture_runtime.py`。

## 真实模型与服务

当前15个真实模型已在工作区SQLite登记、验证、启用，验证后卸载。精确版本与状态见 [部署报告](deployment-validation.json)，业务/管理真实TCP、独立鉴权、进程复用、卸载和重启恢复见 [HTTP报告](http-multimodal-validation.json)。

| 类别 | 已实测结果 | 证据与边界 |
|---|---|---|
| 7个检索模型 | BGE-M3、BGE排序、Chinese-CLIP、BM42、Qwen3 Embedding/排序、OpenAI CLIP真实推理 | [报告](retrieval-migration.json)；相关性样例、CLIP导出余弦一致性、BM42官方样例，未做完整业务检索评测 |
| 4个sherpa模型 | SenseVoice转写、Matcha发声、ERes2Net512维声纹、ZipVoice参考语音克隆 | [报告](sherpa-validation.json)；2段短语音回译CER为0，不代表通用准确率；克隆使用合成参考音，无真人相似度验收 |
| Qwen3.5-4B | OpenVINO GenAI CPU中文回答、简单算术、实际增量文字、取消和再次使用 | [报告](genai-validation.json)；未做通用推理质量基准 |
| SD-Turbo | OpenVINO GenAI CPU实际512×512茶壶图片、重复种子一致、取消等待真实停止 | [报告](media-validation.json)、[样例](../var/media/sd-turbo.png)；运行后RSS约8.67GiB，配置9GiB驻留加1GiB临时 |
| Qwen2.5-VL-7B | OpenVINO GenAI CPU红/蓝图片识别、实际增量文字、取消和再次使用 | [报告](genai-validation.json)；448px图片、实际1009输入+128输出时峰值9875MiB，未做OCR/文档理解质量基准 |
| AnimateDiff-Lightning | epiCRealism底座、CUDA、512×512/16帧真实短视频；复用/取消/登记/HTTP通过 | [视频样例](../var/media/animatediff-lightning.mp4)与[报告](media-validation.json)；运动较小，原始SD1.5失败组合未部署 |

7项检索最大配置容量检查见 [retrieval-capacity.json](retrieval-capacity.json)，当前版本的驻留与总预算均覆盖观察值，旧不足记录保留。4B最大输入实测峰值6164MiB、7B9875MiB；当前全局预算12288MiB主存、10240MiB显存、执行并发1。检索早期质量报告的版本号与后续预算版本可能不同，权重来源和算法不变，当前登记/HTTP/容量报告明确列出当前版本。

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

## 部署与质量限制

- 单机单控制进程；一个数据库不能启动多个Web worker。未安装systemd服务或公网反向代理。
- RAM与显存预算是准入估算，不是操作系统硬隔离；根据冷启动及运行后RSS/显存保留量设置预算。该机器OpenVINO仅检测到CPU，CUDA视频使用RTX 5070 Ti Laptop。
- 原生编译、预填充、音频处理或编码不能立即停止时，会等待实际结束再清理，取消确认可能超过业务超时。
- 全局FIFO可能队头阻塞，不抢占、不按系统压力自动驱逐。模型空闲按策略卸载；当前配置全局并发1。
- 本地进程崩溃会使在途请求失败，不自动重放。HTTP未知远端执行保留隔离额度，需要核实停止；没有接入用户已有HTTP模型服务。
- 模型文件来自受信任管理员；路径/软链接/文件引用受目录限制，不提供恶意模型二进制的系统沙箱。新插件代码需重启，兼容模型配置可运行中添加。
- 图片512×512；视频首版短片，不是长视频服务。音频为整段WAV，未实现实时流式音频或工具调用。
- 已做上述SenseVoice短时内存专项；未做长期持续压力测试、业务质量数据集评测、多租户配额或模型ACL。公开源模型的授权状态见各模型来源记录，不能用技术验收替代许可证审查。
