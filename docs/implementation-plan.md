# 模型迁移与多模态扩展（2026-09-23，已完成）

本轮来源：用户要求移除当前 Qwen3.5-0.8B、Whisper Tiny、旧 VITS 的服务登记；按 `llm-server/src/toml/models.toml` 和 `models.full.toml` 清单接入对应模型，无需沿用其接口/推理代码。常规模型使用 OpenVINO/GenAI，ASR/TTS/声纹/克隆使用 sherpa-onnx；新增真实图像与视频生成。

保持 `API → coordinator → scheduler/lifecycle → worker → task/backend` 边界，新增插件，不改业务主流程。新能力使用现有 `/v1/{capability}` 动态路由。任务插件负责有界输入与预后处理；后端负责本地加载、推理、停止确认和释放。只有验证成功的模型才启用。

分阶段交付：
1. 已盘点：旧仓库主配置6项，full配置补充检索等模型；旧机器挂载目录在本机不存在，需按固定来源下载。OpenVINO仅CPU；NVIDIA RTX5070Ti Laptop 12GB可用于媒体视频，主机15GiB内存。
2. 新增 `ov_genai` 文本/视觉插件；复用或扩展OpenVINO检索插件，补齐BM42真实注意力稀疏输出、Chinese-CLIP双塔。
3. 新增sherpa ASR、TTS、speaker_embeddings、voice_clone任务与后端；固定官方模型和参考录音，实际验证音频输出、识别与声纹相似度。
4. 生图采用OpenVINO GenAI SD-Turbo；视频采用Diffusers CUDA + AnimateDiff-Lightning，真实时序去噪，限制尺寸/帧数/步数，不用重复图片冒充生成视频。
5. 下载与验证脚本记录来源、版本、SHA和质量样例；增加主机/GPU预算及必要资源互斥；单次大模型验证串行，取消/卸载必须等待真实停止。
6. 使用备份后通过管理用例移除旧3个登记，验证并登记新模型、配置默认别名，复验API/进程/排队/取消/重启。部署在本工作区，不自动安装公网系统服务。

备份：`var/backups/pre-migration-20260923.sqlite3` 与 `var/backups/pre-model-migration-source.tar.gz`。旧模型权重按原删除语义保留；源 `llm-server` 不修改。主要新增代码在 `tasks/*`、`backends/*`，公共注册表由主控统一组装；各领域各有独立配置、准备脚本和真实验证报告。

当前执行结果：
- 已完成：插件边界、主存/显存双预算、输出编码期间取消与ffmpeg孤儿清理；旧3模型移除登记，权重保留。
- 已完成：原仓库对应的9个文本/视觉/检索模型采用OpenVINO Runtime/GenAI，4个语音模型采用sherpa-onnx；增加SD-Turbo图片和AnimateDiff-Lightning + epiCRealism视频，共15个真实模型。
- 已完成：全部15模型通过生产配置下登记、验证、默认别名、真实TCP业务/管理调用、单次加载、卸载和重启配置恢复，最终CPU/GPU记账归零。
- 已完成：Qwen中文/算术/色块、检索相关性与CLIP导出一致性、BM42官方样例、合成语音回译/声纹/克隆、PNG/MP4肉眼样例及取消确认。原始SD1.5视频组合质量失败，已切换官方epiCRealism底座并重新验收。
- 已完成：7个检索配置最大batch/token/图片像素容量检查，以及4B/7B最大输入与128生成token容量检查；按实测驻留与临时峰值校准预算，已有配置变更均通过新唯一版本迁移。
- 最终框架回归326项通过，无失败或跳过；全部15模型真实HTTP复验通过；86个环境包依赖一致性检查通过。精确证据及业务质量/部署限制见 `docs/verification.md`。

以下为已完成上一阶段记录，旧模型测试结论不代表本轮新模型已验证。

---

# 实施基线与模块骨架

## 单机生产部署加固阶段（已完成）

用户在完成第一版后要求继续调整到生产化。沿用现有 API、SQLite schema v1、模型权重与登记数据，保留全部已通过的流程行为；本阶段不部署公网或引入分布式组件。

1. 调度器内部集合改为私有，通过有业务含义的查询、暂停与管理许可接口协作，并处理排队请求跨暂停/恢复的竞态。
2. `BackendFeatures` 声明本地/外部生命周期、worker 死亡后执行是否可能继续、是否需要文件、是否为替身、是否提供增量输出。各后端在受信任注册位置声明，核心用例只读契约。
3. `Service` 保留兼容入口；拆出 `RequestCoordinator`、`ModelManager`、`ConfigurationPolicy`、运行时组装和观测模块。SQLite SQL、队列预算、模型 SDK 各自留在所属模块。
4. 加固 HTTP 准入、接收/发送超时、就绪与指标，提供生产配置、systemd/反向代理模板及添加/移除操作手册。
5. 扩展通用流式契约并实测 Qwen 文本增量/取消；执行架构替换测试、完整回归、真实模型 HTTP 验证和有界并发压力验证。结果按实际执行更新，不将短压测当作长期 SLO 承诺。

本阶段新增文件：`configuration.py`、`coordinator.py`、`management.py`、`runtime.py`、`observability.py`、`plugins.py` 与部署/运维文档。核心契约保持明确的输入、终止确认和资源所有权，不通过继承或共享可变字典拆分 Service。

验收结果：169 项自动化测试通过；生产配置下三个真实模型新增/验证/调用/移除通过；真实 Qwen HTTP 增量与断连取消通过；Mock 有界 HTTP 过载通过。部署模板已交付但未安装系统服务，当时的完整证据和质量限制见 `docs/verification-legacy.md`。

来源：本次用户提供的 Mermaid 流程图（START/REG，REQUEST，ADD，REMOVE）；当前仓库检查为空，无已有服务或模型文件，未初始化 Git。用户已确认没有现成 HTTP 服务，因此提供通用适配器及受控 HTTP 集成测试。

## 模块与依赖

`api → coordinator → registry / scheduler / lifecycle → Executor`。
`ProcessExecutor → worker → TaskPlugin + Backend`。具体实现只在组装处选择，核心调度不导入 OpenVINO、HTTP 或分词器。

| 模块 | 公开入口 | 状态归属 |
| --- | --- | --- |
| API | `/v1/{capability}`, `/admin/*` | 鉴权、输入体大小、HTTP/SSE 生命周期 |
| 请求协调器 | `infer`, `execute` | 请求取消与最终清理 |
| 模型管理器 | `add`, `validate`, `load`, `operate`, `maintain` | 管理操作、维护和重试退避 |
| 运行时 | `start`, `close`, `readiness` | 组装、唯一控制锁、持久状态恢复 |
| 登记表 | `resolve`, `register`, `validation`, `enable`, `remove` | SQLite：配置、验证、启用、别名、已登记依赖、事件 |
| 调度器 | `acquire`, `release`, `pause`, `wait_idle`, `drop_resident` | 单锁原子预算、FIFO 有界队列、使用权、驻留/临时资源 |
| 生命周期 | `ensure_loaded`, `unload`, `status` | 单实例加载锁、加载耗时、进程状态、空闲回收 |
| 执行器 | `load`, `stream`, `unload`, `is_alive`, `snapshot` | 一个实例一个复用子进程，取消直到收到停止确认 |
| 任务插件 | `validate`, `prepare → Prepared`, `finish` | BGE Dense / Reranker / Chinese-CLIP / BM42 的输入输出语义 |
| 后端 | `load`, `infer(inputs,cancel)`, `close` | OpenVINO / HTTP / 明确标记的 Mock |

业务模型选择先解析显式名称或版本、默认别名，再检查启用/能力。原子预留驻留预算（每实例一次）+临时预算+全局和模型执行名额，加载后执行。流式传输持有许可直到发送结束且底层确认停止。取消不自动重试，也不提前回收。卸载前暂停准入、拒绝队列、等待使用权归零，超时保留暂停状态并明确报告。外部 HTTP 模型的远端生命周期不伪造；网络故障后远端状态未知时隔离占用，需管理员确认远端停止。

## 阶段

1. 已完成：空仓库检查、骨架说明和稳定契约落盘。以下路径为计划，测试尚未执行。
2. 已完成：SQLite、预算调度、进程复用和 Mock 完整闭环，核心测试通过。
3. 已完成：API/管理鉴权、OpenVINO/HTTP/Transformers/Sherpa 适配、检索与语音/文本插件及配置、可观察性。
4. 已完成：框架边界与故障、真实 TCP 启动/断连/崩溃测试，以及 Qwen/Whisper/VITS 三个真实模型的完整控制流程和取消实测。第一版全量 84 项通过；本轮生产加固另增加回归与真实增量输出，最新精确证据与未验证检索模型见 `docs/verification.md`。

## 工程假设

- 单个控制进程持有数据库旁的独占文件锁；不支持多 Web worker，同机同一数据库重复启动明确拒绝。
- 模型版本配置不可变；更换文件/执行配置必须使用新版本，别名可迁移。文件由管理员放入允许目录，不从业务输入接受路径。
- 内存预算是准入估算，同时记录实际 RSS；不宣称实现操作系统硬内存配额或 GPU 显存测量。
- HTTP 的远端预算独立于本机；这里限制适配器占用和并发。同步 HTTP 正常响应视为远端执行结束；断链/网络超时无法证明停止，进入隔离状态。
- 本阶段新增插件代码需要重启控制进程，不实现动态代码加载或分布式组件。

## 初始验收范围及最终映射位置

下表保留骨架阶段的范围；最终逐分支实现、测试名称和执行证据已移至 `docs/verification.md`。

| 节点/分支 | 实现计划 | 验收计划 |
| --- | --- | --- |
| START/REG | registry, app lifespan | 配置/别名/启用恢复；加载态不伪造恢复 |
| B/C→ERROR | API, coordinator | 分离鉴权、大小、能力、禁用、未知模型 |
| D/Q/E | scheduler | 队列满、等待超时、预算超限、原子预留、无重复驻留计费 |
| F/G→I/FAIL | lifecycle, executor | 并发只加载一次、加载失败、子进程故障 |
| I/OUT/FAIL/CLEAN | coordinator, executor | 普通/流式、取消、停止确认、许可保持、无静默重试 |
| ADD1–ADD6 | management.add, registry | 先持久化待验证，预算内测试，失败保持禁用 |
| RM1–RM9 | management.operate | 依赖/别名检查、队列处理、排空、超时不强杀、三种操作、文件保留 |

测试证据将在完成时写入 `docs/verification.md`；Mock 验证不代表真实模型已支持。
