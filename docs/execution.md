# 执行进程与后端语义

`ProcessExecutor` 使用 `spawn` 创建每个实例的执行进程。控制进程只维护 IPC、进程状态和请求等待器；模型、分词器与原生推理运行时在执行进程中按需初始化。实例重复加载共用加载结果；进程退出后只有下一次新请求会重新创建实例，失败请求本身不会自动重试。

执行进程通过线程池落实 `concurrency`，每次执行持有独立取消事件。`prepare → infer → finish` 完成后发送数据；最后的 `done` 表示该次本地执行已结束。普通调用等成功 `done` 后才交付结果。流式调用可以立即交付数据，但消费者关闭迭代器或任务被取消时，仍须等取消确认或进程死亡。控制层在输出发送完成前继续持有调度许可。

请求输出使用累计字节上限 `max_output_bytes`，包括流式调用的全部片段。进程退出后才释放模型驻留预算，因为原生分配器可能在释放模型对象后仍保留内存。卸载超时报告失败并保留实例，代码不自动强杀。Linux 的父进程死亡信号用于控制进程崩溃后的孤儿清理，不用于管理接口强制排空。

## 后端

- **Mock**：输出均明确标记为测试数据。`delay_s`、`cancel_delay_s`、`load_delay_s`、`stream_chunks`、`chunk_delay_s`、`fail` 和 `fail_load` 仅供故障与边界测试。
- **OpenVINO**：本地 `Core.compile_model`；每次请求创建独立 `InferRequest`。使用 `start_async / wait_for`，取消时先 `cancel` 再 `wait`，不能将发出取消视为完成。SDK 语义依据 [OpenVINO InferRequest 文档](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-request.html)。可选依赖只在选中此后端时导入。`compile_config` 仅接受明确登记的标量执行参数；暂不开放 `CACHE_DIR` 或嵌套设备配置，防止绕过文件目录限制。
- **HTTP**：同步 JSON POST 至配置的 `base_url + infer_path`，输入输出转换由任务插件承担。认证只接受 `auth_env` 指向环境变量；拒绝内嵌凭据、自动重定向和代理环境变量。它只管理本地适配器，远端状态始终是 `external`；适配器关闭不表示远端模型已卸载。
- **OpenVINO GenAI**：Qwen 文字/视觉与 SD-Turbo 使用原生 pipeline；文字由有界队列增量返回，生成线程 join 后才能确认停止。图像在逐步去噪回调检查取消。
- **sherpa-onnx**：SenseVoice、Matcha、ERes2Net、ZipVoice 在独立模型进程执行；只能在原生完成或回调确认后返回取消。
- **Diffusers 视频**：AnimateDiff-Lightning 的时序去噪在 CUDA 执行，停止确认前等待 CUDA 已提交工作完成。显存预算与主存独立预留；加载状态记录实际 CUDA allocated/reserved，详细峰值见媒体验证报告。
- 旧 Transformers / VITS 插件保留为历史兼容代码，旧三个模型的登记已移除，不参与当前默认部署。

Qwen 已实现真实增量文本输出：OpenVINO GenAI 回调将已解码的文字放入有界队列，任务插件整理增量 `delta`；生成结束后提供完整文本、usage、finish_reason。控制端输出队列也有界，慢读触发 `stream_backpressure` 时明确失败，停止确认前保留许可。ASR/TTS 仍缓冲返回，`streaming_mode: buffered`，没有音频增量输出。ProcessExecutor 与真实 HTTP SSE 都有独立实测报告，见 [验证记录](verification.md)。

HTTP 约定远端提供同步终结响应。HTTP 202、网络断开、超时、响应超出上限，以及远端请求期间本地工作进程死亡，都不能证明远端执行已经停止，统一返回 `execution_unknown`。协调器持久化并隔离相关额度，待管理员在远端确认停止后解除。收到本地取消时保持连接，等远端返回；没有远端取消协议时不会伪造已取消。

`tests/test_executor.py` 覆盖实例复用、加载超时与失败、取消确认、迭代器关闭、进程故障、卸载互斥、累计输出限制、HTTP 外部管理及执行状态未知。小型合成 OpenVINO 图测试只有安装真实 Runtime 时才执行；它验证 Runtime 适配器，不证明 BGE / Chinese-CLIP 等真实模型文件已通过验收。

当前迁移验证见 [多模态部署](multimodal-deployment.md)。既有三模型报告只保留为历史证据；各新模型必须分别通过真实权重验证。
