# 多模态部署与模型迁移

本轮从 `llm-server/src/toml/models.toml` 提取启用模型清单，补入 `models.full.toml` 的三个检索模型；不复用旧接口或不合理推理封装，不修改源仓库。旧三个模型已通过正常管理流程移除登记及默认别名，权重默认保留；迁移前 SQLite 备份在 `var/backups/pre-migration-20260923.sqlite3`。

## 环境准备

所有命令在项目根目录执行。实际开发环境为 Linux / Python 3.13，CUDA 视频使用 NVIDIA GPU，OpenVINO 在当前机器上使用 CPU。系统需安装 `curl`、`ffmpeg`。

```bash
uv venv .venv --python 3.13
# GPU 视频的 PyTorch 必须包含相应 CUDA 内核；本机 RTX 5070 Ti 是 sm_120。
uv pip install --python .venv/bin/python 'https://download.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl'
uv pip install --python .venv/bin/python -c requirements-tested.txt -e '.[test,openvino,genai,retrieval,speech,media]'
```

`requirements-tested.txt` 是实际验证环境的完整版本记录（已去除本机临时 wheel 路径），重建时必须使用相应 CUDA wheel 来源，不能把 CPU Torch 当成 CUDA 已准备好。主服务及进程模型插件不依赖系统 Python 的 ML 包。

## 准备权重

```bash
.venv/bin/python scripts/prepare_genai_models.py chat vision
.venv/bin/python scripts/prepare_retrieval_models.py --export-clip \
  --models bge-m3 bge-reranker bm42 chinese-clip qwen3-embedding qwen3-reranker openai-clip
.venv/bin/python scripts/prepare_sherpa_models.py --smoke
.venv/bin/python scripts/prepare_media_models.py image video
```

下载只发生在显式准备脚本中。来源版本固定；大文件用上游 SHA256 验证，分块断点文件验证完成才发布为正式文件。CLIP 与 SD-Turbo 从官方原始权重本地转换为 OpenVINO IR；转换过程与推理分离。语音准备脚本会用本地 Matcha 合成参考语音，并生成带真实验证输入的 `examples/models.sherpa.json`。

模型目录各有来源、摘要及许可证记录。许可证状态见各模型文档；技术测试不表示已获全部商业用途授权。网络慢或断开可以重跑，保留已完整验证的文件和大文件断点。管理员准备阶段默认使用 IPv4 curl；可设置 `MODEL_DOWNLOAD_TRANSPORT=httpx` 使用有界IPv4连接池直连，仍校验Range、长度和SHA。运行时不联网补权重。

## 注册、验证、默认别名

本工作区的持久化状态见 [部署报告](deployment-validation.json)。在另一台机器重建时，先准备权重，再任选一种登记方式。

服务未运行时，使用同一套管理用例的离线控制入口；它也持有数据库独占锁，不能与 HTTP 服务同时运行：

```bash
.venv/bin/python scripts/deploy_models.py \
  examples/models.genai.json examples/models.retrieval-migrated.json \
  examples/models.sherpa.json examples/models.media.json --defaults
```

每个模型实际加载并执行 `validation_input`，成功后启用、建立别名、卸载验证进程以释放预算。失败保留禁用登记与失败原因；修正文件后可重新验证。配置修改要使用新版本，不原地覆盖。

服务运行时，改用管理 HTTP 客户端：

```bash
export BUSINESS_API_KEY="$(openssl rand -hex 24)"
export ADMIN_API_KEY="$(openssl rand -hex 24)"
.venv/bin/python -m model_service --config examples/service.multimodal.json
# 另一终端使用同一 ADMIN_API_KEY；按需登记相应列表。
.venv/bin/python scripts/register_models.py examples/models.genai.json --defaults --unload-after-validation
.venv/bin/python scripts/register_models.py examples/models.sherpa.json --defaults --unload-after-validation
.venv/bin/python scripts/register_models.py examples/models.retrieval-migrated.json --unload-after-validation
.venv/bin/python scripts/register_models.py examples/models.media.json --defaults --unload-after-validation
```

检索配置多个模型支持相同能力：推荐 `default:embeddings` 指向 BGE-M3、`default:rerank` 指向 BGE Reranker、`default:image_embeddings` 指向 Chinese-CLIP。离线部署脚本按列表第一个支持该能力的模型设置默认；普通 `register_models.py --defaults` 会依次更新，登记检索全列表时请省略此参数，再显式选择别名，避免最后一个 CLIP 覆盖默认文本检索。

```bash
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/aliases/default:embeddings \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"bge-m3-int4-ov@d50a3650d1f6"}'
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/aliases/default:rerank \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"bge-reranker-v2-m3-int4-ov@2252a9a291c5"}'
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/aliases/default:image_embeddings \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"chinese-clip-vit-base-patch16@36e679e65c2a-m2560-r1024"}'
curl --fail-with-body -X PUT http://127.0.0.1:8000/admin/aliases/default:sparse_embeddings \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model_id":"bm42-all-minilm-l6-v2-attentions@9695632d760b-m1536"}'
```

同一大模型在使用时不可卸载。管理验证也受预算约束；大量登记使用 `--unload-after-validation`，或者先卸载已空闲的大模型，避免等不到可用预算。

## 调用和删除

[README](../README.md) 给出统一 API、增加/移除流程；[检索](retrieval-migration.md)、[语音](sherpa-models.md)、[媒体](media-models.md) 给出完整输入及文件输出示例。模型路径、设备与推理后端只在管理配置中出现。

```bash
# 仅卸载：保留配置、启用状态，下次业务请求重新加载。
curl --fail-with-body -X POST \
  http://127.0.0.1:8000/admin/models/qwen3.5-4b-ov-awq@0baf3dbd8812/unload \
  -H "Authorization: Bearer $ADMIN_API_KEY"

# 移除：先迁移/删除别名与已登记业务依赖，然后走排空和卸载。
curl --fail-with-body -X DELETE http://127.0.0.1:8000/admin/aliases/default:chat \
  -H "Authorization: Bearer $ADMIN_API_KEY"
curl --fail-with-body -X DELETE \
  http://127.0.0.1:8000/admin/models/qwen3.5-4b-ov-awq@0baf3dbd8812 \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

等待超时明确返回错误并保持暂停状态，不强杀、不删权重。普通移除只删除登记，磁盘文件保留；备份恢复参见 [操作手册](operations.md)。

## 验证与范围

- 框架：`.venv/bin/python -m pytest -q`；测试真实服务控制逻辑和受控故障，不把 Mock 当成模型支持。
- 对话/视觉：`scripts/validate_genai_models.py`，检查真实答案、文字流拼接、取消后复用与峰值 RSS。
- 检索：`scripts/prepare_retrieval_models.py --verify-only --models bge-m3 bge-reranker bm42 chinese-clip qwen3-embedding qwen3-reranker openai-clip`，检查全部7模型的相关性、CLIP 原模型/IR 一致性与 BM42 官方样例。
- 语音：`scripts/prepare_sherpa_models.py --no-download --smoke`，输出 WAV、ASR 回译、声纹相似度和克隆样例。
- 媒体：`scripts/prepare_media_models.py image video --verify-only`，输出 PNG/MP4，记录实际帧、耗时、RSS 和 GPU 峰值。
- 完整登记链路：[deployment-validation.json](deployment-validation.json)；每种模型还有独立真实效果报告，未完成项明确保留失败或待验证状态。

当前主存预算12288MiB、显存预算10240MiB；7B视觉最大输入实测后驻留约9.6GiB。16GiB机器须预留系统和控制进程空间，其他应用占用大时应减少驻留或使用更大内存机器。当前部署最多32个在处理的HTTP请求、8个推理排队请求和1个执行请求，限制接收JSON时控制进程占用；这些上限分别管理。模型执行不与 Web 进程混在一起。单实例只加载一次；内存与显存均在加载前原子预留，驻留预算按实例、临时预算按请求计。原生操作停止后才归还额度；OpenVINO 编译/视觉预处理等不能立即中断的阶段可能延长取消等待。预算不等于操作系统硬隔离，短期功能测试也不等于长期压力与业务质量验收。
