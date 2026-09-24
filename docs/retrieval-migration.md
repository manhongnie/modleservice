# 从 llm-server 迁入的检索模型

只迁入原项目声明的模型，不沿用它的 HTTP 端口、路由或进程管理。旧仓库保持不变。原 `docker-compose.yaml` 指向 `/DATA/AppData/AI_MODELS/llm-server`，此目录在当前环境不存在；因此从对应上游下载固定 revision 的权重。本项目中的路径为 `models/retrieval/`，每个目录保存 `source-manifest.json`，记录来源、版本、许可引用和文件 SHA256。

| 原配置 | 本项目名称 | 任务插件 | 执行后端 | 业务能力 |
|---|---|---|---|---|
| 主配置 embedding | `bge-m3-int4-ov` | `bge_m3_bounded` | OpenVINO Runtime CPU | `embeddings`，1024维 CLS+L2 |
| 主配置 rerank_ov | `bge-reranker-v2-m3-int4-ov` | `bge_reranker_bounded` | OpenVINO Runtime CPU | `rerank`，单logit sigmoid |
| 主配置 clip2 | `chinese-clip-vit-base-patch16` | `dual_clip` | OpenVINO Runtime CPU，双塔同进程 | `embeddings` / `image_embeddings`，512维 |
| 主配置 sparse_embedding2 | `bm42-all-minilm-l6-v2-attentions` | `bm42_local` | OpenVINO Runtime CPU读取ONNX | `sparse_embeddings` |
| full embedding_qwen3_ov | `qwen3-embedding-0.6b-int8-ov` | `qwen3_embedding_ov` | OpenVINO Runtime CPU | `embeddings`，1024维 last-token+L2 |
| full rerank_qwen3_ov | `qwen3-reranker-0.6b-int8-ov` | `qwen3_reranker_ov` | OpenVINO Runtime CPU | `rerank`，yes/no logit概率 |
| full clip | `clip-vit-base-patch16` | `dual_clip` | OpenVINO Runtime CPU，双塔同进程 | `embeddings` / `image_embeddings`，512维 |

full 配置的原始 PyTorch BGE reranker 与主配置的 int4 OV 版来自同一底座，本次部署 int4 OV 版本，不额外重复驻留原始 PyTorch 模型。原 Whisper 属于语音迁移范围，由 sherpa-onnx ASR 替换。

## 准备和校验

```bash
# 主配置的四个检索模型
.venv/bin/python scripts/prepare_retrieval_models.py --export-clip

# full 配置增加的三个模型
.venv/bin/python scripts/prepare_retrieval_models.py \
  --models qwen3-embedding qwen3-reranker openai-clip --export-clip

# 每项在单独进程中做真实模型小样本效果检查，避免测试间遗留驻留内存
.venv/bin/python scripts/prepare_retrieval_models.py --verify-only \
  --models bge-m3 bge-reranker chinese-clip bm42 qwen3-embedding qwen3-reranker openai-clip
```

CLIP 的预处理使用原模型的 processor；准备脚本离线把投影后的文本塔和图像塔分别导出成 OpenVINO IR，使用 FP16 存储权重。服务运行时只编译这些 IR，模型计算不调用 PyTorch。双塔作为一个实例统一加载、预留预算和卸载，不在控制进程中加载模型。

配置见 `examples/models.retrieval-migrated.json`。生产服务启动后按通用管理接口验证再启用，业务使用模型名称选择能力。不要在注册全部检索模型时自动覆盖所有默认别名；各业务可明确选择模型，管理员再选择默认 embedding/reranker。

## 调用

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3-int4-ov","input":{"texts":["北京是中国的首都。","香蕉是一种水果。"]}}'

curl http://127.0.0.1:8000/v1/rerank \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3-int4-ov","input":{"query":"中国的首都是哪里？","documents":["北京是中国的首都。","香蕉是一种水果。"]}}'

curl http://127.0.0.1:8000/v1/sparse_embeddings \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"bm42-all-minilm-l6-v2-attentions","input":{"texts":["Study history and science."],"input_type":"document"}}'
```

CLIP 文本请求使用 `/v1/embeddings` 与 `{"texts":["红色的中国灯笼"]}`。图片请求使用 `/v1/image_embeddings` 与 `{"images":["不含data:前缀的图片base64"]}`，业务不能传模型文件位置或任意图片路径。两类向量在同一个投影空间中归一化，可直接用点积计算余弦相似度。单模态输出字段为 `embeddings`；同时提供 `texts` 和 `images` 时，输出分别为 `text_embeddings` 和 `image_embeddings`。

Qwen3 embedding 检索查询应传 `input_type:"query"`，插件按官方模板添加管理员配置的检索指令；文档默认 `input_type:"document"`，不加指令。Qwen3 reranker 接口与 BGE reranker 相同，插件在执行进程里拼装官方 yes/no 判断模板。

## BM42 边界

BM42 使用 Qdrant 发布的真实 MiniLM attention ONNX，通过 OpenVINO 推理。文档权重来自末层 CLS attention 的 head 均值，按 WordPiece 合并、英文停词过滤、Snowball 词干、同词最大权重，再计算 `log(1+weight)^0.5`；MurmurHash3 仅作为词维度标识。查询权重按官方算法取 1。返回 `requires_idf:true`，检索索引仍需维护语料 IDF；本服务不包含向量库或语料统计。

该上游模型是英文模型，不能把中文输入的有限输出当作中文检索支持。原始 ONNX 只接收 `input_ids`，与官方导出相同；批量 padding 可能影响 attention 权重，固定分块和批处理策略后再评测业务检索效果。

## 验证证据与限制

七个模型均已用本地真实权重在 OpenVINO CPU 上完成小样本检查，没有以 Mock 代替。结果如下：

| 检查 | 实际结果 |
|---|---|
| BGE-M3 中文首都查询 | 北京文档/水果文档相似度 0.662 / 0.348 |
| BGE Reranker 同一正反样本 | 0.98281 / 0.000017 |
| Chinese-CLIP 灯笼图片 | 灯笼/汽车/猫相似度 0.4867 / 0.3524 / 0.3591 |
| BM42 官方模型卡两个例句 | 索引完全相同，权重最大绝对误差 1.23×10⁻⁷ |
| Qwen3 Embedding 中文首都查询 | 0.6703 / 0.1685 |
| Qwen3 Reranker 同一正反样本 | 0.99836 / 0.000032 |
| OpenAI CLIP 灯笼图片 | lantern/car/cat 相似度 0.3222 / 0.1860 / 0.1756 |

两个 CLIP 的导出还用不同于 tracing 的 batch=2、文本长度=8（含 padding）进行 PyTorch/OpenVINO 一致性检查；四个塔的最小输出余弦相似度均大于 0.999999。此证据记录在各模型的 `source-manifest.json` 中。Chinese-CLIP 文本塔使用原始 CLS 隐藏状态后投影，OpenAI CLIP 使用原模型 pooled/EOS 状态后投影，不能交换这两种处理方式。

`docs/retrieval-migration.json` 保存运行时版本、真实输出摘要、加载/推理耗时、进程峰值 RSS 和实际停止结果。首次推理耗时包含分词器/processor 的首次初始化，不是稳定负载下的延迟基准。它是少量正反样本检查，不是检索基准、长时间稳定性测试或准确率承诺。容量限制以配置中的 batch、max_length、并发和预算为准；修改预算需重新测量最大输入。小样本结果不覆盖所有语言、长文本截断和量化误差。

上游参考：

- [BGE-M3 官方模型](https://huggingface.co/BAAI/bge-m3)、[所用 int4 OV 导出](https://huggingface.co/EmbeddedLLM/bge-m3-int4-ov)
- [BGE Reranker 官方模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[所用 int4 OV 导出](https://huggingface.co/EmbeddedLLM/bge-reranker-v2-m3-int4-ov)
- [Chinese-CLIP 官方模型](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [Qdrant BM42 ONNX](https://huggingface.co/Qdrant/all_miniLM_L6_v2_with_attentions)、[官方后处理实现](https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm42.py)
- [Qwen3 Embedding OpenVINO 官方导出与用法](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov)、[Qwen3 Reranker OpenVINO 官方导出与用法](https://huggingface.co/OpenVINO/Qwen3-Reranker-0.6B-int8-ov)
- [OpenAI CLIP 官方模型](https://huggingface.co/openai/clip-vit-base-patch16)

容量另用 `.venv/bin/python scripts/validate_retrieval_capacity.py` 在独立进程中检查配置最大batch/token（CLIP还检查最大解码像素），每个模型执行两次，记录峰值和完成后驻留RSS到 `docs/retrieval-capacity.json`。运行前停止业务服务和其他大模型验证；此检查不等于持续压力测试或硬内存上限保证。

已按容量实测升级BM42、Chinese-CLIP、OpenAI CLIP和Qwen3 Embedding/Reranker的预算配置，采用新的唯一版本登记并通过管理用例移除旧登记，权重未删除。早期质量报告保留当时模型ID；权重来源SHA、任务算法不变，当前版本/预算见配置、容量报告和部署报告。当前7项驻留与总预算均覆盖已测的最大输入样例，旧不足记录保留为校准证据。
