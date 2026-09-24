# OpenVINO GenAI 对话与看图

沿用旧仓库模型清单，使用本项目独立任务插件及 OpenVINO GenAI 后端。当前配置在 [models.genai.json](../examples/models.genai.json)，精确来源、耗时、输出与验证状态见 [genai-validation.json](genai-validation.json)。模型是否已登记并启用，以 [部署报告](deployment-validation.json) 或管理接口为准。

| 能力 | 模型 | 插件 | 后端 |
|---|---|---|---|
| `chat` | `circulus/Qwen3.5-4B-ov-awq` | `ov_chat` | `openvino_genai`，CPU |
| `vision_chat` | `OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov` | `ov_vision_chat` | `openvino_genai`，CPU |

旧 Qwen3.5-0.8B 已移除登记，不是这里的对话模型。4B 使用 VLM 导出布局的原生 pipeline，当前仅对业务开放文字对话；7B 开放有界的图片问答。未开放视频输入、语音输入、工具调用或任意模型路径。

## 调用

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b-ov-awq","input":{"messages":[{"role":"user","content":"请用一句话解释向量检索"}],"max_new_tokens":64}}'

# 真正增量文本，不是将已生成全文切段。
curl -N --fail-with-body http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"input":{"messages":[{"role":"user","content":"请介绍北京"}],"max_new_tokens":64},"stream":true}'
```

业务客户端读本地图片并发送 base64；服务接口不接收客户端文件路径或远程 URL：

```python
import base64, os
from pathlib import Path
import httpx

response = httpx.post("http://127.0.0.1:8000/v1/vision_chat",
    headers={"Authorization": "Bearer " + os.environ["BUSINESS_API_KEY"]},
    json={"model": "qwen2.5-vl-7b-int4-ov", "input": {
        "prompt": "请描述图中的物体。",
        "images": [base64.b64encode(Path("image.png").read_bytes()).decode()],
        "max_new_tokens": 64}}, timeout=180)
response.raise_for_status()
print(response.json()["output"]["text"])
```

输出包含 `text`、`usage` 和 `finish_reason`。SSE 的内容块为 `output.delta`，最后内容块还包含完整 `text`，随后为 `done:true` 终止帧；失败发 `error`，不静默重试。

## 输入及资源边界

示例限制每实例并发1、最多128新token、1024输入token、4096文本字符、每次1张图片，图像在worker内缩小至最长边448。视觉预算根据本地 Qwen processor 的 patch/merge/min_pixels 估算，将文字和图像展开后的 token 都计入准入限制；未知视觉结构需要增加预算规则，不能仅修改名称冒充支持。

图片占位符位于 `user` 消息内，再应用模型模板。不会丢弃图片并降级为纯文字回答。初始化 SDK、图像编码和原生预填充可能不能立刻取消；清理始终等到原生线程停止。模型整个增量输出期间持有执行许可。

文件需要是完整本地 OpenVINO 导出 bundle，包含 tokenizer、语言和相应视觉组件。准备脚本固定源revision及SHA256：

```bash
.venv/bin/python scripts/prepare_genai_models.py chat vision
.venv/bin/python scripts/validate_genai_models.py
```

7B 可用官方 ModelScope 副本传输，但只有文件大小与SHA256和固定HF清单完全相同才复用；这不会改变模型的唯一版本。来源记录保留实际下载端点。源4B量化仓库没有声明许可证字段，不能替发布者虚构授权；来源见 [circulus 模型仓库](https://huggingface.co/circulus/Qwen3.5-4B-ov-awq) 与 [OpenVINO 视觉模型仓库](https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov)。

质量测试是中文问答、简单算术及合成色块识别样例；不等价于通用推理、OCR、文档理解或业务质量基准。

本机最大输入容量实测：4B在接近1024输入token、最多128生成token后RSS约6164MiB，预算6400MiB驻留+768MiB临时；7B以448像素图片、合计1016保守输入token预算（原生1009）及128生成token运行后驻留约9847MiB、峰值9875MiB，预算10240+1024MiB。全局配置预留12288MiB主存；16GiB机器必须给系统与控制进程留空间。两模型均通过红/蓝识别或中文问答、真实增量拼接、取消和后续复用；详见JSON中的各项作用范围。
