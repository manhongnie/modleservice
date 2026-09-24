# 图片与视频生成插件

当前实现通过独立任务插件与后端接入；业务仍调用统一能力接口，不传权重路径。实测结论以 `media-validation.json` 为准。单元测试中的合成像素只验证输入、输出和取消契约，不能证明真实模型推理可用。

| 能力 | 任务插件 | 后端 | 权重 |
|---|---|---|---|
| `image_generation` | `text_to_image` | `openvino_image` | Stability AI SD-Turbo，官方 fp16 权重本地导出 OpenVINO IR |
| `video_generation` | `text_to_video` | `diffusers_video` | epiCRealism（SD 1.5 系列）基座 + ByteDance AnimateDiff-Lightning 4step 时序模块 |

图片使用 [OpenVINO GenAI 的 Text2ImagePipeline](https://openvinotoolkit.github.io/openvino.genai/docs/samples/python/image_generation/)。[SD-Turbo 模型卡](https://huggingface.co/stabilityai/sd-turbo)给出 1–4 步生成和 `guidance_scale=0` 的用法。本项目自行转换官方原始权重，没有把第三方 OpenVINO 导出冒称官方发布。

视频使用 [ByteDance 官方 AnimateDiff-Lightning 4step 权重及配套 Euler 调度方式](https://huggingface.co/ByteDance/AnimateDiff-Lightning)，每次联合去噪多帧，使用实际训练的 temporal motion module。[Diffusers AnimateDiff 文档](https://huggingface.co/docs/diffusers/en/api/pipelines/animatediff)说明时序模块与 SD 图像基座的组合方式。视频不是平移、重复或插值单张图片；MP4 编码仅负责把模型已生成的帧封装成视频。基座固定为官方示例链接的 `emilianJR/epiCRealism` 转换版本；这是社区提供的权重转换，不冒称 ByteDance 发布的基座。原始 SD 1.5 镜像基座在本机组合测试中产生噪声，已从示例配置和默认下载流程移除，失败记录保留供追溯。

## 准备与调用

依赖由统一环境安装：OpenVINO/OpenVINO GenAI 相同发行版本、PyTorch CUDA（NVIDIA 运行视频时）、Diffusers、Accelerate、Transformers、Safetensors、Pillow、NumPy，以及系统 `ffmpeg`。推理进程只读取本地文件，不联网下载模型。

```bash
.venv/bin/python scripts/prepare_media_models.py image video
```

下载脚本固定上游 revision，校验大文件的官方 SHA256，给所有本地文件记录 SHA256；每个模型目录包含来源清单。SD-Turbo 按组件导出，避免同时加载所有模型进行转换。只下载使用 `--download-only`，已有下载只转换图片使用 `image --export-only`。

模型配置在 `examples/models.media.json`。下载与转换完成后，在已启动服务上登记；以下依次提交两条配置，`enable:true` 仍会先执行真实验证，失败时不会开放业务调用。管理令牌和业务令牌须分别设置。

```bash
.venv/bin/python - <<'PYCODE'
import json
from pathlib import Path
for index, config in enumerate(json.loads(Path("examples/models.media.json").read_text())):
    Path(f"/tmp/media-{index}.json").write_text(json.dumps({"config": config, "enable": True}))
PYCODE
curl --fail-with-body http://127.0.0.1:8000/admin/models \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/media-0.json
curl --fail-with-body http://127.0.0.1:8000/admin/models \
  -H "Authorization: Bearer $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/media-1.json
```

采用 `examples/service.multimodal.json` 的较长加载和推理超时；首次 CPU 编译和冷启动可能比复用实例慢很多。业务示例：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/image_generation \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"sd-turbo","input":{"prompt":"A red ceramic teapot on a wooden table, studio photograph","width":512,"height":512,"steps":1,"seed":42}}' \
  > /tmp/image-response.json

curl --fail-with-body http://127.0.0.1:8000/v1/video_generation \
  -H "Authorization: Bearer $BUSINESS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"animatediff-lightning","input":{"prompt":"A small sailboat moving on a calm blue lake, cinematic","width":512,"height":512,"frames":16,"fps":8,"steps":4,"seed":42}}' \
  > /tmp/video-response.json
```

响应 `output.image_base64` 是 PNG，`output.video_base64` 是 H.264 MP4。视频同时返回宽高、帧数、帧率、时长和种子。示例权重主要理解英文提示词。保存结果：

```bash
.venv/bin/python - <<'PYCODE'
import base64, json
from pathlib import Path
for name, field, suffix in [("image", "image_base64", "png"), ("video", "video_base64", "mp4")]:
    response = json.loads(Path(f"/tmp/{name}-response.json").read_text())
    Path(f"/tmp/generated.{suffix}").write_bytes(base64.b64decode(response["output"][field]))
PYCODE
```

卸载保留登记和文件，下次调用重新加载；删除登记会先拒绝新请求、清理等待队列并等待在途执行结束，超时返回明确错误且不强杀。若配置了别名或登记依赖，先解除关联再删除：

```bash
curl --fail-with-body -X POST \
  'http://127.0.0.1:8000/admin/models/animatediff-lightning@027c893e-epic6522-4step/unload?timeout_s=120' \
  -H "Authorization: Bearer $ADMIN_API_KEY"
curl --fail-with-body -X DELETE \
  'http://127.0.0.1:8000/admin/models/animatediff-lightning@027c893e-epic6522-4step?timeout_s=120' \
  -H "Authorization: Bearer $ADMIN_API_KEY"
```

这里删除的是服务登记；模型磁盘文件默认保留。重新添加同一配置即可重新验证并启用。

## 实测记录

完整记录在 `docs/media-validation.json`，真实产物保存在 `var/media/`。重新运行真实本地推理与生成中取消检查：

```bash
.venv/bin/python scripts/prepare_media_models.py image video --verify-only
```

已实测 SD-Turbo 在本机 OpenVINO CPU 上生成 512×512 PNG：加载约 2.4 秒，首次推理约 8.9 秒，复用后约 7.7 秒，相同种子的重复输出一致。已肉眼检查红色茶壶与提示词一致。取消等待当次原生去噪步骤结束，实测约 7.8 秒，取消不是瞬时强杀。

已实测视频在本机 RTX 5070 Ti Laptop / PyTorch CUDA 上生成 512×512、16 帧、8 fps 的 2 秒 MP4。官方人物提示与帆船提示均已查看逐帧联系表：主体清晰、跨帧一致，脸部、姿态或水面有细微变化；不能据此保证大幅动作质量。帆船样例加载约 11.0 秒、冷/热推理约 14.6/14.4 秒，相同种子输出一致，取消确认约 3.4 秒。最大允许帧数已用于真实测试。图片和视频的取消都在真实模型上验证。

验证命令自动写入 `needs_visual_review`，运行成功、非恒定像素或帧间差异不等于内容正确。检查产物后才将 `visual_review` 和最终状态记录为通过。当前报告已完成这一步；统一 HTTP 测试另见 `docs/http-multimodal-validation.json`。

`.venv/bin/pytest -q tests/test_media_generation.py` 的 57 项测试覆盖输入上限、配置/目录约束、输出格式、取消同步以及编码子进程清理。使用合成像素或替身的测试只证明契约，真实模型结论来自上述实测。

## 输入、资源与取消

图片示例固定 512×512，步数 1–4。同一个实例固定尺寸以限制 OpenVINO 原生缓存；另一个尺寸应作为独立配置，重新实测并设置预算。视频示例同样固定 512×512，默认 16 帧，允许 4–16 帧、固定 4 步；推荐已实测的 16 帧。上限为 512×512×16 总像素，不支持通过提高分辨率或帧数绕过预算。帧率只改变播放速度，不增加模型帧数。任务插件限制文本长度、总像素×帧数；图片与视频后端还检查真实 tokenizer 的 token 上限，避免静默截断。

两个模型都要求 `concurrency=1`，默认按需加载，空闲后卸载。实测 512×512 图片的冷/热推理后 RSS 都约 8674 MiB，因此预算为 9216 MiB 驻留加 1024 MiB 请求临时资源；原生缓存不会在请求结束时虚报归还。视频 CPU RSS 实测峰值约 8658 MiB，推理后驻留约 5875 MiB；预算为 6144 MiB 驻留加 4096 MiB 请求资源。CUDA 实测已分配显存峰值约 4472 MiB、缓存保留峰值 5670 MiB，GPU 预算为 6144 MiB 驻留加 2048 MiB 请求资源，包含缓存保留。控制器总 RAM 预算需至少 10240 MiB，GPU 预算至少 8192 MiB。配置中的 RAM/GPU 预算是调度预留，不是系统硬限额。加载、逐步推理、PNG/MP4 整理输出都在模型子进程内完成。取消在去噪步回调中检查；CUDA 路径在成功、取消或失败时均等已提交内核完成，再向控制进程确认结束。编码器使用固定参数和临时目录，超时会终止并等待编码器，业务无法指定文件路径或命令。Linux/WSL 上编码子进程设置父进程死亡信号，模型 worker 意外退出时编码器也会停止；启动时同时检查父进程已死亡的竞态。

媒体输出当前为完整生成后返回单个结果；`stream:true` 只使用统一 SSE 传输完整结果，不宣称逐像素或逐帧实时生成。视频是短时低分辨率动画，不能由此推断长视频、高清、音频生成或商用品质。

## 来源与许可证

- SD-Turbo：`stabilityai/sd-turbo@b261bac6fd2cf515557d5d0707481eafa0485ec2`；保留上游 [LICENSE.md](https://huggingface.co/stabilityai/sd-turbo/blob/b261bac6fd2cf515557d5d0707481eafa0485ec2/LICENSE.md)。该 revision 是 Stability AI Community License，包含商业登记、营收门槛及署名要求；不能当作无条件商用授权。
- epiCRealism 基座：`emilianJR/epiCRealism@6522cf856b8c8e14638a0aaa7bd89b1b098aed17`；[模型卡](https://huggingface.co/emilianJR/epiCRealism/blob/6522cf856b8c8e14638a0aaa7bd89b1b098aed17/README.md)标明 CreativeML Open RAIL-M。这是 ByteDance 官方示例采用的社区 Diffusers 转换。下载原始 safetensors（约 4.27 GB），执行时转为 fp16；motion 权重另约 0.91 GB。
- AnimateDiff-Lightning：`ByteDance/AnimateDiff-Lightning@027c893eec01df7330f5d4b733bc9485ee02e8b2`；[上游许可证](https://huggingface.co/ByteDance/AnimateDiff-Lightning/blob/027c893eec01df7330f5d4b733bc9485ee02e8b2/LICENSE.md)为 CreativeML Open RAIL-M。

Powered by Stability AI. This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved. 本项目将 SD-Turbo 原始权重转换为 OpenVINO IR（fp16 权重），未重新训练模型。
