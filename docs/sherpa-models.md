# sherpa-onnx 语音模型

本项目将 ASR、TTS、声纹向量与零样本语音克隆作为四个独立模型登记，使用相同 `sherpa_onnx` 后端和独立模型执行进程。旧 Whisper/VITS 不再作为默认部署。所有推理都在本机 CPU 完成；运行时不下载模型、不调用外部服务。

| 能力 / 接口 | 模型 | 任务插件 | 返回 |
|---|---|---|---|
| `POST /v1/asr` | SenseVoice Small int8，2024-07-17 导出 | `sherpa_sensevoice` | 文字和模型语言标记 |
| `POST /v1/tts` | Matcha 中文/英文 + 16k Vocos | `sherpa_matcha` | PCM16 WAV base64，16kHz |
| `POST /v1/speaker_embeddings` | 3D-Speaker ERes2Net 中文 16k | `sherpa_speaker` | 单位化声纹向量，可选两段音频的余弦相似度 |
| `POST /v1/voice_clone` | ZipVoice Distill int8 + 24k Vocos | `sherpa_zipvoice` | 使用参考音色生成的 PCM16 WAV base64，24kHz |

声纹接口不提供真实身份识别，也不把相似度自动判定为同一说话人。克隆输入是参考音频及其准确转写；不需要为每个音色注册或训练一个模型。

准备和复测（在项目根目录执行）：

```bash
.venv/bin/python scripts/prepare_sherpa_models.py --smoke
```

按[部署说明](multimodal-deployment.md)启动服务，并在调用端设置与服务相同的 `ADMIN_API_KEY` 后登记：

```bash
.venv/bin/python scripts/register_models.py examples/models.sherpa.json --defaults --unload-after-validation
```

下载器使用官方 sherpa 发布包和发布者的固定 Hugging Face revision，校验源代码中固定的 SHA256，解压时禁止逃逸目录，保存 `models/sherpa/downloads/manifest.json` 和每个模型的 `sha256.json`。它先用真实 Matcha 生成一段参考音频，再将四模型的验证输入写入 `examples/models.sherpa.json`。首次运行无需手动寻找录音；克隆实测使用本地合成音色，未使用发布包内的人物示例录音。每个实测模型在独立进程中运行，结束后回收原生运行时内存。

`--no-download --smoke` 复用已下载的文件。测试输出在 `var/sherpa-validation/`，可播放 `matcha-reference.wav` 和 `zipvoice-clone.wav`；实测耗时、输出信息和进程峰值 RSS 写入 `docs/sherpa-validation.json`。该报告中的真实推理记录与 HTTP 接口测试分开记录，不表示已完成听感或业务准确率验收。重复测试保留首次参考音频，使已登记配置不因随机合成而变化；新 TTS 样本另存为 `matcha-smoke.wav`。报告同时记录输入和输出音频摘要。

业务输入不接受文件路径，只接受音频内容。音频须为单声道 PCM16 WAV，8–48kHz，默认最长30秒；ASR 最短0.1秒，声纹最短0.5秒。克隆参考音频默认1–15秒且必须有转写。每个请求的完整 JSON 输入另有2MiB上限，因此高采样率或两段比较音频可能先达到字节限制；建议使用16kHz录音，较长的双音频比较须缩短片段。TTS 和克隆默认最多200字符，速度0.5–2，克隆扩散步数最多8。默认每实例并发为1，输入/模型/请求内存预算由配置及调度器共同限制。

四个模型当前均已通过真实 TCP HTTP 调用、验证后卸载及重启配置恢复检查，记录见 [HTTP 验证报告](http-multimodal-validation.json)。已设置 `default:asr`、`default:tts`、`default:speaker_embeddings`、`default:voice_clone`，所以下例可省略模型名称；显式选择时在请求顶层添加 `"model": "模型名称@版本"`。

下面在 shell 已设置 `BUSINESS_API_KEY` 时直接调用真实接口：

```python
import base64, json, os, urllib.request
from pathlib import Path

base_url = "http://127.0.0.1:8000"
headers = {"Authorization": "Bearer " + os.environ["BUSINESS_API_KEY"],
           "Content-Type": "application/json"}

def call(capability, payload):
    request = urllib.request.Request(base_url + "/v1/" + capability,
        data=json.dumps({"input": payload}, ensure_ascii=False).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)["output"]

speech = call("tts", {"text": "你好，欢迎使用本地语音服务。"})
Path("reference.wav").write_bytes(base64.b64decode(speech["audio_base64"]))
reference = speech["audio_base64"]
print(call("asr", {"audio_base64": reference}))
print(call("speaker_embeddings", {"audio_base64": reference,
    "compare_audio_base64": reference}))
cloned = call("voice_clone", {"text": "这是一段克隆语音测试。",
    "reference_audio_base64": reference,
    "reference_text": "你好，欢迎使用本地语音服务。", "num_steps": 4})
Path("cloned.wav").write_bytes(base64.b64decode(cloned["audio_base64"]))
```

首轮效果验证使用中文短句，另外通过 ASR 回译检查合成内容，并记录参考/克隆音频的声纹余弦相似度。中英混合、其他语种、噪声、多人录音、长文本和跨语种克隆尚未做质量验收；相似度数值不转换为身份置信度。

ASR/TTS/声纹/克隆首版均为整段输出。即使业务选择 `stream=true`，也不宣称实时音频增量。ASR 和声纹的原生调用只能在完成后确认取消；TTS/克隆在句段回调处协作取消。执行进程在原生调用退出后才发停止确认，不会提前释放调度许可。复杂长句的取消等待可能较长，应依输入限制和实测选择业务超时。

本机真实短句实测（2026-09-23，sherpa-onnx 1.13.8，CPU、2线程；耗时不包含服务排队）：

2026-09-24另补 [SenseVoice内存专项](sensevoice-memory.md)：同一进程1,140次识别，最后700次RSS稳定约486MiB；只关闭模型对象仍保留内存，实际空闲卸载及5轮管理卸载均确认PID退出。下表ASR的369MiB仅来自原先短句，不能当作30秒输入或持续调用的内存上限。

| 能力 | 加载 | 推理 | 进程峰值 RSS | 结果 |
|---|---:|---:|---:|---|
| TTS | 0.938s | 0.157s | 242MiB | 16kHz，2.768秒WAV |
| ASR | 0.810s | 0.128s | 369MiB | 完整识别中文参考句 |
| 声纹 | 0.193s | 0.144s | 185MiB | 512维，同录音相似度1.0 |
| 克隆 | 0.816s | 0.847s | 411MiB | 24kHz，1.992秒WAV |

参考句“你好，欢迎使用本地语音服务。”和克隆句“这是一段克隆语音测试。”都被真实SenseVoice完整识别，去标点字符错误率均为0；参考/克隆声纹余弦相似度为0.7384。这里只有每种合成的一句中文样本，不能据此宣称通用准确率或声音相似度达标。

模型来源和授权信息与软件代码许可分别记录：

- SenseVoice 权重固定为发布者仓库 `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` 的 revision `2365baeacb507f821a0c8120fcee3d484dba7a07`，int8 ONNX SHA256 为 `c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51`。导出说明：[sherpa 官方文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html)。当前原模型卡声明 `model-license`，指向 [FunASR MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)；不将运行时的 Apache-2.0 许可替代模型许可。
- Matcha 中英模型：[官方部署说明](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html)；[发布者模型卡](https://huggingface.co/csukuangfj/matcha-icefall-zh-en/blob/fbe59d77057428c32111d5f3ea4ed67a7d440c61/README.md)指向 `dengcunqin/matcha_tts_zh_en_20251010`，未单独给出明确权重许可字段，商业授权状态未确认。
- ERes2Net：[3D-Speaker 官方项目](https://github.com/modelscope/3D-Speaker)和 sherpa [speaker-recongition-models 发布包](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)。项目源代码许可与训练数据、权重许可不能混同。
- ZipVoice：[官方 sherpa 接入说明](https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html)；[原模型卡](https://huggingface.co/k2-fsa/ZipVoice/blob/4ed45fb6e7e9527b780bef9e097a04bf13fe4e6b/README.md)说明 Distill 模型来自 Emilia。代码与模型分别以发布方文件为准，当前仓库不声称已完成商业授权审核。

Matcha 和 ZipVoice 还分别使用独立的16kHz和24kHz Vocos 权重；下载来源与摘要见 `models/sherpa/downloads/manifest.json`，其权重许可也须单独核对，本轮未完成商业授权审核。

模型不可热换同一版本下的权重：新增目录、使用新版本登记，实际验证成功后切换别名，再按管理流程移除旧登记。新增兼容权重只改配置；新的语音任务/结构只扩展任务插件和后端，不修改 API、调度器或管理主流程。
