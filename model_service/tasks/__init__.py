"""Explicit task registry. Constructing tasks never imports ML runtimes."""
from model_service.contracts import ModelConfig, ServiceError, TaskPlugin
from .core import MockTask, PassthroughTask, SpeechTask, ChatTask
from .embeddings import DenseTask, RerankTask, ChineseClipTask
from .ov_genai import GenAITask
from .sherpa_tasks import SherpaSpeechTask
from .retrieval_ov import BM42Task, DualClipTask, BoundedDenseTask, BoundedRerankTask, QwenEmbeddingTask, QwenRerankerTask
from .media_generation import ImageGenerationTask, VideoGenerationTask


TASKS = {
    "mock": MockTask,
    "http_json": PassthroughTask,
    "bm42_http": PassthroughTask,
    "bge_m3_dense": DenseTask,
    "bge_reranker": RerankTask,
    "chinese_clip": ChineseClipTask,
    "qwen_chat": ChatTask,
    "whisper_asr": SpeechTask,
    "vits_tts": SpeechTask,
    "ov_chat": GenAITask,
    "ov_vision_chat": GenAITask,
    "sherpa_sensevoice": SherpaSpeechTask,
    "sherpa_matcha": SherpaSpeechTask,
    "sherpa_speaker": SherpaSpeechTask,
    "sherpa_zipvoice": SherpaSpeechTask,
    "bge_m3_bounded": BoundedDenseTask,
    "bge_reranker_bounded": BoundedRerankTask,
    "qwen3_embedding_ov": QwenEmbeddingTask,
    "qwen3_reranker_ov": QwenRerankerTask,
    "bm42_local": BM42Task,
    "dual_clip": DualClipTask,
    "text_to_image": ImageGenerationTask,
    "text_to_video": VideoGenerationTask,
}


def create_task(config: ModelConfig, backend_family: str | None = None) -> TaskPlugin:
    factory = TASKS.get(config.task)
    if factory is None:
        raise ServiceError("unknown_task", f"Unknown task plugin: {config.task}")
    from model_service.backends import backend_registration
    family = backend_family or backend_registration(config).task_family
    if config.task == "mock" and family != "mock":
        raise ServiceError("invalid_task_backend", "Mock task requires explicitly selected mock backend")
    if config.task == "bm42_http" and family != "http":
        raise ServiceError("unsupported_format", "bm42_http requires an HTTP backend; select bm42_local for genuine local OpenVINO BM42 inference")
    supported_backends = {
        "http_json": {"http"}, "bge_m3_dense": {"openvino"}, "bge_reranker": {"openvino"},
        "chinese_clip": {"openvino"}, "qwen_chat": {"transformers"},
        "whisper_asr": {"transformers"}, "vits_tts": {"sherpa_tts"},
        "ov_chat": {"openvino_genai"}, "ov_vision_chat": {"openvino_genai"},
        "sherpa_sensevoice": {"sherpa_onnx"}, "sherpa_matcha": {"sherpa_onnx"},
        "sherpa_speaker": {"sherpa_onnx"}, "sherpa_zipvoice": {"sherpa_onnx"},
        "bge_m3_bounded": {"openvino"}, "bge_reranker_bounded": {"openvino"},
        "qwen3_embedding_ov": {"openvino"}, "qwen3_reranker_ov": {"openvino"},
        "bm42_local": {"openvino"}, "dual_clip": {"openvino_clip"},
        "text_to_image": {"openvino_image"}, "text_to_video": {"diffusers_video"},
    }
    if config.task in supported_backends and family not in supported_backends[config.task]:
        raise ServiceError("unsupported_format", f"Task {config.task} does not support backend {config.backend}")
    if config.task in {"qwen_chat", "whisper_asr", "vits_tts"} and config.concurrency != 1:
        raise ServiceError("invalid_config", "Initial generation/speech plugins require concurrency=1")
    if config.task in {"http_json", "bm42_http", "mock"} and len(config.capabilities) != 1:
        raise ServiceError("invalid_config", "Each generic/mock task registration must select one unambiguous capability")
    if config.task == "bm42_http" and config.capabilities != ["sparse_embeddings"]:
        raise ServiceError("unsupported_capability", "BM42 provides sparse_embeddings only")
    return factory(config)


def validate_input(config: ModelConfig, payload: dict, capability: str) -> None:
    if capability not in config.capabilities:
        raise ServiceError("unsupported_capability", f"Model does not provide {capability}")
    create_task(config).validate(payload, capability)
