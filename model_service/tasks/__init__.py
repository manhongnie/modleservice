"""Explicit task registry. Constructing tasks never imports ML runtimes."""
from model_service.contracts import ModelConfig, ServiceError, TaskPlugin
from .core import MockTask, PassthroughTask, SpeechTask, ChatTask
from .embeddings import DenseTask, RerankTask, ChineseClipTask


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
        raise ServiceError("unsupported_format", "BM42 requires a genuine BM42 HTTP implementation; local BM42 is not implemented")
    supported_backends = {
        "http_json": {"http"}, "bge_m3_dense": {"openvino"}, "bge_reranker": {"openvino"},
        "chinese_clip": {"openvino"}, "qwen_chat": {"transformers"},
        "whisper_asr": {"transformers"}, "vits_tts": {"sherpa_tts"},
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
