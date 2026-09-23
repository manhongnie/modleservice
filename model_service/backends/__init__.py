"""Trusted backend registrations; selection and provider policies live at this edge."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Callable, Any

from model_service.contracts import BackendFeatures, ModelConfig, ServiceError, Settings


@dataclass(frozen=True)
class BackendRegistration:
    factory: str | Callable
    features: BackendFeatures | Callable[[ModelConfig], BackendFeatures]
    task_family: str
    validator: Callable[[ModelConfig, Settings | None], None] | None = None

    def describe(self, config: ModelConfig) -> BackendFeatures:
        return self.features(config) if callable(self.features) else self.features

    def create(self, config: ModelConfig):
        factory = self.factory
        if isinstance(factory, str):
            module, name = factory.split(":", 1)
            factory = getattr(import_module(module), name)
        return factory(config)


BACKENDS: dict[str, BackendRegistration] = {}


def register_backend(name: str, factory: str | Callable, *, features: BackendFeatures | Callable,
                     validator: Callable | None = None, task_family: str | None = None) -> None:
    """Called only by trusted application assembly, never from administrator JSON.

    Factories/callbacks must be importable top-level objects for spawned workers.
    Compatible aliases declare an existing task_family; a new input ABI needs a task.
    """
    if name in BACKENDS:
        raise ValueError(f"Backend already registered: {name}")
    BACKENDS[name] = BackendRegistration(factory, features, task_family or name, validator)


def backend_registration(config: ModelConfig) -> BackendRegistration:
    try:
        return BACKENDS[config.backend]
    except KeyError:
        raise ServiceError("unknown_backend", f"Backend is not registered: {config.backend}") from None


def describe_backend(config: ModelConfig) -> BackendFeatures:
    return backend_registration(config).describe(config)


def _finite_options(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ServiceError("invalid_config", "Plugin options must contain finite JSON values")
    if isinstance(value, dict):
        for item in value.values():
            _finite_options(item)
    elif isinstance(value, list):
        for item in value:
            _finite_options(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ServiceError("invalid_config", "Plugin options must contain JSON values only")


def _bounded_number(config, key, minimum, maximum, *, integer=False):
    if key not in config.options:
        return
    value = config.options[key]
    types = {int} if integer else {int, float}
    if type(value) not in types or not minimum <= value <= maximum or (isinstance(value, float) and not math.isfinite(value)):
        raise ServiceError("invalid_config", f"{key} must be {'an integer ' if integer else ''}between {minimum} and {maximum}")


def _http_config(config, settings):
    from urllib.parse import urlsplit
    forbidden = {"headers", "authorization", "api_key", "token", "secret", "password"}
    if forbidden.intersection(key.lower() for key in config.options):
        raise ServiceError("invalid_config", "HTTP credentials must be referenced through auth_env, not persisted inline")
    url = urlsplit(str(config.options.get("base_url", "")))
    try:
        port = url.port
    except ValueError as error:
        raise ServiceError("invalid_config", "HTTP base_url contains an invalid port") from error
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or (port is not None and port == 0):
        raise ServiceError("invalid_config", "HTTP backend requires base_url without embedded credentials")
    path = str(config.options.get("infer_path", "/infer"))
    if not path.startswith("/") or path.startswith("//") or urlsplit(path).scheme or "\\" in path:
        raise ServiceError("invalid_config", "infer_path must be an absolute path on base_url")
    if url.query or url.fragment or urlsplit(path).query or urlsplit(path).fragment:
        raise ServiceError("invalid_config", "HTTP endpoint queries/fragments are not accepted; use auth_env for authentication")
    if settings is not None and url.hostname.lower() not in {host.lower() for host in settings.http_allowed_hosts}:
        raise ServiceError("host_forbidden", "HTTP backend host is not in the administrator allowlist", 403)
    auth_env = config.options.get("auth_env")
    if auth_env is not None and (not isinstance(auth_env, str) or not auth_env.isidentifier()):
        raise ServiceError("invalid_config", "auth_env must name an environment variable")
    _bounded_number(config, "timeout_s", 0.001, 3600)
    _bounded_number(config, "max_response_bytes", 1, 67108864, integer=True)


def _local_config(config, settings):
    for key, low, high in (("threads", 1, 256), ("max_new_tokens", 1, 32768),
                           ("max_input_tokens", 1, 131072), ("max_audio_seconds", 1, 30),
                           ("max_text_chars", 1, 10000), ("speakers", 1, 65536)):
        _bounded_number(config, key, low, high, integer=True)



def _openvino_config(config, settings):
    _local_config(config, settings)
    # compile_model accepts provider properties that can create files, load
    # plugins, or nest device configuration. Until managed cache lifecycle exists,
    # only scalar execution tuning is accepted; no cache/artifact write paths.
    properties = config.options.get("compile_config", {})
    permitted = {"PERFORMANCE_HINT", "NUM_STREAMS", "INFERENCE_NUM_THREADS", "AFFINITY",
                 "ENABLE_CPU_PINNING", "ENABLE_CPU_RESERVATION", "INFERENCE_PRECISION_HINT",
                 "EXECUTION_MODE_HINT", "PERFORMANCE_HINT_NUM_REQUESTS"}
    if not isinstance(properties, dict):
        raise ServiceError("invalid_config", "OpenVINO compile_config must be an object")
    if set(properties) - permitted:
        raise ServiceError("invalid_config", "OpenVINO compile_config only supports execution tuning; cache paths, nested device properties and other file-writing properties are disabled")
    if any(type(value) not in {str, int, float, bool} for value in properties.values()):
        raise ServiceError("invalid_config", "OpenVINO execution tuning properties must be scalar values")


def _mock_config(config, settings):
    for key in ("load_delay_s", "delay_s", "cancel_delay_s", "chunk_delay_s"):
        _bounded_number(config, key, 0, 3600)
    chunks = config.options.get("stream_chunks", 1)
    if not ((type(chunks) is int and 1 <= chunks <= 10000) or (isinstance(chunks, list) and 1 <= len(chunks) <= 10000)):
        raise ServiceError("invalid_config", "Mock stream_chunks must be 1..10000 or a bounded list")


def _transformers_features(config):
    return BackendFeatures(incremental_output=config.task == "qwen_chat")


def _validate_registration(config: ModelConfig, registration: BackendRegistration,
                           settings: Settings | None = None) -> None:
    features = registration.describe(config)
    _finite_options(config.options)
    if features.requires_artifact and not config.path:
        raise ServiceError("invalid_config", "Local model backends require a model path")
    if registration.validator:
        registration.validator(config, settings)


def validate_backend(config: ModelConfig, settings: Settings | None = None) -> None:
    _validate_registration(config, backend_registration(config), settings)


def create_backend(config: ModelConfig, registration: BackendRegistration | None = None):
    registration = registration or backend_registration(config)
    # Spawned workers receive trusted registration metadata explicitly. Validate
    # provider settings there too, without relying on a parent process registry.
    _validate_registration(config, registration)
    return registration.create(config)


register_backend("mock", "model_service.backends.mock:MockBackend", features=BackendFeatures(
    requires_artifact=False, synthetic=True, incremental_output=True), validator=_mock_config)
register_backend("http", "model_service.backends.http:HTTPBackend", features=BackendFeatures(
    management="external", execution_may_outlive_worker=True, requires_artifact=False), validator=_http_config)
register_backend("openvino", "model_service.backends.openvino:OpenVINOBackend", features=BackendFeatures(), validator=_openvino_config)
register_backend("transformers", "model_service.backends.generative:TransformersBackend", features=_transformers_features, validator=_local_config)
register_backend("sherpa_tts", "model_service.backends.speech:SherpaTTSBackend", features=BackendFeatures(), validator=_local_config)
