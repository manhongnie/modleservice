import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from model_service.contracts import ModelConfig, ServiceError
from model_service.tasks import create_task, validate_input


def config(task="mock", capability="embeddings", **kwargs):
    return ModelConfig(name="task-test", version="1", capabilities=[capability], task=task,
                       backend=kwargs.pop("backend", "mock"), validation_input={"texts": ["hello"]}, **kwargs)


@pytest.mark.parametrize("row", json.loads(Path("examples/models.mock.json").read_text()))
def test_all_mock_capabilities_are_explicitly_marked(row):
    cfg = ModelConfig(**row)
    validate_input(cfg, cfg.validation_input, cfg.capabilities[0])
    task = create_task(cfg)
    prepared = task.prepare(cfg.validation_input)
    output = task.finish({"mock": True, "payload": prepared.inputs}, prepared.context)
    assert output["mock"] is True and "Synthetic" in output["notice"]


def test_construction_and_validation_do_not_import_ml_libraries():
    script = """
import json,sys
from pathlib import Path
from model_service.contracts import ModelConfig
from model_service.tasks import create_task,validate_input
for filename in Path('examples').glob('models.*.json'):
 for row in json.loads(Path(filename).read_text()):
  cfg=ModelConfig(**row)
  create_task(cfg)
  validate_input(cfg,cfg.validation_input,cfg.capabilities[0])
assert not {'numpy','torch','transformers','openvino','openvino_genai','diffusers','sherpa_onnx'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": [1]}, {"texts": [""]}, {"texts": ["ok"], "path": "/etc/passwd"}])
def test_rejects_invalid_text_and_business_paths(payload):
    with pytest.raises(ServiceError) as error:
        validate_input(config(), payload, "embeddings")
    assert error.value.code == "invalid_input"


def test_bm42_never_falls_back_to_fake_local_sparse_output():
    with pytest.raises(ServiceError) as error:
        create_task(config(task="bm42_http", capability="sparse_embeddings", backend="openvino"))
    assert error.value.code == "unsupported_format"


def test_unknown_task_fails_explicitly():
    with pytest.raises(ServiceError, match="Unknown task"):
        create_task(config(task="arbitrary.module:class"))


@pytest.mark.parametrize("output", [{"ok": True}, {"sparse_embeddings": [{"indices": [1], "values": []}]},
                                    {"sparse_embeddings": [{"indices": [1], "values": [float("nan")]}]}])
def test_bm42_http_requires_valid_sparse_vectors(output):
    task = create_task(config(task="bm42_http", capability="sparse_embeddings", backend="http"))
    prepared = task.prepare({"texts": ["hello"]})
    with pytest.raises(ServiceError) as error:
        task.finish(output, prepared.context)
    assert error.value.code == "invalid_model_output"


def test_http_dense_checks_batch_shape_and_finite_numbers():
    task = create_task(config(task="http_json", backend="http"))
    prepared = task.prepare({"texts": ["hello", "world"]})
    valid = {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}
    assert task.finish(valid, prepared.context) == valid
    for invalid in ({"ok": True}, {"embeddings": [[1.0]]}, {"embeddings": [[1, 2], [3]]}, {"embeddings": [[float("inf")], [0]]}):
        with pytest.raises(ServiceError):
            task.finish(invalid, prepared.context)


def test_chat_token_limit_and_text_only_boundary():
    cfg = config(task="qwen_chat", capability="chat", backend="transformers", options={"max_new_tokens": 16})
    with pytest.raises(ServiceError):
        validate_input(cfg, {"messages": [{"role": "user", "content": "hello"}], "max_new_tokens": 17}, "chat")
    with pytest.raises(ServiceError):
        validate_input(cfg, {"messages": [{"role": "user", "content": [{"image_url": "http://internal"}]}]}, "chat")


def test_asr_rejects_wrong_rate_and_truncated_wav():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(bytes(100))
    cfg = config(task="whisper_asr", capability="asr", backend="transformers")
    with pytest.raises(ServiceError, match="16000"):
        validate_input(cfg, {"audio_base64": base64.b64encode(buffer.getvalue()).decode()}, "asr")


def test_bge_dense_uses_cls_not_mean_pooling():
    np = pytest.importorskip("numpy")
    task = create_task(config(task="bge_m3_dense", backend="openvino"))
    result = task.finish({"last_hidden_state": np.array([[[3, 4], [900, 0]]])}, {})
    assert result["embeddings"][0] == pytest.approx([0.6, 0.8])
    with pytest.raises(ServiceError, match="last_hidden_state"):
        task.finish({"other": np.ones((1, 2))}, {})


def test_reranker_validates_shape_and_sorts_scores():
    np = pytest.importorskip("numpy")
    task = create_task(config(task="bge_reranker", capability="rerank", backend="openvino"))
    result = task.finish({"logits": np.array([[-2], [2]])}, {"count": 2})
    assert result["results"][0]["index"] == 1
    with pytest.raises(ServiceError):
        task.finish({"logits": np.array([[2]])}, {"count": 2})


def test_tts_waveform_is_encoded_as_wav():
    np = pytest.importorskip("numpy")
    task = create_task(config(task="vits_tts", capability="tts", backend="sherpa_tts"))
    result = task.finish({"samples": np.ones(160, dtype=np.float32) / 2, "sample_rate": 16000}, {})
    with wave.open(io.BytesIO(base64.b64decode(result["audio_base64"]))) as audio:
        assert audio.getnframes() == 160 and audio.getsampwidth() == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "60", 0])
def test_http_options_reject_nonfinite_or_invalid_timeouts(value):
    from model_service.backends import validate_backend
    with pytest.raises(ServiceError) as error:
        validate_backend(config(task="http_json", backend="http", options={"base_url": "http://localhost", "timeout_s": value}))
    assert error.value.code == "invalid_config"


def test_compatible_backend_registration_uses_declared_task_family(monkeypatch):
    from model_service.backends import BACKENDS, backend_registration
    from model_service.plugins import BuiltinPluginCatalog
    from model_service.contracts import Settings
    original = backend_registration(config(backend="http"))
    monkeypatch.setitem(BACKENDS, "custom_remote", original)
    cfg = config(task="http_json", backend="custom_remote", options={"base_url": "http://localhost"})
    catalog = BuiltinPluginCatalog()
    settings = Settings(business_keys=["business"], admin_keys=["admin"])
    catalog.validate(cfg, settings)
    assert catalog.describe(cfg).execution_may_outlive_worker
    assert create_task(cfg).finish({"embeddings": [[1.0]]}, {"payload": cfg.validation_input}) == {"embeddings": [[1.0]]}
    with pytest.raises(ServiceError) as error:
        catalog.validate(cfg, settings.model_copy(update={"http_allowed_hosts": []}))
    assert error.value.code == "host_forbidden"


@pytest.mark.parametrize("properties", [
    {"CACHE_DIR": "/tmp/outside-model-roots"},
    {"CACHE_DIR": "models/cache"},
    {"CACHE_DIR": "models/escaping-symlink"},
    {"cache_dir": "models/cache"},
    {"DEVICE_PROPERTIES": {"CPU": {"CACHE_DIR": "/tmp/outside"}}},
    {"MODEL_PATH": "/tmp/outside"},
    {"PERFORMANCE_HINT": {"CACHE_DIR": "/tmp/outside"}},
    [],
])
def test_openvino_disables_cache_paths_and_nested_compile_properties(properties):
    from model_service.backends import validate_backend
    with pytest.raises(ServiceError) as error:
        validate_backend(config(task="bge_m3_dense", backend="openvino", path="models/unused", options={"compile_config": properties}))
    assert error.value.code == "invalid_config"


def test_openvino_permits_scalar_execution_tuning_without_importing_runtime():
    from model_service.backends import validate_backend
    validate_backend(config(task="bge_m3_dense", backend="openvino", path="models/unused",
        options={"compile_config": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 2}}))


def test_worker_registration_does_not_bypass_backend_config_validation():
    from model_service.backends import backend_registration, create_backend
    cfg = config(task="bge_m3_dense", backend="openvino", path="models/unused",
                 options={"compile_config": {"CACHE_DIR": "/tmp/outside"}})
    with pytest.raises(ServiceError) as error:
        create_backend(cfg, backend_registration(cfg))
    assert error.value.code == "invalid_config"
