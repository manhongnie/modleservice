import base64
import io
import threading
import time
import wave
from types import SimpleNamespace

import pytest

from model_service.backends.sherpa_models import REQUIRED, SherpaONNXBackend, validate_sherpa
from model_service.contracts import ModelConfig, ServiceError
from model_service.tasks.sherpa_tasks import SherpaSpeechTask, TASK_CAPABILITIES


def wav(seconds=1, rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x00\x10" * int(seconds * rate))
    return base64.b64encode(output.getvalue()).decode()


def config(task="sherpa_sensevoice", **updates):
    return ModelConfig(name="speech-test", version="1", backend="sherpa_onnx", task=task,
                       capabilities=[TASK_CAPABILITIES[task]], validation_input={}, **updates)


@pytest.mark.parametrize("task", TASK_CAPABILITIES)
def test_sherpa_config_requires_single_correct_capability(task):
    cfg = config(task)
    with pytest.raises(ServiceError, match="single"):
        SherpaSpeechTask(cfg.model_copy(update={"capabilities": ["wrong"]}))


@pytest.mark.parametrize("task", TASK_CAPABILITIES)
def test_sherpa_tasks_reject_business_paths_and_unknown_fields(task):
    plugin = SherpaSpeechTask(config(task))
    with pytest.raises(ServiceError, match="unsupported fields"):
        plugin.validate({"path": "/tmp/model.wav"}, TASK_CAPABILITIES[task])


@pytest.mark.parametrize("payload", [{"audio_base64": "bad"}, {"audio_base64": wav(.01)},
                                     {"audio_base64": wav(1, 96000)}, {"audio_base64": wav(31)},
                                     {"audio_base64": wav()[:-4]}])
def test_asr_audio_is_bounded_and_complete(payload):
    with pytest.raises(ServiceError):
        SherpaSpeechTask(config()).validate(payload, "asr")


def test_sherpa_uses_no_sdk_during_control_validation():
    import subprocess, sys
    source = '''
import sys
from model_service.contracts import ModelConfig
from model_service.tasks.sherpa_tasks import SherpaSpeechTask
from model_service.backends.sherpa_models import validate_sherpa
config=ModelConfig(name="x",version="1",backend="sherpa_onnx",task="sherpa_matcha",capabilities=["tts"],validation_input={"text":"hello"})
SherpaSpeechTask(config).validate(config.validation_input,"tts")
assert not {"numpy","sherpa_onnx","torch","transformers"}.intersection(sys.modules)
'''
    subprocess.run([sys.executable, "-c", source], check=True)


def test_speaker_returns_normalized_vector_and_similarity():
    plugin = SherpaSpeechTask(config("sherpa_speaker"))
    result = plugin.finish({"embedding": [3, 4], "compare_embedding": [3, 4]}, {})
    assert result["embedding"] == pytest.approx([.6, .8])
    assert result["cosine_similarity"] == pytest.approx(1)
    assert "identity" not in result
    for value in ([0, 0], [float("nan")], [[1, 2]]):
        with pytest.raises(ServiceError, match="embedding"):
            plugin.finish({"embedding": value}, {})


def test_clone_requires_reference_transcript_and_bounded_steps():
    plugin = SherpaSpeechTask(config("sherpa_zipvoice"))
    payload = {"text": "你好", "reference_text": "测试", "reference_audio_base64": wav()}
    plugin.validate(payload, "voice_clone")
    for override in ({"reference_text": ""}, {"reference_audio_base64": wav(.9)}, {"num_steps": 100}, {"speed": float("nan")}, {"num_steps": True}):
        with pytest.raises(ServiceError):
            plugin.validate(payload | override, "voice_clone")


def test_audio_encoding_and_shape_checks():
    plugin = SherpaSpeechTask(config("sherpa_matcha"))
    result = plugin.finish({"samples": [.1] * 160, "sample_rate": 16000}, {})
    assert result["duration_s"] == .01 and result["streaming_mode"] == "buffered"
    with wave.open(io.BytesIO(base64.b64decode(result["audio_base64"]))) as audio:
        assert audio.getnframes() == 160
    for samples in ([], [[0.1]], [float("inf")]):
        with pytest.raises(ServiceError):
            plugin.finish({"samples": samples, "sample_rate": 16000}, {})


@pytest.mark.parametrize("updates", [{"concurrency": 2}, {"device": "GPU"}, {"options": {"threads": 0}},
                                    {"options": {"path": "/tmp/a"}}, {"options": {"max_steps": 999}},
                                    {"options": {"max_audio_seconds": float("inf")}}])
def test_invalid_sherpa_runtime_configuration(updates):
    with pytest.raises(ServiceError):
        validate_sherpa(config(**updates))


@pytest.mark.parametrize("language", [[], {}, None, True, 1])
def test_sherpa_language_type_returns_configuration_error(language):
    with pytest.raises(ServiceError) as error:
        validate_sherpa(config(options={"language": language}))
    assert error.value.code == "invalid_config"


def test_sherpa_rejects_asset_symlinks_outside_model_root(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    target = tmp_path / "outside.onnx"
    target.write_bytes(b"onnx")
    (root / "model.onnx").symlink_to(target)
    with pytest.raises(ServiceError) as error:
        validate_sherpa(config("sherpa_speaker", path=str(root)))
    assert error.value.code == "invalid_model_path"


def test_cancellation_waits_for_native_inference_return():
    backend = SherpaONNXBackend(config("sherpa_matcha"))
    cancel, started, allow_finish = threading.Event(), threading.Event(), threading.Event()
    outcomes = []

    def generate(text, sid, speed, callback):
        started.set()
        assert allow_finish.wait(2)
        assert callback([.1], 1) == 0
        return SimpleNamespace(samples=[.1], sample_rate=16000)

    backend.model = SimpleNamespace(generate=generate)
    def execute():
        try:
            backend.infer({"text": "你好", "speed": 1.0}, cancel)
        except ServiceError as error:
            outcomes.append(error.code)
    worker = threading.Thread(target=execute)
    worker.start()
    assert started.wait(1)
    cancel.set()
    time.sleep(.02)
    assert worker.is_alive() and outcomes == []
    allow_finish.set()
    worker.join(2)
    assert not worker.is_alive() and outcomes == ["cancelled"]


def test_precancelled_request_never_enters_native_runtime():
    backend = SherpaONNXBackend(config())
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ServiceError) as error:
        backend.infer({}, cancel)
    assert error.value.code == "cancelled"


def test_downloader_rejects_existing_corrupt_model_bytes(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_sherpa_test", "scripts/prepare_sherpa_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    directory = tmp_path / "downloads"
    directory.mkdir()
    (directory / "model.onnx").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.fetch(("models", "model.onnx", "0" * 64))
    assert (directory / "model.onnx").read_bytes() == b"corrupt"


def test_downloader_passes_digest_and_verifies_downloaded_bytes(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_sherpa_download_test", "scripts/prepare_sherpa_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    expected = hashlib.sha256(b"model bytes").hexdigest()
    def transport(url, destination, size, expected_sha):
        assert expected_sha == expected
        destination.write_bytes(b"model bytes")
    monkeypatch.setattr(module, "download", transport)
    result = module.fetch(("models", "model.onnx", expected))
    assert result["sha256"] == expected
    assert (tmp_path / "downloads/model.onnx").read_bytes() == b"model bytes"


def test_preparation_rerun_preserves_live_model_files_and_rejects_changed_bundle(tmp_path, monkeypatch):
    import importlib.util
    import tarfile
    spec = importlib.util.spec_from_file_location("prepare_sherpa_bundle_test", "scripts/prepare_sherpa_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(module, "MODEL_ROOT", root)
    archive = tmp_path / "bundle.tar.bz2"
    with tarfile.open(archive, "w:bz2") as writer:
        member = tarfile.TarInfo("bundle/model.onnx")
        member.size = len(b"verified bytes")
        writer.addfile(member, io.BytesIO(b"verified bytes"))
    module.extract_verified(archive, "bundle")
    path = root / "bundle/model.onnx"
    inode = path.stat().st_ino
    module.extract_verified(archive, "bundle")
    assert path.stat().st_ino == inode and path.read_bytes() == b"verified bytes"
    path.write_bytes(b"administrator changed the model")
    with pytest.raises(ValueError, match="differs from pinned release"):
        module.extract_verified(archive, "bundle")
    assert path.read_bytes() == b"administrator changed the model"


def test_preparation_rejects_archive_with_unexpected_destination(tmp_path, monkeypatch):
    import importlib.util
    import tarfile
    spec = importlib.util.spec_from_file_location("prepare_sherpa_archive_test", "scripts/prepare_sherpa_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path)
    archive = tmp_path / "bad.tar.bz2"
    with tarfile.open(archive, "w:bz2") as writer:
        member = tarfile.TarInfo("../outside.onnx")
        member.size = 1
        writer.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unexpected model directory"):
        module.extract_verified(archive, "bundle")


def test_asr_output_removes_runtime_language_tokens():
    plugin = SherpaSpeechTask(config())
    assert plugin.finish({"text": "你好", "language": "<|zh|>"}, {})["language"] == "zh"
    with pytest.raises(ServiceError, match="language"):
        plugin.finish({"text": "hello", "language": 1}, {})
