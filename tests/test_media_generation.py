"""Media boundary tests use synthetic pixels/pipelines; no model support is implied."""
from __future__ import annotations

import base64
from contextlib import nullcontext
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from model_service.backends.media_generation import (DiffusersVideoBackend, OpenVINOImageBackend,
                                                     validate_image_config, validate_video_config)
from model_service.contracts import ModelConfig, ServiceError
from model_service.tasks.media_generation import ImageGenerationTask, VideoGenerationTask


def config(video=False, **changes):
    values = {"name": "media-test", "version": "1", "capabilities": ["video_generation" if video else "image_generation"],
        "task": "text_to_video" if video else "text_to_image", "backend": "diffusers_video" if video else "openvino_image",
        "device": "CUDA" if video else "CPU", "gpu_resident_mb": 2048 if video else 0, "path": "models/synthetic-boundary-test",
        "validation_input": {"prompt": "test"},
        "options": {} if video else {"fixed_width": 128, "fixed_height": 128}}
    return ModelConfig(**(values | changes))


@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("change", [
    {"prompt": " "}, {"prompt": "a" * 501}, {"width": 0}, {"height": 257},
    {"width": True}, {"seed": -1}, {"seed": 1.2}, {"steps": 0}, {"file": "/tmp/output.png"},
])
def test_media_invalid_inputs_are_rejected_before_execution(video, change):
    task = VideoGenerationTask(config(True)) if video else ImageGenerationTask(config())
    with pytest.raises(ServiceError) as error:
        task.validate({"prompt": "a landscape", **change}, task.capability)
    assert error.value.code == "invalid_input"


@pytest.mark.parametrize("change", [{"frames": 0}, {"frames": 17}, {"frames": True}, {"fps": 0},
                                     {"fps": 17}, {"steps": 8}, {"width": 512}])
def test_video_temporal_and_resource_limits(change):
    task = VideoGenerationTask(config(True))
    with pytest.raises(ServiceError):
        task.validate({"prompt": "a waving tree", **change}, task.capability)


def test_total_video_pixel_budget_is_checked_separately():
    task = VideoGenerationTask(config(True, options={"max_width": 512, "max_height": 512}))
    with pytest.raises(ServiceError, match="video budget"):
        task.validate({"prompt": "test", "width": 512, "height": 512, "frames": 16}, task.capability)


def test_image_task_encodes_real_png_dimensions_and_keeps_prompt_out_of_context():
    task = ImageGenerationTask(config())
    prepared = task.prepare({"prompt": "  teapot  ", "width": 128, "height": 128, "seed": 42})
    assert prepared.inputs["prompt"] == "teapot"
    assert "prompt" not in prepared.context
    pixels = np.zeros((1,128,128,3), dtype=np.uint8)
    pixels[:,:,:,0] = 255
    result = task.finish({"images": pixels}, prepared.context)
    with Image.open(io.BytesIO(base64.b64decode(result["image_base64"]))) as decoded:
        assert decoded.size == (128,128)
        assert decoded.getpixel((0,0)) == (255,0,0)
    assert result["format"] == "png" and result["seed"] == 42


def test_media_output_contract_refuses_unexpected_pixel_shape():
    task = ImageGenerationTask(config())
    prepared = task.prepare({"prompt": "test", "width": 128, "height": 128})
    with pytest.raises(ServiceError) as error:
        task.finish({"images": np.zeros((1,64,64,3),dtype=np.uint8)}, prepared.context)
    assert error.value.code == "invalid_model_output"


def test_video_postprocessor_encodes_and_decodes_mp4_without_persistent_temp_files(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("System ffmpeg/ffprobe are required for real MP4 codec testing")
    task = VideoGenerationTask(config(True))
    prepared = task.prepare({"prompt": "test", "width": 128, "height": 128, "frames": 4, "fps": 4})
    # Explicit synthetic frames exercise serialization only, not temporal inference.
    frames = [np.full((128,128,3),value,dtype=np.uint8) for value in [0,80,160,240]]
    output = task.finish({"frames": frames}, prepared.context)
    assert output["frames"] == 4 and output["duration_s"] == 1
    data = base64.b64decode(output["video_base64"])
    assert b"ftyp" in data[:32]
    assert len(data) > 100
    path = tmp_path / "synthetic-codec-test.mp4"
    path.write_bytes(data)
    completed = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                               capture_output=True, check=True, timeout=10)
    stream, = json.loads(completed.stdout)["streams"]
    assert (stream["width"], stream["height"], stream["nb_frames"]) == (128, 128, "4")


def test_image_cancel_callback_is_observed_before_result_is_returned():
    backend, cancel = OpenVINOImageBackend(config()), Event()
    calls = []
    class SyntheticPipeline:
        def generate(self, prompt, **kwargs):
            cancel.set()
            calls.append(kwargs["callback"](0,1,None))
            return SimpleNamespace(data=np.zeros((1,128,128,3),dtype=np.uint8))
    backend.pipeline = SyntheticPipeline()
    backend.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: SimpleNamespace(input_ids=np.zeros((1,4))))
    with pytest.raises(ServiceError) as error:
        backend.infer({"prompt":"test", "width":128, "height":128, "steps":1, "seed":42}, cancel)
    assert error.value.code == "cancelled" and calls == [True]


def test_video_cancel_synchronizes_cuda_before_acknowledging_stop(monkeypatch):
    backend, cancel = DiffusersVideoBackend(config(True)), Event()
    events = []
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, Generator=Generator,
        cuda=SimpleNamespace(synchronize=lambda device: events.append("native_stopped"))))
    class Tokenizer:
        model_max_length = 77
        def __call__(self,*args,**kwargs): return {"input_ids":[1,2]}
    class SyntheticPipeline:
        tokenizer = Tokenizer()
        def __call__(self, **kwargs):
            cancel.set()
            kwargs["callback_on_step_end"](self,0,1,{"latents":None})
            raise AssertionError("callback should abort generation")
    backend.pipeline = SyntheticPipeline()
    with pytest.raises(ServiceError) as error:
        backend.infer({"prompt":"test", "width":128, "height":128, "frames":4,"seed":42},cancel)
    assert error.value.code == "cancelled"
    assert events == ["native_stopped"]


@pytest.mark.parametrize("video,changes", [(False,{"concurrency":2}), (True,{"device":"GPU"}),
    (False,{"options":{"max_width":129}}), (True,{"options":{"max_frames":128}})])
def test_media_configuration_constraints(video,changes):
    with pytest.raises(ServiceError):
        (validate_video_config if video else validate_image_config)(config(video,**changes))


def test_cuda_media_requires_explicit_gpu_budget():
    with pytest.raises(ServiceError, match="gpu_resident_mb"):
        validate_video_config(config(True, gpu_resident_mb=0))


@pytest.mark.parametrize("video", [False, True])
def test_media_bundle_rejects_symlink_to_external_component(tmp_path, video):
    outside = tmp_path / "external"
    outside.mkdir()
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "unet").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ServiceError) as error:
        (validate_video_config if video else validate_image_config)(config(video, path=str(root)))
    assert error.value.code == "path_forbidden"


def test_image_rejects_prompt_above_real_encoder_token_budget():
    backend = OpenVINOImageBackend(config())
    backend.tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: SimpleNamespace(input_ids=np.zeros((1,78))))
    with pytest.raises(ServiceError) as error:
        backend.infer({"prompt": "a long prompt"}, Event())
    assert error.value.code == "input_too_large"


def test_image_instance_accepts_only_its_fixed_shape_to_bound_native_cache():
    task = ImageGenerationTask(config(options={"fixed_width": 512, "fixed_height": 512}))
    assert task.prepare({"prompt": "test"}).inputs["width"] == 512
    with pytest.raises(ServiceError, match="fixed size"):
        task.prepare({"prompt": "test", "width": 256, "height": 256})


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux parent-death guarantee")
def test_encoder_stops_when_its_worker_process_dies(tmp_path):
    import psutil
    marker = tmp_path / "encoder.pid"
    child_code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(60)"
    parent_code = """import subprocess, sys, time
from model_service.tasks.media_generation import _encoder_command
subprocess.Popen(_encoder_command(sys.executable, ["-c", sys.argv[2], sys.argv[1]]))
time.sleep(60)
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(marker), child_code],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    child = None
    def running(process):
        try:
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), "Encoder launcher failed to start"
        child = psutil.Process(int(marker.read_text()))
        parent.terminate()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while running(child) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not running(child)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        parent.stderr.close()
        if child is not None and running(child):
            child.kill()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux parent-death guarantee")
def test_encoder_refuses_to_start_if_worker_died_before_guard_install(tmp_path):
    from model_service.tasks.media_generation import _ENCODER_LAUNCHER
    marker = tmp_path / "must-not-exist"
    command = [sys.executable, "-c", _ENCODER_LAUNCHER, "0", sys.executable, "-c",
               "import sys;open(sys.argv[1],'w').write('started')", str(marker)]
    result = subprocess.run(command, capture_output=True, timeout=5)
    assert result.returncode == 125
    assert not marker.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux parent-death guarantee")
def test_encoder_guard_install_failure_never_executes_encoder(tmp_path):
    import os
    from model_service.tasks.media_generation import _ENCODER_LAUNCHER
    marker = tmp_path / "must-not-exist"
    simulated_failure = "import ctypes;from types import SimpleNamespace;ctypes.CDLL=lambda *a,**k:SimpleNamespace(prctl=lambda *a:-1)\n"
    result = subprocess.run([sys.executable, "-c", simulated_failure + _ENCODER_LAUNCHER,
        str(os.getpid()), sys.executable, "-c", "import sys;open(sys.argv[1],'w').write('started')", str(marker)],
        capture_output=True, timeout=5)
    assert result.returncode != 0 and b"Cannot install encoder parent-death signal" in result.stderr
    assert not marker.exists()


def test_video_model_metadata_cannot_select_unregistered_python_components(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model_index.json").write_text(json.dumps({"unet": ["untrusted_module", "CustomClass"]}))
    with pytest.raises(ServiceError) as error:
        validate_video_config(config(True, path=str(tmp_path)))
    assert error.value.code == "invalid_model_metadata"


@pytest.mark.parametrize("variant", ["../../outside", {}, []])
def test_video_variant_cannot_smuggle_a_component_path(variant):
    with pytest.raises(ServiceError, match="weight_variant"):
        validate_video_config(config(True, options={"weight_variant": variant}))


@pytest.mark.parametrize("name", ["max_width", "threads", "max_prompt_chars", "max_frames"])
def test_video_numeric_limit_null_is_a_config_error(name):
    with pytest.raises(ServiceError) as error:
        validate_video_config(config(True, options={name: None}))
    assert error.value.code == "invalid_config"


def test_verified_video_configuration_defaults_to_bounded_official_shape():
    values = json.loads(Path("examples/models.media.json").read_text())[1]
    model = ModelConfig.model_validate(values)
    validate_video_config(model)
    task = VideoGenerationTask(model)
    prepared = task.prepare({"prompt": "A girl smiling"})
    assert (prepared.inputs["width"], prepared.inputs["height"], prepared.inputs["frames"]) == (512, 512, 16)
    assert prepared.inputs["width"] * prepared.inputs["height"] * prepared.inputs["frames"] == model.options["max_video_pixels"]
    with pytest.raises(ServiceError, match="fixed size"):
        task.prepare({"prompt": "test", "width": 256})


@pytest.mark.parametrize("options", [{"fixed_width": None}, {"fixed_height": 513},
    {"default_frames": None}, {"default_frames": 17}, {"max_video_pixels": 4194305}])
def test_video_native_cache_and_largest_request_limits_are_validated(options):
    with pytest.raises(ServiceError) as error:
        validate_video_config(config(True, options=options))
    assert error.value.code == "invalid_config"
