"""Local image/video inference. Pipeline objects exist only inside model workers."""
from __future__ import annotations

import gc
import json
from pathlib import Path

from model_service.contracts import ServiceError


def _validate_media(config, expected, *, video=False):
    if config.capabilities != [expected] or config.concurrency != 1:
        raise ServiceError("invalid_config", f"Media generation requires capabilities=[{expected!r}] and concurrency=1", 422)
    for name, low, high in (("max_prompt_chars", 1, 1000), ("max_width", 128, 512),
                            ("max_height", 128, 512), ("threads", 1, 64)):
        value = config.options.get(name)
        if name in config.options and (type(value) is not int or not low <= value <= high):
            raise ServiceError("invalid_config", f"{name} must be an integer between {low} and {high}", 422)
    for name in ("max_width", "max_height"):
        if name in config.options and config.options[name] % 64:
            raise ServiceError("invalid_config", f"{name} must be a multiple of 64", 422)
    if video:
        for name, low, high in (("max_frames", 4, 16), ("max_video_pixels", 128*128*4, 512*512*16)):
            value = config.options.get(name)
            if name in config.options and (type(value) is not int or not low <= value <= high):
                raise ServiceError("invalid_config", f"{name} must be an integer between {low} and {high}", 422)


def _check_bundle(config):
    root = Path(config.path).resolve()
    for item in root.rglob("*"):
        if item.is_symlink() and not item.resolve().is_relative_to(root):
            raise ServiceError("path_forbidden", "Media model components must stay inside their model directory", 403)


def validate_image_config(config, settings=None):
    _validate_media(config, "image_generation")
    _check_bundle(config)
    for dimension in ("width", "height"):
        value = config.options.get("fixed_" + dimension, 512)
        if type(value) is not int or not 128 <= value <= config.options.get("max_" + dimension, 512) or value % 64:
            raise ServiceError("invalid_config", "Image fixed dimensions must be multiples of 64 inside configured limits", 422)
    if config.device.upper() != "CPU":
        raise ServiceError("unsupported_device", "This verified OpenVINO image configuration supports CPU only", 422)
    if config.task != "text_to_image":
        raise ServiceError("invalid_task_backend", "openvino_image requires text_to_image", 422)


def validate_video_config(config, settings=None):
    _validate_media(config, "video_generation", video=True)
    _check_bundle(config)
    for dimension in ("width", "height"):
        if "fixed_" + dimension in config.options:
            value = config.options["fixed_" + dimension]
            if type(value) is not int or not 128 <= value <= config.options.get("max_" + dimension, 256) or value % 64:
                raise ServiceError("invalid_config", "Video fixed dimensions must be multiples of 64 inside configured limits", 422)
    frames = config.options.get("default_frames", 8)
    if type(frames) is not int or not 4 <= frames <= config.options.get("max_frames", 16):
        raise ServiceError("invalid_config", "Video default_frames must be inside configured frame limits", 422)
    if config.device.upper().startswith("CUDA") and config.gpu_resident_mb <= 0:
        raise ServiceError("invalid_config", "CUDA video models require a positive gpu_resident_mb budget", 422)
    variant = config.options.get("weight_variant", "fp16")
    if variant is not None and variant != "fp16":
        raise ServiceError("invalid_config", "Video weight_variant must be fp16 or null for original safetensors", 422)
    if config.device.upper() not in {"CPU", "CUDA", "CUDA:0"}:
        raise ServiceError("unsupported_device", "Video device must be CPU or CUDA", 422)
    if config.task != "text_to_video":
        raise ServiceError("invalid_task_backend", "diffusers_video requires text_to_video", 422)
    index_path = Path(config.path) / "base/model_index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except (ValueError, OSError) as error:
            raise ServiceError("invalid_model_metadata", "Could not read video component metadata", 422) from error
        components = {"unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"], "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"]}
        if not isinstance(index, dict) or any(index.get(key) != value for key, value in components.items()):
            raise ServiceError("invalid_model_metadata", "Video base must use the registered SD1.5 component classes", 422)


def _check_cancel(cancel):
    if cancel.is_set():
        raise ServiceError("cancelled", "Media inference stopped after cancellation", 499)


class OpenVINOImageBackend:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.tokenizer = None
        self.max_prompt_tokens = 77

    def load(self):
        validate_image_config(self.config)
        try:
            import openvino_genai as genai
        except ImportError as error:
            raise ServiceError("missing_dependency", "Install matching OpenVINO and openvino-genai releases", 503) from error
        root = Path(self.config.path)
        self.max_prompt_tokens = int(json.loads((root / "text_encoder/config.json").read_text())["max_position_embeddings"])
        self.tokenizer = genai.Tokenizer(str(root / "tokenizer"))
        self.pipeline = genai.Text2ImagePipeline(self.config.path, "CPU",
            INFERENCE_NUM_THREADS=int(self.config.options.get("threads", 4)))
        return {"backend": "openvino_image", "device": "CPU", "management": "local",
                "pipeline": "SD-Turbo", "streaming": "buffered"}

    def infer(self, inputs, cancel):
        _check_cancel(cancel)
        tokens = self.tokenizer.encode(inputs["prompt"], truncation=False)
        if tokens.input_ids.shape[-1] > self.max_prompt_tokens:
            raise ServiceError("input_too_large", "Prompt exceeds the image text encoder token limit", 422)
        def on_step(step, total_steps, latents):
            return cancel.is_set()
        # Returning from generate confirms the native request is no longer running.
        images = self.pipeline.generate(inputs["prompt"], width=inputs["width"], height=inputs["height"],
            num_inference_steps=inputs["steps"], num_images_per_prompt=1, guidance_scale=0.0,
            rng_seed=inputs["seed"], callback=on_step)
        _check_cancel(cancel)
        return {"images": images.data.copy()}

    def close(self):
        self.pipeline = None
        self.tokenizer = None
        gc.collect()
        return {"released": True}


class DiffusersVideoBackend:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.device = "cpu" if config.device.upper() == "CPU" else "cuda:0"

    def load(self):
        validate_video_config(self.config)
        try:
            import torch
            from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
            from safetensors.torch import load_file
        except ImportError as error:
            raise ServiceError("missing_dependency", "Install CUDA PyTorch, diffusers, accelerate and safetensors", 503) from error
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ServiceError("unsupported_device", "CUDA video inference requires a working CUDA PyTorch installation", 503)
        torch.set_num_threads(int(self.config.options.get("threads", 4)))
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        root = Path(self.config.path)
        adapter = MotionAdapter().to(dtype=dtype)
        weights = load_file(root / "animatediff_lightning_4step_diffusers.safetensors", device="cpu")
        adapter.load_state_dict(weights)
        del weights
        # Supply fixed optional components directly, so model_index metadata cannot
        # select arbitrary Python module/class names through Diffusers auto-loading.
        scheduler_config = json.loads((root / "base/scheduler/scheduler_config.json").read_text())
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config, timestep_spacing="trailing", beta_schedule="linear")
        self.pipeline = AnimateDiffPipeline.from_pretrained(root / "base", motion_adapter=adapter,
            torch_dtype=dtype, variant=self.config.options.get("weight_variant", "fp16"), local_files_only=True, use_safetensors=True,
            scheduler=scheduler, feature_extractor=None, image_encoder=None)
        self.pipeline.vae.enable_slicing()
        self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        info = {"backend": "diffusers_video", "device": self.device, "management": "local",
                "pipeline": "AnimateDiff-Lightning-4step", "temporal_model": True, "streaming": "buffered"}
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
            info.update(gpu_allocated_mb=round(torch.cuda.memory_allocated(self.device) / 1048576, 3),
                        gpu_reserved_mb=round(torch.cuda.memory_reserved(self.device) / 1048576, 3))
        return info

    def infer(self, inputs, cancel):
        import torch
        _check_cancel(cancel)
        # Check the encoder's actual token budget rather than silently truncate.
        ids = self.pipeline.tokenizer(inputs["prompt"], truncation=False)["input_ids"]
        if len(ids) > self.pipeline.tokenizer.model_max_length:
            raise ServiceError("input_too_large", "Prompt exceeds the video text encoder token limit", 422)
        def on_step(pipeline, step, timestep, callback_kwargs):
            _check_cancel(cancel)
            return callback_kwargs
        try:
            with torch.inference_mode():
                output = self.pipeline(prompt=inputs["prompt"], width=inputs["width"], height=inputs["height"],
                    num_frames=inputs["frames"], num_inference_steps=4, guidance_scale=1.0,
                    generator=torch.Generator(device="cpu").manual_seed(inputs["seed"]),
                    callback_on_step_end=on_step, decode_chunk_size=1)
            _check_cancel(cancel)
            return {"frames": output.frames[0]}
        finally:
            # CUDA launches are asynchronous; stop is acknowledged only after all
            # launched kernels complete, including cancellation and error paths.
            if self.device.startswith("cuda"):
                torch.cuda.synchronize(self.device)

    def close(self):
        self.pipeline = None
        gc.collect()
        if self.device.startswith("cuda"):
            try:
                import torch
            except ImportError:
                return {"released": True}
            if torch.cuda.is_initialized():
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
        return {"released": True}
