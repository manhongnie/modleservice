"""Bounded media task contracts; model runtimes are imported only in workers."""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile

from model_service.contracts import Prepared, ServiceError


# Run this tiny launcher in a fresh interpreter, then replace it with ffmpeg.
# preexec_fn would be unsafe inside the worker's multi-threaded native runtime.
_ENCODER_LAUNCHER = """import ctypes, os, signal, sys
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    raise RuntimeError("Cannot install encoder parent-death signal")
if os.getppid() != int(sys.argv[1]):
    os._exit(125)
os.execv(sys.argv[2], sys.argv[2:])
"""


def _encoder_command(executable: str, arguments: list[str]) -> list[str]:
    if sys.platform.startswith("linux"):
        return [sys.executable, "-c", _ENCODER_LAUNCHER, str(os.getpid()), executable, *arguments]
    return [executable, *arguments]


def _invalid(message):
    raise ServiceError("invalid_input", message, 422)


class ImageGenerationTask:
    capability = "image_generation"
    video = False

    def __init__(self, config):
        self.config = config

    def validate(self, payload, capability):
        if capability != self.capability:
            raise ServiceError("unsupported_capability", f"This task supports {self.capability} only", 422)
        if not isinstance(payload, dict):
            _invalid("Media generation input must be an object")
        allowed = {"prompt", "width", "height", "steps", "seed"}
        if self.video:
            allowed |= {"frames", "fps"}
        if set(payload) - allowed:
            _invalid("Unknown media generation input field")
        prompt = payload.get("prompt")
        limit = int(self.config.options.get("max_prompt_chars", 500))
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > limit:
            _invalid(f"prompt must contain 1..{limit} characters")
        default = 256 if self.video else 512
        for dimension in ("width", "height"):
            maximum = int(self.config.options.get("max_" + dimension, default))
            fixed = int(self.config.options.get("fixed_" + dimension, default))
            value = payload.get(dimension, fixed)
            if (not self.video or "fixed_" + dimension in self.config.options) and value != fixed:
                _invalid(f"{dimension} must equal this model instance fixed size {fixed}")
            if type(value) is not int or value < 128 or value > maximum or value % 64:
                _invalid(f"{dimension} must be a multiple of 64 between 128 and {maximum}")
        steps = payload.get("steps", 4 if self.video else 1)
        if type(steps) is not int or (steps != 4 if self.video else not 1 <= steps <= 4):
            _invalid("AnimateDiff-Lightning uses exactly 4 steps" if self.video else "SD-Turbo steps must be 1..4")
        seed = payload.get("seed", 0)
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            _invalid("seed must be an integer between 0 and 2147483647")
        if self.video:
            frames = payload.get("frames", self.config.options.get("default_frames", 8))
            fps = payload.get("fps", 8)
            maximum = int(self.config.options.get("max_frames", 16))
            if type(frames) is not int or not 4 <= frames <= maximum:
                _invalid(f"frames must be an integer between 4 and {maximum}")
            if type(fps) is not int or not 1 <= fps <= 16:
                _invalid("fps must be an integer between 1 and 16")
            pixels = payload.get("width", self.config.options.get("fixed_width", default)) * payload.get("height", self.config.options.get("fixed_height", default)) * frames
            if pixels > int(self.config.options.get("max_video_pixels", 256 * 256 * 16)):
                _invalid("The requested width, height and frames exceed the model video budget")

    def prepare(self, payload):
        self.validate(payload, self.capability)
        default = 256 if self.video else 512
        inputs = {"prompt": payload["prompt"].strip(),
                  "width": payload.get("width", int(self.config.options.get("fixed_width", default))),
                  "height": payload.get("height", int(self.config.options.get("fixed_height", default))), "steps": payload.get("steps", 4 if self.video else 1),
                  "seed": payload.get("seed", 0)}
        if self.video:
            inputs.update(frames=payload.get("frames", self.config.options.get("default_frames", 8)), fps=payload.get("fps", 8))
        return Prepared(inputs, {name: value for name, value in inputs.items() if name != "prompt"})

    def finish(self, outputs, context):
        import numpy as np
        from PIL import Image
        images = outputs.get("images")
        if images is None or len(images) != 1:
            raise ServiceError("invalid_model_output", "Image backend must return exactly one image", 502)
        array = np.asarray(images[0])
        if array.shape != (context["height"], context["width"], 3) or array.dtype != np.uint8:
            raise ServiceError("invalid_model_output", "Image backend returned an unexpected image shape or type", 502)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return {"image_base64": base64.b64encode(buffer.getvalue()).decode(), "format": "png",
                "width": context["width"], "height": context["height"], "seed": context["seed"]}


class VideoGenerationTask(ImageGenerationTask):
    capability = "video_generation"
    video = True

    def finish(self, outputs, context):
        import numpy as np
        encoder = shutil.which("ffmpeg")
        if encoder is None:
            raise ServiceError("missing_dependency", "Video encoding requires the ffmpeg system executable", 503)
        frames = outputs.get("frames")
        if frames is None or len(frames) != context["frames"]:
            raise ServiceError("invalid_model_output", "Video backend returned an unexpected frame count", 502)
        arrays = [np.asarray(frame) for frame in frames]
        expected = (context["height"], context["width"], 3)
        if any(array.shape != expected or array.dtype != np.uint8 for array in arrays):
            raise ServiceError("invalid_model_output", "Video backend returned an unexpected frame shape or type", 502)
        # The encoder owns a temporary file. No business-supplied path is used;
        # successful and failed requests both remove the file before completion.
        with tempfile.TemporaryDirectory(prefix="model-video-") as directory:
            path = Path(directory) / "result.mp4"
            command = _encoder_command(encoder, ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                       "-pixel_format", "rgb24", "-video_size", f"{context['width']}x{context['height']}",
                       "-framerate", str(context["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)])
            try:
                # subprocess.run kills and waits for the encoder on timeout. The
                # request remains pinned until this postprocessing has returned.
                subprocess.run(command, input=b"".join(array.tobytes() for array in arrays),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=60)
            except (subprocess.SubprocessError, OSError) as error:
                raise ServiceError("media_encoding_failed", "Video encoding did not complete", 502) from error
            data = path.read_bytes()
        return {"video_base64": base64.b64encode(data).decode(), "format": "mp4",
                "width": context["width"], "height": context["height"],
                "frames": context["frames"], "fps": context["fps"],
                "duration_s": context["frames"] / context["fps"], "seed": context["seed"]}
