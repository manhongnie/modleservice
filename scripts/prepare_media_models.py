"""Download pinned upstream media weights; optionally export SD-Turbo to OpenVINO.

This is an explicit administrator action. Serving never downloads model artifacts.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import threading
import tempfile
import time

try:
    from scripts.download_models import download, read_url
except ModuleNotFoundError:
    from download_models import download, read_url

SOURCES = {
    "image": {"repo": "stabilityai/sd-turbo", "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
              "directory": "sd-turbo-source", "license": "stabilityai-community (see pinned LICENSE.md)"},
    "video_base": {"repo": "emilianJR/epiCRealism", "revision": "6522cf856b8c8e14638a0aaa7bd89b1b098aed17",
                   "directory": "animatediff-lightning-epic/base", "license": "creativeml-openrail-m",
                   "provenance": "Diffusers conversion referenced by the official AnimateDiff-Lightning usage example"},
    "video_motion": {"repo": "ByteDance/AnimateDiff-Lightning",
                     "revision": "027c893eec01df7330f5d4b733bc9485ee02e8b2",
                     "directory": "animatediff-lightning-epic", "license": "creativeml-openrail-m"},
}


def selected(name: str, filename: str) -> bool:
    if name == "video_motion":
        return filename in {"animatediff_lightning_4step_diffusers.safetensors", "LICENSE.md", "README.md"}
    if filename in {"model_index.json", "README.md", "LICENSE", "LICENSE.md"}:
        return True
    if filename.split("/", 1)[0] not in {"tokenizer", "scheduler", "text_encoder", "unet", "vae"}:
        return False
    weights_suffix = ".safetensors" if name == "video_base" else ".fp16.safetensors"
    return filename.endswith((".json", ".txt", weights_suffix))


def download_source(name: str, root: Path):
    source = SOURCES[name]
    repo, revision = source["repo"], source["revision"]
    target = root / source["directory"]
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(read_url(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
    if manifest["sha"] != revision:
        raise RuntimeError("Upstream revision does not match pinned media source")
    files = []
    for item in manifest["siblings"]:
        filename = item["rfilename"]
        if not selected(name, filename):
            continue
        path = (target / filename).resolve()
        if not path.is_relative_to(target.resolve()):
            raise ValueError("Upstream manifest escapes destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo} {filename} ({item.get('size', 0)} bytes)", flush=True)
        download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", path,
                        item.get("size"), item.get("lfs", {}).get("sha256"))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": filename, "size_bytes": path.stat().st_size, "sha256": digest})
    (target / "source.json").write_text(json.dumps({**source, "url": f"https://huggingface.co/{repo}", "files": files}, indent=2))
    return target


def _export_sd_turbo_component(source: Path, target: Path, component: str):
    """Export each component separately to keep peak CPU memory bounded."""
    import gc
    import openvino as ov
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from openvino_tokenizers import convert_tokenizer

    torch.set_num_threads(4)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model_index.json", "LICENSE.md", "README.md", "source.json"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    for config_folder in ("tokenizer", "scheduler"):
        shutil.copytree(source / config_folder, target / config_folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.lock", "*.partial"))

    def save(component, converted, config_source):
        folder = target / component
        folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="export-", dir=folder) as temporary:
            staged = Path(temporary)
            ov.save_model(converted, staged / "openvino_model.xml", compress_to_fp16=True)
            shutil.copy2(source / config_source / "config.json", staged / "config.json")
            # Publish XML last: its presence is the per-component completion flag.
            (staged / "openvino_model.bin").replace(folder / "openvino_model.bin")
            (staged / "config.json").replace(folder / "config.json")
            (staged / "openvino_model.xml").replace(folder / "openvino_model.xml")

    if component == "text_encoder" and not (target / "text_encoder/openvino_model.xml").exists():
        print("Exporting SD-Turbo CLIP text encoder", flush=True)
        model = CLIPTextModel.from_pretrained(source / "text_encoder", variant="fp16", dtype=torch.float32,
                                             local_files_only=True, attn_implementation="eager").eval()
        class TextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input_ids):
                return self.model(input_ids=input_ids, return_dict=False)[0]
        wrapper = TextEncoder(model)
        converted = ov.convert_model(wrapper, example_input=torch.ones((1, 77), dtype=torch.int64),
                                     input=[("input_ids", [1, 77])])
        converted.output(0).get_tensor().set_names({"last_hidden_state"})
        save("text_encoder", converted, "text_encoder")
        del converted, wrapper, model
        gc.collect()

    if component == "unet" and not (target / "unet/openvino_model.xml").exists():
        print("Exporting SD-Turbo UNet", flush=True)
        model = UNet2DConditionModel.from_pretrained(source / "unet", variant="fp16", torch_dtype=torch.float32,
                                                   local_files_only=True, use_safetensors=True).eval()
        class Denoiser(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, sample, timestep, encoder_hidden_states):
                return self.model(sample, timestep, encoder_hidden_states, return_dict=False)[0]
        wrapper = Denoiser(model)
        example = (torch.randn(1,4,32,32), torch.tensor([1], dtype=torch.int64), torch.randn(1,77,1024))
        converted = ov.convert_model(wrapper, example_input=example,
                                     input=[("sample", [1,4,-1,-1]), ("timestep", [1]), ("encoder_hidden_states", [1,77,1024])])
        for index, name in enumerate(("sample", "timestep", "encoder_hidden_states")):
            converted.input(index).get_tensor().set_names({name})
        converted.output(0).get_tensor().set_names({"out_sample"})
        save("unet", converted, "unet")
        del converted, wrapper, model, example
        gc.collect()

    if component == "vae_decoder" and not (target / "vae_decoder/openvino_model.xml").exists():
        print("Exporting SD-Turbo VAE decoder", flush=True)
        model = AutoencoderKL.from_pretrained(source / "vae", variant="fp16", torch_dtype=torch.float32,
                                              local_files_only=True, use_safetensors=True).eval()
        class Decoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, latent_sample):
                return self.model.decode(latent_sample, return_dict=False)[0]
        wrapper = Decoder(model)
        converted = ov.convert_model(wrapper, example_input=torch.randn(1,4,32,32),
                                     input=[("latent_sample", [1,4,-1,-1])])
        converted.output(0).get_tensor().set_names({"sample"})
        save("vae_decoder", converted, "vae")
        del converted, wrapper, model
        gc.collect()

    if component == "tokenizer" and not (target / "tokenizer/openvino_tokenizer.xml").exists():
        print("Exporting SD-Turbo tokenizer", flush=True)
        tokenizer = CLIPTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
        converted = convert_tokenizer(tokenizer, with_detokenizer=False)
        ov.save_model(converted, target / "tokenizer/openvino_tokenizer.xml")
    completed = (target / "tokenizer/openvino_tokenizer.xml").exists() and all(
        (target / folder / "openvino_model.xml").exists() for folder in ("text_encoder", "unet", "vae_decoder"))
    if component != "tokenizer" or not completed:
        return
    (target / "NOTICE.txt").write_text("This Stability AI Model is licensed under the Stability AI Community License, "
        "Copyright © Stability AI Ltd. All Rights Reserved.\nPowered by Stability AI.\n"
        "Modified: original SD-Turbo checkpoint exported to OpenVINO IR with fp16 weight storage; no retraining.\n")
    files = []
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name != "export.json":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": str(path.relative_to(target)), "sha256": digest, "size_bytes": path.stat().st_size})
    (target / "export.json").write_text(json.dumps({"source": SOURCES["image"], "openvino": ov.__version__,
          "format": "OpenVINO IR fp16 weights", "components": ["text_encoder", "unet", "vae_decoder", "tokenizer"], "files": files}, indent=2))


def export_sd_turbo(source: Path, target: Path):
    # A fresh process per component releases native conversion caches as well as
    # Python tensors before the next component starts on small-memory hosts.
    for component in ("text_encoder", "unet", "vae_decoder", "tokenizer"):
        child = multiprocessing.get_context("spawn").Process(
            target=_export_sd_turbo_component, args=(source, target, component))
        child.start()
        child.join()
        if child.exitcode:
            raise RuntimeError(f"SD-Turbo {component} OpenVINO export failed")


def _verify_one(name: str, root: str, video_device: str | None = None, video_frames: int | None = None):
    """Run one real backend in a fresh process and retain its generated artifact."""
    from datetime import datetime, timezone
    import psutil
    import gc
    import numpy as np
    from model_service.backends.media_generation import OpenVINOImageBackend, DiffusersVideoBackend
    from model_service.contracts import ModelConfig, ServiceError
    from model_service.tasks.media_generation import ImageGenerationTask, VideoGenerationTask

    configs = json.loads(Path("examples/models.media.json").read_text())
    values = configs[0 if name == "image" else 1]
    values["path"] = str(Path(root) / ("sd-turbo-ov" if name == "image" else "animatediff-lightning-epic"))
    if name == "video" and video_device:
        values["device"] = video_device
        if video_device == "CPU":
            values.update(gpu_resident_mb=0, gpu_request_mb=0)
    config = ModelConfig.model_validate(values)
    gpu = name == "video" and config.device.upper().startswith("CUDA")
    task = ImageGenerationTask(config) if name == "image" else VideoGenerationTask(config)
    backend = OpenVINOImageBackend(config) if name == "image" else DiffusersVideoBackend(config)
    payload = dict(config.validation_input)
    if name == "image":
        payload.update(width=512, height=512)
    elif video_frames is not None:
        payload["frames"] = video_frames
    prepared = task.prepare(payload)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    finished = threading.Event()
    def sample_memory():
        while not finished.wait(.05):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    record = {"status": "failed", "tested_at": datetime.now(timezone.utc).isoformat(),
              "model_id": config.model_id, "device": config.device, "input": payload,
              "test_scope": "real local backend and task postprocessing; service HTTP is validated separately"}
    try:
        began = time.monotonic()
        info = backend.load()
        record["load_seconds"] = round(time.monotonic() - began, 3)
        record["loaded_rss_mb"] = round(process.memory_info().rss / 1048576, 3)
        record["backend"] = info
        if gpu:
            import torch
            torch.cuda.reset_peak_memory_stats()
            record["loaded_gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1048576, 3)
        began = time.monotonic()
        raw = backend.infer(prepared.inputs, threading.Event())
        record["inference_seconds"] = round(time.monotonic() - began, 3)
        output = task.finish(raw, prepared.context)
        key = "image_base64" if name == "image" else "video_base64"
        data = base64.b64decode(output[key])
        artifact = Path("var/media") / ("sd-turbo.png" if name == "image" else "animatediff-lightning.mp4")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(data)
        record.update(status="needs_visual_review", runtime_checks="passed", artifact=str(artifact), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest(), output={key: value for key, value in output.items() if not key.endswith("_base64")})
        pixels = raw["images"] if name == "image" else raw["frames"]
        arrays = [np.asarray(frame) for frame in pixels]
        record["pixel_stddev"] = round(float(np.std(arrays[0])), 4)
        if record["pixel_stddev"] <= 1:
            raise RuntimeError("Generated artifact has nearly constant pixels")
        if name == "video":
            record["distinct_generated_frames"] = len({hashlib.sha256(array.tobytes()).hexdigest() for array in arrays})
            record["adjacent_frame_mean_absolute_difference"] = round(float(np.mean([
                np.abs(first.astype(np.float32)-second.astype(np.float32)).mean()
                for first, second in zip(arrays, arrays[1:])])), 4)
            if record["distinct_generated_frames"] < 2:
                raise RuntimeError("Temporal pipeline returned identical frames")
            if gpu:
                record["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 3)
                record["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1048576, 3)
        gc.collect()
        record["rss_after_inference_mb"] = round(process.memory_info().rss / 1048576, 3)
        if name in {"image", "video"}:
            record["cold_peak_rss_mb"] = round(peak[0] / 1048576, 3)
            peak[0] = process.memory_info().rss
            began = time.monotonic()
            warm = backend.infer(prepared.inputs, threading.Event())
            record["warm_inference_seconds"] = round(time.monotonic() - began, 3)
            record["repeated_seed_same_pixels"] = bool(np.array_equal(raw["images"] if name == "image" else np.asarray(raw["frames"]), warm["images"] if name == "image" else np.asarray(warm["frames"])))
            del warm
            gc.collect()
            record["rss_after_warm_inference_mb"] = round(process.memory_info().rss / 1048576, 3)
            record["warm_peak_rss_mb"] = round(peak[0] / 1048576, 3)
            peak[0] = max(peak[0], int(record["cold_peak_rss_mb"] * 1048576))
        # The same real model must acknowledge cancellation only after its native
        # denoising step has stopped; the framework separately tests permit ownership.
        cancel = threading.Event()
        timer = threading.Timer(.05, cancel.set)
        cancelled_at = time.monotonic()
        timer.start()
        try:
            backend.infer(prepared.inputs, cancel)
            raise RuntimeError("Real media cancellation did not stop the request")
        except ServiceError as error:
            if error.code != "cancelled":
                raise
            record["real_cancellation"] = {"status": "passed", "error_code": error.code,
                "stop_acknowledged_seconds": round(time.monotonic() - cancelled_at, 3)}
        finally:
            timer.cancel()
            timer.join()
        gc.collect()
        record["rss_after_cancellation_mb"] = round(process.memory_info().rss / 1048576, 3)
        if gpu:
            record["peak_gpu_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1048576, 3)
            record["peak_gpu_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1048576, 3)
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        finished.set()
        sampler.join()
        record["peak_rss_mb"] = round(peak[0] / 1048576, 3)
        backend.close()
        report_path = Path("docs/media-validation.json")
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        record_key = "real_" + name + "_inference"
        if report.get(record_key):
            report.setdefault("previous_real_runs", []).append(report[record_key])
        report[record_key] = record
        report["status"] = "real_models_validated" if all((report.get("real_" + kind + "_inference") or {}).get("status") == "passed" for kind in ("image", "video")) else "partial_real_validation"
        report["note"] = "Per-model records distinguish real inference from synthetic boundary tests. Successful execution and nonconstant pixels do not prove useful generated content; inspect artifacts before marking visual_review/status passed."
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["image", "video"])
    parser.add_argument("--root", type=Path, default=Path("models/media"))
    parser.add_argument("--video-device", choices=["CPU", "CUDA"], help="Override video device for real verification only")
    parser.add_argument("--video-frames", type=int, choices=[4, 8, 16], help="Override frames for real video verification only")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Run real local inference and write var/media plus docs/media-validation.json")
    args = parser.parse_args()
    if args.verify_only:
        for name in args.models:
            child = multiprocessing.get_context("spawn").Process(target=_verify_one, args=(name, str(args.root), args.video_device, args.video_frames))
            child.start()
            child.join()
            if child.exitcode:
                raise RuntimeError(f"Real {name} model verification failed; inspect docs/media-validation.json")
        return
    if not args.export_only:
        for name in args.models:
            for source in (["image"] if name == "image" else ["video_base", "video_motion"]):
                download_source(source, args.root)
    if "image" in args.models and not args.download_only:
        export_sd_turbo(args.root / SOURCES["image"]["directory"], args.root / "sd-turbo-ov")


if __name__ == "__main__":
    main()
