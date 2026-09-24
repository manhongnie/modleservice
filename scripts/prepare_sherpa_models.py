#!/usr/bin/env python3
"""Fetch pinned sherpa models; verify hashes, prepare real references and smoke tests.

Run from the repository root. Existing downloads are hashed before reuse.
No code from model repositories is imported. --smoke uses separate processes per
model so native allocations are released between the four model families.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_models import download
MODEL_ROOT = ROOT / "models/sherpa"
URL_ROOT = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
# Release asset digests, fixed in source. ERes2Net uses the publisher's Hugging Face LFS SHA256 because its GitHub asset predates the digest field.
ASSETS = [
    ("huggingface", "sensevoice-model.int8.onnx", "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
    ("tts-models", "matcha-icefall-zh-en.tar.bz2", "271b804af570400d3bcdcb53bf6e53cc9f75180ee763b9f13eb5eaf2b0d086ef"),
    ("tts-models", "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2", "77219c8b40f4ee8d73a7f902305ff6c1128ef9b54461c41b4ca6ed890b6c2803"),
    ("speaker-recongition-models", "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx", "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"),
    ("vocoder-models", "vocos-16khz-univ.onnx", "b599142a1fb8ff03de3e84ac35ff537c619e56f4267a6fe894851a42844acf9e"),
    ("vocoder-models", "vocos_24khz.onnx", "bcb3b970e384161c4d634f0bb9e999ff1c471b34c9bc0b1049a5014065ed3cc0"),
    ("huggingface", "sensevoice-tokens.txt", "f449eb28dc567533d7fa59be34e2abca8784f771850c78a47fb731a31429a1dc"),
]
MIRRORS = {
    "sensevoice-model.int8.onnx": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/2365baeacb507f821a0c8120fcee3d484dba7a07/model.int8.onnx",
    "sensevoice-tokens.txt": "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/2365baeacb507f821a0c8120fcee3d484dba7a07/tokens.txt",
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx": "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "vocos-16khz-univ.onnx": "https://huggingface.co/csukuangfj/sherpa-onnx-vocoders/resolve/ba83e216a24b2b756cccd01291cbd9953ba46079/vocos-16khz-univ.onnx",
}
ASSET_SIZES = dict(zip((asset[1] for asset in ASSETS), (239233841, 79033838, 109162785, 39593761, 53882848, 54157409, 315894)))
NAMES = {
    "asr": ("sensevoice-small-int8", "2024-07-17-c71f0ce00bec", "sherpa_sensevoice", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17", 768, 256),
    "tts": ("matcha-zh-en", "271b804af570", "sherpa_matcha", "matcha-icefall-zh-en", 512, 256),
    "speaker_embeddings": ("eres2net-speaker-zh", "3dspeaker-16k", "sherpa_speaker", "3dspeaker-eres2net", 256, 128),
    "voice_clone": ("zipvoice-distill-int8", "77219c8b40f4", "sherpa_zipvoice", "sherpa-onnx-zipvoice-distill-int8-zh-en-emilia", 1024, 1024),
}
REFERENCE_TEXT = "你好，欢迎使用本地语音服务。"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def extract_verified(archive_path, folder_name):
    target = MODEL_ROOT / folder_name
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        if any(Path(member.name).parts[0] != folder_name for member in members):
            raise ValueError("Archive contains an unexpected model directory")
        if target.exists():
            # A preparation rerun must never rewrite files used by active workers.
            for member in members:
                if member.isfile():
                    existing = (MODEL_ROOT / member.name).resolve()
                    if not existing.is_relative_to(target.resolve()) or not existing.is_file():
                        raise ValueError("Existing model bundle is incomplete or escaped its directory")
                    with archive.extractfile(member) as source:
                        expected = hashlib.file_digest(source, "sha256").hexdigest()
                    if digest(existing) != expected:
                        raise ValueError("Existing model bundle differs from pinned release; use a new version directory")
            return
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=MODEL_ROOT) as temporary:
            archive.extractall(temporary, filter="data")
            (Path(temporary) / folder_name).rename(target)


def fetch(asset):
    tag, name, expected = asset
    downloads = MODEL_ROOT / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    destination = downloads / name
    upstream_url = MIRRORS[name] if tag == "huggingface" else URL_ROOT + tag + "/" + name
    url = MIRRORS.get(name, upstream_url)
    if not destination.is_file():
        print("Downloading", name, flush=True)
        for attempt in range(3):
            try:
                download(url, destination, ASSET_SIZES.get(name), expected)
                break
            except OSError:
                if attempt == 2:
                    raise
                # Completed byte ranges survive; retry only the sparse holes.
                time.sleep(2)
    actual = digest(destination)
    if actual != expected:
        raise ValueError(f"Existing download checksum mismatch: {name}")
    if name == "sensevoice-tokens.txt":
        raw = destination.read_bytes()
        if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != "2cfc92fc2ff26aaa690b7c01fd96b41109413881":
            raise ValueError("SenseVoice tokens do not match publisher Git object")
    if name.endswith(".tar.bz2"):
        extract_verified(destination, name.removesuffix(".tar.bz2"))
    return {"url": url, "upstream_url": upstream_url, "file": str(destination.relative_to(ROOT)), "sha256": actual,
            "publisher_digest_verified": True}


def install_assets(download=True):
    if download:
        with ThreadPoolExecutor(max_workers=3) as pool:
            manifest = list(pool.map(fetch, ASSETS))
        (MODEL_ROOT / "downloads/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    downloads = MODEL_ROOT / "downloads"
    for _, name, expected in ASSETS:
        if not (downloads / name).is_file() or digest(downloads / name) != expected:
            raise ValueError(f"Download integrity check failed: {name}")
    destinations = [
        ("sensevoice-model.int8.onnx", MODEL_ROOT / NAMES["asr"][3] / "model.int8.onnx"),
        ("sensevoice-tokens.txt", MODEL_ROOT / NAMES["asr"][3] / "tokens.txt"),
        ("vocos-16khz-univ.onnx", MODEL_ROOT / NAMES["tts"][3] / "vocos-16khz-univ.onnx"),
        ("vocos_24khz.onnx", MODEL_ROOT / NAMES["voice_clone"][3] / "vocos_24khz.onnx"),
        (ASSETS[3][1], MODEL_ROOT / NAMES["speaker_embeddings"][3] / "model.onnx"),
    ]
    for name, destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != digest(downloads / name):
                raise ValueError(f"Existing model asset differs from pinned source: {destination}; use a new version directory")
        else:
            shutil.copyfile(downloads / name, destination)
    # Record every runtime file to make native model deployments auditable.
    for _, _, _, directory, _, _ in NAMES.values():
        folder = MODEL_ROOT / directory
        manifest = {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "sha256.json"}
        (folder / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


def model_config(capability, payload):
    name, version, task, directory, resident, temporary = NAMES[capability]
    options = {"threads": 2}
    if capability in {"asr", "speaker_embeddings"}:
        options["max_audio_seconds"] = 30
    if capability in {"tts", "voice_clone"}:
        options["max_text_chars"] = 200
    if capability == "voice_clone":
        options.update(max_reference_seconds=15, max_steps=8)
    return {"name": name, "version": version, "capabilities": [capability], "task": task, "backend": "sherpa_onnx",
            "path": str((MODEL_ROOT / directory).relative_to(ROOT)), "device": "CPU", "concurrency": 1,
            "resident_mb": resident, "request_mb": temporary, "max_input_bytes": 2097152,
            "load_policy": "on_demand", "idle_seconds": 60, "options": options, "validation_input": payload}


def run_model(config, output_path=None):
    import resource
    from threading import Event
    from model_service.contracts import ModelConfig
    from model_service.backends.sherpa_models import SherpaONNXBackend
    from model_service.tasks.sherpa_tasks import SherpaSpeechTask
    cfg = ModelConfig(**config)
    plugin, backend = SherpaSpeechTask(cfg), SherpaONNXBackend(cfg)
    plugin.validate(cfg.validation_input, cfg.capabilities[0])
    started = time.monotonic()
    metadata = backend.load()
    loaded = time.monotonic()
    try:
        prepared = plugin.prepare(cfg.validation_input)
        output = plugin.finish(backend.infer(prepared.inputs, Event()), prepared.context)
        finished = time.monotonic()
        if output_path and "audio_base64" in output:
            Path(output_path).write_bytes(base64.b64decode(output["audio_base64"]))
        summary = {key: value for key, value in output.items() if key not in {"audio_base64", "embedding"}}
        report = {"model_id": cfg.model_id, "real_model": True, "metadata": metadata,
                  "load_seconds": loaded - started, "inference_seconds": finished - loaded,
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "output": summary}
        for field in ("audio_base64", "compare_audio_base64", "reference_audio_base64"):
            if field in cfg.validation_input:
                report[field.replace("base64", "sha256")] = hashlib.sha256(base64.b64decode(cfg.validation_input[field])).hexdigest()
        if "audio_base64" in output:
            import io, wave
            import numpy as np
            raw = base64.b64decode(output["audio_base64"])
            with wave.open(io.BytesIO(raw)) as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            report["audio_sha256"] = hashlib.sha256(raw).hexdigest()
            report["rms"] = float(np.sqrt(np.mean(samples ** 2)))
            report["peak_amplitude"] = float(np.max(np.abs(samples)))
            if report["rms"] <= 1e-6:
                raise ValueError("Model produced effectively silent audio")
        if "embedding" in output:
            report["embedding_sha256"] = hashlib.sha256(json.dumps(output["embedding"]).encode()).hexdigest()
        return report
    finally:
        backend.close()


def subprocess_model(config, label, output_path=None):
    directory = ROOT / "var/sherpa-validation"
    directory.mkdir(parents=True, exist_ok=True)
    cfg_path, report_path = directory / f"{label}.config.json", directory / f"{label}.report.json"
    cfg_path.write_text(json.dumps(config, ensure_ascii=False))
    command = [sys.executable, __file__, "--worker", str(cfg_path), "--report", str(report_path)]
    if output_path:
        command += ["--audio-output", str(output_path)]
    subprocess.run(command, cwd=ROOT, check=True, timeout=300)
    return json.loads(report_path.read_text())


def character_error_rate(expected, actual):
    expected = "".join(character for character in expected if character.isalnum())
    actual = "".join(character for character in actual if character.isalnum())
    previous = list(range(len(actual) + 1))
    for i, left in enumerate(expected, 1):
        current = [i]
        for j, right in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / max(1, len(expected))


def prepare_examples_and_smoke(smoke=False):
    directory = ROOT / "var/sherpa-validation"
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "matcha-reference.wav"
    tts = model_config("tts", {"text": REFERENCE_TEXT})
    reports = []
    if not reference.is_file():
        reports.append(subprocess_model(tts, "tts", reference))
    elif smoke:
        # Preserve the registered validation payload across repeated smoke runs.
        reports.append(subprocess_model(tts, "tts", directory / "matcha-smoke.wav"))
    reference_base64 = base64.b64encode(reference.read_bytes()).decode()
    configs = [tts,
               model_config("asr", {"audio_base64": reference_base64}),
               model_config("speaker_embeddings", {"audio_base64": reference_base64, "compare_audio_base64": reference_base64}),
               model_config("voice_clone", {"text": "这是一段克隆语音测试。", "reference_audio_base64": reference_base64,
                                            "reference_text": REFERENCE_TEXT, "num_steps": 4})]
    (ROOT / "examples/models.sherpa.json").write_text(json.dumps(configs, ensure_ascii=False, indent=2) + "\n")
    if smoke:
        for config in configs[1:]:
            capability = config["capabilities"][0]
            output = directory / "zipvoice-clone.wav" if capability == "voice_clone" else None
            reports.append(subprocess_model(config, capability, output))
        clone_base64 = base64.b64encode((directory / "zipvoice-clone.wav").read_bytes()).decode()
        clone_asr = subprocess_model(model_config("asr", {"audio_base64": clone_base64}), "clone_asr")
        clone_speaker = subprocess_model(model_config("speaker_embeddings", {"audio_base64": reference_base64,
            "compare_audio_base64": clone_base64}), "clone_speaker")
        report = {"tested_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "scope": "Real sherpa-onnx CPU inference through production task and backend, separate process per model; not HTTP or perceptual quality certification",
                  "reference_source": "Saved Matcha synthetic reference voice generated locally, not a recording of a real person; reference stays stable between runs",
                  "reference_audio_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                  "models": reports,
                  "roundtrip_checks": {"tts_expected_text": REFERENCE_TEXT,
                      "clone_expected_text": configs[-1]["validation_input"]["text"],
                      "clone_asr": clone_asr, "reference_clone_similarity": clone_speaker,
                      "tts_asr_character_error_rate": character_error_rate(REFERENCE_TEXT, reports[1]["output"]["text"]),
                      "clone_asr_character_error_rate": character_error_rate(configs[-1]["validation_input"]["text"], clone_asr["output"]["text"]),
                      "comparison_note": "Punctuation removed; one synthetic Chinese phrase per model; not a benchmark or human listening score"}}
        (ROOT / "docs/sherpa-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-download", action="store_true", help="Reuse already checked assets")
    parser.add_argument("--smoke", action="store_true", help="Run all four real models sequentially")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--audio-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        result = run_model(json.loads(args.worker.read_text()), args.audio_output)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return
    install_assets(not args.no_download)
    prepare_examples_and_smoke(args.smoke)


if __name__ == "__main__":
    main()
