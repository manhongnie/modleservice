"""Explicit administrator download step. Inference never downloads model artifacts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import urllib.request


HF_MODELS = {
    "qwen": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "qwen3.5-0.8b"),
    "asr": ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "whisper-tiny"),
}
TTS_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2"


def download(url, target, expected_size=None, expected_sha256=None):
    # Parallel invocations must never truncate a shared in-progress download.
    with target.with_suffix(target.suffix + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if expected_size is not None and target.stat().st_size != expected_size:
                raise IOError(f"Existing artifact has incorrect size: {target}")
            if expected_sha256:
                with target.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != expected_sha256:
                        raise IOError(f"Existing artifact SHA256 mismatch: {target}")
            return
        _download(url, target, expected_size, expected_sha256)


def _download(url, target, expected_size, expected_sha256):
    temporary = target.with_suffix(target.suffix + ".partial")
    if expected_size and expected_size > 16 * 1024 * 1024:
        # Resolve the signed artifact URL once, then fetch bounded ranges. Each
        # thread owns a disjoint file offset; only a verified file is published.
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            resolved_url = response.url
        chunk = 8 * 1024 * 1024
        resume = temporary.exists() and temporary.stat().st_size == expected_size and expected_sha256
        with temporary.open("r+b" if resume else "wb") as output:
            fd = output.fileno()
            if not resume:
                os.ftruncate(fd, expected_size)
            offsets = list(range(0, expected_size, chunk))
            if resume and hasattr(os, "SEEK_HOLE"):
                # A range is written only after receiving its complete body. Sparse
                # holes therefore identify unfinished ranges from an interrupted
                # parallel run. The full upstream SHA still gates publication.
                pending = set()
                position = 0
                while position < expected_size:
                    try:
                        hole = os.lseek(fd, position, os.SEEK_HOLE)
                        if hole >= expected_size:
                            break
                        try:
                            end = os.lseek(fd, hole, os.SEEK_DATA)
                        except OSError:
                            end = expected_size
                        pending.update(range(hole // chunk * chunk, end, chunk))
                        position = end
                    except OSError:
                        pending = set(offsets)
                        break
                offsets = sorted(pending)

            def part(begin):
                end = min(expected_size - 1, begin + chunk - 1)
                request = urllib.request.Request(resolved_url, headers={"Range": f"bytes={begin}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise IOError("Server did not honor bounded download range")
                    data = response.read(chunk + 1)
                    if len(data) != end - begin + 1:
                        raise IOError("Incomplete download range")
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], begin + offset)

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(part, offsets))
        with temporary.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise IOError("Downloaded model SHA256 did not match upstream manifest")
        temporary.replace(target)
        return
    request = urllib.request.Request(url, headers={"User-Agent": "shared-model-service/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
        declared = response.headers.get("Content-Length")
        if declared is not None and output.tell() != int(declared):
            raise IOError(f"Incomplete download of {target.name}")
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise IOError(f"Downloaded artifact has incorrect size: {target.name}")
    if expected_sha256:
        with temporary.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected_sha256:
                raise IOError(f"Downloaded artifact SHA256 mismatch: {target.name}")
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["qwen", "asr", "tts"])
    parser.add_argument("--root", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        if name == "tts":
            archive = args.root / "vits-icefall-zh-aishell3.tar.bz2"
            if not archive.exists():
                print("Downloading", TTS_URL, flush=True)
                download(TTS_URL, archive)
            with tarfile.open(archive) as source:
                if any(member.name.split("/")[0] != "vits-icefall-zh-aishell3" for member in source.getmembers()):
                    raise ValueError("TTS archive contains an unexpected target directory")
                source.extractall(args.root, filter="data")
            with (args.root / "vits-icefall-zh-aishell3/model.onnx").open("rb") as source:
                sha = hashlib.file_digest(source, "sha256").hexdigest()
            (args.root / "vits-icefall-zh-aishell3/source.json").write_text(json.dumps({"url": TTS_URL, "model_sha256": sha}, indent=2))
            continue
        repo, revision, folder = HF_MODELS[name]
        target = args.root / folder
        target.mkdir(exist_ok=True)
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true", timeout=30) as response:
            manifest = json.load(response)
        for entry in manifest["siblings"]:
            filename = entry["rfilename"]
            if not filename.endswith((".json", ".safetensors", ".txt", ".jinja")):
                continue
            path = (target / filename).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Remote manifest entry escaped model directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Preparing", repo, filename, flush=True)
            download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", path,
                     entry.get("size"), entry.get("lfs", {}).get("sha256"))
        hashes = {item["rfilename"]: item["lfs"]["sha256"] for item in manifest["siblings"] if item.get("lfs") and (target / item["rfilename"]).exists()}
        (target / "source.json").write_text(json.dumps({"repo": repo, "revision": revision, "sha256": hashes}, indent=2))


if __name__ == "__main__":
    main()
