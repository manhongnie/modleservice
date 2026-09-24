"""Explicit administrator download step. Inference never downloads model artifacts."""
from __future__ import annotations
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from threading import Lock
import time
import urllib.request


HF_MODELS = {
    "qwen": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "qwen3.5-0.8b"),
    "asr": ("openai/whisper-tiny", "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "whisper-tiny"),
}
TTS_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2"

_httpx_client = None
_httpx_client_lock = Lock()
_httpx_redirects = {}


def _pooled_client():
    """The optional transport owns one bounded, thread-safe IPv4 connection pool."""
    global _httpx_client
    with _httpx_client_lock:
        if _httpx_client is None:
            import httpx
            workers = max(1, min(16, int(os.environ.get('MODEL_DOWNLOAD_WORKERS', '4'))))
            limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers,
                                 keepalive_expiry=60)
            transport = httpx.HTTPTransport(local_address='0.0.0.0', limits=limits, trust_env=False)
            _httpx_client = httpx.Client(transport=transport, trust_env=False, follow_redirects=True,
                                         timeout=httpx.Timeout(30, connect=10, pool=10),
                                         headers={'Accept-Encoding': 'identity'})
            atexit.register(_httpx_client.close)
        return _httpx_client


def _read_url_httpx(url, *, head=False, byte_range=None, max_bytes=None):
    import httpx
    headers = {}
    expected = None
    if byte_range is not None:
        start, end = map(int, byte_range.split('-'))
        expected = end - start + 1
        if start < 0 or expected <= 0:
            raise ValueError('Invalid bounded download range')
        headers['Range'] = 'bytes=' + byte_range
    limit = min(expected, max_bytes) if expected is not None and max_bytes is not None else (expected if expected is not None else max_bytes)
    client = _pooled_client()
    for attempt in range(3):
        deadline = time.monotonic() + 120
        try:
            with _httpx_client_lock:
                destination = _httpx_redirects.get(url, url) if not head else url
            with client.stream('HEAD' if head else 'GET', destination, headers=headers) as response:
                response.raise_for_status()
                if head:
                    if str(response.url) != url:
                        with _httpx_client_lock:
                            _httpx_redirects[url] = str(response.url)
                    return str(response.url).encode()
                # Validate status and advertised bounds before reading any body.
                advertised = response.headers.get('Content-Length')
                if advertised and limit is not None and int(advertised) > limit:
                    raise IOError('Server exceeded bounded download length')
                if expected is not None:
                    if response.status_code != 206:
                        raise IOError('Server did not honor bounded download range')
                    if advertised and int(advertised) != expected:
                        raise IOError('Unexpected bounded download length')
                    span = response.headers.get('Content-Range', '')
                    parsed = re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)', span)
                    if (parsed is None or int(parsed[1]) != start or int(parsed[2]) != end
                            or (parsed[3] != '*' and int(parsed[3]) <= end)):
                        raise IOError('Unexpected Content-Range')
                # Headers have established a valid endpoint already. Preserve
                # it even if this response's body later hits a transient timeout.
                if str(response.url) != url:
                    with _httpx_client_lock:
                        current = _httpx_redirects.get(url)
                        if current is None or current == destination:
                            _httpx_redirects[url] = str(response.url)
                result = bytearray()
                for chunk in response.iter_raw():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Artifact read exceeded its deadline')
                    if limit is not None and len(result) + len(chunk) > limit:
                        raise IOError('Server exceeded bounded download length')
                    result.extend(chunk)
                if expected is not None and len(result) != expected:
                    raise IOError('Incomplete download range')
                return bytes(result)
        except (httpx.HTTPError, TimeoutError) as error:
            # Transient failures must not discard a known CDN endpoint for all
            # workers and send them through the slow origin again. Only an
            # authorization failure can indicate that its signed URL expired.
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in {401, 403}:
                with _httpx_client_lock:
                    if _httpx_redirects.get(url) == destination:
                        _httpx_redirects.pop(url, None)
            if attempt == 2:
                raise IOError('Artifact transport failed: ' + type(error).__name__) from error
            time.sleep(.5 * (attempt + 1))


def read_url(url: str, *, head: bool = False, byte_range: str | None = None,
             max_bytes: int | None = None) -> bytes:
    """Administrator-only IPv4 transport; curl verifies TLS and bounds retries/time."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError('max_bytes must be a nonnegative integer')
    transport = os.environ.get('MODEL_DOWNLOAD_TRANSPORT', 'curl')
    if transport == 'httpx':
        return _read_url_httpx(url, head=head, byte_range=byte_range, max_bytes=max_bytes)
    if transport != 'curl':
        raise ValueError('MODEL_DOWNLOAD_TRANSPORT must be curl or httpx')
    command = ["curl", "-4", "--fail", "--silent", "--show-error", "--location",
               "--connect-timeout", "10", "--max-time", "120"]
    if head:
        command += ["--head", "--output", os.devnull, "--write-out", "%{url_effective}"]
    elif byte_range is not None:
        command += ["--range", byte_range, "--write-out", "\n%{http_code}"]
    if byte_range is not None:
        start, end = map(int, byte_range.split('-'))
        command += ['--max-filesize', str(end - start + 1)]
    elif max_bytes is not None:
        command += ['--max-filesize', str(max(1, max_bytes))]
    # A fresh stdout buffer for every attempt: curl --retry can append a second
    # response to already-written stdout after an interrupted first response.
    for attempt in range(3):
        completed = subprocess.run([*command, url], capture_output=True, timeout=130)
        if completed.returncode == 0:
            break
        if attempt == 2:
            raise IOError('Artifact transport failed: ' + completed.stderr.decode(errors='replace')[-500:])
        time.sleep(.5 * (attempt + 1))
    body = completed.stdout
    if byte_range is None and max_bytes is not None and len(body) > max_bytes:
        raise IOError('Server exceeded bounded download length')
    if byte_range is not None:
        body, status = body.rsplit(b"\n", 1)
        if status != b"206":
            raise IOError("Server did not honor bounded download range")
    return body


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


def _quarantine_corrupt_partial(temporary, actual_sha256):
    # Keep evidence but remove the bad allocation map from resumable state.
    # In particular, a fully allocated corrupt file has no SEEK_HOLE ranges and
    # would otherwise fail on every restart without downloading any new bytes.
    destination = temporary.with_name(f'.{temporary.stem[:64]}.corrupt-{time.time_ns()}-{actual_sha256}')
    temporary.replace(destination)
    return destination.name


def _download(url, target, expected_size, expected_sha256):
    temporary = target.with_suffix(target.suffix + ".partial")
    if expected_size and expected_size > 16 * 1024 * 1024:
        # Resolve the signed artifact URL once, then fetch bounded ranges. Each
        # thread owns a disjoint file offset; only a verified file is published.
        resolved_url = read_url(url, head=True).decode().strip()
        if os.environ.get('MODEL_DOWNLOAD_TRANSPORT') == 'httpx':
            # HEAD has populated origin -> CDN. Keep origin as the request key
            # so a later 401/403 can obtain a fresh signed redirect.
            resolved_url = url
        chunk = max(1, min(32, int(os.environ.get('MODEL_DOWNLOAD_CHUNK_MB', '2')))) * 1024 * 1024
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
                data = read_url(resolved_url, byte_range=f"{begin}-{end}")
                if len(data) != end - begin + 1:
                    raise IOError("Incomplete download range")
                offset = 0
                while offset < len(data):
                    offset += os.pwrite(fd, data[offset:], begin + offset)

            if offsets and os.environ.get('MODEL_DOWNLOAD_TRANSPORT') == 'httpx':
                # Some providers redirect GET but not HEAD. Resolve that once
                # with a single byte before opening a pool of large GETs.
                read_url(resolved_url, byte_range=f'{offsets[0]}-{offsets[0]}')
            with ThreadPoolExecutor(max_workers=max(1, min(16, int(os.environ.get('MODEL_DOWNLOAD_WORKERS', '4'))))) as pool:
                pending = {pool.submit(part, begin) for begin in offsets}
                failures = []
                for completed in as_completed(pending):
                    pending.remove(completed)
                    try:
                        completed.result()
                    except OSError as error:
                        # One intermittent range failure must not cancel every
                        # later range and discard a working warm connection pool.
                        # Failed holes remain unpublished for the bounded retry.
                        failures.append(str(error))
                if failures:
                    raise IOError(failures[0])
        with temporary.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            isolated = _quarantine_corrupt_partial(temporary, actual)
            raise IOError(f'Downloaded model SHA256 did not match upstream manifest; isolated as {isolated}')
        temporary.replace(target)
        return
    temporary.write_bytes(read_url(url, max_bytes=expected_size))
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise IOError(f"Downloaded artifact has incorrect size: {target.name}")
    if expected_sha256:
        with temporary.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected_sha256:
            isolated = _quarantine_corrupt_partial(temporary, actual)
            raise IOError(f'Downloaded artifact SHA256 mismatch: {target.name}; isolated as {isolated}')
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
