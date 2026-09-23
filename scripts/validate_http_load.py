"""Controlled Mock overload/admission check over real HTTP, never a real-model benchmark.

Uses its own temporary development registry, credentials, free port and CLI process.
The 200-request / 16-client burst intentionally exceeds four execution slots plus
an eight-entry queue. Rejections are expected, are counted and are never retried.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import psutil


REQUESTS = 200
CONCURRENCY = 16
EXECUTIONS = 4
QUEUE_SIZE = 8


def process_running(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


async def exercise(base_url: str, business_key: str, admin_key: str) -> dict:
    samples: list[dict] = []
    latencies: list[float] = []
    statuses: Counter = Counter()
    errors: Counter = Counter()
    admin_headers = {"Authorization": "Bearer " + admin_key}
    business_headers = {"Authorization": "Bearer " + business_key}
    stop = asyncio.Event()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(base_url=base_url, timeout=15,
                                 limits=httpx.Limits(max_connections=CONCURRENCY + 4)) as client:
        config = {"name": "http-load-mock", "version": "1", "task": "mock", "backend": "mock",
                  "capabilities": ["embeddings"], "validation_input": {"texts": ["validation"]},
                  "concurrency": EXECUTIONS, "resident_mb": 64, "request_mb": 4,
                  "idle_seconds": 3600, "options": {"delay_s": 0.1, "load_delay_s": 0.02}}
        registered = await client.post("/admin/models", headers=admin_headers,
                                       json={"config": config, "enable": True})
        registered.raise_for_status()

        async def get_status() -> dict:
            response = await client.get("/admin/status", headers=admin_headers)
            response.raise_for_status()
            return response.json()

        async def monitor() -> None:
            while not stop.is_set():
                status = await get_status()
                ledger = status["scheduler"]
                samples.append({name: ledger[name] for name in (
                    "active_count", "queue_length", "temporary_mb", "reserved_mb", "resident_instances")})
                assert ledger["active_count"] <= EXECUTIONS, ledger
                assert ledger["queue_length"] <= QUEUE_SIZE, ledger
                assert ledger["temporary_mb"] <= EXECUTIONS * config["request_mb"], ledger
                assert ledger["reserved_mb"] <= 128, ledger
                assert ledger["resident_instances"] == 1, ledger
                await asyncio.sleep(0.005)

        async def request(index: int) -> None:
            async with semaphore:
                began = time.monotonic()
                response = await client.post("/v1/embeddings", headers=business_headers,
                                             json={"model": "http-load-mock@1",
                                                   "input": {"texts": [f"synthetic-{index}"]}})
                latencies.append(time.monotonic() - began)
                statuses[str(response.status_code)] += 1
                data = response.json()
                if response.status_code == 200:
                    assert data["mock"] is True and data["done"] is True, data
                else:
                    code = data["error"]["code"]
                    errors[code] += 1
                    assert (response.status_code, code) in {(429, "queue_full"), (504, "queue_timeout")}, data

        polling = asyncio.create_task(monitor())
        began = time.monotonic()
        try:
            await asyncio.gather(*(request(index) for index in range(REQUESTS)))
            elapsed = time.monotonic() - began
            deadline = time.monotonic() + 5
            while True:
                final = await get_status()
                ledger = final["scheduler"]
                if ledger["active_count"] == ledger["queue_length"] == ledger["temporary_mb"] == 0:
                    break
                assert time.monotonic() < deadline, ledger
                await asyncio.sleep(0.01)
        finally:
            stop.set()
            await polling

        assert sum(statuses.values()) == REQUESTS
        assert statuses["200"] > 0 and statuses["429"] > 0, statuses
        assert samples and max(item["active_count"] for item in samples) == EXECUTIONS, samples
        assert max(item["queue_length"] for item in samples) == QUEUE_SIZE, samples
        workers = list(final["workers"].values())
        assert len(workers) == 1 and workers[0]["alive"] and workers[0]["load_count"] == 1, workers
        assert not ledger["quarantined_requests"]
        ordered = sorted(latencies)

        def percentile(fraction: float) -> float:
            return round(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)] * 1000, 3)

        return {
            "scope": "Explicit Mock only: HTTP overload admission, bounded queue and resource cleanup; not real-model throughput or quality",
            "requested": REQUESTS, "client_concurrency": CONCURRENCY,
            "execution_capacity": EXECUTIONS, "queue_capacity": QUEUE_SIZE,
            "mock_execution_delay_s": 0.1, "duration_s": round(elapsed, 4),
            "http_status_counts": dict(sorted(statuses.items())), "error_code_counts": dict(sorted(errors.items())),
            "latency_ms_all_requests_including_rejections": {
                "p50": percentile(0.5), "p95": percentile(0.95), "max": round(max(ordered) * 1000, 3)},
            "sampling_interval_s": 0.005, "samples": len(samples),
            "observed_peaks": {name: max(sample[name] for sample in samples) for name in samples[0]},
            "scheduler_after_requests": ledger,
            "worker": {"pid": workers[0]["pid"], "load_count": workers[0]["load_count"],
                       "load_count_includes_registration_validation": True},
            "checks": {"bounded_active_and_queue": True, "bounded_budget": True,
                       "both_success_and_overload_rejection_observed": True,
                       "no_automatic_request_retries": True, "single_worker_load": True,
                       "active_queue_temporary_returned_to_zero": True},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/http-load-validation.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="model-http-load-") as temporary:
        directory = Path(temporary)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        business_key, admin_key = secrets.token_hex(24), secrets.token_hex(24)
        config = directory / "service.json"
        config.write_text(json.dumps({
            "database": str(directory / "registry.sqlite3"), "model_roots": [temporary],
            "deployment_mode": "development", "allow_mock": True,
            "max_executions": EXECUTIONS, "queue_size": QUEUE_SIZE,
            "total_memory_mb": 128, "queue_timeout_s": 2, "execution_timeout_s": 10,
            "load_timeout_s": 10, "drain_timeout_s": 10, "max_http_requests": 64,
            "maintenance_interval_s": 1,
        }))
        log_path = directory / "controller.log"
        report = None
        children: list[psutil.Process] = []
        with log_path.open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "model_service", "--config", str(config),
                                        "--host", "127.0.0.1", "--port", str(port)],
                                       env=dict(os.environ, BUSINESS_API_KEY=business_key, ADMIN_API_KEY=admin_key),
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                url = f"http://127.0.0.1:{port}"
                deadline = time.monotonic() + 15
                with httpx.Client(timeout=0.5) as probe:
                    while True:
                        try:
                            if probe.get(url + "/ready").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if process.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("Temporary Mock controller failed to become ready")
                        time.sleep(0.025)
                report = asyncio.run(exercise(url, business_key, admin_key))
            finally:
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    # Only failure cleanup for this script's isolated Mock test processes.
                    process.kill()
                    process.wait(timeout=5)
                    raise RuntimeError("Mock test controller failed graceful shutdown") from None
                finally:
                    _, alive = psutil.wait_procs(children, timeout=5)
                    leftover = [child for child in alive if process_running(child)]
                    for child in leftover:
                        child.kill()  # Failed test cleanup, never an administrative model operation.
                    if leftover:
                        psutil.wait_procs(leftover, timeout=5)
                    assert not leftover, "Mock test leaked child processes"
        assert process.returncode in (0, -signal.SIGTERM), process.returncode
        assert "Application shutdown complete." in log_path.read_text()
        assert report is not None
        report.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                      transport="real TCP HTTP to one isolated uvicorn CLI controller",
                      mock=True, status="passed_mock_http_overload")
        report["shutdown"] = {"returncode": process.returncode, "application_shutdown_completed": True,
                              "controller_exited": True, "remaining_child_processes": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "http_status_counts": report["http_status_counts"],
                      "observed_peaks": report["observed_peaks"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
