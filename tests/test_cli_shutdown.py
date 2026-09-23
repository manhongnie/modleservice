"""CLI shutdown must wait for native work beyond the per-attempt drain deadline."""
from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from model_service.api import _close_until_stopped
from model_service.contracts import ModelConfig, ServiceError
from model_service.registry import Registry


@pytest.mark.asyncio
async def test_lifespan_close_retries_only_known_stop_deadlines(caplog):
    class SlowStop:
        def __init__(self):
            self.calls = []

        async def close(self):
            self.calls.append(time.monotonic())
            if len(self.calls) == 1:
                raise ServiceError("shutdown_timeout", "sensitive-path-must-not-be-logged")
            if len(self.calls) == 2:
                raise ServiceError("unload_timeout", "sensitive-path-must-not-be-logged")

    service = SlowStop()
    await _close_until_stopped(service, retry_interval_s=0.01)
    assert len(service.calls) == 3
    assert service.calls[1] - service.calls[0] >= 0.009
    assert service.calls[2] - service.calls[1] >= 0.009
    assert "shutdown_timeout" in caplog.text and "unload_timeout" in caplog.text
    assert "sensitive-path-must-not-be-logged" not in caplog.text

    class BrokenStop:
        calls = 0

        async def close(self):
            self.calls += 1
            raise ServiceError("unexpected_storage_error", "requires investigation")

    broken = BrokenStop()
    with pytest.raises(ServiceError, match="requires investigation"):
        await _close_until_stopped(broken, retry_interval_s=0.01)
    assert broken.calls == 1


def test_cli_shutdown_during_resident_native_load_waits_and_exits_cleanly(tmp_path):
    # Seed persisted, previously validated metadata; no real model is represented.
    # This exercises restart maintenance loading rather than an HTTP admin request.
    database = tmp_path / "registry.sqlite3"
    model = ModelConfig(name="slow-resident-mock", version="1", capabilities=["embeddings"],
                        task="mock", backend="mock", load_policy="resident",
                        validation_input={"texts": ["check"]}, options={"load_delay_s": 1.2})
    registry = Registry(str(database))
    registry.register(model)
    registry.validation(model.model_id, True)
    registry.enable(model.model_id, True)
    registry.close()
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"database": str(database), "model_roots": [str(tmp_path)],
                                  "maintenance_interval_s": 0.01, "drain_timeout_s": 0.05,
                                  "load_timeout_s": 0.05}))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    business_key, admin_key = secrets.token_hex(24), secrets.token_hex(24)
    log_path = tmp_path / "controller.log"
    process = None
    children = []
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen([sys.executable, "-m", "model_service", "--config", str(config),
                                        "--host", "127.0.0.1", "--port", str(port)],
                                       env=dict(os.environ, BUSINESS_API_KEY=business_key, ADMIN_API_KEY=admin_key),
                                       stdout=log, stderr=subprocess.STDOUT)
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=0.5) as client:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        response = client.get("/admin/status", headers={"Authorization": "Bearer " + admin_key})
                        if response.status_code == 200:
                            status = response.json()
                            workers = list(status["workers"].values())
                            if workers and workers[0]["alive"]:
                                assert status["models"][0]["load"]["state"] in {"loading", "load_pending"}
                                worker = psutil.Process(workers[0]["pid"])
                                break
                    except httpx.HTTPError:
                        pass
                    assert process.poll() is None, log_path.read_text()
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.01)
            children = psutil.Process(process.pid).children(recursive=True)
            began = time.monotonic()
            process.terminate()
            # Per-attempt drain deadline has elapsed, but native load has not.
            time.sleep(0.3)
            assert process.poll() is None, log_path.read_text()
            assert worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE
            assert '"event":"shutdown_waiting"' in log_path.read_text()
            process.wait(timeout=8)
            assert time.monotonic() - began >= 0.9
            assert process.returncode in (0, -signal.SIGTERM)
            output = log_path.read_text()
            assert "Application shutdown complete." in output
            assert "Application shutdown failed" not in output
            _, alive = psutil.wait_procs(children, timeout=3)
            assert all(child.status() == psutil.STATUS_ZOMBIE for child in alive)
            # Successful cleanup must also release the database ownership lock.
            if sys.platform.startswith("linux"):
                import fcntl
                with open(str(database) + ".lock", "a+") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            if process is not None and process.poll() is None:
                process.kill()  # Isolated failed-test cleanup, never a model management action.
                process.wait(timeout=5)
            for child in children:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(children, timeout=3)
