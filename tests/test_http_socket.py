"""Real socket smoke test: ASGI disconnect through the complete control/worker path."""
import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from model_service.contracts import Settings
from model_service.service import Service


@pytest.fixture
def server(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"database": str(tmp_path / "models.db"), "model_roots": [str(tmp_path)],
                                  "maintenance_interval_s": 10, "drain_timeout_s": 5}))
    env = dict(os.environ, BUSINESS_API_KEY=secrets.token_hex(16), ADMIN_API_KEY=secrets.token_hex(16))
    logfile = (tmp_path / "service.log").open("w+")
    process = subprocess.Popen([sys.executable, "-m", "model_service", "--config", str(config),
                                "--port", str(port)], env=env, stdout=logfile, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                response = httpx.get(url + "/health", timeout=.2)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if process.poll() is not None or time.monotonic() > deadline:
                logfile.seek(0)
                pytest.fail(logfile.read())
            time.sleep(.03)
        yield url, env, process, config
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Test failure cleanup only; no service management operation force-kills.
            process.kill()
            process.wait()
        logfile.close()


async def test_real_socket_startup_stream_disconnect_and_metrics(server):
    url, env, _, _ = server
    admin = {"Authorization": "Bearer " + env["ADMIN_API_KEY"]}
    business = {"Authorization": "Bearer " + env["BUSINESS_API_KEY"]}
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        response = await client.post("/admin/models", headers=admin, json={"config": {
            "name": "socket-mock", "version": "1", "task": "mock", "backend": "mock",
            "capabilities": ["embeddings"], "validation_input": {"texts": ["check"]},
            "options": {"stream_chunks": 50, "chunk_delay_s": .02, "cancel_delay_s": .06}
        }})
        assert response.status_code == 200, response.text
        await client.put("/admin/aliases/default:embeddings", headers=admin, json={"model_id": "socket-mock@1"})
        async with client.stream("POST", "/v1/embeddings", headers=business,
                                 json={"input": {"texts": ["socket test"]}, "stream": True}) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    assert json.loads(line[5:])["mock"] is True
                    break
            status = (await client.get("/admin/status", headers=admin)).json()
            assert status["scheduler"]["active_count"] == 1
        deadline = time.monotonic() + 3
        while True:
            status = (await client.get("/admin/status", headers=admin)).json()
            if not status["scheduler"]["active_count"]:
                break
            assert time.monotonic() < deadline
            await asyncio.sleep(.025)
        assert status["scheduler"]["resident_instances"] == 1
        worker = next(iter(status["workers"].values()))
        assert worker["load_count"] == 1 and worker["rss_mb"] > 0
        assert status["events"][0]["kind"] in {"request_finished", "execution_finished"}


@pytest.mark.skipif(sys.platform != "linux", reason="Parent-death behavior is Linux-specific")
async def test_controller_crash_stops_child_and_restart_recovers_registry(server):
    url, env, process, config_path = server
    admin = {"Authorization": "Bearer " + env["ADMIN_API_KEY"]}
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        response = await client.post("/admin/models", headers=admin, json={"config": {
            "name": "crash-mock", "version": "1", "task": "mock", "backend": "mock",
            "capabilities": ["embeddings"], "validation_input": {"texts": ["check"]}
        }})
        assert response.status_code == 200
        status = (await client.get("/admin/status", headers=admin)).json()
        worker_pid = next(iter(status["workers"].values()))["pid"]
        process.kill()  # Inject controller crash, not a management operation.
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while psutil.pid_exists(worker_pid):
            if psutil.Process(worker_pid).status() == psutil.STATUS_ZOMBIE:
                break  # Exited; awaiting OS parent reaping, no execution or RSS remains.
            assert time.monotonic() < deadline, "Orphan worker remained alive after controller death"
            await asyncio.sleep(.025)
    values = json.loads(config_path.read_text())
    values.update(business_keys=[env["BUSINESS_API_KEY"]], admin_keys=[env["ADMIN_API_KEY"]])
    restored = Service(Settings(**values))
    await restored.start()
    try:
        status = await restored.status()
        assert status["models"][0]["enabled"]
        assert status["models"][0]["load"]["state"] == "unloaded"
        result = [item async for item in restored.infer("embeddings", "crash-mock", {"texts": ["after crash"]},
                                                       asyncio.Event(), "after-controller-crash")]
        assert result[0]["mock"]
        new_pid = next(iter((await restored.status())["workers"].values()))["pid"]
        assert new_pid != worker_pid
    finally:
        await restored.close()
