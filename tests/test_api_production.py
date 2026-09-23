from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from model_service import __main__ as cli
from model_service.api import HTTPBoundary, InferenceResponse, create_app
from model_service.contracts import Settings


def settings(tmp_path: Path, **changes) -> Settings:
    data = dict(database=str(tmp_path / "models.sqlite3"), model_roots=[str(tmp_path)],
                business_keys=["business-test-key"], admin_keys=["admin-test-key"])
    return Settings(**(data | changes))


def scope(path="/v1/chat", authorization=b"Bearer business-test-key"):
    return {"type": "http", "asgi": {"version": "3.0"}, "method": "POST", "path": path,
            "raw_path": path.encode(), "query_string": b"", "scheme": "http",
            "server": ("test", 80), "client": ("127.0.0.1", 1234),
            "headers": [(b"authorization", authorization), (b"content-type", b"application/json")]}


@pytest.mark.asyncio
@pytest.mark.parametrize("path,key", [("/v1/chat", b"Bearer admin-test-key"),
                                       ("/admin/models", b"Bearer business-test-key"),
                                       ("/v1/chat", b"")])
async def test_unauthorized_request_never_reads_slow_body(tmp_path, path, key):
    app = create_app(settings(tmp_path))
    sent = []

    async def receive():
        raise AssertionError("Unauthenticated body was consumed")

    async def send(message):
        sent.append(message)

    await app(scope(path, key), receive, send)
    assert sent[0]["status"] == 401
    assert dict(sent[0]["headers"])[b"www-authenticate"] == b"Bearer"


@pytest.mark.asyncio
async def test_total_body_deadline_is_not_reset_by_chunks(tmp_path):
    app = create_app(settings(tmp_path, body_timeout_s=0.035))
    sent = []
    received = 0

    async def receive():
        nonlocal received
        await asyncio.sleep(0.02)
        received += 1
        return {"type": "http.request", "body": b" ", "more_body": True}

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope(), receive, send), 0.5)
    assert sent[0]["status"] == 408
    assert received < 3
    assert json.loads(sent[-1]["body"])["error"]["code"] == "body_timeout"


@pytest.mark.asyncio
async def test_http_capacity_rejects_without_reading_and_returns_slot_after_cleanup(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    async def downstream(scope, receive, send):
        if scope["path"] != "/health":
            started.set()
            await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    app = HTTPBoundary(downstream, settings(tmp_path, max_http_requests=1))
    sent = []

    async def first_receive():
        return {"type": "http.request", "body": b"{}"}

    async def never_receive():
        raise AssertionError("A rejected request body was consumed")

    async def send(message):
        sent.append(message)

    first = asyncio.create_task(app(scope(), first_receive, send))
    await started.wait()
    await app(scope(), never_receive, send)
    assert sent[0]["status"] == 503
    assert dict(sent[0]["headers"])[b"Retry-After".lower()] == b"1"
    assert app.active == 1
    # Health remains reachable even when business/admin slots are all occupied.
    public_scope = scope("/health", b"")
    public_scope["method"] = "GET"
    await app(public_scope, never_receive, send)
    assert sent[-2]["status"] == 200
    assert app.active == 1
    release.set()
    await first
    assert app.active == 0
    await app(scope(), first_receive, send)
    assert sent[-2]["status"] == 200


@pytest.mark.asyncio
async def test_readiness_and_admin_metrics_authentication(tmp_path):
    class Service:
        ready = True

        async def readiness(self):
            return {"status": "ready" if self.ready else "not_ready"}

        async def metrics(self):
            return "# TYPE model_service_requests_total counter\nmodel_service_requests_total 2\n"

        async def status(self):
            return {"models": [{"id": "example@1"}]}

    app = create_app(settings(tmp_path))
    service = app.state.service = Service()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/ready")).json() == {"status": "ready"}
        service.ready = False
        assert (await client.get("/ready")).status_code == 503
        for path in ["/admin/models", "/admin/metrics"]:
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path, headers={"Authorization": "Bearer business-test-key"})).status_code == 401
        response = await client.get("/admin/metrics", headers={"Authorization": "Bearer admin-test-key"})
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "model_service_requests_total 2" in response.text
        response = await client.get("/admin/models", headers={"Authorization": "Bearer admin-test-key"})
        assert response.json() == {"models": [{"id": "example@1"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_output_timeout_cancels_and_waits_for_underlying_stop(streaming):
    cancel, allow_stop, released = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sends = 0

    async def iterator():
        try:
            yield {"output": {"text": "first"}, "done": True}
        finally:
            assert cancel.is_set()
            await allow_stop.wait()
            released.set()

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        nonlocal sends
        sends += 1
        if message["type"] == "http.response.body":
            await asyncio.Event().wait()

    task = asyncio.create_task(InferenceResponse(iterator(), cancel, streaming, write_timeout_s=0.02)(
        {"type": "http", "state": {}}, receive, send))
    await asyncio.wait_for(cancel.wait(), 0.5)
    await asyncio.sleep(0.02)
    assert not task.done()
    assert not released.is_set()
    allow_stop.set()
    await asyncio.wait_for(task, 0.5)
    assert released.is_set()
    assert sends == 2  # No retry or error write to an unresponsive transport.


def test_production_config_requires_long_distinct_keys_and_mock_disabled(tmp_path):
    with pytest.raises(ValidationError):
        settings(tmp_path, deployment_mode="production", allow_mock=False)
    with pytest.raises(ValidationError):
        settings(tmp_path, deployment_mode="production", business_keys=["b" * 48], admin_keys=["a" * 48])
    configured = settings(tmp_path, deployment_mode="production", allow_mock=False,
                          business_keys=["b" * 48], admin_keys=["a" * 48])
    assert configured.deployment_mode == "production"


def test_check_config_exits_without_starting_server_or_revealing_keys(tmp_path, monkeypatch, capsys):
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"deployment_mode": "production", "allow_mock": False}))
    business, admin = "business-secret-" + "b" * 40, "admin-secret-" + "a" * 40
    monkeypatch.setenv("BUSINESS_API_KEY", business)
    monkeypatch.setenv("ADMIN_API_KEY", admin)
    monkeypatch.setattr(sys, "argv", ["model_service", "--config", str(path), "--check-config"])
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: pytest.fail("Server was started"))
    cli.main()
    output = capsys.readouterr().out
    assert "Configuration is valid" in output
    assert business not in output
    assert admin not in output


def test_cli_uses_one_worker_bounded_connections_and_safe_shutdown(tmp_path, monkeypatch):
    path = tmp_path / "service.json"
    path.write_text("{}")
    monkeypatch.setenv("BUSINESS_API_KEY", "business-test-key")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-test-key")
    monkeypatch.setattr(sys, "argv", ["model_service", "--config", str(path)])
    import uvicorn
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    cli.main()
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["workers"] == 1
    assert calls[0]["limit_concurrency"] == 144
    assert calls[0]["timeout_graceful_shutdown"] is None
    assert calls[0]["access_log"] is False
