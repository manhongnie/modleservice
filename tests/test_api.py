from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from model_service.__main__ import load_settings
from model_service.api import InferenceResponse, create_app
from model_service.contracts import ServiceError, Settings


def settings(tmp_path: Path, **changes) -> Settings:
    return Settings(database=str(tmp_path / "models.sqlite3"),
                    model_roots=[str(tmp_path)], business_keys=["business-test-key"],
                    admin_keys=["admin-test-key"], **changes)


class StubService:
    def __init__(self):
        self.released = False
        self.started = False
        self.cancel_seen = False
        self.calls = []

    async def infer(self, **kwargs):
        self.started = True
        self.calls.append(kwargs)
        try:
            if kwargs["payload"].get("fail"):
                raise ServiceError("load_failed", "Secret model path: /private/model.xml", 503)
            yield {"request_id": kwargs["request_id"], "model": "example@v1", "output": {"text": "ok"}, "done": True}
        finally:
            self.released = True

    async def status(self):
        return {"models": [], "queue_length": 0}

    async def reconcile(self, request_id, confirmed_stopped):
        self.calls.append((request_id, confirmed_stopped))
        return {"reconciled": confirmed_stopped}


@pytest.mark.asyncio
async def test_authentication_is_separate_and_docs_are_not_exposed(tmp_path):
    app = create_app(settings(tmp_path))
    service = StubService()
    app.state.service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        for path in ["/docs", "/redoc", "/openapi.json"]:
            assert (await client.get(path)).status_code == 404
        assert (await client.get("/admin/status")).status_code == 401
        assert (await client.get("/admin/status", headers={"Authorization": "Bearer business-test-key"})).status_code == 401
        assert (await client.get("/admin/status", headers={"Authorization": "Bearer admin-test-key"})).status_code == 200
        body = {"input": {"text": "hello"}}
        assert (await client.post("/v1/chat", json=body, headers={"Authorization": "Bearer admin-test-key"})).status_code == 401
        assert not service.started
        response = await client.post("/v1/chat", json=body, headers={"Authorization": "Bearer business-test-key"})
        assert response.status_code == 200
        assert response.json()["output"] == {"text": "ok"}
        assert response.headers["x-request-id"] == response.json()["request_id"]
        assert service.released


@pytest.mark.asyncio
async def test_chunked_body_limit_counts_actual_bytes(tmp_path):
    app = create_app(settings(tmp_path, max_body_bytes=64))
    service = StubService()
    app.state.service = service

    async def body():
        yield b'{"input":{"text":"'
        for _ in range(8):
            yield b"abcdefghij"
        yield b'"}}'

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat", content=body(), headers={
            "Authorization": "Bearer business-test-key", "Content-Type": "application/json",
            # Deliberately incorrect Content-Length must not bypass byte counting.
            "Content-Length": "1",
        })
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "input_too_large"
    assert not service.started


@pytest.mark.asyncio
async def test_business_errors_hide_backend_paths_and_stream_preflight(tmp_path):
    app = create_app(settings(tmp_path))
    app.state.service = StubService()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for streaming in [False, True]:
            response = await client.post("/v1/chat", json={"input": {"fail": True}, "stream": streaming},
                                         headers={"Authorization": "Bearer business-test-key"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "load_failed"
            assert "private" not in response.text
            assert "Secret" not in response.text


@pytest.mark.asyncio
async def test_validation_errors_do_not_echo_input(tmp_path):
    app = create_app(settings(tmp_path))
    app.state.service = StubService()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat", json={"input": "sensitive-document-content"},
                                     headers={"Authorization": "Bearer business-test-key"})
    assert response.status_code == 422
    assert "sensitive-document-content" not in response.text


@pytest.mark.asyncio
async def test_reconcile_is_admin_only_and_does_not_invent_confirmation(tmp_path):
    app = create_app(settings(tmp_path))
    service = StubService()
    app.state.service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/admin/requests/request-1/reconcile", json={"confirmed_stopped": True},
                                     headers={"Authorization": "Bearer business-test-key"})
        assert response.status_code == 401
        response = await client.post("/admin/requests/request-1/reconcile", json={"confirmed_stopped": False},
                                     headers={"Authorization": "Bearer admin-test-key"})
    assert response.status_code == 200
    assert service.calls == [("request-1", False)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_permit_is_held_until_response_body_is_sent(streaming):
    cancel = asyncio.Event()
    released = False
    body_sent = False

    async def iterator():
        nonlocal released
        try:
            yield {"output": {"text": "result"}, "done": True}
        finally:
            assert body_sent
            released = True

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        nonlocal body_sent
        if message["type"] == "http.response.body" and message.get("body"):
            await asyncio.sleep(0.01)
            assert not released
            body_sent = True

    await InferenceResponse(iterator(), cancel, streaming)({"type": "http", "state": {}}, receive, send)
    assert released
    assert body_sent


@pytest.mark.asyncio
async def test_stream_disconnect_waits_for_underlying_stop():
    cancel = asyncio.Event()
    stop_acknowledged = asyncio.Event()
    allow_stop = asyncio.Event()
    receive_queue = asyncio.Queue()
    released = False
    sent = []

    async def iterator():
        nonlocal released
        try:
            yield {"output": {"text": "first"}}
            await cancel.wait()
            await allow_stop.wait()
            stop_acknowledged.set()
        finally:
            assert stop_acknowledged.is_set()
            released = True

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            await receive_queue.put({"type": "http.disconnect"})

    task = asyncio.create_task(InferenceResponse(iterator(), cancel, True)(
        {"type": "http", "state": {}}, receive_queue.get, send))
    await asyncio.wait_for(cancel.wait(), 1)
    await asyncio.sleep(0.02)
    assert not task.done()
    assert not released
    allow_stop.set()
    await asyncio.wait_for(task, 1)
    assert released
    assert len([message for message in sent if message.get("body")]) == 1


@pytest.mark.asyncio
async def test_server_task_cancellation_does_not_abort_stop_confirmation():
    cancel = asyncio.Event()
    started = asyncio.Event()
    allow_stop = asyncio.Event()
    released = False

    async def iterator():
        nonlocal released
        try:
            started.set()
            await cancel.wait()
            await allow_stop.wait()
            return
            yield  # Makes this a generator which emits no result on cancellation.
        finally:
            assert allow_stop.is_set()
            released = True

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        raise AssertionError("A cancelled request must not send a response")

    task = asyncio.create_task(InferenceResponse(iterator(), cancel, True)(
        {"type": "http", "state": {}}, receive, send))
    await started.wait()
    task.cancel()
    await asyncio.wait_for(cancel.wait(), 1)
    await asyncio.sleep(0.02)
    assert not released
    assert not task.done()
    allow_stop.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert released


@pytest.mark.asyncio
async def test_stream_failure_after_partial_output_is_explicit_and_never_retried():
    cancel = asyncio.Event()
    attempts = 0
    sent = []

    async def iterator():
        nonlocal attempts
        attempts += 1
        yield {"output": {"text": "partial"}}
        raise ServiceError("worker_crashed", "/private/worker.exe crashed", 502)

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await InferenceResponse(iterator(), cancel, True)({"type": "http", "state": {}}, receive, send)
    body = b"".join(message.get("body", b"") for message in sent).decode()
    assert "event: chunk" in body
    assert "event: error" in body
    assert "worker_crashed" in body
    assert "/private" not in body
    assert attempts == 1


def test_config_keys_are_environment_injected_and_distinct(tmp_path, monkeypatch):
    path = tmp_path / "service.json"
    path.write_text(json.dumps({"database": str(tmp_path / "database.db")}))
    monkeypatch.setenv("MODEL_SERVICE_CONFIG", str(path))
    monkeypatch.delenv("BUSINESS_API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        load_settings()
    monkeypatch.setenv("BUSINESS_API_KEY", "test-business-secret")
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-secret")
    loaded = load_settings()
    assert loaded.business_keys == ["test-business-secret"]
    assert loaded.admin_keys == ["test-admin-secret"]
    monkeypatch.setenv("ADMIN_API_KEY", "test-business-secret")
    with pytest.raises(ValidationError):
        load_settings()

