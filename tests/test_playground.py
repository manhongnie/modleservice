"""Browser playground boundary and the business-safe, read-only model catalog."""
from __future__ import annotations

import json

import httpx
import pytest

from model_service.api import create_app
from model_service.contracts import ModelConfig, Settings
from model_service.service import Service


BUSINESS = {"Authorization": "Bearer business-test-key"}
ADMIN = {"Authorization": "Bearer admin-test-key"}


def settings(tmp_path, **changes):
    return Settings(database=str(tmp_path / "models.sqlite3"), model_roots=[str(tmp_path)],
                    business_keys=["business-test-key"], admin_keys=["admin-test-key"],
                    maintenance_interval_s=60, **changes)


def model(name="browser-mock", **changes):
    values = dict(name=name, version="v1", capabilities=["chat"], task="mock", backend="mock",
                  resident_mb=16, request_mb=4,
                  validation_input={"messages": [{"role": "user", "content": "validate"}]})
    return ModelConfig(**(values | changes))


@pytest.fixture
async def service(tmp_path):
    instance = Service(settings(tmp_path))
    await instance.start()
    try:
        yield instance
    finally:
        await instance.close()


def register_ready(service, config):
    """Restore persisted registry state without loading model weights."""
    service.registry.register(config)
    service.registry.validation(config.model_id, True)
    service.registry.enable(config.model_id, True)


async def test_playground_is_public_with_restricted_resource_headers(tmp_path):
    app = create_app(settings(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for path, content_type in [("/", "text/html"), ("/playground.css", "text/css"),
                                   ("/playground.js", "javascript")]:
            response = await client.get(path)
            assert response.status_code == 200, response.text
            assert content_type in response.headers["content-type"]
            assert response.content
            assert response.headers["x-content-type-options"] == "nosniff"
            assert "no-store" in response.headers["cache-control"]
            csp = response.headers["content-security-policy"]
            for directive in ["default-src 'none'", "script-src 'self'", "style-src 'self'",
                              "connect-src 'self'", "img-src 'self' blob: data:",
                              "media-src 'self' blob:"]:
                assert directive in csp
            assert "unsafe-inline" not in csp
            assert "business-test-key" not in response.text
            assert "admin-test-key" not in response.text
            if path == "/":
                assert "Model Service" in response.text


@pytest.mark.parametrize("method,path", [
    ("POST", "/"), ("POST", "/playground.js"), ("GET", "/playground.js/extra"),
    ("GET", "/playground.css/extra"), ("GET", "/static/api.py"),
    ("GET", "/static/../api.py"), ("GET", "/model_service/api.py"),
    ("GET", "/examples/service.multimodal.json"), ("GET", "/README.md"),
    ("GET", "/docs"), ("GET", "/openapi.json"),
])
async def test_public_page_does_not_expose_other_paths_or_methods(tmp_path, method, path):
    app = create_app(settings(tmp_path))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.request(method, path)
    assert response.status_code == 404
    assert "Model Service" not in response.text


@pytest.mark.parametrize("authorization", [b"", b"Bearer admin-test-key", b"Bearer wrong-key"])
async def test_unauthorized_catalog_never_consumes_request_body(tmp_path, authorization):
    app = create_app(settings(tmp_path))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "method": "GET", "path": "/v1/models",
             "raw_path": b"/v1/models", "query_string": b"", "scheme": "http",
             "server": ("test", 80), "client": ("127.0.0.1", 1234),
             "headers": [(b"authorization", authorization)]}
    sent = []

    async def receive():
        raise AssertionError("An unauthorized catalog request consumed its body")

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    assert sent[0]["status"] == 401
    assert dict(sent[0]["headers"])[b"www-authenticate"] == b"Bearer"


async def test_empty_catalog_comes_from_registry(service, tmp_path):
    app = create_app(service.settings)
    app.state.service = service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/v1/models", headers=BUSINESS)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["vary"] == "Authorization"
    assert response.json() == {"models": [], "max_body_bytes": service.settings.max_body_bytes}
    assert (await service.status())["workers"] == {}


async def test_catalog_excludes_models_unavailable_for_inference(service):
    ready = model("ready")
    register_ready(service, ready)
    disabled = model("disabled")
    register_ready(service, disabled)
    service.registry.enable(disabled.model_id, False)
    service.registry.register(model("unvalidated"))
    for state in ["validating", "validation_failed", "draining", "reconciliation_required"]:
        config = model(state)
        register_ready(service, config)
        service.registry.management(config.model_id, state)
    register_ready(service, model("removed-plugin", backend="plugin-no-longer-installed"))

    response = await service.available_models()
    assert [entry["model_id"] for entry in response["models"]] == [ready.model_id]


async def test_catalog_excludes_temporarily_paused_models_until_resumed(service):
    config = model("temporarily-paused")
    register_ready(service, config)
    assert [entry["model_id"] for entry in (await service.available_models())["models"]] == [config.model_id]

    await service.scheduler.pause(config.instance_key)
    row = service.registry.get(config.model_id)
    assert row["enabled"] and row["validated"] and row["management_state"] == "ready"
    assert (await service.available_models())["models"] == []

    await service.scheduler.resume(config.instance_key)
    assert [entry["model_id"] for entry in (await service.available_models())["models"]] == [config.model_id]


async def test_catalog_returns_only_safe_fields_and_performs_no_loads_or_writes(service, monkeypatch):
    config = model("safe-public-name", path="/private/weights-secret.bin", max_input_bytes=4096,
                   validation_input={"text": "private-validation-sample"}, options={
                       "max_new_tokens": 64, "max_width": 512, "default_fps": 8.0,
                       "fixed_width": 256, "max_prompt_chars": 1024, "max_batch_size": 8,
                       "max_text_chars": "private-string-value", "max_audio_seconds": True,
                       "max_steps": -1, "max_batch": {"value": 8},
                       "api_key": "private-provider-key", "threads": 16,
                       "base_url": "http://private-provider", "mock": False})
    register_ready(service, config)
    service.registry.validation(config.model_id, True, "private-validation-error")
    before_changes = service.registry.db.total_changes
    before_events = service.registry.events()

    def unexpected(*args, **kwargs):
        raise AssertionError("Reading the model catalog must not validate or load a model")

    with monkeypatch.context() as patch:
        patch.setattr(service.runtime.policy, "validate", unexpected)
        patch.setattr(service.runtime.catalog, "validate", unexpected)
        patch.setattr(service.runtime.executor, "load", unexpected)
        response = await service.available_models()

    assert response == {"max_body_bytes": service.settings.max_body_bytes, "models": [{
        "model_id": config.model_id, "name": config.name, "version": "v1", "capabilities": ["chat"],
        "mock": True, "limits": {"max_input_bytes": 4096, "max_new_tokens": 64, "max_width": 512,
                                  "default_fps": 8.0, "fixed_width": 256, "max_prompt_chars": 1024,
                                  "max_batch_size": 8}}]}
    assert "private-" not in json.dumps(response)
    assert service.registry.db.total_changes == before_changes
    assert service.registry.events() == before_events
    assert (await service.status())["workers"] == {}


async def test_mock_indicator_uses_backend_features_and_respects_mock_policy(service):
    register_ready(service, model("synthetic", options={"mock": False}))
    register_ready(service, model("provider", task="http", backend="http", options={"mock": True}))
    entries = {entry["name"]: entry for entry in (await service.available_models())["models"]}
    assert entries["synthetic"]["mock"] is True
    assert entries["provider"]["mock"] is False
    service.settings.allow_mock = False
    assert [entry["name"] for entry in (await service.available_models())["models"]] == ["provider"]


async def test_catalog_selection_can_call_existing_inference_api(tmp_path):
    app = create_app(settings(tmp_path))
    config = model("registered-browser-model")
    async with app.router.lifespan_context(app):
        service = app.state.service
        await service.add(config, enable=False)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/v1/models", headers=BUSINESS)).json()["models"] == []
            await service.enable(config.model_id)
            entry = (await client.get("/v1/models", headers=BUSINESS)).json()["models"][0]
            assert entry["model_id"] == config.model_id
            assert entry["mock"] is True
            payload = {"model": entry["model_id"], "input": {
                "messages": [{"role": "user", "content": "hello from the browser"}]}}
            response = await client.post("/v1/" + entry["capabilities"][0], json=payload, headers=BUSINESS)
            assert response.status_code == 200, response.text
            assert response.json()["output"]["text"] == "MOCK: hello from the browser"
            assert response.json()["mock"] is True
            assert (await client.post("/v1/chat", json=payload, headers=ADMIN)).status_code == 401
            assert (await client.get("/admin/models", headers=BUSINESS)).status_code == 401
            assert (await client.get("/admin/models", headers=ADMIN)).status_code == 200
