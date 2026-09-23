"""HTTP integration with the real control plane and explicitly Mock child workers."""
from __future__ import annotations

import json

import httpx
import pytest

from model_service.api import create_app
from model_service.contracts import Settings


def configured(tmp_path):
    return Settings(database=str(tmp_path / "models.sqlite3"), model_roots=[str(tmp_path / "models")],
                    business_keys=["business-test-key"], admin_keys=["admin-test-key"],
                    maintenance_interval_s=60)


def model_config():
    return {"name": "http-integration-mock", "version": "v1", "capabilities": ["chat"],
            "task": "mock", "backend": "mock", "resident_mb": 16, "request_mb": 4,
            "validation_input": {"messages": [{"role": "user", "content": "validate"}]}}


@pytest.mark.asyncio
async def test_http_lifecycle_with_real_control_plane_and_mock_worker(tmp_path):
    settings = configured(tmp_path)
    app = create_app(settings)
    admin = {"Authorization": "Bearer admin-test-key"}
    business = {"Authorization": "Bearer business-test-key"}
    model_id = "http-integration-mock@v1"
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/admin/models", json={"config": model_config()}, headers=admin)
            assert response.status_code == 200, response.text
            assert response.json()["enabled"]
            assert (await client.put("/admin/aliases/default:chat", json={"model_id": model_id}, headers=admin)).status_code == 200
            response = await client.post("/v1/chat", json={"input": {"messages": [{"role": "user", "content": "hello"}]}}, headers=business)
            assert response.status_code == 200, response.text
            assert response.json()["mock"] is True
            assert response.json()["output"]["text"] == "MOCK: hello"
            state = (await client.get("/admin/status", headers=admin)).json()
            assert state["models"][0]["active_requests"] == 0
            assert list(state["workers"].values())[0]["load_count"] == 1
            response = await client.post(f"/admin/models/{model_id}/remove", headers=admin)
            assert response.status_code == 409
            assert (await client.delete("/admin/aliases/default:chat", headers=admin)).status_code == 200
            assert (await client.put("/admin/dependencies/business-a", json={"model_id": model_id}, headers=admin)).status_code == 200
            assert (await client.delete(f"/admin/models/{model_id}", headers=admin)).status_code == 409
            assert (await client.delete("/admin/dependencies/business-a", headers=admin)).status_code == 200
            assert (await client.post(f"/admin/models/{model_id}/disable", headers=admin)).status_code == 200

    restored = create_app(settings)
    async with restored.router.lifespan_context(restored):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored), base_url="http://test") as client:
            state = (await client.get("/admin/status", headers=admin)).json()
            assert len(state["models"]) == 1
            assert state["models"][0]["enabled"] is False
            assert (await client.delete(f"/admin/models/{model_id}", headers=admin)).status_code == 200
            assert (await client.get("/admin/status", headers=admin)).json()["models"] == []


@pytest.mark.asyncio
async def test_management_rejects_paths_outside_model_roots(tmp_path):
    app = create_app(configured(tmp_path))
    secret = tmp_path / "outside-model-roots.bin"
    secret.write_bytes(b"not-a-model")
    config = {**model_config(), "path": str(secret)}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/admin/models", json={"config": config},
                                         headers={"Authorization": "Bearer admin-test-key"})
            assert response.status_code == 403
            assert response.json()["error"]["code"] == "path_forbidden"
            state = (await client.get("/admin/status", headers={"Authorization": "Bearer admin-test-key"})).json()
            assert state["models"] == []
    assert secret.read_bytes() == b"not-a-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,metadata", [
    ("model.safetensors.index.json", {"metadata": {}, "weight_map": {"weight": "../outside.bin"}}),
    ("tokenizer_config.json", {"tokenizer_file": "../outside.bin"}),
    ("processor_config.json", {"processor": {"vocab_file": "../outside.bin"}}),
])
async def test_management_rejects_model_metadata_references_outside_allowed_roots(tmp_path, filename, metadata):
    root = tmp_path / "models"
    root.mkdir()
    (root / filename).write_text(json.dumps(metadata))
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not-a-real-model")
    app = create_app(configured(tmp_path))
    admin = {"Authorization": "Bearer admin-test-key"}
    config = {**model_config(), "path": str(root), "backend": "transformers", "task": "qwen_chat"}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/admin/models", json={"config": config}, headers=admin)
            assert response.status_code == 403, response.text
            assert response.json()["error"]["code"] == "path_forbidden"
            state = (await client.get("/admin/status", headers=admin)).json()
            assert state["models"] == []
            assert state["workers"] == {}
    assert outside.read_bytes() == b"not-a-real-model"
