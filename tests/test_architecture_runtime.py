"""Use-case replacement, lifecycle independence and recovery regressions.

The injected provider below is explicitly a controlled test double, not a model.
"""
import asyncio
import ast
from pathlib import Path
import sqlite3
import time

import pytest

from model_service.contracts import BackendFeatures, ModelConfig, ServiceError, Settings
from model_service.service import Service


class ControlledCatalog:
    def __init__(self, *, remote=False, synthetic=False):
        self.features = BackendFeatures(management="external" if remote else "local",
                                       execution_may_outlive_worker=remote,
                                       requires_artifact=False, synthetic=synthetic)

    def describe(self, config):
        return self.features

    def validate(self, config, settings):
        assert config.backend == "independent-provider"

    def validate_input(self, config, payload, capability):
        if not isinstance(payload.get("text"), str):
            raise ServiceError("invalid_input", "text required", 422)


class ControlledExecutor:
    def __init__(self):
        self.loaded = set()
        self.loads = 0
        self.fail_load = False

    async def load(self, config):
        self.loads += 1
        if self.fail_load:
            raise ServiceError("load_failed", "Test provider unavailable", 503)
        self.loaded.add(config.instance_key)
        return {"state": "loaded"}

    def is_alive(self, key):
        return key in self.loaded

    async def stream(self, key, payload, request_id, stream, cancel):
        if payload["text"] == "unknown":
            raise ServiceError("execution_unknown", "Controlled remote uncertainty", 502)
        yield {"text": payload["text"]}

    async def unload(self, key, timeout_s=30):
        self.loaded.discard(key)
        return {"test_double_closed": True}

    async def close(self, timeout_s=30):
        self.loaded.clear()

    def snapshot(self):
        return {key: {"alive": True, "rss_bytes": 0} for key in self.loaded}


def model(**changes):
    return ModelConfig(name="custom", version="1", task="independent-task", backend="independent-provider",
                       capabilities=["chat"], validation_input={"text": "validation"},
                       resident_mb=20, request_mb=5, **changes)


def settings(tmp_path, **changes):
    values = dict(database=str(tmp_path / "registry.db"), model_roots=[str(tmp_path)],
                  business_keys=["b" * 32], admin_keys=["a" * 32], maintenance_interval_s=100)
    values.update(changes)
    return Settings(**values)


async def collect(service, config, request_id="test", payload=None):
    return [result async for result in service.infer("chat", config.model_id, payload or {"text": "hello"},
                                                      asyncio.Event(), request_id)]


async def eventually(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        await asyncio.sleep(.005)


async def test_injected_provider_needs_no_core_names_and_preserves_external_quarantine(tmp_path):
    service = Service(settings(tmp_path), ControlledExecutor(), ControlledCatalog(remote=True))
    await service.start()
    config = model()
    try:
        await service.add(config)
        result = await collect(service, config)
        assert result[0]["mock"] is False and result[0]["output"]["text"] == "hello"
        assert (await service.status())["models"][0]["load"]["state"] == "external"
        with pytest.raises(ServiceError, match="Controlled remote"):
            await collect(service, config, "uncertain", {"text": "unknown"})
        assert service.scheduler.has_quarantine(config.instance_key)
        assert service.registry.uncertainties()[0]["request_id"] == "uncertain"
        assert (await service.readiness())["status"] == "not_ready"
        await service.reconcile("uncertain", True)
        assert (await service.readiness())["status"] == "ready"
        assert "model_service_requests_total{outcome=\"execution_unknown\"} 1" in await service.metrics()
    finally:
        await service.close()


async def test_preload_does_not_enable_and_same_object_restart_resets_reservations(tmp_path):
    service = Service(settings(tmp_path), ControlledExecutor(), ControlledCatalog())
    await service.start()
    config = model()
    try:
        await service.add(config, enable=False)
        await service.operate(config.model_id, "unload")
        state = await service.load(config.model_id)
        assert not state["enabled"] and state["load"]["state"] == "loaded"
        with pytest.raises(ServiceError) as error:
            await collect(service, config)
        assert error.value.code == "model_disabled"
        await service.close()
        await service.start()
        assert service.scheduler.snapshot()["reserved_mb"] == 0
        assert service.lifecycle.status(config)["state"] == "unloaded"
        await service.enable(config.model_id)
        await collect(service, config)
    finally:
        await service.close()


async def test_revalidation_invalidates_queued_business_request(tmp_path):
    service = Service(settings(tmp_path, max_executions=1), ControlledExecutor(), ControlledCatalog())
    await service.start()
    config = model()
    await service.add(config)
    holder = await service.scheduler.acquire(config.model_copy(update={"name": "other"}), "blocker", asyncio.Event())
    queued = asyncio.create_task(collect(service, config, "old-business"))
    validating = None
    try:
        await eventually(lambda: service.scheduler.has_waiters(config.instance_key))
        validating = asyncio.create_task(service.validate(config.model_id))
        with pytest.raises(ServiceError) as error:
            await queued
        assert error.value.code == "model_paused"
        await service.scheduler.release(holder)
        await validating
        assert service.registry.get(config.model_id)["enabled"]
        assert service.scheduler.snapshot()["queue_length"] == 0
    finally:
        await service.scheduler.release(holder)
        await asyncio.gather(*(task for task in [queued, validating] if task), return_exceptions=True)
        await service.close()


async def test_duplicate_id_cannot_replace_original_cancellation_registration(tmp_path):
    service = Service(settings(tmp_path), ControlledExecutor(), ControlledCatalog())
    await service.start()
    config = model()
    await service.add(config)
    cancel = asyncio.Event()
    original = service.infer("chat", config.model_id, {"text": "hello"}, cancel, "same")
    await anext(original)
    try:
        with pytest.raises(ServiceError) as error:
            await collect(service, config, "same")
        assert error.value.code == "duplicate_request"
        assert service.runtime.coordinator.pending_count() == 1
        service.runtime.coordinator.begin_shutdown()
        assert cancel.is_set()
    finally:
        await original.aclose()
        await service.close()


async def test_resident_load_failure_backs_off_and_reports_not_ready(tmp_path):
    executor = ControlledExecutor()
    service = Service(settings(tmp_path, maintenance_interval_s=.01, resident_retry_seconds=1),
                      executor, ControlledCatalog())
    await service.start()
    config = model(load_policy="resident")
    try:
        await service.add(config)
        executor.fail_load = True
        await service.operate(config.model_id, "unload")
        await eventually(lambda: executor.loads == 2)
        await asyncio.sleep(.08)
        assert executor.loads == 2
        assert service.scheduler.snapshot()["active_count"] == 0
        assert (await service.readiness())["status"] == "not_ready"
    finally:
        await service.close()


async def test_failed_startup_releases_lock_and_can_be_retried(tmp_path):
    configured = settings(tmp_path)
    with sqlite3.connect(configured.database) as db:
        db.execute("PRAGMA user_version=99")
    failed = Service(configured)
    with pytest.raises(RuntimeError, match="schema"):
        await failed.start()
    with sqlite3.connect(configured.database) as db:
        db.execute("PRAGMA user_version=1")
    recovered = Service(configured)
    await recovered.start()
    await recovered.close()


async def test_concurrent_close_is_idempotent_and_releases_ownership(tmp_path):
    configured = settings(tmp_path)
    service = Service(configured, ControlledExecutor(), ControlledCatalog())
    await service.start()
    await service.add(model())
    await asyncio.gather(service.close(), service.close(), service.close())
    replacement = Service(configured)
    await replacement.start()
    await replacement.close()


async def test_production_rejects_synthetic_provider_by_metadata_not_name(tmp_path):
    service = Service(settings(tmp_path, deployment_mode="production", allow_mock=False),
                      ControlledExecutor(), ControlledCatalog(synthetic=True))
    await service.start()
    try:
        with pytest.raises(ServiceError) as error:
            await service.add(model())
        assert error.value.code == "mock_disabled"
        assert service.registry.all() == []
        assert Path(service.settings.database).stat().st_mode & 0o777 == 0o600
    finally:
        await service.close()


def test_core_module_boundaries_do_not_depend_on_backend_names_or_scheduler_storage():
    """Prevent reintroducing the concrete coupling removed in this change."""
    for name in ("service", "coordinator", "management", "configuration"):
        tree = ast.parse(Path(f"model_service/{name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in {"backends", "tasks", "executor", "torch", "openvino", "httpx"}
                               for part in (node.module or "").split(".")), (name, node.module)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
                if node.value.attr == "scheduler":
                    assert not node.attr.startswith("_") and node.attr not in {"active", "queue", "quarantined", "residents"}
                if node.value.attr == "config":
                    assert node.attr != "backend"
