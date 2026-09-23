"""Remote intent and quarantine differ while concurrent HTTP calls are in flight."""
from __future__ import annotations

import asyncio

import pytest

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service


class ControlledHTTPExecutor:
    """Fake HTTP execution outcomes; exercises the real persistent coordinator."""

    def __init__(self):
        self.loaded = set()
        self.gates = {name: asyncio.Event() for name in ("A", "B")}
        self.entered = {name: asyncio.Event() for name in ("A", "B")}

    async def load(self, config):
        self.loaded.add(config.instance_key)
        return {"management": "external"}

    def is_alive(self, key):
        return key in self.loaded

    async def stream(self, key, payload, request_id, stream, cancel):
        if request_id in self.gates:
            self.entered[request_id].set()
            await self.gates[request_id].wait()
        if request_id == "A":
            raise ServiceError("execution_unknown", "Simulated unknown remote completion", 502)
        yield {"text": "controlled HTTP result"}

    async def unload(self, key, timeout_s=30):
        self.loaded.discard(key)
        return {"adapter_closed": True, "management": "external"}

    def snapshot(self):
        return {}

    async def close(self, timeout_s=30):
        self.loaded.clear()


def settings(tmp_path):
    return Settings(database=str(tmp_path / "registry.db"), model_roots=[str(tmp_path)],
                    business_keys=["business"], admin_keys=["admin"], maintenance_interval_s=100)


def model():
    return ModelConfig(name="remote", version="1", task="http_json", backend="http",
                       concurrency=2, capabilities=["chat"], resident_mb=20, request_mb=5,
                       options={"base_url": "http://localhost"},
                       validation_input={"messages": [{"role": "user", "content": "test"}]})


async def request(service, config, request_id):
    return [result async for result in service.infer("chat", config.model_id, config.validation_input,
                                                     asyncio.Event(), request_id)]


async def fail_one_request(service, executor, config):
    failed = asyncio.create_task(request(service, config, "A"))
    await executor.entered["A"].wait()
    executor.gates["A"].set()
    with pytest.raises(ServiceError) as error:
        await failed
    assert error.value.code == "execution_unknown"


@pytest.mark.asyncio
async def test_reconcile_unknown_request_while_other_remote_request_is_still_running(tmp_path):
    executor = ControlledHTTPExecutor()
    service = Service(settings(tmp_path), executor=executor)
    await service.start()
    config = model()
    await service.add(config)
    running = asyncio.create_task(request(service, config, "B"))
    try:
        await executor.entered["B"].wait()
        await fail_one_request(service, executor, config)
        assert len((await service.status())["uncertain_requests"]) == 2
        await service.reconcile("A", True)
        executor.gates["B"].set()
        await running
        status = await service.status()
        assert status["uncertain_requests"] == []
        assert status["models"][0]["management_state"] == "ready"
        assert (await request(service, config, "C"))[0]["output"]["text"]
    finally:
        executor.gates["B"].set()
        await asyncio.gather(running, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_reconcile_preserves_explicit_drain_intent(tmp_path, restart):
    executor = ControlledHTTPExecutor()
    configured = settings(tmp_path)
    service = Service(configured, executor=executor)
    await service.start()
    config = model()
    await service.add(config)
    await fail_one_request(service, executor, config)
    with pytest.raises(ServiceError) as error:
        await service.operate(config.model_id, "unload", timeout_s=0.01)
    assert error.value.code == "drain_timeout"
    if restart:
        await service.close()
        service = Service(configured, executor=ControlledHTTPExecutor())
        await service.start()
    try:
        assert (await service.status())["models"][0]["management_state"] == "draining"
        await service.reconcile("A", True)
        status = await service.status()
        assert status["uncertain_requests"] == []
        assert status["models"][0]["management_state"] == "draining"
        with pytest.raises(ServiceError) as error:
            await request(service, config, "C")
        assert error.value.code == "model_paused"
        await service.operate(config.model_id, "unload")
        assert (await request(service, config, "D"))[0]["output"]["text"]
    finally:
        await service.close()
