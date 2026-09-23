"""A shutdown deadline reports maintenance that still owns a native model load."""
import asyncio
import time

import pytest

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service


async def eventually(predicate, timeout=5):
    until = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= until:
            raise AssertionError("Runtime transition did not occur")
        await asyncio.sleep(.005)


async def test_resident_load_shutdown_timeout_retains_lock_then_can_retry(tmp_path):
    settings = Settings(database=str(tmp_path / "models.sqlite"), model_roots=[str(tmp_path)],
        business_keys=["business"], admin_keys=["admin"], maintenance_interval_s=.01,
        drain_timeout_s=.05, load_timeout_s=.05)
    service = Service(settings)
    await service.start()
    model = ModelConfig(name="resident-loading", version="1", capabilities=["chat"],
        task="mock", backend="mock", load_policy="resident", resident_mb=20, request_mb=5,
        options={"load_delay_s": .5}, validation_input={"messages": [{"role": "user", "content": "hi"}]})
    # Represent a persisted validated registration whose worker starts at boot.
    service.registry.register(model)
    service.registry.validation(model.model_id, True)
    service.registry.enable(model.model_id, True)
    try:
        await eventually(lambda: bool(service.runtime.executor.snapshot()))
        began = time.monotonic()
        with pytest.raises(ServiceError) as error:
            await service.close()
        assert error.value.code == "shutdown_timeout"
        assert time.monotonic() - began < .35
        assert service.scheduler.snapshot()["active_count"] == 1
        assert service.registry.get(model.model_id)["enabled"]
        assert service.runtime.executor.is_alive(model.instance_key)

        competing = Service(settings)
        with pytest.raises(ServiceError) as error:
            await competing.start()
        assert error.value.code == "controller_exists"

        await eventually(lambda: service.scheduler.snapshot()["active_count"] == 0)
        # The loader has acknowledged completion, so a subsequent graceful close
        # can release the process and database ownership without force killing.
        await service.close()
        assert not service.runtime.executor.is_alive(model.instance_key)
        assert service.registry is None
        restarted = Service(settings)
        await restarted.start()
        try:
            assert restarted.registry.get(model.model_id)["enabled"]
        finally:
            await restarted.close()
    finally:
        # Give OS process teardown a generous budget when recovering a failed
        # assertion; the production deadline assertions above use the original value.
        settings.drain_timeout_s = 2
        await service.close()
