"""Regression tests for resource ownership during shutdown and slow native loads."""
from __future__ import annotations

import asyncio
import contextlib
import time

import pytest

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.executor import ProcessExecutor
from model_service.service import Service


def settings(tmp_path, **changes):
    return Settings(database=str(tmp_path / "models.db"), model_roots=[str(tmp_path)],
                    business_keys=["business"], admin_keys=["administrator"],
                    maintenance_interval_s=60, **changes)


def config(**options):
    return ModelConfig(name="race-mock", version="1", capabilities=["chat"], task="mock", backend="mock",
                       resident_mb=20, request_mb=5,
                       validation_input={"messages": [{"role": "user", "content": "test"}]},
                       options=options)


async def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("State transition did not happen before the deadline")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_load_timeout_does_not_release_lease_before_native_loading_finishes(tmp_path):
    configured = settings(tmp_path, load_timeout_s=0.03)
    executor = ProcessExecutor(load_timeout_s=0.03)
    service = Service(configured, executor=executor)
    await service.start()
    registered = asyncio.create_task(service.add(config(load_delay_s=0.25)))
    try:
        await eventually(lambda: bool(executor.snapshot()))
        await asyncio.sleep(0.07)  # Past timeout, but before simulated native loading finishes.
        snapshot = service.scheduler.snapshot()
        assert snapshot["active_count"] == 1
        assert snapshot["temporary_mb"] == 5
        assert snapshot["resident_mb"] == 20
        assert not registered.done()
        result = (await asyncio.gather(registered, return_exceptions=True))[0]
        assert isinstance(result, ServiceError)
        assert result.code == "load_timeout"
        assert service.scheduler.snapshot()["active_count"] == 0
    finally:
        await asyncio.gather(registered, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_active_request_stop_before_closing_registry(tmp_path):
    configured = settings(tmp_path, drain_timeout_s=2)
    executor = ProcessExecutor()
    service = Service(configured, executor=executor)
    await service.start()
    model = config(delay_s=0.15, cancel_delay_s=0.12)
    await service.add(model)
    cancel = asyncio.Event()

    async def infer():
        async for _ in service.infer("chat", model.model_id, model.validation_input, cancel, "shutdown-request"):
            pass

    request = asyncio.create_task(infer())
    await eventually(lambda: any(instance["busy"] for instance in executor.snapshot().values()))
    before = time.monotonic()
    try:
        close_result = (await asyncio.gather(service.close(), return_exceptions=True))[0]
        request_result = (await asyncio.gather(request, return_exceptions=True))[0]
        assert close_result is None, repr(close_result)
        assert time.monotonic() - before >= 0.10
        assert isinstance(request_result, ServiceError)
        assert request_result.code == "cancelled"
        assert service.scheduler.snapshot()["active_count"] == 0
        assert not any(instance["alive"] for instance in executor.snapshot().values())
    finally:
        cancel.set()
        await asyncio.gather(request, return_exceptions=True)
        with contextlib.suppress(ServiceError):
            await executor.close()
        with contextlib.suppress(ServiceError):
            await service.close()


@pytest.mark.asyncio
async def test_shutdown_timeout_preserves_control_lock_and_registry_until_retry(tmp_path):
    configured = settings(tmp_path, drain_timeout_s=0.03)
    service = Service(configured)
    await service.start()
    model = config()
    await service.add(model)
    cancel = asyncio.Event()
    iterator = service.infer("chat", model.model_id, model.validation_input, cancel, "slow-consumer")
    try:
        first = await anext(iterator)
        assert first["mock"]
        # Simulate a client still holding the last result before ASGI completes sending.
        with pytest.raises(ServiceError) as error:
            await service.close()
        assert error.value.code == "shutdown_timeout"
        assert cancel.is_set()
        assert (await service.status())["scheduler"]["active_count"] == 1

        competing = Service(configured)
        with pytest.raises(ServiceError) as error:
            await competing.start()
        assert error.value.code == "controller_exists"
    finally:
        await iterator.aclose()
        await service.close()

    # A successful retry releases ownership, allowing normal restart.
    restarted = Service(configured)
    await restarted.start()
    try:
        assert len((await restarted.status())["models"]) == 1
    finally:
        await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_timeout", [0.03, 1.0])
async def test_shutdown_accounts_for_admin_validation_waiting_in_resource_queue(tmp_path, drain_timeout):
    configured = settings(tmp_path, total_memory_mb=100, queue_timeout_s=0.15,
                          drain_timeout_s=drain_timeout)
    service = Service(configured)
    await service.start()
    resident = config().model_copy(update={"resident_mb": 90})
    queued = resident.model_copy(update={"name": "queued-admin-validation"})
    await service.add(resident)
    adding = asyncio.create_task(service.add(queued))
    closing = None
    try:
        await eventually(lambda: service.scheduler.snapshot()["queue_length"] == 1)
        assert service.scheduler.snapshot()["active_count"] == 0
        closing = asyncio.create_task(service.close())
        await asyncio.sleep(0.005)
        with pytest.raises(ServiceError) as error:
            await service.set_alias("during-shutdown", resident.model_id)
        assert error.value.code == "shutting_down"

        if drain_timeout < configured.queue_timeout_s:
            with pytest.raises(ServiceError) as error:
                await closing
            assert error.value.code == "shutdown_timeout"
            assert len((await service.status())["models"]) == 2
            competing = Service(configured)
            with pytest.raises(ServiceError) as error:
                await competing.start()
            assert error.value.code == "controller_exists"

        result = (await asyncio.gather(adding, return_exceptions=True))[0]
        assert isinstance(result, ServiceError), repr(result)
        assert result.code == "queue_timeout"
        if drain_timeout >= configured.queue_timeout_s:
            await closing
        else:
            await service.close()
    finally:
        await asyncio.gather(adding, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await service.close()

    restarted = Service(configured)
    await restarted.start()
    try:
        failed = next(row for row in (await restarted.status())["models"]
                      if row["config"]["name"] == queued.name)
        assert not failed["enabled"]
        assert not failed["validated"]
        assert failed["management_state"] == "validation_failed"
    finally:
        await restarted.close()
