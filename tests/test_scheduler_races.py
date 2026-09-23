"""Admission and lifecycle regressions using controlled, synthetic executors."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from model_service.contracts import BackendFeatures, ModelConfig, ServiceError, Settings
from model_service.lifecycle import Lifecycle
from model_service.scheduler import Scheduler


def settings(**changes):
    return Settings(business_keys=["business"], admin_keys=["admin"], **changes)


def model(**changes):
    values = {"name": "race-model", "version": "1", "capabilities": ["chat"],
              "backend": "mock", "task": "mock", "resident_mb": 20, "request_mb": 5,
              "validation_input": {"messages": [{"role": "user", "content": "test"}]}}
    return ModelConfig(**(values | changes))


async def wait_until_queued(scheduler, key=None):
    async with asyncio.timeout(2):
        while not scheduler.has_waiters(key):
            await asyncio.sleep(0)


async def test_pause_resume_rejects_waiters_from_before_pause():
    scheduler, config = Scheduler(settings()), model()
    lease = await scheduler.acquire(config, "active", asyncio.Event())
    waiting = asyncio.create_task(scheduler.acquire(config, "old-waiter", asyncio.Event()))
    await wait_until_queued(scheduler)
    # Neither uncontended lock acquisition yields to the old waiter. A boolean
    # paused flag alone loses this management event and admits the old request.
    await scheduler.pause(config.instance_key)
    await scheduler.resume(config.instance_key)
    await scheduler.release(lease)
    with pytest.raises(ServiceError) as error:
        await waiting
    assert error.value.code == "model_paused"
    assert not scheduler.has_waiters()
    fresh = await scheduler.acquire(config, "fresh-request", asyncio.Event())
    await scheduler.release(fresh)


async def test_validation_can_reserve_while_business_admission_remains_paused():
    scheduler, config = Scheduler(settings()), model()
    await scheduler.pause(config.instance_key)
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(config, "business", asyncio.Event())
    assert error.value.code == "model_paused"
    lease = await scheduler.acquire(config, "validation", asyncio.Event(), administrative=True)
    assert scheduler.is_paused(config.instance_key)
    assert scheduler.normal_active_count() == 1
    await scheduler.release(lease)
    assert scheduler.is_paused(config.instance_key)


async def test_administrative_requests_obey_fifo_and_budget_limits():
    scheduler = Scheduler(settings(max_executions=1, total_memory_mb=100))
    first, second = model(), model(name="other")
    held = await scheduler.acquire(first, "running", asyncio.Event())
    waiter = asyncio.create_task(scheduler.acquire(second, "business", asyncio.Event()))
    await wait_until_queued(scheduler, second.instance_key)
    await scheduler.pause(first.instance_key)
    validation = asyncio.create_task(scheduler.acquire(first, "validation", asyncio.Event(), administrative=True))
    await asyncio.sleep(0)
    await scheduler.release(held)
    business = await waiter
    assert not validation.done()
    await scheduler.release(business)
    await scheduler.release(await validation)
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(model(resident_mb=100), "oversized-validation", asyncio.Event(), administrative=True)
    assert error.value.code == "resource_limit"


async def test_idle_claim_is_atomic_and_does_not_claim_other_management_pause():
    scheduler, config = Scheduler(settings()), model()
    held = await scheduler.acquire(config, "running", asyncio.Event())
    assert not await scheduler.try_pause_idle(config.instance_key)
    await scheduler.release(held)
    assert await scheduler.try_pause_idle(config.instance_key)
    assert not await scheduler.try_pause_idle(config.instance_key)
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(config, "racing-request", asyncio.Event())
    assert error.value.code == "model_paused"
    await scheduler.resume(config.instance_key)
    admitted = await scheduler.acquire(config, "after-maintenance", asyncio.Event())
    await scheduler.release(admitted)


async def test_idle_claim_refuses_model_with_queued_work_but_no_active_work():
    scheduler = Scheduler(settings(max_executions=1))
    first, queued = model(), model(name="queued-model")
    held = await scheduler.acquire(first, "running", asyncio.Event())
    waiting = asyncio.create_task(scheduler.acquire(queued, "queued", asyncio.Event()))
    await wait_until_queued(scheduler, queued.instance_key)
    assert scheduler.count(queued.instance_key) == 0
    assert not await scheduler.try_pause_idle(queued.instance_key)
    assert not scheduler.is_paused(queued.instance_key)
    await scheduler.release(held)
    await scheduler.release(await waiting)


async def test_duplicate_request_id_cannot_queue_for_a_different_model():
    scheduler = Scheduler(settings(max_executions=1))
    config, other = model(), model(name="other")
    held = await scheduler.acquire(config, "held", asyncio.Event())
    waiting = asyncio.create_task(scheduler.acquire(config, "duplicate", asyncio.Event()))
    await wait_until_queued(scheduler)
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(other, "duplicate", asyncio.Event())
    assert error.value.code == "duplicate_request"
    await scheduler.release(held)
    await scheduler.release(await waiting)


async def test_quarantine_queries_are_snapshots_and_keep_resources_until_reconciled():
    scheduler, config = Scheduler(settings()), model()
    lease = await scheduler.acquire(config, "remote-unknown", asyncio.Event())
    await scheduler.quarantine(lease)
    assert scheduler.has_quarantine(config.instance_key)
    assert scheduler.normal_active_count() == 0
    assert scheduler.count(config.instance_key) == 1
    ids = scheduler.quarantine_request_ids()
    ids.clear()
    snapshot = scheduler.snapshot()
    snapshot["quarantined_requests"].clear()
    snapshot["active_by_model"].clear()
    assert scheduler.quarantine_request_ids() == ["remote-unknown"]
    with pytest.raises(ServiceError) as error:
        await scheduler.resume(config.instance_key)
    assert error.value.code == "execution_unknown"
    with pytest.raises(ServiceError) as error:
        await scheduler.wait_idle(config.instance_key, 0)
    assert error.value.code == "drain_timeout"
    assert scheduler.snapshot()["reserved_mb"] == 25
    await scheduler.reconcile("remote-unknown")
    assert scheduler.snapshot()["reserved_mb"] == 20
    assert not scheduler.has_quarantine(config.instance_key)
    await scheduler.resume(config.instance_key)


async def test_stale_release_cannot_remove_a_reused_request_id():
    scheduler, config = Scheduler(settings()), model()
    old = await scheduler.acquire(config, "reused-id", asyncio.Event())
    await scheduler.release(old)
    current = await scheduler.acquire(config, "reused-id", asyncio.Event())
    await scheduler.release(old)
    assert scheduler.count(config.instance_key) == 1
    await scheduler.release(current)


async def test_expired_waiter_cannot_claim_resources_released_at_its_deadline(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("model_service.scheduler.time", SimpleNamespace(monotonic=lambda: now[0]))
    scheduler, config = Scheduler(settings()), model()
    held = await scheduler.acquire(config, "held", asyncio.Event())
    waiting = asyncio.create_task(scheduler.acquire(config, "expired", asyncio.Event(), timeout_s=1))
    await wait_until_queued(scheduler)
    now[0] += 2
    await scheduler.release(held)
    with pytest.raises(ServiceError) as error:
        await waiting
    assert error.value.code == "queue_timeout"
    assert scheduler.count(config.instance_key) == 0


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
async def test_invalid_timeouts_never_admit_or_change_resource_accounting(timeout):
    scheduler, config = Scheduler(settings()), model()
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(config, "invalid-timeout", asyncio.Event(), timeout_s=timeout)
    assert error.value.code == "invalid_timeout"
    assert scheduler.snapshot()["reserved_mb"] == 0
    with pytest.raises(ServiceError) as error:
        await scheduler.wait_idle(config.instance_key, timeout)
    assert error.value.code == "invalid_timeout"


@pytest.mark.parametrize("field", ["resident_mb", "request_mb"])
async def test_negative_budgets_rejected_at_config_and_scheduler_boundary(field):
    with pytest.raises(ValidationError):
        model(**{field: -1})
    # Trusted Python integrations may use model_copy, which bypasses validation.
    invalid = model().model_copy(update={field: -1})
    scheduler = Scheduler(settings())
    with pytest.raises(ServiceError) as error:
        await scheduler.acquire(invalid, "invalid-budget", asyncio.Event())
    assert error.value.code == "resource_limit"
    assert scheduler.snapshot()["reserved_mb"] == 0


class ControlledExecutor:
    """Synthetic executor: it proves lifecycle semantics, not real model support."""

    def __init__(self):
        self.alive = False
        self.load_calls = 0
        self.ready = asyncio.Event()
        self.load_started = asyncio.Event()
        self.unload_timeout = False
        self.report_state = "loaded"

    async def load(self, config):
        self.load_calls += 1
        self.alive = True
        self.load_started.set()
        await self.ready.wait()
        return {"state": self.report_state, "metadata": {"synthetic": True}}

    def is_alive(self, key):
        return self.alive

    async def unload(self, key, timeout_s):
        if self.unload_timeout:
            raise ServiceError("unload_timeout", "Synthetic slow unload", 409)
        self.alive = False
        return {"process_exited": True}


def lifecycle(executor, **features):
    return Lifecycle(executor, lambda config: BackendFeatures(**features))


async def test_lifecycle_concurrent_calls_load_one_instance_and_return_copies():
    executor, config = ControlledExecutor(), model()
    manager = lifecycle(executor)
    loading = [asyncio.create_task(manager.ensure_loaded(config)) for _ in range(12)]
    await executor.load_started.wait()
    executor.ready.set()
    results = await asyncio.gather(*loading)
    assert executor.load_calls == 1
    results[0]["metadata"].clear()
    assert manager.status(config)["metadata"] == {"synthetic": True}
    assert results[1]["metadata"] == {"synthetic": True}
    await manager.unload(config, 1)


async def test_unload_timeout_never_reuses_the_still_living_worker():
    executor, config = ControlledExecutor(), model()
    executor.ready.set()
    manager = lifecycle(executor)
    await manager.ensure_loaded(config)
    executor.unload_timeout = True
    with pytest.raises(ServiceError) as error:
        await manager.unload(config, .01)
    assert error.value.code == "unload_timeout"
    assert manager.status(config)["state"] == "unload_pending"
    with pytest.raises(ServiceError) as error:
        await manager.ensure_loaded(config)
    assert error.value.code == "model_unavailable"
    assert executor.load_calls == 1
    executor.unload_timeout = False
    await manager.unload(config, 1)
    await manager.ensure_loaded(config)
    assert executor.load_calls == 2
    await manager.unload(config, 1)


async def test_failed_alive_worker_is_not_converted_to_loaded():
    executor, config = ControlledExecutor(), model()
    executor.ready.set()
    executor.report_state = "failed"
    manager = lifecycle(executor)
    with pytest.raises(ServiceError) as error:
        await manager.ensure_loaded(config)
    assert error.value.code == "model_unavailable"
    assert manager.status(config)["state"] == "failed"
    executor.report_state = "loaded"
    with pytest.raises(ServiceError):
        await manager.ensure_loaded(config)
    assert executor.load_calls == 1
    await manager.unload(config, 1)
    await manager.ensure_loaded(config)
    await manager.unload(config, 1)


async def test_external_lifecycle_is_declared_by_capabilities_not_backend_name():
    executor, config = ControlledExecutor(), model(backend="custom-remote")
    executor.ready.set()
    manager = lifecycle(executor, management="external", requires_artifact=False)
    assert manager.status(config)["state"] == "external"
    await manager.ensure_loaded(config)
    assert manager.status(config)["adapter_state"] == "loaded"
    await manager.unload(config, 1)
    assert manager.status(config)["state"] == "external"
    assert manager.status(config)["adapter_state"] == "unloaded"


async def test_settle_load_retains_ownership_across_repeated_cancellation():
    executor, config = ControlledExecutor(), model()
    manager = lifecycle(executor)
    initial = asyncio.create_task(manager.ensure_loaded(config))
    await executor.load_started.wait()
    initial.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initial
    assert manager.status(config)["state"] == "load_pending"
    settling = asyncio.create_task(manager.settle_load(config))
    await asyncio.sleep(0)
    settling.cancel()
    await asyncio.sleep(0)
    assert not settling.done()
    executor.ready.set()
    await settling
    assert manager.status(config)["state"] == "loaded"
    await manager.unload(config, 1)
