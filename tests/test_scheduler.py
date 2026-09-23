import asyncio

import pytest

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.scheduler import Scheduler


def settings(**kwargs):
    return Settings(business_keys=["business"], admin_keys=["admin"], **kwargs)


def config(**kwargs):
    return ModelConfig(name="model", version="v1", capabilities=["embeddings"], task="mock", backend="mock",
                       validation_input={"texts": ["test"]}, **kwargs)


async def test_resident_memory_reserved_once_and_temporary_released():
    scheduler = Scheduler(settings(total_memory_mb=140, max_executions=4))
    model = config(concurrency=4, resident_mb=100, request_mb=10)
    leases = await asyncio.gather(*(scheduler.acquire(model, str(i), asyncio.Event()) for i in range(4)))
    assert scheduler.snapshot()["resident_mb"] == 100
    assert scheduler.snapshot()["temporary_mb"] == 40
    with pytest.raises(ServiceError, match="in use"):
        await scheduler.drop_resident(model.instance_key)
    await asyncio.gather(*(scheduler.release(lease) for lease in leases))
    assert scheduler.snapshot()["reserved_mb"] == 100
    await scheduler.drop_resident(model.instance_key)
    assert scheduler.snapshot()["reserved_mb"] == 0


async def test_queue_overflow_timeout_cancel_and_pause():
    scheduler = Scheduler(settings(queue_size=1, queue_timeout_s=0.08))
    model = config()
    lease = await scheduler.acquire(model, "one", asyncio.Event())
    waiter = asyncio.create_task(scheduler.acquire(model, "two", asyncio.Event()))
    await asyncio.sleep(0.01)
    with pytest.raises(ServiceError) as caught:
        await scheduler.acquire(model, "three", asyncio.Event())
    assert caught.value.code == "queue_full"
    with pytest.raises(ServiceError) as caught:
        await waiter
    assert caught.value.code == "queue_timeout"
    cancel = asyncio.Event()
    waiter = asyncio.create_task(scheduler.acquire(model, "four", cancel))
    await asyncio.sleep(0.01)
    cancel.set()
    with pytest.raises(ServiceError) as caught:
        await waiter
    assert caught.value.code == "cancelled"
    waiter = asyncio.create_task(scheduler.acquire(model, "five", asyncio.Event()))
    await asyncio.sleep(0.01)
    await scheduler.pause(model.instance_key)
    with pytest.raises(ServiceError) as caught:
        await waiter
    assert caught.value.code == "model_paused"
    assert scheduler.snapshot()["queue_length"] == 0
    await scheduler.release(lease)


async def test_permanent_budget_limit_and_temporary_shortage():
    scheduler = Scheduler(settings(total_memory_mb=100, queue_timeout_s=0.03))
    with pytest.raises(ServiceError) as caught:
        await scheduler.acquire(config(resident_mb=90, request_mb=11), "too-large", asyncio.Event())
    assert caught.value.code == "resource_limit"
    a = config(resident_mb=80, request_mb=10)
    first = await scheduler.acquire(a, "first", asyncio.Event())
    await scheduler.release(first)
    b = a.model_copy(update={"name": "other"})
    with pytest.raises(ServiceError) as caught:
        await scheduler.acquire(b, "other", asyncio.Event())
    assert caught.value.code == "queue_timeout"
    assert scheduler.snapshot()["reserved_mb"] == 80


async def test_fifo_and_drain_timeout_do_not_release_active_lease():
    scheduler = Scheduler(settings())
    model = config()
    first = await scheduler.acquire(model, "first", asyncio.Event())
    second = asyncio.create_task(scheduler.acquire(model, "second", asyncio.Event()))
    third = asyncio.create_task(scheduler.acquire(model, "third", asyncio.Event()))
    await asyncio.sleep(0.01)
    with pytest.raises(ServiceError) as caught:
        await scheduler.wait_idle(model.instance_key, .01)
    assert caught.value.code == "drain_timeout"
    assert scheduler.snapshot()["active_count"] == 1
    await scheduler.release(first)
    two = await second
    assert not third.done()
    await scheduler.release(two)
    await scheduler.release(await third)
