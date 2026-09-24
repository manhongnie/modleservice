import asyncio
import pytest
from model_service.contracts import ModelConfig, Settings, ServiceError
from model_service.scheduler import Scheduler


def model(name='video', **changes):
    return ModelConfig(name=name, version='1', task='mock', backend='mock', capabilities=['video_generation'],
                       validation_input={}, resident_mb=100, request_mb=20,
                       gpu_resident_mb=600, gpu_request_mb=100, **changes)


async def test_gpu_admission_is_atomic_with_host_budget_and_residency_is_not_double_counted():
    scheduler = Scheduler(Settings(business_keys=['b'], admin_keys=['a'], total_gpu_memory_mb=800))
    config = model(concurrency=2)
    one = await scheduler.acquire(config, 'one', asyncio.Event())
    two = await scheduler.acquire(config, 'two', asyncio.Event())
    assert scheduler.snapshot()['gpu_resident_mb'] == 600
    assert scheduler.snapshot()['gpu_temporary_mb'] == 200
    with pytest.raises(ServiceError) as caught:
        await scheduler.acquire(model('other'), 'other', asyncio.Event(), timeout_s=.01)
    assert caught.value.code == 'queue_timeout'
    assert scheduler.snapshot()['resident_instances'] == 1
    await scheduler.release(one)
    await scheduler.release(two)
    assert scheduler.snapshot()['gpu_reserved_mb'] == 600
    await scheduler.drop_resident(config.instance_key)
    assert scheduler.snapshot()['gpu_reserved_mb'] == 0


async def test_gpu_over_budget_never_loads_or_reserves_host_memory():
    scheduler = Scheduler(Settings(business_keys=['b'], admin_keys=['a'], total_gpu_memory_mb=500))
    with pytest.raises(ServiceError) as caught:
        await scheduler.acquire(model(), 'request', asyncio.Event())
    assert caught.value.code == 'resource_limit'
    assert scheduler.snapshot()['reserved_mb'] == scheduler.snapshot()['gpu_reserved_mb'] == 0


async def test_recovered_uncertainty_retains_gpu_reservation_even_after_capacity_change():
    scheduler = Scheduler(Settings(business_keys=['b'], admin_keys=['a'], total_gpu_memory_mb=0))
    config = model()
    await scheduler.restore_quarantine(config, 'uncertain')
    assert scheduler.snapshot()['gpu_reserved_mb'] == 700
    await scheduler.reconcile('uncertain')
    assert scheduler.snapshot()['gpu_reserved_mb'] == 600
    await scheduler.drop_resident(config.instance_key)
    assert scheduler.snapshot()['gpu_reserved_mb'] == 0
