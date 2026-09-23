import asyncio
import os
import signal
import time
import uuid

import pytest

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service


def model(**kwargs):
    values = dict(name="mock-dense", version="v1", task="mock", backend="mock",
                  capabilities=["embeddings"], validation_input={"texts": ["test"]},
                  resident_mb=20, request_mb=5)
    values.update(kwargs)
    return ModelConfig(**values)


@pytest.fixture
async def service(tmp_path):
    settings = Settings(database=str(tmp_path / "registry.db"), model_roots=[str(tmp_path)],
                        business_keys=["business"], admin_keys=["admin"],
                        maintenance_interval_s=100, queue_timeout_s=.5)
    service = Service(settings)
    await service.start()
    yield service
    await service.close()


async def collect(service, config, *, cancel=None, stream=False):
    return [item async for item in service.infer(config.capabilities[0], config.model_id,
                    config.validation_input, cancel or asyncio.Event(), uuid.uuid4().hex, stream)]


async def test_single_load_under_concurrent_requests_and_unload_reload(service):
    config = model(concurrency=4, options={"delay_s": .05})
    await service.add(config)
    initial = (await service.status())["workers"][config.instance_key]["load_count"]
    await asyncio.gather(*(collect(service, config) for _ in range(8)))
    status = await service.status()
    assert status["workers"][config.instance_key]["load_count"] == initial == 1
    assert status["scheduler"]["active_count"] == 0
    assert status["scheduler"]["resident_mb"] == 20
    await service.operate(config.model_id, "unload")
    assert service.registry.get(config.model_id)["enabled"]
    await collect(service, config)
    assert (await service.status())["workers"][config.instance_key]["load_count"] == 2


async def test_failed_validation_stays_disabled_and_is_persisted(service):
    config = model(options={"fail": True})
    with pytest.raises(ServiceError):
        await service.add(config)
    row = service.registry.get(config.model_id)
    assert not row["enabled"] and not row["validated"] and row["validation_error"]
    with pytest.raises(ServiceError) as caught:
        await service.enable(config.model_id)
    assert caught.value.code == "not_validated"


async def test_stream_pin_cancel_and_drain_timeout(service):
    config = model(options={"stream_chunks": 8, "chunk_delay_s": .06})
    await service.add(config)
    cancel = asyncio.Event()
    iterator = service.infer("embeddings", config.model_id, config.validation_input, cancel, "stream-test", True)
    first = await anext(iterator)
    assert first["mock"]
    assert (await service.status())["scheduler"]["active_count"] == 1
    with pytest.raises(ServiceError) as caught:
        await service.operate(config.model_id, "remove", .02)
    assert caught.value.code == "drain_timeout"
    assert service.lifecycle.is_alive(config.instance_key)
    assert service.registry.get(config.model_id)["management_state"] == "draining"
    cancel.set()
    await iterator.aclose()
    assert (await service.status())["scheduler"]["active_count"] == 0
    await service.operate(config.model_id, "remove")
    assert service.registry.all() == []


async def test_cancel_does_not_release_before_stop_confirmation(service):
    config = model(options={"delay_s": .12, "cancel_delay_s": .12})
    await service.add(config)
    cancel = asyncio.Event()
    running = asyncio.create_task(collect(service, config, cancel=cancel))
    await asyncio.sleep(.04)
    cancel.set()
    await asyncio.sleep(.025)
    assert (await service.status())["scheduler"]["active_count"] == 1
    with pytest.raises(ServiceError) as caught:
        await running
    assert caught.value.code == "cancelled"
    assert (await service.status())["scheduler"]["active_count"] == 0


async def test_execution_deadline_reports_timeout_after_underlying_stop(service):
    config = model(options={"delay_s": .1, "cancel_delay_s": .04})
    await service.add(config)
    service.settings.execution_timeout_s = .02
    with pytest.raises(ServiceError) as caught:
        await collect(service, config)
    assert caught.value.code == "execution_timeout" and caught.value.status == 504
    assert (await service.status())["scheduler"]["active_count"] == 0


async def test_remove_checks_dependencies_aliases_and_preserves_files(service, tmp_path):
    asset = tmp_path / "weights.bin"
    asset.write_bytes(b"fake asset")
    config = model(path=str(asset))
    await service.add(config)
    await service.set_alias("default:embeddings", config.model_id)
    await service.add_dependency("search-service", config.model_id)
    with pytest.raises(ServiceError) as caught:
        await service.operate(config.model_id, "remove")
    assert caught.value.code == "model_referenced"
    await service.delete_alias("default:embeddings")
    await service.delete_dependency("search-service")
    await service.operate(config.model_id, "disable")
    with pytest.raises(ServiceError) as caught:
        await collect(service, config)
    assert caught.value.code == "model_disabled"
    await service.enable(config.model_id)
    await service.operate(config.model_id, "remove")
    assert asset.read_bytes() == b"fake asset"


async def test_worker_crash_is_reported_no_retry_and_next_request_recovers(service):
    config = model(options={"delay_s": .2})
    await service.add(config)
    running = asyncio.create_task(collect(service, config))
    await asyncio.sleep(.06)
    pid = (await service.status())["workers"][config.instance_key]["pid"]
    os.kill(pid, signal.SIGKILL)  # Test injects failure; service itself never force kills.
    with pytest.raises(ServiceError) as caught:
        await running
    assert caught.value.code == "process_failed"
    assert (await service.status())["scheduler"]["reserved_mb"] == 0
    result = await collect(service, config)
    assert result[0]["mock"]
    assert (await service.status())["workers"][config.instance_key]["load_count"] == 2


async def test_restart_restores_registry_but_not_loaded_state(service):
    config = model()
    await service.add(config)
    await service.set_alias("default:embeddings", config.model_id)
    await service.close()
    restored = Service(service.settings)
    await restored.start()
    try:
        status = await restored.status()
        assert status["models"][0]["enabled"]
        assert status["models"][0]["load"]["state"] == "unloaded"
        assert status["aliases"]["default:embeddings"] == config.model_id
        result = [item async for item in restored.infer("embeddings", None, config.validation_input, asyncio.Event(), "restored")]
        assert result[0]["mock"]
    finally:
        await restored.close()


async def test_second_controller_refused(service):
    other = Service(service.settings)
    with pytest.raises(ServiceError) as caught:
        await other.start()
    assert caught.value.code == "controller_exists"


async def test_path_restriction_and_immutable_version(service, tmp_path):
    with pytest.raises(ServiceError) as caught:
        await service.add(model(path="/etc/passwd"))
    assert caught.value.code == "path_forbidden"
    link = tmp_path / "escape"
    link.symlink_to("/etc/passwd")
    with pytest.raises(ServiceError) as caught:
        await service.add(model(path=str(tmp_path)))
    assert caught.value.code == "path_forbidden"
    entry = tmp_path / "model.xml"
    entry.write_text("<net/>")
    with pytest.raises(ServiceError) as caught:
        await service.add(model(path=str(entry)))
    assert caught.value.code == "path_forbidden"  # Escaping sibling assets are rejected too.
    config = model()
    await service.add(config)
    with pytest.raises(ServiceError) as caught:
        await service.add(config)
    assert caught.value.code == "version_exists"


async def test_remote_uncertainty_survives_restart_and_requires_confirmation(service):
    config = model()
    await service.add(config)
    # Exercise durable recovery independent of a remote server by injecting the
    # same persisted intent that is written before every HTTP execution.
    service.registry.uncertain("remote-request", config.model_id, "controller crashed")
    await service.close()
    restored = Service(service.settings)
    await restored.start()
    try:
        assert (await restored.status())["scheduler"]["active_count"] == 1
        with pytest.raises(ServiceError) as caught:
            await restored.reconcile("remote-request", False)
        assert caught.value.code == "confirmation_required"
        await restored.reconcile("remote-request", True)
        assert (await restored.status())["scheduler"]["active_count"] == 0
        assert restored.registry.uncertainties() == []
    finally:
        await restored.close()


@pytest.mark.parametrize("policy", ["on_demand", "resident"])
async def test_idle_unload_and_resident_restart_policies(tmp_path, policy):
    settings = Settings(database=str(tmp_path / "policies.db"), model_roots=[str(tmp_path)],
                        business_keys=["business"], admin_keys=["admin"], maintenance_interval_s=.02)
    config = model(load_policy=policy, idle_seconds=.04)
    service = Service(settings)
    await service.start()
    try:
        await service.add(config)
        if policy == "resident":
            await service.close()
            service = Service(settings)
            await service.start()
        target = "unloaded" if policy == "on_demand" else "loaded"
        deadline = time.monotonic() + 3
        while service.lifecycle.status(config)["state"] != target:
            assert time.monotonic() < deadline
            await asyncio.sleep(.02)
        status = await service.status()
        assert status["scheduler"]["active_count"] == 0
        assert status["scheduler"]["resident_mb"] == (0 if policy == "on_demand" else config.resident_mb)
        assert service.registry.get(config.model_id)["enabled"]
    finally:
        await service.close()
