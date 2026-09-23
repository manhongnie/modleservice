from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event

import pytest

from model_service.backends.http import HTTPBackend
from model_service.contracts import ModelConfig, ServiceError
from model_service.executor import ProcessExecutor


def config(**overrides):
    values = dict(name="worker-test", version="1", capabilities=["embeddings"], task="mock",
                  backend="mock", validation_input={"texts": ["hello"]}, concurrency=2)
    values.update(overrides)
    return ModelConfig(**values)


async def collect(executor, cfg, request_id="r", cancel=None, stream=False):
    return [chunk async for chunk in executor.stream(
        cfg.instance_key, {"texts": ["hello"]}, request_id, stream, cancel or asyncio.Event())]


async def test_process_load_is_single_flight_and_reused():
    executor = ProcessExecutor()
    cfg = config(options={"load_delay_s": 0.15})
    try:
        results = await asyncio.gather(*(executor.load(cfg) for _ in range(5)))
        assert len({item["pid"] for item in results}) == 1
        assert all(item["load_count"] == 1 for item in results)
        first, second = await asyncio.gather(collect(executor, cfg, "one"), collect(executor, cfg, "two"))
        assert first[0]["mock"] and second[0]["mock"]
        snapshot = executor.snapshot()[cfg.instance_key]
        assert snapshot["completed"] == 2 and snapshot["rss_bytes"] > 0
    finally:
        await executor.close()


async def test_cancellation_waits_for_backend_stop():
    executor = ProcessExecutor()
    cfg = config(options={"delay_s": 2, "cancel_delay_s": 0.2})
    try:
        await executor.load(cfg)
        cancel = asyncio.Event()
        inference = asyncio.create_task(collect(executor, cfg, cancel=cancel))
        await asyncio.sleep(0.08)
        before = time.monotonic()
        cancel.set()
        with pytest.raises(ServiceError) as error:
            await inference
        assert error.value.code == "cancelled"
        assert time.monotonic() - before >= 0.18
        assert executor.snapshot()[cfg.instance_key]["busy"] == 0
    finally:
        await executor.close()


async def test_task_cancellation_waits_for_backend_stop():
    executor = ProcessExecutor()
    cfg = config(options={"delay_s": 2, "cancel_delay_s": 0.15})
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        await asyncio.sleep(0.06)
        before = time.monotonic()
        inference.cancel()
        with pytest.raises(asyncio.CancelledError):
            await inference
        assert time.monotonic() - before >= 0.13
        assert executor.snapshot()[cfg.instance_key]["busy"] == 0
    finally:
        await executor.close()


async def test_closing_stream_waits_for_stop_and_retains_process():
    executor = ProcessExecutor()
    cfg = config(options={"stream_chunks": 100, "chunk_delay_s": 0.03, "cancel_delay_s": 0.15})
    try:
        await executor.load(cfg)
        cancel = asyncio.Event()
        stream = executor.stream(cfg.instance_key, {"texts": ["hello"]}, "stream", True, cancel)
        assert (await anext(stream))["mock"]
        before = time.monotonic()
        await stream.aclose()
        assert time.monotonic() - before >= 0.13
        assert cancel.is_set()
        assert executor.is_alive(cfg.instance_key)
        assert executor.snapshot()[cfg.instance_key]["busy"] == 0
    finally:
        await executor.close()


async def test_process_fault_is_reported_without_retry():
    executor = ProcessExecutor()
    cfg = config(options={"delay_s": 2})
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        await asyncio.sleep(0.08)
        os.kill(executor.snapshot()[cfg.instance_key]["pid"], signal.SIGKILL)
        with pytest.raises(ServiceError) as error:
            await asyncio.wait_for(inference, 3)
        assert error.value.code == "process_failed"
        assert executor.snapshot()[cfg.instance_key]["load_count"] == 1
        assert not executor.is_alive(cfg.instance_key)
    finally:
        await executor.close()


async def test_unload_refuses_active_inference():
    executor = ProcessExecutor()
    cfg = config(options={"delay_s": 0.2})
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        await asyncio.sleep(0.04)
        with pytest.raises(ServiceError) as error:
            await executor.unload(cfg.instance_key)
        assert error.value.code == "model_in_use"
        assert executor.is_alive(cfg.instance_key)
        await inference
        result = await executor.unload(cfg.instance_key)
        assert result["process_exited"] and not executor.is_alive(cfg.instance_key)
    finally:
        await executor.close()


async def test_load_timeout_keeps_process_visible_until_it_finishes():
    executor = ProcessExecutor(load_timeout_s=0.05)
    cfg = config(options={"load_delay_s": 0.25})
    try:
        with pytest.raises(ServiceError) as error:
            await executor.load(cfg)
        assert error.value.code == "load_timeout"
        assert executor.snapshot()[cfg.instance_key]["state"] == "loading"
        assert executor.is_alive(cfg.instance_key)
        executor.load_timeout_s = 2
        assert (await executor.load(cfg))["load_count"] == 1
    finally:
        await executor.close()


async def test_load_failure_confirms_process_exit():
    executor = ProcessExecutor()
    cfg = config(options={"fail_load": True})
    try:
        with pytest.raises(ServiceError) as error:
            await executor.load(cfg)
        assert error.value.code == "load_failed"
        assert not executor.is_alive(cfg.instance_key)
    finally:
        await executor.close()


async def test_output_limit_stops_stream_without_retry():
    executor = ProcessExecutor(max_output_bytes=200)
    cfg = config(options={"stream_chunks": ["x" * 1000]})
    try:
        await executor.load(cfg)
        with pytest.raises(ServiceError) as error:
            await collect(executor, cfg, stream=True)
        assert error.value.code == "output_too_large"
        assert executor.snapshot()[cfg.instance_key]["busy"] == 0
    finally:
        await executor.close()


async def test_partial_stream_failure_is_reported_without_retry():
    executor = ProcessExecutor(max_output_bytes=200)
    cfg = config(options={"stream_chunks": ["first", "x" * 500]})
    try:
        await executor.load(cfg)
        iterator = executor.stream(cfg.instance_key, {"texts": ["hello"]}, "partial", True, asyncio.Event())
        assert (await anext(iterator))["delta"] == "first"
        with pytest.raises(ServiceError) as error:
            await anext(iterator)
        assert error.value.code == "output_too_large"
        state = executor.snapshot()[cfg.instance_key]
        assert state["load_count"] == 1 and state["completed"] == 1 and state["busy"] == 0
    finally:
        await executor.close()


class _Handler(BaseHTTPRequestHandler):
    delay = 0.0
    response_status = 200
    entered = Event()

    def do_POST(self):
        self.entered.set()
        time.sleep(self.delay)
        length = int(self.headers["Content-Length"])
        content = json.dumps({"echo": json.loads(self.rfile.read(length))}).encode()
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except BrokenPipeError:
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def http_service():
    _Handler.delay = 0.0
    _Handler.response_status = 200
    _Handler.entered.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_adapter_reports_external_management(http_service):
    backend = HTTPBackend(config(backend="http", task="http_json", options={"base_url": http_service}))
    assert backend.load()["remote_state"] == "external"
    assert backend.infer({"text": "hello"}, Event()) == {"echo": {"text": "hello"}}
    closed = backend.close()
    assert closed["adapter_closed"] and closed["remote_state"] == "external"
    assert "unloaded" not in closed


def test_inline_http_credentials_are_rejected(http_service):
    from model_service.backends import validate_backend
    with pytest.raises(ServiceError) as error:
        validate_backend(config(backend="http", options={"base_url": http_service, "api_key": "do-not-store"}))
    assert error.value.code == "invalid_config"


def test_http_cancel_waits_for_response(http_service):
    _Handler.delay = 0.2
    backend = HTTPBackend(config(backend="http", task="http_json", options={"base_url": http_service}))
    backend.load()
    cancel = Event()
    timer = threading.Timer(0.04, cancel.set)
    timer.start()
    try:
        before = time.monotonic()
        with pytest.raises(ServiceError) as error:
            backend.infer({}, cancel)
        assert error.value.code == "cancelled"
        assert time.monotonic() - before >= 0.19
    finally:
        timer.join()
        backend.close()


def test_http_timeout_is_unknown_remote_execution(http_service):
    _Handler.delay = 0.2
    backend = HTTPBackend(config(backend="http", task="http_json", options={"base_url": http_service, "timeout_s": 0.03}))
    backend.load()
    try:
        with pytest.raises(ServiceError) as error:
            backend.infer({}, Event())
        assert error.value.code == "execution_unknown"
    finally:
        backend.close()


def test_http_accepted_is_unknown_remote_execution(http_service):
    _Handler.response_status = 202
    backend = HTTPBackend(config(backend="http", task="http_json", options={"base_url": http_service}))
    backend.load()
    try:
        with pytest.raises(ServiceError) as error:
            backend.infer({}, Event())
        assert error.value.code == "execution_unknown"
    finally:
        backend.close()


async def test_http_worker_death_retains_unknown_remote_status(http_service):
    _Handler.delay = 0.3
    executor = ProcessExecutor()
    cfg = config(backend="http", task="http_json", options={"base_url": http_service})
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        assert await asyncio.to_thread(_Handler.entered.wait, 2)
        os.kill(executor.snapshot()[cfg.instance_key]["pid"], signal.SIGKILL)
        with pytest.raises(ServiceError) as error:
            await inference
        assert error.value.code == "execution_unknown"
    finally:
        await executor.close()


async def test_cancelled_http_task_still_reports_unknown_if_remote_times_out(http_service):
    _Handler.delay = 0.3
    executor = ProcessExecutor()
    cfg = config(backend="http", task="http_json", options={"base_url": http_service, "timeout_s": 0.12})
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        assert await asyncio.to_thread(_Handler.entered.wait, 2)
        before = time.monotonic()
        inference.cancel()
        with pytest.raises(ServiceError) as error:
            await inference
        assert error.value.code == "execution_unknown"
        assert time.monotonic() - before >= 0.09
    finally:
        await executor.close()


def test_real_openvino_small_tensor_graph(tmp_path):
    ov = pytest.importorskip("openvino")
    np = pytest.importorskip("numpy")
    from openvino import opset13 as ops
    from model_service.backends.openvino import OpenVINOBackend
    tensor = ops.parameter([1, 3], np.float32, name="input")
    output = ops.multiply(tensor, ops.constant(np.array([2.0], dtype=np.float32)))
    output.output(0).get_tensor().set_names({"output"})
    graph = ov.Model([output], [tensor], "tiny_test_graph")
    path = tmp_path / "model.xml"
    ov.save_model(graph, str(path))
    backend = OpenVINOBackend(config(backend="openvino", path=str(path)))
    try:
        loaded = backend.load()
        assert loaded["backend"] == "openvino" and loaded["device"] == "CPU"
        actual = backend.infer({"input": [[1, 2, 3]]}, Event())
        np.testing.assert_allclose(actual["output"], [[2, 4, 6]])
    finally:
        backend.close()


async def test_different_remote_registration_preserves_external_failure_semantics(http_service, monkeypatch):
    """A second compatible provider needs no new coordinator/worker branch."""
    from model_service.backends import BACKENDS, backend_registration
    from model_service.plugins import BuiltinPluginCatalog
    original = backend_registration(config(backend="http"))
    monkeypatch.setitem(BACKENDS, "another_remote", original)
    cfg = config(backend="another_remote", task="http_json", options={"base_url": http_service})
    assert BuiltinPluginCatalog().describe(cfg).management == "external"
    _Handler.delay = 0.3
    executor = ProcessExecutor()
    try:
        await executor.load(cfg)
        inference = asyncio.create_task(collect(executor, cfg))
        assert await asyncio.to_thread(_Handler.entered.wait, 2)
        os.kill(executor.snapshot()[cfg.instance_key]["pid"], signal.SIGKILL)
        with pytest.raises(ServiceError) as error:
            await inference
        assert error.value.code == "execution_unknown"
        assert executor.snapshot()[cfg.instance_key]["load_count"] == 1
    finally:
        await executor.close()
