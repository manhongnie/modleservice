"""Asynchronous control-side supervision of reusable model worker processes."""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import Any, AsyncIterator

import psutil

from model_service.backends import backend_registration, describe_backend
from model_service.contracts import BackendFeatures, ModelConfig, ServiceError
from model_service.worker import worker_main


@dataclass
class _Pending:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=8))
    overflow: bool = False
    done: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


@dataclass
class _Instance:
    config: ModelConfig
    process: Any
    connection: Connection
    features: BackendFeatures
    pending: dict[str, _Pending] = field(default_factory=dict)
    reader: asyncio.Task | None = None
    state: str = "loading"
    info: dict[str, Any] = field(default_factory=dict)
    inference_seconds: float = 0
    completed: int = 0
    load_count: int = 1
    closed: bool = False


def _receive(connection: Connection):
    if connection.poll(0.05):
        return connection.recv()
    return None


def _raise_error(message: dict[str, Any]) -> None:
    error = message.get("error")
    if error:
        raise ServiceError(error["code"], error["message"], error.get("status", 503))


async def _acknowledged(future: asyncio.Future):
    """Defer task cancellation until the worker acknowledges termination."""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
    return future.result()


class ProcessExecutor:
    def __init__(self, load_timeout_s: float = 120, max_output_bytes: int = 8388608):
        self.load_timeout_s = load_timeout_s
        self.max_output_bytes = max_output_bytes
        self._context = mp.get_context("spawn")
        self._instances: dict[str, _Instance] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._load_counts: dict[str, int] = {}

    def _send(self, instance: _Instance, command: dict[str, Any]) -> None:
        try:
            instance.connection.send(command)
        except (OSError, EOFError) as exc:
            raise ServiceError("process_failed", "Model worker is unavailable", 503) from exc

    async def _reader(self, instance: _Instance) -> None:
        try:
            while not instance.closed:
                message = await asyncio.to_thread(_receive, instance.connection)
                if message is None:
                    if not instance.process.is_alive():
                        break
                    continue
                pending = instance.pending.get(message["id"])
                if pending is None:
                    continue
                if message["type"] == "done":
                    if message["id"] == "__load__":
                        if message.get("error"):
                            instance.state = "failed"
                            while instance.process.is_alive():
                                await asyncio.sleep(0.01)
                        else:
                            instance.state = "loaded"
                            instance.info = message.get("result", {})
                    elif message["id"] == "__unload__":
                        if not message.get("error"):
                            instance.state = "unloaded"
                    else:
                        instance.inference_seconds = message.get("inference_seconds", 0)
                        instance.completed += 1
                    if not pending.done.done():
                        pending.done.set_result(message)
                if message["type"] == "chunk" and pending.overflow:
                    continue
                if pending.queue.full():
                    pending.overflow = True
                    if message["type"] == "chunk":
                        try:
                            self._send(instance, {"op": "cancel", "id": message["id"]})
                        except ServiceError:
                            pass  # Reader will confirm process death before completion.
                        continue
                    # Terminal acknowledgement must remain readable even when a
                    # consumer disconnected. This request explicitly fails if the
                    # bounded channel filled; no partial output is silently retried.
                    pending.queue.get_nowait()
                if message["type"] == "done" and pending.overflow:
                    if (message.get("error") or {}).get("code") != "execution_unknown":
                        message = {**message, "error": {"code": "stream_backpressure",
                            "message": "Stream consumer exceeded the bounded output buffer", "status": 503}}
                pending.queue.put_nowait(message)
        except (EOFError, OSError, ConnectionResetError):
            pass
        finally:
            # A broken channel is not itself proof that native execution stopped.
            # Local occupancy is released only after the OS confirms process death.
            while instance.process.is_alive() and not instance.closed:
                await asyncio.sleep(0.01)
            if instance.state != "unloaded":
                instance.state = "failed"
            for name, pending in instance.pending.items():
                if not pending.done.done():
                    remote_unknown = instance.features.execution_may_outlive_worker and not name.startswith("__")
                    failure = {"type": "done", "error": {
                        "code": "execution_unknown" if remote_unknown else "process_failed",
                        "message": "Worker exited; external completion is unknown" if remote_unknown else
                                   "Model worker exited; request was not retried", "status": 503}}
                    pending.done.set_result(failure)
                    if pending.queue.full():
                        pending.queue.get_nowait()
                    pending.queue.put_nowait(failure)

    async def load(self, config: ModelConfig) -> dict[str, Any]:
        key = config.instance_key
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            instance = self._instances.get(key)
            if instance is None or not instance.process.is_alive():
                if instance is not None:
                    await self._dispose(instance)
                parent, child = self._context.Pipe()
                process = self._context.Process(
                    target=worker_main,
                    args=(child, config.model_dump(), os.getpid(), self.max_output_bytes, backend_registration(config)),
                    name=f"model-{config.name}-{config.version}", daemon=False,
                )
                instance = _Instance(config, process, parent, describe_backend(config))
                instance.pending["__load__"] = _Pending()
                self._load_counts[key] = self._load_counts.get(key, 0) + 1
                instance.load_count = self._load_counts[key]
                self._instances[key] = instance
                try:
                    process.start()
                except (OSError, RuntimeError) as exc:
                    parent.close()
                    child.close()
                    self._instances.pop(key, None)
                    raise ServiceError("process_start_failed", "Model worker could not be started", 503) from exc
                child.close()
                instance.reader = asyncio.create_task(self._reader(instance))
            pending = instance.pending["__load__"]
            try:
                message = await asyncio.wait_for(asyncio.shield(pending.done), self.load_timeout_s)
            except asyncio.TimeoutError as exc:
                # No compile API promises to stop immediately. Keep the process and
                # reservation visible so the lifecycle layer can reconcile it.
                raise ServiceError("load_timeout", "Model is still loading; reservation remains held", 504) from exc
            except asyncio.CancelledError:
                await _acknowledged(pending.done)
                raise
            _raise_error(message)
            return {**instance.info, "load_count": instance.load_count, "state": instance.state}

    async def stream(self, key: str, payload: dict[str, Any], request_id: str,
                     stream: bool, cancel: asyncio.Event) -> AsyncIterator[dict[str, Any]]:
        instance = self._instances.get(key)
        if instance is None or instance.state != "loaded" or not instance.process.is_alive():
            raise ServiceError("process_failed", "Model worker is not loaded or has exited", 503)
        if request_id in instance.pending:
            raise ServiceError("duplicate_request", "Request identifier is already running", 409)
        pending = _Pending()
        instance.pending[request_id] = pending

        async def watch_cancel():
            await cancel.wait()
            if not pending.done.done():
                self._send(instance, {"op": "cancel", "id": request_id})

        watcher = None
        submitted = False
        buffered = []
        try:
            self._send(instance, {"op": "infer", "id": request_id, "payload": payload, "stream": stream})
            submitted = True
            watcher = asyncio.create_task(watch_cancel())
            while True:
                message = await pending.queue.get()
                if message["type"] == "done":
                    _raise_error(message)
                    for output in buffered:
                        if not cancel.is_set():
                            yield output
                    break
                if not cancel.is_set():
                    if stream:
                        yield message["result"]
                    else:
                        buffered.append(message["result"])
        finally:
            terminal = None
            if submitted and not pending.done.done():
                cancel.set()
                try:
                    self._send(instance, {"op": "cancel", "id": request_id})
                except ServiceError:
                    pass  # Reader confirms process death before marking done.
                terminal = await _acknowledged(pending.done)
            elif pending.done.done():
                terminal = pending.done.result()
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            instance.pending.pop(request_id, None)
            if terminal and (terminal.get("error") or {}).get("code") == "execution_unknown":
                _raise_error(terminal)

    async def unload(self, key: str, timeout_s: float = 30) -> dict[str, Any]:
        instance = self._instances.get(key)
        if instance is None:
            return {"state": "unloaded", "already_unloaded": True}
        if not instance.process.is_alive():
            await self._dispose(instance)
            self._instances.pop(key, None)
            return {"state": "unloaded", "process_exited": True}
        if any(not value.done.done() for name, value in instance.pending.items() if not name.startswith("__")):
            raise ServiceError("model_in_use", "Model still has active executions", 409)
        pending = instance.pending.get("__unload__")
        if pending is None or pending.done.done():
            pending = _Pending()
            instance.pending["__unload__"] = pending
            self._send(instance, {"op": "unload", "id": "__unload__"})
        try:
            message = await asyncio.wait_for(asyncio.shield(pending.done), timeout_s)
        except asyncio.TimeoutError as exc:
            raise ServiceError("unload_timeout", "Model worker has not stopped; no force kill was attempted", 409) from exc
        if message.get("error") and not instance.process.is_alive():
            await self._dispose(instance)
            self._instances.pop(key, None)
            return {"state": "unloaded", "process_exited": True}
        _raise_error(message)
        # An unload acknowledgement alone is not enough: native allocators may retain
        # memory until process exit. Confirm exit before releasing resident accounting.
        deadline = time.monotonic() + timeout_s
        while instance.process.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if instance.process.is_alive():
            raise ServiceError("unload_timeout", "Model process has not exited; reservation remains held", 409)
        result = message.get("result", {})
        await self._dispose(instance)
        self._instances.pop(key, None)
        return {**result, "state": "unloaded", "process_exited": True}

    async def _dispose(self, instance: _Instance) -> None:
        instance.closed = True
        if instance.reader is not None:
            await instance.reader
        instance.connection.close()
        instance.process.join(timeout=0)
        instance.process.close()

    def is_alive(self, key: str) -> bool:
        instance = self._instances.get(key)
        return instance is not None and instance.process.is_alive()

    def snapshot(self) -> dict[str, Any]:
        result = {}
        for key, instance in self._instances.items():
            alive = instance.process.is_alive()
            rss = 0
            if alive:
                try:
                    rss = psutil.Process(instance.process.pid).memory_info().rss
                except psutil.Error:
                    pass
            result[key] = {**instance.info, "pid": instance.process.pid,
                "state": instance.state if alive else "failed", "alive": alive,
                "rss_bytes": rss, "rss_mb": round(rss / 1048576, 3),
                "load_count": instance.load_count,
                "output_buffer_limit": 8,
                "buffered_chunks": sum(pending.queue.qsize() for name, pending in instance.pending.items() if not name.startswith("__")),
                "load_seconds": instance.info.get("load_seconds"),
                "busy": sum(not pending.done.done() for name, pending in instance.pending.items() if not name.startswith("__")),
                "inference_seconds": instance.inference_seconds, "completed": instance.completed}
        return result

    async def close(self, timeout_s: float = 30) -> None:
        failures = []
        for key in list(self._instances):
            try:
                await self.unload(key, timeout_s)
            except ServiceError as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
