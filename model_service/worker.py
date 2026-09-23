"""One reusable model instance per spawned process, with cooperative cancellation."""
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection
from threading import Event, Lock
from typing import Any

from model_service.backends import create_backend, backend_registration, BackendRegistration
from model_service.contracts import ModelConfig, ServiceError
from model_service.tasks import create_task


def _error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ServiceError):
        return {"code": exc.code, "message": exc.message, "status": exc.status}
    # Tracebacks may expose model input or credentials. Keep public errors bounded.
    return {"code": "inference_failed", "message": f"Worker failed: {type(exc).__name__}", "status": 503}


def _parent_death_signal(parent_pid: int) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise RuntimeError("Could not install parent-death signal")
        # Parent may die between spawn and prctl. Do not leave an orphan behind.
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)


def worker_main(connection: Connection, raw_config: dict[str, Any], parent_pid: int,
                max_output_bytes: int, registration: BackendRegistration | None = None) -> None:
    _parent_death_signal(parent_pid)
    send_lock = Lock()
    jobs_lock = Lock()
    jobs: dict[str, Event] = {}
    backend = None
    pool = None

    def send(message: dict[str, Any]) -> None:
        with send_lock:
            connection.send(message)

    try:
        config = ModelConfig.model_validate(raw_config)
        registration = registration or backend_registration(config)
        features = registration.describe(config)
        task = create_task(config, backend_family=registration.task_family)
        backend = create_backend(config, registration)
        before = time.monotonic()
        info = backend.load()
        info.update(load_seconds=time.monotonic() - before, pid=os.getpid())
        send({"id": "__load__", "type": "done", "result": info})
        pool = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="inference")

        def run(request_id: str, payload: dict[str, Any], stream: bool, cancel: Event) -> None:
            before = time.monotonic()
            failure = None
            emitted_bytes = 0

            def emit(result: dict[str, Any]) -> None:
                nonlocal emitted_bytes
                emitted_bytes += len(json.dumps(result, ensure_ascii=False).encode())
                if emitted_bytes > max_output_bytes:
                    raise ServiceError("output_too_large", "Model output exceeds the configured limit", 413)
                send({"id": request_id, "type": "chunk", "result": result})

            try:
                if cancel.is_set():
                    raise ServiceError("cancelled", "Cancelled before inference", 499)
                prepared = task.prepare(payload)
                if stream and features.incremental_output:
                    # Keep generator close in this scope: emit/decoding failures and
                    # cancellation must stop generation before sending terminal ack.
                    from contextlib import closing
                    with closing(backend.stream(prepared.inputs, cancel)) as chunks:
                        for output in chunks:
                            if cancel.is_set():
                                raise ServiceError("cancelled", "Execution stopped after cancellation", 499)
                            result = task.finish_chunk(output, prepared.context)
                            if result is not None:
                                emit(result)
                else:
                    output = backend.infer(prepared.inputs, cancel)
                    if cancel.is_set():
                        raise ServiceError("cancelled", "Execution stopped after cancellation", 499)
                    result = task.finish(output, prepared.context)
                    if stream:
                        result = {**result, "streaming_mode": "buffered"}
                    emit(result)
            except BaseException as exc:
                failure = _error(exc)
            finally:
                with jobs_lock:
                    jobs.pop(request_id, None)
                send({"id": request_id, "type": "done", "error": failure,
                      "inference_seconds": time.monotonic() - before})

        while True:
            command = connection.recv()
            operation = command["op"]
            if operation == "infer":
                request_id = command["id"]
                with jobs_lock:
                    if request_id in jobs or len(jobs) >= config.concurrency:
                        send({"id": request_id, "type": "done", "error": {
                            "code": "executor_busy", "message": "Worker execution capacity exceeded", "status": 503}})
                        continue
                    cancel = Event()
                    jobs[request_id] = cancel
                pool.submit(run, request_id, command["payload"], command["stream"], cancel)
            elif operation == "cancel":
                with jobs_lock:
                    if command["id"] in jobs:
                        jobs[command["id"]].set()
            elif operation == "unload":
                with jobs_lock:
                    active = bool(jobs)
                if active:
                    send({"id": "__unload__", "type": "done", "error": {
                        "code": "model_in_use", "message": "Model still has active executions", "status": 409}})
                    continue
                pool.shutdown(wait=True)
                pool = None
                result = backend.close()
                backend = None
                send({"id": "__unload__", "type": "done", "result": result})
                return
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except BaseException as exc:
        try:
            send({"id": "__load__", "type": "done", "error": _error(exc)})
        except (OSError, EOFError):
            pass
    finally:
        with jobs_lock:
            for cancel in jobs.values():
                cancel.set()
        if pool is not None:
            pool.shutdown(wait=True)
        if backend is not None:
            backend.close()
        connection.close()
