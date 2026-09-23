"""HTTP boundary: authentication, bounded bodies, transport and disconnects.

Model lifecycle and resource policy belong to Service, never to HTTP handlers.
"""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import math
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse, Response

from .contracts import ModelConfig, ServiceError, Settings

logger = logging.getLogger("model_service.http")


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, max_length=256)
    input: dict[str, Any]
    stream: bool = False


class AddModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: ModelConfig
    enable: bool = True


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=256)


class ReconcileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed_stopped: bool


_PUBLIC_ERRORS = {
    "unauthorized": "A valid API key is required for this interface.",
    "model_not_found": "The requested model does not exist.",
    "not_found": "The requested model or alias does not exist.",
    "model_disabled": "The requested model is disabled.",
    "model_draining": "The requested model is not accepting requests.",
    "model_paused": "The requested model is paused for management or reconciliation.",
    "ambiguous_model": "Select a versioned model name or a configured alias.",
    "capability_mismatch": "The model does not support this capability.",
    "unsupported_capability": "The model does not support this capability.",
    "invalid_input": "The input is invalid for this capability.",
    "input_too_large": "The request exceeds the input size limit.",
    "queue_full": "The waiting queue is full.",
    "queue_timeout": "The request timed out waiting for resources.",
    "resource_limit": "The request exceeds the configured resource budget.",
    "resource_exhausted": "The request exceeds the configured resource budget.",
    "execution_timeout": "The inference execution timed out.",
    "cancelled": "The inference request was cancelled.",
    "load_failed": "The model could not be loaded.",
    "load_timeout": "Model loading timed out; its resource reservation remains held.",
    "execution_failed": "The model execution failed.",
    "inference_failed": "The model execution failed.",
    "process_failed": "The model execution process failed.",
    "worker_failed": "The model execution process failed.",
    "worker_crashed": "The model execution process failed.",
    "execution_unknown": "The remote execution status is unknown; resources remain reserved.",
    "output_too_large": "The model output exceeds the configured output limit.",
    "shutting_down": "The service is shutting down.",
}


def error_body(error: ServiceError, *, administrative: bool = False) -> dict[str, Any]:
    message = error.message if administrative else _PUBLIC_ERRORS.get(
        error.code, "The request could not be completed."
    )
    return {"error": {"code": error.code, "message": message}}


def _authenticated(scope: dict, keys: list[str]) -> bool:
    headers = [value for name, value in scope.get("headers", []) if name.lower() == b"authorization"]
    # Duplicate credentials are ambiguous and rejected before consuming any body.
    authorization = headers[0] if len(headers) == 1 else b""
    scheme, _, supplied = authorization.partition(b" ")
    matched = False
    for key in keys:
        matched |= hmac.compare_digest(supplied, key.encode("utf-8"))
    return scheme.lower() == b"bearer" and bool(supplied) and matched


class HTTPBoundary:
    """Authenticate before reading, bound admission/body time, and audit metadata only.

    Counter changes contain no await, so admission is atomic within the single
    control process/event loop. An admitted request keeps its slot through cleanup.
    """

    def __init__(self, app: Any, settings: Settings):
        self.app, self.settings = app, settings
        self.active = 0

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        status = 500
        admitted = False

        async def logged_send(message: dict) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-request-id", request_id.encode("ascii"))
                ]
            await asyncio.wait_for(send(message), timeout=self.settings.write_timeout_s)

        async def reject(code: str, message: str, http_status: int) -> None:
            headers = {"Connection": "close"}
            if http_status == 401:
                headers["WWW-Authenticate"] = "Bearer"
            if http_status == 503:
                headers["Retry-After"] = "1"
            await JSONResponse({"error": {"code": code, "message": message}},
                               status_code=http_status, headers=headers)(scope, receive, logged_send)

        try:
            path = scope.get("path", "")
            if path in {"/health", "/ready"} and scope.get("method") == "GET":
                await self.app(scope, receive, logged_send)
                return
            is_admin = path == "/admin" or path.startswith("/admin/")
            is_business = path.startswith("/v1/")
            if not is_admin and not is_business:
                await reject("not_found", "The requested interface does not exist.", 404)
                return
            if not _authenticated(scope, self.settings.admin_keys if is_admin else self.settings.business_keys):
                await reject("unauthorized", _PUBLIC_ERRORS["unauthorized"], 401)
                return
            if self.active >= self.settings.max_http_requests:
                await reject("http_capacity", "The HTTP request capacity is exhausted.", 503)
                return
            self.active += 1
            admitted = True
            chunks: list[bytes] = []
            size = 0
            try:
                # This is a total receive deadline; slow chunks cannot reset it.
                async with asyncio.timeout(self.settings.body_timeout_s):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            status = 499
                            return
                        if message["type"] != "http.request":
                            continue
                        body = message.get("body", b"")
                        size += len(body)
                        if size > self.settings.max_body_bytes:
                            break
                        chunks.append(body)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await reject("body_timeout", "The request body receive deadline expired.", 408)
                return
            if size > self.settings.max_body_bytes:
                await reject("input_too_large", "Request body is too large.", 413)
                return
            body = b"".join(chunks)
            delivered = False

            async def bounded_receive() -> dict:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, logged_send)
        except (OSError, TimeoutError):
            # A client that cannot accept output gets no second write attempt.
            status = 499
        finally:
            if admitted:
                self.active -= 1
            route = scope.get("route")
            logger.info(json.dumps({
                "event": "http_request", "request_id": request_id,
                "method": scope.get("method"),
                "route": getattr(route, "path", "unmatched"), "status": status,
            }, separators=(",", ":")))


async def _finish(task: asyncio.Task) -> Any:
    """Wait for cleanup even when an ASGI parent task is cancelled."""
    with anyio.CancelScope(shield=True):
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    break
        return task.result()


class _Disconnected(Exception):
    pass


class InferenceResponse(Response):
    """Keep the Service iterator alive until every yielded item has been sent.

    Pulling a generator is shielded from server cancellation: a disconnect signals
    the cooperative cancellation event and cleanup waits for executor confirmation.
    """

    media_type = "application/json"

    def __init__(self, iterator: Any, cancel: asyncio.Event, streaming: bool, write_timeout_s: float = 30):
        super().__init__(content=b"")
        self.iterator, self.cancel, self.streaming = iterator, cancel, streaming
        self.write_timeout_s = write_timeout_s

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        started = False
        disconnected = asyncio.Event()
        pull: asyncio.Task | None = None

        async def watch_disconnect() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected.set()
                    self.cancel.set()
                    return

        watcher = asyncio.create_task(watch_disconnect())

        async def safe_send(message: dict) -> None:
            if disconnected.is_set():
                raise _Disconnected()
            sent = asyncio.create_task(send(message))
            try:
                done, _ = await asyncio.wait({sent, watcher}, timeout=self.write_timeout_s,
                                             return_when=asyncio.FIRST_COMPLETED)
                if sent in done:
                    await sent
                    return
                raise _Disconnected()
            finally:
                if not sent.done():
                    sent.cancel()
                with contextlib.suppress(asyncio.CancelledError, OSError):
                    await sent

        async def next_item() -> dict:
            nonlocal pull
            pull = asyncio.create_task(anext(self.iterator))
            return await asyncio.shield(pull)

        async def send_error(error: ServiceError) -> None:
            nonlocal started
            if disconnected.is_set():
                return
            if not started:
                await JSONResponse(error_body(error), status_code=error.status)(scope, receive, safe_send)
                started = True
            elif self.streaming:
                data = json.dumps(error_body(error), ensure_ascii=False, separators=(",", ":"))
                await safe_send({"type": "http.response.body", "body": f"event: error\ndata: {data}\n\n".encode(), "more_body": False})

        try:
            try:
                first = await next_item()
            except StopAsyncIteration:
                raise ServiceError("empty_output", "The executor returned no output.", 502)
            if self.streaming:
                await safe_send({"type": "http.response.start", "status": 200, "headers": [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"cache-control", b"no-cache"), (b"x-accel-buffering", b"no"),
                ]})
                started = True
                item = first
                while True:
                    event = "result" if item.get("done", False) else "chunk"
                    data = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    await safe_send({"type": "http.response.body", "body": f"event: {event}\ndata: {data}\n\n".encode(), "more_body": True})
                    try:
                        item = await next_item()
                    except StopAsyncIteration:
                        break
                await safe_send({"type": "http.response.body", "body": b"", "more_body": False})
            else:
                # The Service non-streaming contract yields one complete result.
                await JSONResponse(first)(scope, receive, safe_send)
                started = True
        except ServiceError as error:
            with contextlib.suppress(_Disconnected, OSError):
                await send_error(error)
        except (_Disconnected, OSError):
            self.cancel.set()
        except asyncio.CancelledError:
            self.cancel.set()
            raise
        except Exception:
            # Never send exception text, model paths or backend options to business callers.
            logger.error("inference_transport_failed request_id=%s", scope.get("state", {}).get("request_id"))
            with contextlib.suppress(_Disconnected, OSError):
                await send_error(ServiceError("internal_error", "The request could not be completed.", 500))
        finally:
            self.cancel.set()
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            if pull is not None and not pull.done():
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await _finish(pull)
            # Retrieving a completed task's exception avoids unhandled-task warnings.
            if pull is not None and pull.done():
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    pull.result()
            await _finish(asyncio.create_task(self.iterator.aclose()))


async def _close_until_stopped(service: Any, retry_interval_s: float = 1) -> None:
    """Keep the controller alive while a known slow stop still owns resources.

    Exiting the lifespan early would let the event loop close the executor reader
    and trigger parent-death cleanup before a native load/unload finishes. Only
    explicit, retryable stop deadlines are retried; other errors remain visible.
    """
    while True:
        try:
            await service.close()
            return
        except ServiceError as error:
            if error.code not in {"shutdown_timeout", "unload_timeout"}:
                raise
            logger.warning(json.dumps({"event": "shutdown_waiting", "code": error.code,
                                       "retry_in_s": retry_interval_s}, separators=(",", ":")))
            await asyncio.sleep(retry_interval_s)


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        from .__main__ import load_settings
        settings = load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .service import Service
        service = Service(settings)
        app.state.service = service
        await service.start()
        try:
            yield
        finally:
            await _close_until_stopped(service)

    app = FastAPI(title="Shared model service", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.add_middleware(HTTPBoundary, settings=settings)

    def authenticate(request: Request, keys: list[str]) -> None:
        if not _authenticated(request.scope, keys):
            raise ServiceError("unauthorized", _PUBLIC_ERRORS["unauthorized"], 401)

    async def business_auth(request: Request) -> None:
        authenticate(request, settings.business_keys)

    async def admin_auth(request: Request) -> None:
        authenticate(request, settings.admin_keys)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError) -> JSONResponse:
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return JSONResponse(error_body(error, administrative=request.url.path.startswith("/admin/")),
                            status_code=error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": {"code": "invalid_request", "message": "Request body or parameters are invalid."}}, status_code=422)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> Response:
        result = await request.app.state.service.readiness()
        return JSONResponse(result, status_code=200 if result.get("status") == "ready" else 503)

    @app.post("/v1/{capability}", dependencies=[Depends(business_auth)])
    async def infer(capability: str, body: InferenceRequest, request: Request) -> Response:
        cancel = asyncio.Event()
        iterator = request.app.state.service.infer(
            capability=capability, model=body.model, payload=body.input, cancel=cancel,
            request_id=request.state.request_id, stream=body.stream,
        )
        return InferenceResponse(iterator, cancel, body.stream, settings.write_timeout_s)

    @app.get("/admin/status", dependencies=[Depends(admin_auth)])
    async def status(request: Request) -> dict:
        return await request.app.state.service.status()

    @app.get("/admin/metrics", dependencies=[Depends(admin_auth)])
    async def metrics(request: Request) -> Response:
        result = await request.app.state.service.metrics()
        return Response(result, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/admin/models", dependencies=[Depends(admin_auth)])
    async def models(request: Request) -> dict:
        result = await request.app.state.service.status()
        return {"models": result["models"]}

    @app.post("/admin/requests/{request_id}/reconcile", dependencies=[Depends(admin_auth)])
    async def reconcile(request_id: str, body: ReconcileRequest, request: Request) -> dict:
        return await request.app.state.service.reconcile(request_id, confirmed_stopped=body.confirmed_stopped)

    @app.post("/admin/models", dependencies=[Depends(admin_auth)])
    async def add_model(body: AddModelRequest, request: Request) -> dict:
        return await request.app.state.service.add(body.config, enable=body.enable)

    @app.post("/admin/models/{model_id}/{operation}", dependencies=[Depends(admin_auth)])
    async def model_operation(model_id: str, operation: Literal["validate", "load", "enable", "unload", "disable", "remove"],
                              request: Request, timeout_s: float | None = None, enable: bool = True) -> dict:
        service = request.app.state.service
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > 3600):
            raise ServiceError("invalid_timeout", "timeout_s must be in (0, 3600].", 422)
        if operation == "validate":
            return await service.validate(model_id, enable=enable)
        if operation == "load":
            return await service.load(model_id)
        if operation == "enable":
            return await service.enable(model_id)
        return await service.operate(model_id, operation, timeout_s=timeout_s)

    @app.delete("/admin/models/{model_id}", dependencies=[Depends(admin_auth)])
    async def remove_model(model_id: str, request: Request, timeout_s: float | None = None) -> dict:
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > 3600):
            raise ServiceError("invalid_timeout", "timeout_s must be in (0, 3600].", 422)
        return await request.app.state.service.operate(model_id, "remove", timeout_s=timeout_s)

    @app.put("/admin/aliases/{alias}", dependencies=[Depends(admin_auth)])
    async def set_alias(alias: str, body: TargetRequest, request: Request) -> dict:
        return await request.app.state.service.set_alias(alias, body.model_id)

    @app.delete("/admin/aliases/{alias}", dependencies=[Depends(admin_auth)])
    async def delete_alias(alias: str, request: Request) -> dict:
        return await request.app.state.service.delete_alias(alias)

    @app.put("/admin/dependencies/{dependency}", dependencies=[Depends(admin_auth)])
    async def add_dependency(dependency: str, body: TargetRequest, request: Request) -> dict:
        return await request.app.state.service.add_dependency(dependency, body.model_id)

    @app.delete("/admin/dependencies/{dependency}", dependencies=[Depends(admin_auth)])
    async def delete_dependency(dependency: str, request: Request) -> dict:
        return await request.app.state.service.delete_dependency(dependency)

    return app
