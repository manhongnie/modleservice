"""Business request orchestration through stable plugin and execution contracts."""
from __future__ import annotations
import asyncio
import contextlib
import json
import time

from .contracts import ModelConfig, ServiceError


async def finish_shielded(awaitable):
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class RequestCoordinator:
    def __init__(self, settings, registry, scheduler, lifecycle, executor, policy, observer):
        self.settings, self.registry, self.scheduler = settings, registry, scheduler
        self.lifecycle, self.executor, self.policy, self.observer = lifecycle, executor, policy, observer
        self._cancellations = {}
        self._closing = False

    def begin_shutdown(self):
        self._closing = True
        for event in self._cancellations.values():
            event.set()

    def pending_count(self):
        return len(self._cancellations)

    async def infer(self, capability: str, model: str | None, payload: dict,
                    cancel: asyncio.Event, request_id: str, stream: bool = False):
        began = time.monotonic()
        outcome = "rejected"
        model_id = None
        execution = None
        if request_id in self._cancellations:
            raise ServiceError("duplicate_request", "Request ID is already active or queued", 409)
        self._cancellations[request_id] = cancel
        try:
            if self._closing:
                raise ServiceError("shutting_down", "Service is shutting down", 503)
            row = self.registry.resolve(model, capability)
            config = row["config"]
            model_id = config.model_id
            if not row["enabled"]:
                raise ServiceError("model_disabled", "Model is disabled", 409)
            self.policy.validate_input(config, payload, capability)
            self.observer.event("request_started", request_id=request_id, model=model_id, capability=capability, stream=stream)
            execution = self.execute(config, payload, cancel, request_id, stream)
            async for output in execution:
                if output is None or not stream:
                    outcome = "completed"
                yield {"request_id": request_id, "model": model_id, "output": output,
                       "mock": self.policy.catalog.describe(config).synthetic, "done": output is None or not stream}
            outcome = "completed"
        except ServiceError as exc:
            outcome = exc.code
            raise
        except (asyncio.CancelledError, GeneratorExit):
            if outcome != "completed":
                outcome = "cancelled"
            raise
        finally:
            cancel.set()
            if execution is not None:
                await finish_shielded(execution.aclose())
            self._cancellations.pop(request_id, None)
            self.observer.event("request_finished", request_id=request_id, model=model_id, outcome=outcome,
                        duration_seconds=time.monotonic() - began, resources=self.scheduler.snapshot())

    async def execute(self, config: ModelConfig, payload: dict, cancel: asyncio.Event,
                       request_id: str, stream: bool, *, administrative: bool = False):
        lease = await self.scheduler.acquire(config, request_id, cancel, administrative=administrative)
        began = time.monotonic()
        inference_began = None
        iterator = None
        unknown = False
        remote_intent = False
        timed_out = False

        async def deadline():
            nonlocal timed_out
            await asyncio.sleep(self.settings.execution_timeout_s)
            timed_out = True
            cancel.set()

        timer = asyncio.create_task(deadline())
        try:
            # Repeat path checks on every load after admin-controlled file changes.
            if not self.lifecycle.is_alive(config.instance_key):
                self.policy.validate(config)
            await self.lifecycle.ensure_loaded(config)
            if cancel.is_set():
                raise ServiceError("execution_timeout" if timed_out else "cancelled", "Request stopped before inference", 504 if timed_out else 499)
            if self.policy.catalog.describe(config).execution_may_outlive_worker:
                self.registry.uncertain(request_id, config.model_id, "Remote execution may survive controller restart")
                remote_intent = True
            inference_began = time.monotonic()
            iterator = self.executor.stream(config.instance_key, payload, request_id, stream, cancel)
            async for output in iterator:
                if timed_out:
                    raise ServiceError("execution_timeout", "Execution exceeded deadline; cancellation requested", 504)
                if cancel.is_set():
                    raise ServiceError("cancelled", "Request was cancelled", 499)
                if len(json.dumps(output).encode()) > self.settings.max_output_bytes:
                    raise ServiceError("output_too_large", "Output exceeds service limit", 502)
                yield output
            if timed_out:
                raise ServiceError("execution_timeout", "Execution exceeded deadline and has stopped", 504)
            if stream:
                # Keep the lease across transmission of the terminal SSE frame too.
                yield None
        except ServiceError as exc:
            unknown = exc.code == "execution_unknown" or (remote_intent and exc.code == "process_failed")
            if timed_out and exc.code == "cancelled":
                raise ServiceError("execution_timeout", "Execution deadline exceeded; underlying execution has stopped", 504) from exc
            raise
        finally:
            cancel.set()
            timer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await timer

            async def cleanup():
                nonlocal unknown
                if inference_began is None:
                    await self.lifecycle.settle_load(config)
                try:
                    if iterator is not None:
                        await iterator.aclose()
                except ServiceError as exc:
                    unknown = unknown or exc.code in ("execution_unknown", "process_failed") and remote_intent
                    if not unknown:
                        self.observer.event("cleanup_error", model=config.model_id, request_id=request_id, code=exc.code)
                if unknown:
                    self.registry.uncertain(request_id, config.model_id, "Remote completion unknown; administrator must confirm stopped")
                    if self.registry.get(config.model_id)["management_state"] != "draining":
                        self.registry.management(config.model_id, "reconciliation_required")
                    await self.scheduler.quarantine(lease)
                else:
                    if remote_intent:
                        self.registry.clear_uncertainty(request_id)
                    await self.scheduler.release(lease)
                    if not self.lifecycle.is_alive(config.instance_key) and not self.scheduler.count(config.instance_key):
                        await self.scheduler.drop_resident(config.instance_key)
                self.lifecycle.touch(config.instance_key)
                self.observer.event("execution_finished", request_id=request_id, model=config.model_id,
                            execution_seconds=time.monotonic() - began,
                            inference_seconds=None if inference_began is None else time.monotonic() - inference_began,
                            completion_known=not unknown,
                            worker=self.executor.snapshot().get(config.instance_key, {}))

            await finish_shielded(cleanup())
