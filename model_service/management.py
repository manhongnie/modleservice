"""Model administration and maintenance, independent of HTTP or model SDKs."""
from __future__ import annotations
import asyncio
import json
import math
import re
import time
import uuid
from contextlib import asynccontextmanager

from .contracts import ModelConfig, ServiceError
from .coordinator import finish_shielded


# Only numerical input hints belong in the business-facing model directory.
# Paths, credentials, validation samples and arbitrary provider options stay private.
_PUBLIC_INPUT_LIMITS = frozenset({
    "max_new_tokens", "max_text_chars", "max_input_tokens", "max_length", "max_batch", "max_batch_size",
    "max_audio_seconds", "max_reference_seconds", "max_steps", "max_width", "max_height",
    "max_frames", "max_images", "max_image_pixels", "default_steps", "default_frames",
    "default_fps", "fixed_width", "fixed_height", "max_prompt_chars", "max_video_pixels",
})


class ModelManager:
    def __init__(self, settings, registry, scheduler, lifecycle, executor, policy, coordinator, observer):
        self.settings, self.registry, self.scheduler = settings, registry, scheduler
        self.lifecycle, self.executor, self.policy = lifecycle, executor, policy
        self.coordinator, self.observer = coordinator, observer
        self._admin_lock = asyncio.Lock()
        self._closing = False
        self._retry_after = {}

    def begin_shutdown(self):
        self._closing = True

    @asynccontextmanager
    async def shutdown_guard(self, timeout_s):
        try:
            await asyncio.wait_for(self._admin_lock.acquire(), timeout_s)
        except TimeoutError:
            raise ServiceError("shutdown_timeout", "Management operation has not finished; controller lock and registry remain open", 409) from None
        try:
            yield
        finally:
            self._admin_lock.release()

    @asynccontextmanager
    async def _manage(self):
        if self._closing:
            raise ServiceError("shutting_down", "Service is shutting down", 503)
        async with self._admin_lock:
            if self._closing:
                raise ServiceError("shutting_down", "Service is shutting down", 503)
            yield

    async def add(self, config: ModelConfig, enable: bool = True) -> dict:
        async with self._manage():
            self.policy.validate(config)
            self.registry.register(config)
            self.observer.event("model_registered", model=config.model_id)
            return await self._validate(config.model_id, enable)

    async def validate(self, model_id: str, enable: bool = True) -> dict:
        async with self._manage():
            return await self._validate(model_id, enable)

    async def _validate(self, model_id: str, enable: bool) -> dict:
        row = self.registry.get(model_id)
        config = row["config"]
        if self.scheduler.count(config.instance_key):
            raise ServiceError("model_busy", "Cannot validate while model is in use", 409)
        if self.scheduler.has_quarantine(config.instance_key):
            raise ServiceError("execution_unknown", "Reconcile remote execution first", 409)
        self.policy.validate(config)
        await self.scheduler.pause(config.instance_key)
        # The admin lock prevents conflicting management updates; enabled models
        # remain unavailable to business during validation.
        self.registry.enable(model_id, False)
        self.registry.management(model_id, "validating")
        cancel = asyncio.Event()
        request_id = "validation-" + uuid.uuid4().hex
        try:
            await self.scheduler.wait_idle(config.instance_key, self.settings.drain_timeout_s)
            async for _ in self.coordinator.execute(config, config.validation_input, cancel, request_id, False, administrative=True):
                pass
        except BaseException as exc:
            self.registry.validation(model_id, False, str(exc))
            if self.registry.get(model_id)["management_state"] != "reconciliation_required":
                self.registry.management(model_id, "validation_failed")
            await self.scheduler.pause(config.instance_key)
            self.observer.event("validation_failed", model=model_id, code=getattr(exc, "code", type(exc).__name__))
            raise
        self.registry.validation(model_id, True)
        self.registry.enable(model_id, enable)
        self.registry.management(model_id, "ready")
        if enable:
            await self.scheduler.resume(config.instance_key)
        self.observer.event("validation_passed", model=model_id, enabled=enable, mock=self.policy.catalog.describe(config).synthetic)
        return self._model_status(self.registry.get(model_id))

    async def enable(self, model_id: str) -> dict:
        async with self._manage():
            row = self.registry.get(model_id)
            self.policy.validate(row["config"])
            state = self.lifecycle.status(row["config"])
            if state.get("adapter_state", state["state"]) in {"unloading", "unload_pending"}:
                raise ServiceError("model_unavailable", "Complete pending unloading before enabling", 409)
            await self.scheduler.resume(row["config"].instance_key)
            try:
                self.registry.enable(model_id, True)
            except BaseException:
                await self.scheduler.pause(row["config"].instance_key)
                raise
            self.registry.management(model_id, "ready")
            self.observer.event("model_enabled", model=model_id)
            return self._model_status(self.registry.get(model_id))

    async def load(self, model_id: str) -> dict:
        async with self._manage():
            return await self._load(model_id)

    async def _load(self, model_id: str, *, maintenance: bool = False) -> dict:
        row = self.registry.get(model_id)
        if not row["validated"]:
            raise ServiceError("not_validated", "Validate the model before loading", 409)
        config = row["config"]
        if row["management_state"] != "ready" or self.scheduler.has_quarantine(config.instance_key):
            raise ServiceError("model_unavailable", "Finish draining or reconciliation before loading", 409)
        self.policy.validate(config)
        lease = await self.scheduler.acquire(config, "load-" + uuid.uuid4().hex, asyncio.Event(),
                                             timeout_s=.01 if maintenance else None, administrative=True)
        try:
            await self.lifecycle.ensure_loaded(config)
        finally:
            async def finish_load():
                await self.lifecycle.settle_load(config)
                await self.scheduler.release(lease)
                if not self.lifecycle.is_alive(config.instance_key) and not self.scheduler.count(config.instance_key):
                    await self.scheduler.drop_resident(config.instance_key)
            await finish_shielded(finish_load())
        self.observer.event("model_loaded", model=model_id, state=self.lifecycle.status(config))
        return self._model_status(row)

    async def operate(self, model_id: str, operation: str, timeout_s: float | None = None) -> dict:
        if operation not in {"unload", "disable", "remove"}:
            raise ServiceError("invalid_operation", "Expected unload, disable or remove", 422)
        timeout_s = self.settings.drain_timeout_s if timeout_s is None else timeout_s
        if not math.isfinite(timeout_s) or timeout_s <= 0 or timeout_s > 3600:
            raise ServiceError("invalid_timeout", "Drain timeout must be in (0,3600]", 422)
        async with self._manage():
            row = self.registry.get(model_id)
            config = row["config"]
            refs = self.registry.references(model_id)
            if operation in {"remove", "disable"} and any(refs.values()):
                raise ServiceError("model_referenced", "Reassign/delete aliases and registered dependencies first: " + json.dumps(refs), 409)
            self.registry.management(model_id, "draining")
            await self.scheduler.pause(config.instance_key)
            self.observer.event("model_draining", model=model_id, operation=operation)
            await self.scheduler.wait_idle(config.instance_key, timeout_s)
            result = await self.lifecycle.unload(config, timeout_s)
            await self.scheduler.drop_resident(config.instance_key)
            if operation == "remove":
                self.registry.remove(model_id)
            elif operation == "disable":
                self.registry.enable(model_id, False)
                self.registry.management(model_id, "ready")
            else:
                self.registry.management(model_id, "ready")
                if row["enabled"]:
                    await self.scheduler.resume(config.instance_key)
            self.observer.event("model_" + operation, model=model_id, release=result)
            return {"model_id": model_id, "operation": operation, "files_deleted": False, "release": result,
                    "state": None if operation == "remove" else self._model_status(self.registry.get(model_id))}

    async def set_alias(self, alias: str, target: str) -> dict:
        if not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", alias):
            raise ServiceError("invalid_alias", "Invalid alias", 422)
        async with self._manage():
            self.registry.set_alias(alias, target)
            self.observer.event("alias_set", alias=alias, model=target)
            return {"alias": alias, "model_id": target}

    async def delete_alias(self, alias: str) -> dict:
        async with self._manage():
            self.registry.delete_alias(alias)
            self.observer.event("alias_deleted", alias=alias)
            return {"deleted": alias}

    async def add_dependency(self, dependency: str, model_id: str) -> dict:
        if not dependency or len(dependency) > 128:
            raise ServiceError("invalid_dependency", "Dependency name must contain 1 to 128 characters", 422)
        async with self._manage():
            self.registry.add_dependency(dependency, model_id)
            self.observer.event("dependency_set", dependency=dependency, model=model_id)
            return {"dependency": dependency, "model_id": model_id}

    async def delete_dependency(self, dependency: str) -> dict:
        async with self._manage():
            self.registry.delete_dependency(dependency)
            self.observer.event("dependency_deleted", dependency=dependency)
            return {"deleted": dependency}

    async def reconcile(self, request_id: str, confirmed_stopped: bool) -> dict:
        if not confirmed_stopped:
            raise ServiceError("confirmation_required", "Confirm remote execution stopped before releasing its reservation", 409)
        async with self._manage():
            records = [u for u in self.registry.uncertainties() if u["request_id"] == request_id]
            if not records:
                raise ServiceError("request_not_found", "No uncertain execution with this request ID", 404)
            model_id = records[0]["model_id"]
            await self.scheduler.reconcile(request_id)
            self.registry.clear_uncertainty(request_id)
            row = self.registry.get(model_id)
            config = row["config"]
            if not self.lifecycle.is_alive(config.instance_key) and not self.scheduler.count(config.instance_key):
                await self.scheduler.drop_resident(config.instance_key)
            remaining_quarantine = self.scheduler.has_quarantine(config.instance_key)
            if not remaining_quarantine and row["management_state"] == "reconciliation_required":
                self.registry.management(model_id, "ready")
                if row["enabled"]:
                    await self.scheduler.resume(config.instance_key)
            self.observer.event("remote_stop_confirmed", request_id=request_id, model=model_id)
            return {"request_id": request_id, "reconciled": True}

    def _model_status(self, row: dict) -> dict:
        config = row["config"]
        try:
            load = self.lifecycle.status(config)
        except ServiceError as exc:
            load = {"state": "unavailable", "error": exc.code}
        return {**row, "config": config.model_dump(), "load": load,
                "active_requests": self.scheduler.count(config.instance_key),
                "references": self.registry.references(config.model_id)}

    async def status(self) -> dict:
        return {"models": [self._model_status(row) for row in self.registry.all()],
                "scheduler": self.scheduler.snapshot(), "workers": self.executor.snapshot(),
                "aliases": self.registry.aliases(), "dependencies": self.registry.dependencies(),
                "uncertain_requests": self.registry.uncertainties(), "events": self.registry.events()}

    async def available_models(self) -> dict:
        """List callable models without loading them or exposing administrative state."""
        models = []
        if self._closing:
            return {"models": models, "max_body_bytes": self.settings.max_body_bytes}
        for row in self.registry.all():
            if not row["enabled"] or not row["validated"] or row["management_state"] != "ready":
                continue
            config = row["config"]
            if self.scheduler.is_paused(config.instance_key):
                continue
            try:
                features = self.policy.catalog.describe(config)
            except ServiceError:
                continue
            if features.synthetic and not self.settings.allow_mock:
                continue
            limits = {name: value for name, value in config.options.items()
                      if name in _PUBLIC_INPUT_LIMITS and type(value) in (int, float)
                      and 0 < value <= 2**53 - 1}
            models.append({"model_id": config.model_id, "name": config.name,
                           "version": config.version, "capabilities": list(config.capabilities),
                           "mock": features.synthetic,
                           "limits": {**limits, "max_input_bytes": config.max_input_bytes}})
        return {"models": models, "max_body_bytes": self.settings.max_body_bytes}

    async def maintain(self):
        while not self._closing:
            await asyncio.sleep(self.settings.maintenance_interval_s)
            if self._admin_lock.locked():
                continue
            async with self._admin_lock:
                for row in self.registry.all():
                    config = row["config"]
                    key = config.instance_key
                    if self.scheduler.count(key):
                        continue
                    if not self.lifecycle.is_alive(key):
                        await self.scheduler.drop_resident(key)
                    if row["management_state"] != "ready" or not row["enabled"]:
                        continue
                    try:
                        if time.monotonic() < self._retry_after.get(key, 0):
                            continue
                        if config.load_policy == "resident" and not self.lifecycle.is_alive(key):
                            # Do not block maintenance waiting behind the request queue.
                            if self.scheduler.has_waiters():
                                continue
                            await self._load(config.model_id, maintenance=True)
                        elif config.load_policy == "on_demand" and self.lifecycle.is_alive(key) and self.lifecycle.idle(key) >= config.idle_seconds:
                            if not await self.scheduler.try_pause_idle(key):
                                continue
                            self.registry.management(config.model_id, "draining")
                            await self.lifecycle.unload(config, self.settings.drain_timeout_s)
                            await self.scheduler.drop_resident(key)
                            self.registry.management(config.model_id, "ready")
                            await self.scheduler.resume(key)
                            self.observer.event("idle_unloaded", model=config.model_id)
                    except ServiceError as exc:
                        self._retry_after[key] = time.monotonic() + self.settings.resident_retry_seconds
                        self.observer.event("maintenance_error", model=config.model_id, code=exc.code)
