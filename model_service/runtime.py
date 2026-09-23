"""Single-controller composition, recovery, readiness and graceful shutdown."""
from __future__ import annotations

import asyncio
import fcntl
import os
import time
from pathlib import Path

from .configuration import ConfigurationPolicy
from .contracts import Executor, PluginCatalog, ServiceError, Settings
from .coordinator import RequestCoordinator
from .lifecycle import Lifecycle
from .management import ModelManager
from .observability import Observability
from .registry import Registry
from .scheduler import Scheduler


class ApplicationRuntime:
    def __init__(self, settings: Settings, executor: Executor | None = None,
                 catalog: PluginCatalog | None = None):
        self.settings, self.executor, self.catalog = settings, executor, catalog
        self.registry = None
        self.scheduler = Scheduler(settings)
        self.lifecycle = None
        self._lock_file = None
        self._maintenance = None
        self._close_lock = asyncio.Lock()
        self._started = False
        self._closing = False

    async def start(self):
        if self._lock_file is not None:
            raise ServiceError("already_started", "Controller has already started", 409)
        path = Path(self.settings.database).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(str(path) + ".lock", "a+")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            self._lock_file = None
            raise ServiceError("controller_exists", "Another controller owns this database; use one Web worker", 409) from None
        try:
            self.scheduler = Scheduler(self.settings)
            if self.settings.deployment_mode == "production":
                os.fchmod(self._lock_file.fileno(), 0o600)
                descriptor = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(descriptor)
                path.chmod(0o600)
            self.registry = Registry(str(path))
            if self.executor is None:
                from .executor import ProcessExecutor
                self.executor = ProcessExecutor(load_timeout_s=self.settings.load_timeout_s,
                                                max_output_bytes=self.settings.max_output_bytes)
            if self.catalog is None:
                from .plugins import BuiltinPluginCatalog
                self.catalog = BuiltinPluginCatalog()
            self.lifecycle = Lifecycle(self.executor, self.catalog.describe)
            self.policy = ConfigurationPolicy(self.settings, self.catalog)
            self.observer = Observability(self.registry)
            self.coordinator = RequestCoordinator(self.settings, self.registry, self.scheduler,
                                                  self.lifecycle, self.executor, self.policy, self.observer)
            self.manager = ModelManager(self.settings, self.registry, self.scheduler, self.lifecycle,
                                        self.executor, self.policy, self.coordinator, self.observer)
            for row in self.registry.all():
                if row["management_state"] != "ready" or not row["enabled"]:
                    await self.scheduler.pause(row["config"].instance_key)
            for uncertain in self.registry.uncertainties():
                row = self.registry.get(uncertain["model_id"])
                config = row["config"]
                await self.scheduler.restore_quarantine(config, uncertain["request_id"])
                if row["management_state"] != "draining":
                    self.registry.management(config.model_id, "reconciliation_required")
            self.observer.event("startup", recovered_uncertainties=len(self.registry.uncertainties()))
            self._maintenance = asyncio.create_task(self.manager.maintain(), name="model-maintenance")
            self._closing = False
            self._started = True
        except BaseException:
            if self.registry:
                self.registry.close()
                self.registry = None
            self._unlock()
            raise

    async def readiness(self) -> dict:
        reasons = []
        if not self._started or self._closing:
            reasons.append("controller_not_accepting_requests")
        if self._maintenance is None or self._maintenance.done():
            reasons.append("maintenance_unavailable")
        if self.registry is not None and self._started:
            for row in self.registry.all():
                if not row["enabled"]:
                    continue
                config = row["config"]
                try:
                    features = self.catalog.describe(config)
                    state = self.lifecycle.status(config)
                    if features.synthetic and not self.settings.allow_mock:
                        reasons.append(f"{config.model_id}:mock_disabled")
                    if row["management_state"] != "ready" or self.scheduler.is_paused(config.instance_key):
                        reasons.append(f"{config.model_id}:not_accepting_requests")
                    elif config.load_policy == "resident" and state.get("adapter_state", state["state"]) != "loaded":
                        reasons.append(f"{config.model_id}:resident_not_loaded")
                except ServiceError as exc:
                    reasons.append(f"{config.model_id}:{exc.code}")
        return {"status": "not_ready" if reasons else "ready", "reasons": reasons}

    async def metrics(self) -> str:
        return self.observer.metrics(self.scheduler.snapshot(), self.executor.snapshot())

    def _unlock(self):
        fcntl.flock(self._lock_file, fcntl.LOCK_UN)
        self._lock_file.close()
        self._lock_file = None

    async def close(self):
        async with self._close_lock:
            await self._close()

    async def _close(self):
        if self._lock_file is None:
            return
        self._closing = True
        self.manager.begin_shutdown()
        self.coordinator.begin_shutdown()
        if self._maintenance:
            self._maintenance.cancel()
            done, _ = await asyncio.wait({self._maintenance}, timeout=self.settings.drain_timeout_s)
            if not done:
                raise ServiceError("shutdown_timeout", "Model maintenance has not stopped; controller lock and registry remain open", 409)
            try:
                await self._maintenance
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self.observer.event("maintenance_stopped", error=type(exc).__name__)
        async with self.manager.shutdown_guard(self.settings.drain_timeout_s):
            deadline = time.monotonic() + self.settings.drain_timeout_s
            while self.coordinator.pending_count() or self.scheduler.normal_active_count():
                if time.monotonic() >= deadline:
                    raise ServiceError("shutdown_timeout", "Requests have not stopped; controller lock and registry remain open", 409)
                await asyncio.sleep(.01)
            await self.executor.close(self.settings.drain_timeout_s)
            self.observer.event("shutdown", resources=self.scheduler.snapshot())
            self.registry.close()
            self.registry = None
            self._started = False
            self._unlock()
