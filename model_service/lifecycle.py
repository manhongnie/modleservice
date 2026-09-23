"""Model-instance state and single-flight loading; no budget or database access."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from copy import deepcopy

from .contracts import BackendFeatures, Executor, ModelConfig, ServiceError


class Lifecycle:
    def __init__(self, executor: Executor,
                 describe: Callable[[ModelConfig], BackendFeatures] | None = None):
        self.executor = executor
        if describe is None:
            # Backward-compatible standalone construction. The service composition
            # root supplies its configured catalog explicitly.
            from .plugins import BuiltinPluginCatalog
            describe = BuiltinPluginCatalog().describe
        self._describe = describe
        self._locks: dict[str, asyncio.Lock] = {}
        self._states: dict[str, dict] = {}
        self._last_used: dict[str, float] = {}

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    async def ensure_loaded(self, config: ModelConfig) -> dict:
        key = config.instance_key
        async with self._lock(key):
            state = self._states.get(key, {})
            alive = self.executor.is_alive(key)
            if alive and state.get("state") == "loaded":
                return deepcopy(state)
            if alive and state.get("state") in {"unloading", "unload_pending", "unloaded", "failed"}:
                raise ServiceError("model_unavailable", "Existing model worker is not ready; complete unloading before reuse", 409)
            began = time.monotonic()
            self._states[key] = {"state": "loading", "load_seconds": None}
            try:
                info = await self.executor.load(config)
            except BaseException:
                self._states[key] = {"state": "load_pending" if self.executor.is_alive(key) else "failed",
                                     "load_seconds": time.monotonic() - began}
                raise
            if info.get("state", "loaded") != "loaded" or not self.executor.is_alive(key):
                self._states[key] = {"state": "failed", "load_seconds": time.monotonic() - began}
                raise ServiceError("model_unavailable", "Model worker did not confirm a live loaded instance", 503)
            state = {**info, "state": "loaded", "load_seconds": time.monotonic() - began}
            self._states[key] = state
            self.touch(key)
            return deepcopy(state)

    async def unload(self, config: ModelConfig, timeout_s: float) -> dict:
        key = config.instance_key
        async with self._lock(key):
            self._states[key] = {**self._states.get(key, {}), "state": "unloading"}
            try:
                result = await self.executor.unload(key, timeout_s)
                if self.executor.is_alive(key):
                    raise ServiceError("unload_timeout", "Model worker has not exited; reservation remains held", 409)
            except BaseException:
                self._states[key] = {**self._states[key],
                                     "state": "unload_pending" if self.executor.is_alive(key) else "failed"}
                raise
            self._states[key] = {"state": "unloaded", "last_release": deepcopy(result)}
            self.touch(key)
            return result

    async def settle_load(self, config: ModelConfig) -> None:
        """Compile timeouts do not mean native loading stopped. Keep the pin.

        Most loaders cannot interrupt compilation. Cleanup waits until the worker
        reports ready or exits, without silently killing it or retrying inference.
        """
        async def settle() -> None:
            while self._states.get(config.instance_key, {}).get("state") == "load_pending" and self.is_alive(config.instance_key):
                try:
                    await self.ensure_loaded(config)
                except Exception:
                    if self.is_alive(config.instance_key):
                        await asyncio.sleep(.02)

        pending = asyncio.create_task(settle())
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Cancellation at any await, including the retry delay, cannot
                # abandon a native load that still owns its request lease.
                continue
        pending.result()

    def touch(self, key: str) -> None:
        self._last_used[key] = time.monotonic()

    def idle(self, key: str) -> float:
        return time.monotonic() - self._last_used.get(key, time.monotonic())

    def is_alive(self, key: str) -> bool:
        return self.executor.is_alive(key)

    def status(self, config: ModelConfig) -> dict:
        state = deepcopy(self._states.get(config.instance_key, {"state": "unloaded"}))
        if state["state"] in {"loaded", "loading", "load_pending", "unloading", "unload_pending"} and not self.is_alive(config.instance_key):
            state["state"] = "failed"
        if self._describe(config).management == "external":
            state = {**state, "adapter_state": state["state"], "state": "external", "remote_lifecycle": "externally_managed"}
        return state
