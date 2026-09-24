"""One atomic ledger for admission, request pins and resident reservations."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from .contracts import ModelConfig, ServiceError, Settings


@dataclass(frozen=True)
class Lease:
    request_id: str
    key: str
    model_id: str
    temporary_mb: int
    acquired_at: float
    gpu_temporary_mb: int = 0


class Scheduler:
    """Own resource state; callers use queries, never mutate the ledger.

    Synchronous queries are snapshots on the controller's event loop. Decisions
    that grant resources or change admission always take the same condition lock.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._condition = asyncio.Condition()
        self._residents: dict[str, int] = {}
        self._gpu_residents: dict[str, int] = {}
        self._active: dict[str, Lease] = {}
        self._paused: set[str] = set()
        self._queue: list[tuple[str, str]] = []
        self._quarantined: set[str] = set()
        self._pause_epochs: dict[str, int] = {}

    def _count(self, key: str) -> int:
        return sum(lease.key == key for lease in self._active.values())

    def _used(self) -> int:
        return sum(self._residents.values()) + sum(lease.temporary_mb for lease in self._active.values())

    def _gpu_used(self) -> int:
        return sum(self._gpu_residents.values()) + sum(lease.gpu_temporary_mb for lease in self._active.values())

    def _fits(self, config: ModelConfig) -> bool:
        return (len(self._active) < self.settings.max_executions
                and self._count(config.instance_key) < config.concurrency
                and self._used() + config.request_mb + (0 if config.instance_key in self._residents else config.resident_mb)
                <= self.settings.total_memory_mb
                and self._gpu_used() + config.gpu_request_mb + (0 if config.instance_key in self._gpu_residents else config.gpu_resident_mb)
                <= self.settings.total_gpu_memory_mb)

    def _grant(self, config: ModelConfig, request_id: str) -> Lease:
        self._residents.setdefault(config.instance_key, config.resident_mb)
        self._gpu_residents.setdefault(config.instance_key, config.gpu_resident_mb)
        lease = Lease(request_id, config.instance_key, config.model_id, config.request_mb, time.monotonic(), config.gpu_request_mb)
        self._active[request_id] = lease
        return lease

    def _pause(self, key: str) -> None:
        self._paused.add(key)
        # A waiting request must observe that admission was stopped even when a
        # fast management operation resumes it before that waiter runs again.
        self._pause_epochs[key] = self._pause_epochs.get(key, 0) + 1
        self._condition.notify_all()

    @staticmethod
    def _validate_timeout(timeout_s: float) -> None:
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ServiceError("invalid_timeout", "Timeout must be finite and nonnegative", 422)

    def _validate_budget(self, config: ModelConfig) -> None:
        if min(config.resident_mb, config.request_mb, config.gpu_resident_mb, config.gpu_request_mb) < 0:
            raise ServiceError("resource_limit", "Resource budgets must be nonnegative", 422)
        if config.resident_mb + config.request_mb > self.settings.total_memory_mb:
            raise ServiceError("resource_limit", "Model resident plus request budget exceeds service limit", 422)
        if config.gpu_resident_mb + config.gpu_request_mb > self.settings.total_gpu_memory_mb:
            raise ServiceError("resource_limit", "Model GPU resident plus request budget exceeds service limit", 422)

    async def acquire(self, config: ModelConfig, request_id: str, cancel: asyncio.Event,
                      timeout_s: float | None = None, *, administrative: bool = False) -> Lease:
        self._validate_budget(config)
        timeout = self.settings.queue_timeout_s if timeout_s is None else timeout_s
        self._validate_timeout(timeout)
        deadline = time.monotonic() + timeout
        ticket = (request_id, config.instance_key)
        async with self._condition:
            if request_id in self._active or any(waiter_id == request_id for waiter_id, _ in self._queue):
                raise ServiceError("duplicate_request", "Request ID is already active or queued", 409)
            epoch = self._pause_epochs.get(config.instance_key, 0)
            try:
                while True:
                    if cancel.is_set():
                        raise ServiceError("cancelled", "Request cancelled before execution", 499)
                    if not administrative and (config.instance_key in self._paused
                            or epoch != self._pause_epochs.get(config.instance_key, 0)):
                        raise ServiceError("model_paused", "Model is paused for management or reconciliation", 409)
                    if ticket in self._queue and time.monotonic() >= deadline:
                        raise ServiceError("queue_timeout", "Timed out waiting for resources", 504)
                    first = not self._queue or self._queue[0] == ticket
                    if first and self._fits(config):
                        return self._grant(config, request_id)
                    if ticket not in self._queue:
                        if len(self._queue) >= self.settings.queue_size:
                            raise ServiceError("queue_full", "Waiting queue is full", 429)
                        self._queue.append(ticket)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ServiceError("queue_timeout", "Timed out waiting for resources", 504)
                    try:
                        await asyncio.wait_for(self._condition.wait(), min(0.05, remaining))
                    except TimeoutError:
                        pass
            finally:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                self._condition.notify_all()

    async def release(self, lease: Lease) -> None:
        async with self._condition:
            # A delayed cleanup cannot release a newer request that reused an ID.
            if self._active.get(lease.request_id) == lease:
                self._active.pop(lease.request_id)
                self._quarantined.discard(lease.request_id)
                self._condition.notify_all()

    async def quarantine(self, lease: Lease) -> None:
        async with self._condition:
            if self._active.get(lease.request_id) != lease:
                raise ServiceError("request_not_active", "Cannot quarantine a released request", 409)
            self._quarantined.add(lease.request_id)
            self._pause(lease.key)

    async def restore_quarantine(self, config: ModelConfig, request_id: str) -> None:
        # Recovery restores the old reservation even when current configured
        # capacity is lower: unknown remote execution still owns those resources.
        async with self._condition:
            if request_id in self._active or any(waiter_id == request_id for waiter_id, _ in self._queue):
                raise ServiceError("duplicate_request", "Request ID is already active or queued", 409)
            self._grant(config, request_id)
            self._quarantined.add(request_id)
            self._pause(config.instance_key)

    async def reconcile(self, request_id: str) -> None:
        async with self._condition:
            lease = self._active.get(request_id)
            if lease:
                if request_id not in self._quarantined:
                    raise ServiceError("not_quarantined", "Request is still under normal execution", 409)
                self._active.pop(request_id)
                self._quarantined.discard(request_id)
                self._condition.notify_all()

    async def pause(self, key: str) -> None:
        async with self._condition:
            self._pause(key)

    async def try_pause_idle(self, key: str) -> bool:
        """Claim an unused model for maintenance without racing new admission."""
        async with self._condition:
            if key in self._paused or self._count(key) or self.has_waiters(key):
                return False
            self._pause(key)
            return True

    async def resume(self, key: str) -> None:
        async with self._condition:
            if self.has_quarantine(key):
                raise ServiceError("execution_unknown", "Confirm remote completion before resuming", 409)
            self._paused.discard(key)
            self._condition.notify_all()

    async def wait_idle(self, key: str, timeout_s: float) -> None:
        self._validate_timeout(timeout_s)
        deadline = time.monotonic() + timeout_s
        async with self._condition:
            while self._count(key):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ServiceError("drain_timeout", f"Model remains paused with {self._count(key)} active request(s); no process was killed", 409)
                try:
                    await asyncio.wait_for(self._condition.wait(), remaining)
                except TimeoutError:
                    pass

    async def drop_resident(self, key: str) -> None:
        async with self._condition:
            if self._count(key):
                raise ServiceError("model_busy", "Cannot release resident resources while model is in use", 409)
            self._residents.pop(key, None)
            self._gpu_residents.pop(key, None)
            self._condition.notify_all()

    def count(self, key: str) -> int:
        return self._count(key)

    def normal_active_count(self) -> int:
        return len(self._active) - len(self._quarantined)

    def has_waiters(self, key: str | None = None) -> bool:
        return bool(self._queue) if key is None else any(waiter_key == key for _, waiter_key in self._queue)

    def has_quarantine(self, key: str) -> bool:
        return any(self._active[request_id].key == key for request_id in self._quarantined)

    def quarantine_request_ids(self) -> list[str]:
        return sorted(self._quarantined)

    def is_paused(self, key: str) -> bool:
        return key in self._paused

    def snapshot(self) -> dict:
        return {"queue_length": len(self._queue), "queue_capacity": self.settings.queue_size,
                "active_count": len(self._active), "execution_capacity": self.settings.max_executions,
                "resident_mb": sum(self._residents.values()),
                "temporary_mb": sum(lease.temporary_mb for lease in self._active.values()),
                "reserved_mb": self._used(), "budget_mb": self.settings.total_memory_mb,
                "gpu_resident_mb": sum(self._gpu_residents.values()),
                "gpu_temporary_mb": sum(lease.gpu_temporary_mb for lease in self._active.values()),
                "gpu_reserved_mb": self._gpu_used(), "gpu_budget_mb": self.settings.total_gpu_memory_mb,
                "resident_instances": len(self._residents), "quarantined_requests": sorted(self._quarantined),
                "active_by_model": {lease.model_id: self._count(lease.key) for lease in self._active.values()}}
