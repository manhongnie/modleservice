"""Stable contracts shared by control plane, workers, task plugins and backends."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from threading import Event
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
    version: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
    capabilities: list[str] = Field(min_length=1)
    task: str
    backend: str
    path: str = ""
    device: str = "CPU"
    load_policy: str = Field(default="on_demand", pattern=r"^(on_demand|resident)$")
    idle_seconds: float = Field(default=300, ge=0)
    concurrency: int = Field(default=1, ge=1, le=64)
    resident_mb: int = Field(default=256, ge=0)
    request_mb: int = Field(default=32, ge=0)
    max_input_bytes: int = Field(default=1048576, ge=1, le=67108864)
    options: dict[str, Any] = Field(default_factory=dict)
    validation_input: dict[str, Any]

    @property
    def model_id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def instance_key(self) -> str:
        # Version and execution configuration are immutable. Aliases resolve before this.
        identity = self.model_dump(exclude={"load_policy", "idle_seconds", "validation_input"})
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    database: str = "var/models.sqlite3"
    model_roots: list[str] = Field(default_factory=lambda: ["models"])
    business_keys: list[str] = Field(min_length=1)
    admin_keys: list[str] = Field(min_length=1)
    total_memory_mb: int = Field(default=4096, ge=1)
    max_executions: int = Field(default=4, ge=1)
    queue_size: int = Field(default=32, ge=0)
    queue_timeout_s: float = Field(default=10, gt=0)
    execution_timeout_s: float = Field(default=60, gt=0)
    load_timeout_s: float = Field(default=120, gt=0)
    drain_timeout_s: float = Field(default=30, gt=0)
    max_body_bytes: int = Field(default=2097152, ge=1)
    max_output_bytes: int = Field(default=8388608, ge=1)
    maintenance_interval_s: float = Field(default=1, gt=0)
    http_allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])


    deployment_mode: Literal["development", "production"] = "development"
    allow_mock: bool = True
    max_http_requests: int = Field(default=128, ge=1)
    body_timeout_s: float = Field(default=15, gt=0, le=300)
    write_timeout_s: float = Field(default=30, gt=0, le=600)
    resident_retry_seconds: float = Field(default=30, ge=1, le=3600)

    @model_validator(mode="after")
    def distinct_keys(self):
        if set(self.business_keys) & set(self.admin_keys) or any(not k for k in self.business_keys + self.admin_keys):
            raise ValueError("business and admin keys must be nonempty and disjoint")
        if self.deployment_mode == "production":
            if any(len(key) < 32 for key in self.business_keys + self.admin_keys):
                raise ValueError("production API keys must contain at least 32 characters")
            if self.allow_mock:
                raise ValueError("production mode requires allow_mock=false")
        return self


@dataclass(frozen=True)
class BackendFeatures:
    """Execution semantics declared in the trusted backend registration."""
    management: Literal["local", "external"] = "local"
    execution_may_outlive_worker: bool = False
    requires_artifact: bool = True
    synthetic: bool = False
    incremental_output: bool = False


class PluginCatalog(Protocol):
    def describe(self, config: ModelConfig) -> BackendFeatures: ...
    def validate(self, config: ModelConfig, settings: Settings) -> None: ...
    def validate_input(self, config: ModelConfig, payload: dict[str, Any], capability: str) -> None: ...


@dataclass
class Prepared:
    inputs: dict[str, Any]
    context: dict[str, Any]


class TaskPlugin(Protocol):
    def validate(self, payload: dict[str, Any], capability: str) -> None: ...
    def prepare(self, payload: dict[str, Any]) -> Prepared: ...
    def finish(self, outputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]: ...


class Backend(Protocol):
    """All methods run in worker. infer must not finish until underlying work stops.

    An HTTP transport failure with unknown remote completion must be reported as
    execution_unknown; the coordinator quarantines that reservation for reconciliation.
    """
    def load(self) -> dict[str, Any]: ...
    def infer(self, inputs: dict[str, Any], cancel: Event) -> dict[str, Any]: ...
    def close(self) -> dict[str, Any]: ...


class Executor(Protocol):
    async def load(self, config: ModelConfig) -> dict[str, Any]: ...
    def stream(self, key: str, payload: dict[str, Any], request_id: str, stream: bool, cancel: Any): ...
    async def unload(self, key: str, timeout_s: float = 30) -> dict[str, Any]: ...
    def snapshot(self) -> dict[str, Any]: ...
    def is_alive(self, key: str) -> bool: ...
    async def close(self, timeout_s: float = 30) -> None: ...


class StreamingBackend(Protocol):
    """stream().close() must wait for underlying generation to stop before returning."""
    def stream(self, inputs: dict[str, Any], cancel: Event): ...


class StreamingTask(Protocol):
    """Translate backend chunks; per-request decoding state belongs in context."""
    def finish_chunk(self, outputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any] | None: ...
