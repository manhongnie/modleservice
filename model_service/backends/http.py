"""Synchronous JSON adapter for an externally managed inference service."""
from __future__ import annotations

import os
from threading import Event
from typing import Any
from urllib.parse import urlsplit

import httpx

from model_service.contracts import ModelConfig, ServiceError


class HTTPBackend:
    def __init__(self, config: ModelConfig):
        self.options = config.options
        self.client: httpx.Client | None = None

    def load(self) -> dict[str, Any]:
        base = str(self.options.get("base_url", ""))
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ServiceError("invalid_config", "HTTP backend requires a base_url without embedded credentials")
        path = str(self.options.get("infer_path", "/infer"))
        if not path.startswith("/") or path.startswith("//") or urlsplit(path).scheme:
            raise ServiceError("invalid_config", "infer_path must be an absolute path on base_url")
        self.url = base.rstrip("/") + path
        headers = {}
        auth_env = self.options.get("auth_env")
        if auth_env:
            secret = os.environ.get(str(auth_env))
            if not secret:
                raise ServiceError("missing_secret", f"Environment variable {auth_env} is required", 503)
            headers["Authorization"] = f"Bearer {secret}"
        self.client = httpx.Client(
            timeout=float(self.options.get("timeout_s", 60)), headers=headers,
            follow_redirects=False, trust_env=False,
        )
        return {"backend": "http", "management": "external", "remote_state": "external", "adapter_ready": True}

    def infer(self, inputs: dict[str, Any], cancel: Event) -> dict[str, Any]:
        if cancel.is_set():
            raise ServiceError("cancelled", "Cancelled before contacting external service", 499)
        if self.client is None:
            raise ServiceError("not_loaded", "HTTP adapter is not initialized", 503)
        # Closing a socket does not prove a remote inference stopped. A cancellation
        # therefore waits for the response. Transport failure has unknown completion.
        try:
            with self.client.stream("POST", self.url, json=inputs) as response:
                max_bytes = int(self.options.get("max_response_bytes", 8 * 1024 * 1024))
                content = bytearray()
                for block in response.iter_bytes():
                    content.extend(block)
                    if len(content) > max_bytes:
                        raise ServiceError("execution_unknown", "External response exceeded limit; remote completion is unknown", 502)
                status = response.status_code
        except httpx.HTTPError as exc:
            raise ServiceError("execution_unknown", "External transport failed; remote completion is unknown", 502) from exc
        if status == 202:
            raise ServiceError("execution_unknown", "External service accepted asynchronous work; completion is unknown", 502)
        if cancel.is_set():
            raise ServiceError("cancelled", "External service responded after cancellation; execution is stopped", 499)
        if status >= 300:
            raise ServiceError("backend_error", f"External service returned HTTP {status}", 502)
        try:
            import json
            result = json.loads(content)
        except (ValueError, UnicodeError) as exc:
            raise ServiceError("invalid_backend_output", "External service returned invalid JSON", 502) from exc
        if not isinstance(result, dict):
            raise ServiceError("invalid_backend_output", "External response must be a JSON object", 502)
        return result

    def close(self) -> dict[str, Any]:
        if self.client is not None:
            self.client.close()
            self.client = None
        return {"backend": "http", "adapter_closed": True, "management": "external", "remote_state": "external"}
