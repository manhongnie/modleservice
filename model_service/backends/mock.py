"""Explicit test double. No output from this backend represents model inference."""
from __future__ import annotations

import time
from threading import Event
from typing import Any

from model_service.contracts import ModelConfig, ServiceError


class MockBackend:
    def __init__(self, config: ModelConfig):
        self.options = config.options

    def load(self) -> dict[str, Any]:
        time.sleep(float(self.options.get("load_delay_s", 0)))
        if self.options.get("fail_load"):
            raise ServiceError("load_failed", "Mock configured to fail loading", 503)
        return {"backend": "mock", "mock": True, "management": "local"}

    def infer(self, inputs: dict[str, Any], cancel: Event) -> dict[str, Any]:
        if cancel.wait(float(self.options.get("delay_s", 0))):
            # Simulate a backend that needs time to acknowledge cancellation.
            time.sleep(float(self.options.get("cancel_delay_s", 0)))
            raise ServiceError("cancelled", "Mock execution stopped", 499)
        if self.options.get("fail"):
            raise ServiceError("inference_failed", "Mock configured to fail inference", 503)
        return {"mock": True, "payload": inputs}

    def stream(self, inputs: dict[str, Any], cancel: Event):
        output = self.infer(inputs, cancel)
        chunks = self.options.get("stream_chunks", 1)
        count = chunks if isinstance(chunks, int) else len(chunks)
        for index in range(count):
            if cancel.wait(float(self.options.get("chunk_delay_s", 0))):
                time.sleep(float(self.options.get("cancel_delay_s", 0)))
                raise ServiceError("cancelled", "Mock output stream stopped", 499)
            if self.options.get("fail_stream_after") == index:
                raise ServiceError("inference_failed", "Mock stream failed after partial output", 503)
            if isinstance(chunks, int):
                yield {**output, "chunk_index": index}
            else:
                yield {"chunk": chunks[index], "chunk_index": index}

    def close(self) -> dict[str, Any]:
        return {"backend": "mock", "mock": True, "unloaded": True}
