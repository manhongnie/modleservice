"""Streaming contracts: real token production semantics without model weights."""
import asyncio
import threading
import time
from threading import Event

import pytest

from model_service.backends.generative import TransformersBackend
from model_service.contracts import ModelConfig, ServiceError
from model_service.executor import ProcessExecutor
from model_service.tasks.core import ChatTask


def config(**changes):
    return ModelConfig(**dict(name="stream-test", version="1", capabilities=["chat"],
        task="qwen_chat", backend="transformers", path="unused", validation_input={"messages": [
        {"role": "user", "content": "hello"}]}, **changes))


class FakeGeneratingModel:
    def __init__(self, count=1000, failure=False, stop_delay=0.1):
        self.count, self.failure, self.stop_delay = count, failure, stop_delay
        self.stopped = Event()
        self.produced = 0

    def generate(self, input_ids, streamer, stopping_criteria, **kwargs):
        import torch
        try:
            streamer.put(input_ids)
            for _ in range(self.count):
                streamer.put(torch.tensor([42]))
                self.produced += 1
                if self.failure:
                    raise ValueError("generation failed after token")
                if any(criterion(None, None) for criterion in stopping_criteria):
                    break
            streamer.end()
        finally:
            time.sleep(self.stop_delay)
            self.stopped.set()


def test_generation_is_incremental_bounded_and_close_confirms_stop():
    torch = pytest.importorskip("torch")
    backend = TransformersBackend(config())
    backend.model = FakeGeneratingModel()
    cancel = Event()
    stream = backend.stream({"input_ids": torch.tensor([[1, 2]])}, cancel)
    first = next(stream)
    assert first == {"token_ids": [42], "final": False}
    assert not backend.model.stopped.is_set()
    time.sleep(0.1)
    assert backend.model.produced <= 10  # One consumed item plus eight queue slots.
    before = time.monotonic()
    stream.close()
    assert time.monotonic() - before >= 0.09
    assert backend.model.stopped.is_set() and cancel.is_set()
    assert not any(t.name == "token-generation" and t.is_alive() for t in threading.enumerate())


def test_generation_error_after_partial_output_is_not_retried():
    torch = pytest.importorskip("torch")
    backend = TransformersBackend(config())
    backend.model = FakeGeneratingModel(failure=True, stop_delay=0.01)
    stream = backend.stream({"input_ids": torch.tensor([[1, 2]])}, Event())
    assert next(stream)["token_ids"] == [42]
    with pytest.raises(ValueError):
        next(stream)
    assert backend.model.produced == 1 and backend.model.stopped.is_set()


def test_chat_stream_does_not_emit_partial_utf8_and_final_usage():
    task = ChatTask(config())

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {1: "你\ufffd", 2: "你好", 3: "你好世界"}[len(ids)]

    task.tokenizer = Tokenizer()
    context = {"prompt_tokens": 5}
    chunks = [task.finish_chunk({"token_ids": list(range(size)), "final": size == 3}, context) for size in (1, 2, 3)]
    assert "".join(chunk["delta"] for chunk in chunks) == "你好世界"
    assert chunks[-1]["text"] == "你好世界"
    assert chunks[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}
    assert all("\ufffd" not in chunk["delta"] for chunk in chunks)


@pytest.mark.parametrize("chunk_count", [9, 10000])
async def test_slow_output_consumer_is_bounded_and_execution_stops(chunk_count):
    cfg = ModelConfig(name="slow", version="1", task="mock", backend="mock", capabilities=["chat"],
        validation_input={"messages": [{"role": "user", "content": "test"}]},
        options={"stream_chunks": chunk_count, "chunk_delay_s": 0.001, "cancel_delay_s": 0.05})
    executor = ProcessExecutor()
    try:
        await executor.load(cfg)
        stream = executor.stream(cfg.instance_key, cfg.validation_input, "slow", True, asyncio.Event())
        assert (await anext(stream))["mock"]
        await asyncio.sleep(0.2)
        state = executor.snapshot()[cfg.instance_key]
        assert state["buffered_chunks"] <= state["output_buffer_limit"]
        with pytest.raises(ServiceError) as error:
            async for _ in stream:
                pass
        assert error.value.code == "stream_backpressure"
        assert executor.snapshot()[cfg.instance_key]["busy"] == 0
        assert executor.snapshot()[cfg.instance_key]["load_count"] == 1
    finally:
        await executor.close()


async def test_worker_reports_backend_partial_error_without_retry():
    cfg = ModelConfig(name="failing", version="1", task="mock", backend="mock", capabilities=["chat"],
        validation_input={"messages": [{"role": "user", "content": "test"}]},
        options={"stream_chunks": 10, "fail_stream_after": 1})
    executor = ProcessExecutor()
    try:
        await executor.load(cfg)
        stream = executor.stream(cfg.instance_key, cfg.validation_input, "failing", True, asyncio.Event())
        assert (await anext(stream))["chunk_index"] == 0
        with pytest.raises(ServiceError) as error:
            await anext(stream)
        assert error.value.code == "inference_failed"
        assert executor.snapshot()[cfg.instance_key]["completed"] == 1
    finally:
        await executor.close()


def test_generation_reports_token_limit_finish_reason():
    torch = pytest.importorskip("torch")
    backend = TransformersBackend(config())
    backend.model = FakeGeneratingModel(count=2, stop_delay=0)
    chunks = list(backend.stream({"input_ids": torch.tensor([[1]]), "max_new_tokens": 2}, Event()))
    assert chunks[-1] == {"token_ids": [42, 42], "final": True, "finish_reason": "length"}
    assert backend.model.stopped.is_set()
