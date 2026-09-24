"""Cancellation while encoding must wait for completion and suppress late output."""
from queue import Queue, Empty
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from model_service import worker
from model_service.contracts import ModelConfig, BackendFeatures


@pytest.mark.parametrize('mode', ['ordinary', 'buffered', 'incremental'])
def test_cancel_during_postprocessing_has_no_success_chunk(monkeypatch, mode):
    commands, messages = Queue(), Queue()
    entered, release, finished = Event(), Event(), Event()
    class Connection:
        def recv(self): return commands.get(timeout=5)
        def send(self, message): messages.put(message)
        def close(self): pass
    class Backend:
        def load(self): return {}
        def infer(self, inputs, cancel):
            self.cancel = cancel
            return {'value': 1}
        def stream(self, inputs, cancel):
            yield self.infer(inputs, cancel)
        def close(self): return {'released': True}
    backend = Backend()
    class Task:
        def prepare(self, payload): return SimpleNamespace(inputs=payload, context={})
        def finish(self, output, context):
            entered.set()
            assert release.wait(3)
            finished.set()
            return output
        finish_chunk = finish
    monkeypatch.setattr(worker, '_parent_death_signal', lambda _: None)
    monkeypatch.setattr(worker, 'create_backend', lambda *_: backend)
    monkeypatch.setattr(worker, 'create_task', lambda *_args, **_kwargs: Task())
    config = ModelConfig(name='test', version='1', backend='mock', task='mock',
                         capabilities=['embeddings'], validation_input={'texts': ['x']})
    registration = SimpleNamespace(task_family='mock', describe=lambda _: BackendFeatures(incremental_output=mode == 'incremental'))
    thread = Thread(target=worker.worker_main, args=(Connection(), config.model_dump(), 0, 1000, registration))
    thread.start()
    try:
        assert messages.get(timeout=2)['id'] == '__load__'
        commands.put({'op': 'infer', 'id': 'r', 'payload': {}, 'stream': mode != 'ordinary'})
        assert entered.wait(2)
        commands.put({'op': 'cancel', 'id': 'r'})
        assert backend.cancel.wait(2)
        with pytest.raises(Empty): messages.get(timeout=.05)
        release.set()
        terminal = messages.get(timeout=2)
        assert finished.is_set()
        assert terminal['type'] == 'done' and terminal['error']['code'] == 'cancelled'
        assert messages.empty()  # No success chunk escaped during finish/encoding.
    finally:
        release.set()
        commands.put({'op': 'unload'})
        thread.join(3)
        assert not thread.is_alive()
