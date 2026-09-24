"""Real SenseVoice repeated-request, cancellation and process-reclamation probe.

Uses an isolated SQLite registry and the production Service/worker path. Does not
change the deployed registry, trim malloc arenas, or force collection per request.
Reports observations, not proof of absence of leaks over arbitrary runtimes.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import platform
import secrets
import statistics
import tempfile
from threading import Event, Thread
import time
import wave

import psutil

from model_service.contracts import ModelConfig, ServiceError, Settings
from model_service.service import Service


def summarize(samples):
    values = [sample['worker_rss_mb'] for sample in samples]
    count = len(values)
    window = min(20, count // 2)
    xs = list(range(count))
    xm, ym = statistics.mean(xs), statistics.mean(values)
    denominator = sum((x - xm) ** 2 for x in xs)
    return {'requests': count, 'rss_first_mb': values[0], 'rss_last_mb': values[-1],
            'rss_min_mb': min(values), 'rss_max_mb': max(values),
            'first_window_median_mb': statistics.median(values[:window]),
            'last_window_median_mb': statistics.median(values[-window:]),
            'window_median_growth_mb': statistics.median(values[-window:]) - statistics.median(values[:window]),
            'linear_slope_mb_per_request': sum((x - xm) * (y - ym) for x, y in zip(xs, values)) / denominator,
            'control_rss_first_mb': samples[0]['control_rss_mb'],
            'control_rss_last_mb': samples[-1]['control_rss_mb']}


def wave_at_duration(reference, seconds):
    with wave.open(io.BytesIO(base64.b64decode(reference)), 'rb') as audio:
        rate = audio.getframerate()
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        pcm = audio.readframes(audio.getnframes())
    needed = int(rate * seconds) * 2
    pcm = (pcm * math.ceil(needed / len(pcm)))[:needed]
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        audio.writeframes(pcm)
    return {'audio_base64': base64.b64encode(buffer.getvalue()).decode()}


def memory(process):
    try:
        value = process.memory_full_info()
        return {'rss_mb': value.rss / 1048576, 'uss_mb': value.uss / 1048576,
                'pss_mb': value.pss / 1048576}
    except psutil.NoSuchProcess:
        return {'exited': True}


async def run(args):
    config = next(ModelConfig(**row) for row in json.loads(Path('examples/models.sherpa.json').read_text())
                  if row['task'] == 'sherpa_sensevoice')
    root = Path('var/sensevoice-memory')
    root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'started_unix': time.time(), 'python': platform.python_version(),
              'platform': platform.platform(),
              'model_id': config.model_id, 'runtime': 'sherpa-onnx CPU, actual installed version recorded at load',
              'configuration': config.model_dump(exclude={'validation_input'}),
              'scope': 'Real production Service and reusable spawned worker, isolated temporary SQLite. Tiled synthetic speech tests allocation shapes; not an ASR accuracy or multi-day endurance benchmark.',
              'reference_sha256': hashlib.sha256(base64.b64decode(config.validation_input['audio_base64'])).hexdigest(),
              'samples_path': str(samples_path), 'phases': [], 'unload_cycles': [], 'cancellations': [],
              'measurement_notes': [
                  'Memory fields ending in _mb use MiB (1024 ** 2 bytes).',
                  'Per-request RSS is sampled after inference completion; it is not a native allocation trace.',
                  'The 20ms worker peak sampler attaches after initial validation and after each reload inference; cold loading is excluded.',
                  'Validation audio is omitted from saved snapshots; its SHA256 is recorded separately.',
                  'No per-request gc.collect(), malloc_trim(), provider changes, or production database writes.',
              ]}
    def without_audio(value):
        if isinstance(value, dict):
            return {key: without_audio(item) for key, item in value.items() if key != 'validation_input'}
        if isinstance(value, list):
            return [without_audio(item) for item in value]
        return value
    def save():
        report_path.write_text(json.dumps(without_audio(report), ensure_ascii=False, indent=2) + '\n')
    control = psutil.Process()
    monitored = [None]
    stop_monitor = Event()
    peak = [0]
    def monitor():
        while not stop_monitor.wait(.02):
            if monitored[0] is not None:
                try:
                    peak[0] = max(peak[0], monitored[0].memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
    thread = Thread(target=monitor)
    payloads = {n: wave_at_duration(config.validation_input['audio_base64'], n) for n in (1, 5, 15, 30)}
    counter = 0
    with tempfile.TemporaryDirectory(prefix='registry-', dir=root) as temporary, samples_path.open('w') as output:
        settings = json.loads(Path('examples/service.multimodal.json').read_text())
        settings.update(database=str(Path(temporary) / 'models.sqlite'),
                        business_keys=[secrets.token_hex(24)], admin_keys=[secrets.token_hex(24)])
        service = Service(Settings(**settings))
        await service.start()
        thread.start()
        async def infer(payload, cancel=None):
            nonlocal counter
            counter += 1
            values = [v async for v in service.infer('asr', config.model_id, payload,
                      cancel or asyncio.Event(), f'memory-{counter}')]
            assert len(values) == 1 and values[0]['done'] and not values[0]['mock']
            return values[0]['output']['text']
        def worker():
            return service.runtime.executor.snapshot()[config.instance_key]
        def assert_no_requests():
            state = service.scheduler.snapshot()
            assert state['active_count'] == state['queue_length'] == state['temporary_mb'] == 0, state
            assert worker()['busy'] == worker()['buffered_chunks'] == 0
        async def phase(name, durations, count):
            readings = []
            before = time.monotonic()
            for i in range(count):
                duration = durations[i % len(durations)]
                text = await infer(payloads[duration])
                state = worker()
                assert state['pid'] == report['initial_worker']['pid'] and state['load_count'] == 1
                assert_no_requests()
                item = {'phase': name, 'index': i, 'request': counter, 'audio_seconds': duration,
                        'worker_rss_mb': state['rss_bytes'] / 1048576,
                        'control_rss_mb': control.memory_info().rss / 1048576, 'text_chars': len(text)}
                readings.append(item)
                output.write(json.dumps(item) + '\n')
                if (i + 1) % 50 == 0:
                    output.flush()
                    print(name, i + 1, 'RSS MiB', round(item['worker_rss_mb'], 2), flush=True)
            summary = {'name': name, 'audio_durations_s': durations, 'seconds': time.monotonic() - before,
                       **summarize(readings), 'worker_memory': memory(monitored[0]),
                       'worker_completed': worker()['completed']}
            report['phases'].append(summary)
            save()
            print('PHASE', json.dumps(summary), flush=True)
        try:
            await service.add(config)
            state = worker()
            report['initial_worker'] = state
            monitored[0] = psutil.Process(state['pid'])
            report['control_before'] = memory(control)
            save()
            await phase('warmup', [1, 5, 15, 30], args.warmup)
            await phase('fixed_short', [1], args.short)
            await phase('fixed_maximum', [30], args.long)
            await phase('mixed_durations', [1, 5, 15, 30], args.mixed)
            await phase('short_after_maximum', [1], args.short)
            await phase('maximum_repeat', [30], args.long)
            for i in range(args.cancel):
                event = asyncio.Event()
                pending = asyncio.create_task(infer(payloads[30], event))
                deadline = time.monotonic() + 10
                while not worker()['busy']:
                    if pending.done() or time.monotonic() > deadline:
                        raise AssertionError('Inference did not reach the worker')
                    await asyncio.sleep(.002)
                await asyncio.sleep(.03)
                before = time.monotonic()
                event.set()
                try:
                    await pending
                except ServiceError as error:
                    assert error.code == 'cancelled', error.code
                else:
                    raise AssertionError('Expected cancellation during maximum-duration decode')
                assert_no_requests()
                report['cancellations'].append({'confirmed_after_s': time.monotonic() - before,
                    'rss_mb': worker()['rss_bytes'] / 1048576})
            report['after_cancellation'] = {'text': await infer(payloads[1]), 'same_pid': worker()['pid'] == state['pid']}
            assert report['after_cancellation']['same_pid']
            # Observe the actual configured idle interval, without accelerating it.
            report['before_idle'] = {'worker': worker(), 'memory': memory(monitored[0])}
            began = time.monotonic()
            print('Waiting for actual idle unload', config.idle_seconds, 'seconds', flush=True)
            while service.lifecycle.is_alive(config.instance_key) or service.scheduler.snapshot()['reserved_mb']:
                if time.monotonic() - began > config.idle_seconds + 15:
                    raise AssertionError('Idle worker was not unloaded')
                await asyncio.sleep(.2)
            assert not monitored[0].is_running()
            assert service.scheduler.snapshot()['reserved_mb'] == 0
            report['idle_unload'] = {'configured_seconds': config.idle_seconds, 'observed_seconds': time.monotonic() - began,
                                    'pid_exited': True, 'scheduler': service.scheduler.snapshot()}
            monitored[0] = None
            save()
            for i in range(args.cycles):
                await infer(payloads[30])
                state = worker()
                process = psutil.Process(state['pid'])
                monitored[0] = process
                await infer(payloads[1])
                before = memory(process)
                began = time.monotonic()
                release = await service.operate(config.model_id, 'unload')
                assert not process.is_running()
                assert service.scheduler.snapshot()['reserved_mb'] == 0
                report['unload_cycles'].append({'cycle': i, 'pid': state['pid'], 'memory_before': before,
                    'unload_seconds': time.monotonic() - began, 'process_exited': True, 'release': release})
                monitored[0] = None
                save()
            report['requests_including_cancelled'] = counter
            report['control_after'] = memory(control)
            report['scheduler_final'] = service.scheduler.snapshot()
            report['status'] = 'completed_real_memory_probe'
        except BaseException as error:
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            try:
                await service.close()
            except BaseException as error:
                report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
                raise
            finally:
                stop_monitor.set()
                thread.join()
                report['sampled_worker_peak_rss_mb'] = peak[0] / 1048576
                report['finished_unix'] = time.time()
                save()
    print('REPORT', str(report_path), report['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='docs/sensevoice-memory-validation.json')
    parser.add_argument('--samples', default='var/sensevoice-memory/samples.jsonl')
    parser.add_argument('--warmup', type=int, default=40)
    parser.add_argument('--short', type=int, default=300)
    parser.add_argument('--long', type=int, default=100)
    parser.add_argument('--mixed', type=int, default=300)
    parser.add_argument('--cancel', type=int, default=10)
    parser.add_argument('--cycles', type=int, default=5)
    args = parser.parse_args()
    if min(args.warmup, args.short, args.long, args.mixed) < 4 or args.cancel < 1 or args.cycles < 1:
        parser.error('Phase lengths must be >=4, cancellation/reload counts >=1')
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))
