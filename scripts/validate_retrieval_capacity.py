"""Measure real retrieval bundles at configured input bounds in isolated processes.

Stop the service and other native model tests first. This is a capacity sample,
not a throughput benchmark, hard memory guarantee, or quality evaluation.
"""
import argparse
import base64
import gc
import io
import json
import math
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
import time


def probe(config):
    import psutil
    from model_service.backends import create_backend
    from model_service.tasks import create_task
    batch = config.options.get('max_batch_size', 8)
    length = config.options.get('max_length', 512)
    long_text = '北京知识图谱 graph search science. ' * length
    payload = ({'query': '知识图谱是什么？', 'documents': [long_text] * batch}
               if config.capabilities == ['rerank'] else {'texts': [long_text] * batch})
    if config.task == 'dual_clip':
        from PIL import Image
        edge = math.isqrt(config.options.get('max_image_pixels', 16777216))
        with Image.new('RGB', (edge, edge), color=(64, 128, 192)) as picture:
            buffer = io.BytesIO()
            picture.save(buffer, format='PNG')
        payload['images'] = [base64.b64encode(buffer.getvalue()).decode()] * batch
        buffer.close()
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode())
    assert payload_bytes <= config.max_input_bytes, (payload_bytes, config.max_input_bytes)
    task, backend = create_task(config), create_backend(config)
    for capability in config.capabilities:
        task.validate(payload, capability)
    process = psutil.Process()
    stopped = Event()
    peaks = [process.memory_info().rss]
    def sample():
        while not stopped.wait(.01):
            peaks[0] = max(peaks[0], process.memory_info().rss)
    monitor = Thread(target=sample)
    monitor.start()
    record = {'model_id': config.model_id, 'timestamp_unix': time.time(),
              'batch': batch, 'max_length': length, 'payload_bytes': payload_bytes,
              'resident_budget_mb': config.resident_mb, 'request_budget_mb': config.request_mb,
              'runs': []}
    try:
        started = time.monotonic()
        backend.load()
        record['load_seconds'] = time.monotonic() - started
        for index in range(2):
            started = time.monotonic()
            prepared = task.prepare(payload)
            tensor_inputs = prepared.inputs.get('text', prepared.inputs)
            ids = tensor_inputs['input_ids']
            assert tuple(ids.shape) == (batch, length), tuple(ids.shape)
            record['input_shape'] = list(ids.shape)
            if 'image' in prepared.inputs:
                record['decoded_image_pixels_each'] = edge * edge
                record['image_tensor_shape'] = list(prepared.inputs['image']['pixel_values'].shape)
            result = backend.infer(prepared.inputs, Event())
            task.finish(result, prepared.context)
            del result, prepared, tensor_inputs, ids
            gc.collect()
            record['runs'].append({'index': index, 'seconds': time.monotonic() - started,
                                   'steady_rss_mb': process.memory_info().rss / 1048576})
        record['peak_rss_mb'] = peaks[0] / 1048576
        record['resident_budget_covers_observed'] = max(r['steady_rss_mb'] for r in record['runs']) <= config.resident_mb
        record['total_budget_covers_observed'] = record['peak_rss_mb'] <= config.resident_mb + config.request_mb
        record['status'] = ('passed_real_capacity_sample' if record['resident_budget_covers_observed']
                            and record['total_budget_covers_observed'] else 'budget_needs_calibration')
    finally:
        try:
            backend.close()
        finally:
            stopped.set()
            monitor.join()
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Exact configured name; default all seven')
    parser.add_argument('--child', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from model_service.contracts import ModelConfig
    configs = [ModelConfig(**row) for row in json.loads(Path('examples/models.retrieval-migrated.json').read_text())]
    if args.child:
        probe(next(config for config in configs if config.name == args.child))
        return
    path = Path('docs/retrieval-capacity.json')
    report = json.loads(path.read_text()) if path.exists() else {'results': {}}
    report['current_model_ids'] = [config.model_id for config in configs]
    report.update(scope='Real configured maximum token/batch input and CLIP maximum decoded image pixels; two sequential requests per fresh process. No concurrent models or throughput/quality benchmark.')
    for config in configs:
        if args.model and config.name not in args.model:
            continue
        result = subprocess.run([sys.executable, __file__, '--child', config.name], capture_output=True, text=True)
        if result.returncode:
            report['results'][config.model_id] = {'status': 'failed', 'stderr': result.stderr[-6000:]}
        else:
            report['results'][config.model_id] = json.loads(result.stdout.splitlines()[-1])
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(config.model_id, report['results'][config.model_id], flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
