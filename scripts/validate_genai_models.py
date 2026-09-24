"""Real GenAI output, incremental cancellation/reuse and peak-RSS verification."""
import argparse
import base64
from copy import deepcopy
import importlib.metadata
import io
import json
import multiprocessing as mp
from pathlib import Path
import resource
from threading import Event, enumerate as threads
import time
import traceback


def worker(raw, destination, capacity=False):
    from model_service.contracts import ModelConfig, ServiceError
    from model_service.backends.ov_genai import OpenVINOGenAIBackend
    from model_service.tasks.ov_genai import GenAITask
    import psutil
    cfg = ModelConfig.model_validate(raw)
    backend, task = OpenVINOGenAIBackend(cfg), GenAITask(cfg)
    report = {'model': cfg.model_id, 'backend': cfg.backend, 'timestamp_unix': time.time(),
              'versions': {name: importlib.metadata.version(name) for name in ('openvino', 'openvino-genai', 'openvino-tokenizers')},
              'scope': 'Real local target weights; direct task/backend in an isolated process. HTTP lifecycle is tested separately.'}
    try:
        report['source'] = json.loads((Path(cfg.path) / 'source.json').read_text())
        started = time.monotonic()
        report['load'] = backend.load()
        report['load_seconds'] = time.monotonic() - started
        report['rss_after_load_mb'] = psutil.Process().memory_info().rss / 1024**2
        original_pipeline_id = id(backend.pipeline)
        started = time.monotonic()
        prepared_contexts = []
        def infer(payload):
            task.validate(payload, cfg.capabilities[0])
            prepared = task.prepare(payload)
            output = task.finish(backend.infer(prepared.inputs, Event()), prepared.context)
            actual_tokens = output.get('usage', {}).get('prompt_tokens')
            evidence = {**prepared.context, 'native_input_tokens': actual_tokens}
            if actual_tokens is not None:
                evidence['bound_covers_native_tokens'] = actual_tokens <= prepared.context['total_input_token_bound']
            prepared_contexts.append(evidence)
            assert evidence.get('bound_covers_native_tokens', True), evidence
            return output
        report['output'] = infer(cfg.validation_input)
        report['inference_seconds'] = time.monotonic() - started
        report['rss_after_first_inference_mb'] = psutil.Process().memory_info().rss / 1024**2
        if cfg.task == 'ov_chat':
            arithmetic = {'messages': [{'role': 'user', 'content': '17加25等于多少？只回答数字。'}], 'max_new_tokens': 32}
            report['arithmetic'] = infer(arithmetic)
            report['quality_assertions'] = {'greeting': '你好' in report['output']['text'],
                                            'arithmetic': '42' in report['arithmetic']['text']}
            stream_payload = arithmetic
            long_payload = {'messages': [{'role': 'user', 'content': '请依次列出从一到二十的中文数字。'}], 'max_new_tokens': 64}
        else:
            from PIL import Image
            buffer = io.BytesIO()
            Image.new('RGB', (224, 224), 'blue').save(buffer, format='PNG')
            blue_payload = deepcopy(cfg.validation_input)
            blue_payload['images'] = [base64.b64encode(buffer.getvalue()).decode('ascii')]
            report['blue_square'] = infer(blue_payload)
            report['quality_assertions'] = {
                'red_square': '红' in report['output']['text'] or 'red' in report['output']['text'].lower(),
                'blue_square': '蓝' in report['blue_square']['text'] or 'blue' in report['blue_square']['text'].lower(),
                'image_changes_answer': report['output']['text'] != report['blue_square']['text'],
            }
            stream_payload = blue_payload
            long_payload = deepcopy(blue_payload)
            long_payload.update(prompt='请详细描述这张图片的颜色，并列出十种同色的常见物品。', max_new_tokens=64)
        task.validate(stream_payload, cfg.capabilities[0])
        prepared = task.prepare(stream_payload)
        chunks = [task.finish_chunk(chunk, prepared.context) for chunk in backend.stream(prepared.inputs, Event())]
        text = ''.join(chunk['delta'] for chunk in chunks)
        report['stream'] = {'chunks': len(chunks), 'text': text,
                            'matches_final': text == chunks[-1]['text']}
        task.validate(long_payload, cfg.capabilities[0])
        cancel = Event()
        generation = backend.stream(task.prepare(long_payload).inputs, cancel)
        try:
            first = next(generation)
            started = time.monotonic()
            cancel.set()
        finally:
            generation.close()  # Waits for the native producer thread to stop.
        native_threads_stopped = not any(thread.name == 'openvino-generation' for thread in threads())
        report['cancel'] = {'first_delta': first['delta'], 'stop_seconds': time.monotonic() - started,
                            'native_producer_stopped': native_threads_stopped}
        report['after_cancel'] = infer(cfg.validation_input)
        report['reuse'] = {'same_pipeline': id(backend.pipeline) == original_pipeline_id,
                           'responded_after_cancel': bool(report['after_cancel']['text'])}
        if capacity:
            bounded_payload = deepcopy(cfg.validation_input)
            bounded_payload['max_new_tokens'] = cfg.options.get('max_new_tokens', 128)
            if cfg.task == 'ov_vision_chat':
                from PIL import Image
                buffer = io.BytesIO()
                edge = cfg.options.get('image_edge', 448)
                Image.new('RGB', (edge, edge), 'blue').save(buffer, format='PNG')
                bounded_payload['images'] = [base64.b64encode(buffer.getvalue()).decode('ascii')]
            sentence = '这是一段用于容量测试的输入。'
            instruction = '\n请依次列出从1到500的整数，持续输出，不要提前停止。'
            low, high = 0, cfg.options.get('max_text_chars', 4096) // len(sentence)
            while low < high:
                middle = (low + high + 1) // 2
                prompt = sentence * middle + instruction
                if cfg.task == 'ov_chat':
                    bounded_payload['messages'] = [{'role': 'user', 'content': prompt}]
                else:
                    bounded_payload['prompt'] = prompt
                try:
                    task.validate(bounded_payload, cfg.capabilities[0])
                    task.prepare(bounded_payload)
                except ServiceError as error:
                    if error.code != 'input_too_large':
                        raise
                    high = middle - 1
                else:
                    low = middle
            prompt = sentence * low + instruction
            if cfg.task == 'ov_chat':
                bounded_payload['messages'] = [{'role': 'user', 'content': prompt}]
            else:
                bounded_payload['prompt'] = prompt
            started = time.monotonic()
            capacity_output = infer(bounded_payload)
            report['capacity'] = {'input_bound': prepared_contexts[-1],
                                  'image_edge': cfg.options.get('image_edge') if cfg.task == 'ov_vision_chat' else None,
                                  'usage': capacity_output.get('usage'),
                                  'finish_reason': capacity_output.get('finish_reason'),
                                  'seconds': time.monotonic() - started,
                                  'rss_after_capacity_mb': psutil.Process().memory_info().rss / 1024**2,
                                  'text_preview': capacity_output['text'][:160]}
            assert prepared_contexts[-1]['total_input_token_bound'] >= cfg.options.get('max_input_tokens', 1024) - 32
        report['input_token_bounds'] = prepared_contexts
        report['rss_after_reuse_mb'] = psutil.Process().memory_info().rss / 1024**2
        assert report['stream']['matches_final'] and report['stream']['chunks'] > 1
        assert native_threads_stopped and all(report['reuse'].values())
        assert all(report['quality_assertions'].values()), report['quality_assertions']
        report['status'] = 'passed_real_model'
    except BaseException as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        report['peak_rss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['release'] = backend.close()
        Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*')
    parser.add_argument('--capacity', action='store_true', help='Also exercise near-limit input and configured generation/image bounds')
    args = parser.parse_args()
    report_path = Path('docs/genai-validation.json')
    reports = json.loads(report_path.read_text()) if report_path.exists() else {}
    for raw in json.loads(Path('examples/models.genai.json').read_text()):
        if args.models and raw['name'] not in args.models:
            continue
        destination = Path('var') / (raw['name'] + '-validation.json')
        destination.parent.mkdir(exist_ok=True)
        destination.unlink(missing_ok=True)
        process = mp.get_context('spawn').Process(target=worker, args=(raw, str(destination), args.capacity))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise SystemExit(f'Model worker exited unexpectedly: {process.exitcode}')
        report = json.loads(destination.read_text())
        reports[raw['name']] = report
        report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
        print(raw['name'], report['status'], report.get('output'), flush=True)
        if report['status'] != 'passed_real_model':
            raise SystemExit(report.get('error'))


if __name__ == '__main__':
    main()
