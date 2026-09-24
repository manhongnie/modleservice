"""Local OpenVINO GenAI pipeline; bounded streaming and confirmed cancellation."""
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Thread

from model_service.contracts import ServiceError


def validate_genai(config, settings=None):
    if config.task not in {'ov_chat', 'ov_vision_chat'} or config.concurrency != 1:
        raise ServiceError('invalid_config', 'GenAI generation requires a supported task and concurrency=1', 422)
    if config.device != 'CPU':
        raise ServiceError('unsupported_device', 'This deployment validates GenAI on CPU only', 422)
    pipeline = config.options.get('pipeline', 'vlm')
    if not isinstance(pipeline, str) or pipeline not in {'llm', 'vlm'}:
        raise ServiceError('invalid_config', 'GenAI pipeline must be llm or vlm', 422)
    if config.task == 'ov_vision_chat' and config.options.get('pipeline', 'vlm') != 'vlm':
        raise ServiceError('invalid_config', 'Vision chat requires a VLM pipeline', 422)
    bounds = {'threads': (1, 24), 'max_input_tokens': (1, 4096), 'max_new_tokens': (1, 512),
              'max_images': (1, 2), 'image_edge': (28, 672), 'max_image_pixels': (1, 16777216),
              'max_text_chars': (1, 32768)}
    for name, (low, high) in bounds.items():
        value = config.options.get(name)
        if value is not None and (type(value) is not int or not low <= value <= high):
            raise ServiceError('invalid_config', f'{name} must be an integer between {low} and {high}', 422)
    if set(config.options) - set(bounds) - {'pipeline'}:
        raise ServiceError('invalid_config', 'Unknown GenAI options; arbitrary compile/cache paths are not accepted', 422)


class OpenVINOGenAIBackend:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.genai = None

    def load(self):
        try:
            import openvino_genai as genai
        except ImportError as error:
            raise ServiceError('missing_dependency', 'Install matching openvino-genai and openvino runtime', 503) from error
        root = Path(self.config.path).resolve()
        if not root.is_dir():
            raise ServiceError('model_files_missing', 'GenAI requires a complete exported local bundle', 422)
        for entry in root.rglob('*'):
            if entry.is_symlink() and not entry.resolve().is_relative_to(root):
                raise ServiceError('path_forbidden', 'GenAI artifacts must remain inside their model bundle', 403)
        validate_genai(self.config)
        self.genai = genai
        kind = self.config.options.get('pipeline', 'vlm')
        factory = genai.VLMPipeline if kind == 'vlm' else genai.LLMPipeline
        self.pipeline = factory(str(root), self.config.device,
            INFERENCE_NUM_THREADS=int(self.config.options.get('threads', 4)),
            NUM_STREAMS=1, PERFORMANCE_HINT='LATENCY')
        return {'backend': 'openvino_genai', 'device': self.config.device,
                'runtime_version': genai.__version__, 'pipeline': kind, 'streaming': 'incremental'}

    def _generate(self, inputs, cancel, callback):
        if cancel.is_set():
            raise ServiceError('cancelled', 'Cancelled before generation', 499)
        gen = self.pipeline.get_generation_config()
        gen.max_new_tokens = inputs['max_new_tokens']
        gen.do_sample = False
        gen.apply_chat_template = False  # The task already formats complete messages.
        kwargs = dict(generation_config=gen, streamer=callback)
        if self.config.options.get('pipeline', 'vlm') == 'vlm':
            import openvino as ov
            kwargs['images'] = [ov.Tensor(image) for image in inputs.get('images', [])]
        result = self.pipeline.generate(inputs['prompt'], **kwargs)
        if cancel.is_set():
            raise ServiceError('cancelled', 'Native GenAI generation has stopped', 499)
        if isinstance(result, str):
            text, usage = result, {}
        else:
            text = result.texts[0]
            metrics = result.perf_metrics
            usage = {'prompt_tokens': metrics.get_num_input_tokens(),
                     'completion_tokens': metrics.get_num_generated_tokens()}
        return {'text': text, 'usage': usage,
                'finish_reason': 'length' if usage.get('completion_tokens', 0) >= inputs['max_new_tokens'] else 'stop'}

    def infer(self, inputs, cancel):
        return self._generate(inputs, cancel, lambda _: self.genai.StreamingStatus.CANCEL if cancel.is_set()
                              else self.genai.StreamingStatus.RUNNING)

    def stream(self, inputs, cancel):
        queue = Queue(maxsize=8)
        stopped = Event()
        result, errors = [], []

        def callback(delta):
            if delta:
                while not cancel.is_set():
                    try:
                        queue.put(delta, timeout=.05)
                        break
                    except Full:
                        continue
            return self.genai.StreamingStatus.CANCEL if cancel.is_set() else self.genai.StreamingStatus.RUNNING

        def generate():
            try:
                result.append(self._generate(inputs, cancel, callback))
            except BaseException as error:
                errors.append(error)
            finally:
                stopped.set()

        producer = Thread(target=generate, name='openvino-generation', daemon=False)
        producer.start()
        complete = False
        emitted = ''
        try:
            while not stopped.is_set() or not queue.empty():
                if cancel.is_set():
                    raise ServiceError('cancelled', 'GenAI cancellation requested', 499)
                try:
                    delta = queue.get(timeout=.05)
                except Empty:
                    continue
                emitted += delta
                yield {'delta': delta, 'streaming_mode': 'incremental', 'finish_reason': None}
            producer.join()
            if errors:
                raise errors[0]
            if cancel.is_set():
                raise ServiceError('cancelled', 'Native GenAI generation has stopped', 499)
            final = result[0]
            if not final['text'].startswith(emitted):
                raise ServiceError('invalid_model_output', 'Generated text does not match emitted stream', 502)
            complete = True
            yield {**final, 'delta': final['text'][len(emitted):], 'streaming_mode': 'incremental'}
        finally:
            if not complete:
                cancel.set()
            producer.join()  # No worker done acknowledgement before native termination.

    def close(self):
        self.pipeline = None
        self.genai = None
        import gc
        gc.collect()
        return {'released': True, 'management': 'local'}
