"""Optional CPU Transformers backend. Models and imports live only in workers."""
from model_service.contracts import ServiceError


class TransformersBackend:
    def __init__(self, config):
        self.config, self.model = config, None

    def load(self):
        try:
            import torch
            from transformers import Qwen3_5ForConditionalGeneration, WhisperForConditionalGeneration
        except ImportError as error:
            raise ServiceError("missing_dependency", "Install generative extras: CPU torch and Transformers >= 5.2", 503) from error
        if self.config.device.upper() != "CPU":
            raise ServiceError("unsupported_device", "Initial Transformers backend supports CPU only")
        torch.set_num_threads(int(self.config.options.get("threads", 4)))
        if self.config.task == "qwen_chat":
            # Official checkpoint includes vision weights; API accepts text only.
            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                self.config.path, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="eager").eval()
        elif self.config.task == "whisper_asr":
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.path, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="eager").eval()
        else:
            raise ServiceError("unsupported_task", "Transformers backend supports qwen_chat and whisper_asr")
        return {"backend": "transformers", "device": "CPU", "streaming": "incremental" if self.config.task == "qwen_chat" else "buffered"}

    def infer(self, inputs, cancel):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        if cancel.is_set():
            raise ServiceError("cancelled", "Request cancelled", 499)

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set()

        inputs = dict(inputs)
        maximum = inputs.pop("max_new_tokens", int(self.config.options.get("max_new_tokens", 128)))
        kwargs = {"max_new_tokens": maximum, "do_sample": False, "stopping_criteria": StoppingCriteriaList([Cancelled()])}
        if self.config.task == "whisper_asr":
            kwargs.update(task="transcribe", language=self.config.options.get("language", "zh"))
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        if cancel.is_set():
            raise ServiceError("cancelled", "Generation stopped after cancellation", 499)
        if self.config.task == "qwen_chat":
            return {"token_ids": generated[0, inputs["input_ids"].shape[-1]:].tolist()}
        return {"token_ids": generated.tolist()}

    def stream(self, inputs, cancel):
        """Generate token chunks with a bounded producer and confirmed stop on close.

        Only token IDs cross this boundary. ChatTask owns prompt construction and
        Unicode decoding. ASR deliberately uses the buffered infer contract.
        """
        import torch
        from queue import Empty, Full, Queue
        from threading import Event, Thread
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer

        if self.config.task != "qwen_chat":
            raise ServiceError("streaming_unsupported", "This task only supports buffered output")
        if cancel.is_set():
            raise ServiceError("cancelled", "Request cancelled", 499)
        output = Queue(maxsize=8)
        stopped = Event()
        failure = []

        class TokenStreamer(BaseStreamer):
            first = True

            def put(self, value):
                if self.first:
                    self.first = False  # generate() sends the complete prompt first.
                    return
                tokens = value.reshape(-1).tolist()
                while not cancel.is_set():
                    try:
                        output.put(tokens, timeout=0.05)
                        return
                    except Full:
                        continue
                raise ServiceError("cancelled", "Generation stopped after cancellation", 499)

            def end(self):
                pass  # Returning from generate(), not end(), confirms native stop.

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel.is_set()

        kwargs = dict(inputs)
        maximum = kwargs.pop("max_new_tokens", int(self.config.options.get("max_new_tokens", 128)))

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs, max_new_tokens=maximum, do_sample=False,
                        streamer=TokenStreamer(), stopping_criteria=StoppingCriteriaList([Cancelled()]))
            except BaseException as error:
                failure.append(error)
            finally:
                stopped.set()

        thread = Thread(target=generate, name="token-generation", daemon=False)
        thread.start()
        ids = []
        completed = False
        try:
            while not stopped.is_set() or not output.empty():
                if cancel.is_set():
                    raise ServiceError("cancelled", "Generation stopped after cancellation", 499)
                try:
                    ids.extend(output.get(timeout=0.05))
                except Empty:
                    continue
                yield {"token_ids": list(ids), "final": False}
            thread.join()
            if failure:
                raise failure[0]
            if cancel.is_set():
                raise ServiceError("cancelled", "Generation stopped after cancellation", 499)
            completed = True
            yield {"token_ids": list(ids), "final": True, "finish_reason": "length" if len(ids) >= maximum else "stop"}
        finally:
            if not completed:
                cancel.set()
            # Cooperative cancellation can wait for a native kernel. No done
            # acknowledgement or budget release is safe before this join returns.
            thread.join()

    def close(self):
        self.model = None
        import gc
        gc.collect()
        return {"released": True}
