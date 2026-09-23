"""Business payload validation, HTTP boundaries and explicitly fake test tasks."""
from __future__ import annotations

import base64
import hashlib
import io
import math
import wave
from typing import Any

from model_service.contracts import ModelConfig, Prepared, ServiceError


CAPABILITIES = {"embeddings", "rerank", "image_embeddings", "sparse_embeddings", "chat", "asr", "tts"}


def bad(message: str) -> None:
    raise ServiceError("invalid_input", message, 422)


def texts(payload: dict) -> list[str]:
    value = payload.get("texts", payload.get("input"))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or len(value) > 128 or any(not isinstance(x, str) or not x.strip() for x in value):
        bad("texts/input must contain 1..128 nonempty strings")
    return value


def validate_payload(config: ModelConfig, payload: dict, capability: str) -> None:
    if not isinstance(payload, dict):
        bad("payload must be an object")
    if capability not in CAPABILITIES:
        raise ServiceError("unsupported_capability", f"Unsupported capability: {capability}")
    # Business callers cannot choose files, model configuration or destinations.
    forbidden = {"path", "model_path", "backend", "device", "url", "endpoint", "options"}
    if forbidden.intersection(payload):
        bad("Business input must not contain file paths, destinations or backend configuration")
    if capability in {"embeddings", "sparse_embeddings"}:
        texts(payload)
    elif capability == "rerank":
        if not isinstance(payload.get("query"), str) or not payload["query"].strip():
            bad("query must be a nonempty string")
        texts({"texts": payload.get("documents")})
    elif capability == "image_embeddings":
        images = payload.get("images")
        if not isinstance(images, list) or not images or len(images) > 16:
            bad("images must contain 1..16 base64 encoded images")
        for image in images:
            decode_base64(image, "image")
    elif capability == "chat":
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) > 64:
            bad("messages must contain 1..64 messages")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
                bad("Only system/user/assistant text messages are supported")
        maximum = int(config.options.get("max_new_tokens", 256))
        tokens = payload.get("max_new_tokens", maximum)
        if type(tokens) is not int or not 1 <= tokens <= maximum:
            bad(f"max_new_tokens must be between 1 and {maximum}")
    elif capability == "asr":
        audio = decode_base64(payload.get("audio_base64"), "audio_base64")
        # Parse bounded WAV headers in control plane, without importing numpy/torch.
        try:
            with wave.open(io.BytesIO(audio)) as source:
                if source.getcomptype() != "NONE" or source.getsampwidth() != 2 or source.getnchannels() != 1:
                    bad("ASR accepts mono PCM16 WAV only")
                if source.getframerate() != 16000:
                    bad("ASR accepts 16000 Hz WAV only")
                if not 0 < source.getnframes() <= int(config.options.get("max_audio_seconds", 30)) * 16000:
                    bad("Audio duration is empty or exceeds model limit")
                expected = source.getnframes() * 2
                if len(source.readframes(source.getnframes())) != expected:
                    bad("WAV audio is truncated")
        except (wave.Error, EOFError) as error:
            bad(f"Invalid PCM WAV: {error}")
    elif capability == "tts":
        if not isinstance(payload.get("text"), str) or not payload["text"].strip() or len(payload["text"]) > int(config.options.get("max_text_chars", 300)):
            bad("text must be nonempty and within the configured character limit")
        speaker = payload.get("speaker_id", 0)
        speed = payload.get("speed", 1.0)
        if type(speaker) is not int or speaker < 0 or speaker >= int(config.options.get("speakers", 174)):
            bad("speaker_id is out of range")
        if type(speed) not in {int, float} or not math.isfinite(speed) or not 0.5 <= speed <= 2:
            bad("speed must be between 0.5 and 2")


def decode_base64(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        bad(f"{name} must be nonempty base64")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        bad(f"{name} is not valid base64")


class PassthroughTask:
    def __init__(self, config: ModelConfig):
        self.config = config

    def validate(self, payload: dict, capability: str) -> None:
        validate_payload(self.config, payload, capability)

    def prepare(self, payload: dict) -> Prepared:
        return Prepared(dict(payload), {"payload": payload})

    def finish(self, outputs: dict, context: dict) -> dict:
        capability = self.config.capabilities[0]
        payload = context.get("payload", {})

        def invalid():
            raise ServiceError("invalid_model_output", f"HTTP response does not satisfy the {capability} contract", 502)

        def number(value):
            return type(value) in {int, float} and math.isfinite(value)

        if not isinstance(outputs, dict):
            invalid()
        if capability in {"embeddings", "image_embeddings"}:
            vectors = outputs.get("embeddings")
            expected = len(payload["images"]) if capability == "image_embeddings" else len(texts(payload))
            if not isinstance(vectors, list) or len(vectors) != expected or any(
                not isinstance(v, list) or not v or any(not number(n) for n in v) for v in vectors
            ) or len({len(v) for v in vectors}) != 1:
                invalid()
        elif capability == "sparse_embeddings":
            vectors = outputs.get("sparse_embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(texts(payload)):
                invalid()
            for vector in vectors:
                if not isinstance(vector, dict):
                    invalid()
                indices, values = vector.get("indices"), vector.get("values")
                if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
                    invalid()
                if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices) or any(not number(v) for v in values):
                    invalid()
        elif capability == "rerank":
            results = outputs.get("results")
            if not isinstance(results, list) or len(results) != len(payload["documents"]):
                invalid()
            seen = set()
            for row in results:
                if not isinstance(row, dict) or type(row.get("index")) is not int or not 0 <= row["index"] < len(payload["documents"]) or not number(row.get("score")):
                    invalid()
                seen.add(row["index"])
            if len(seen) != len(results):
                invalid()
        elif capability in {"chat", "asr"}:
            if not isinstance(outputs.get("text"), str):
                invalid()
        elif capability == "tts":
            if outputs.get("format") != "wav" or type(outputs.get("sample_rate")) is not int or not 8000 <= outputs["sample_rate"] <= 192000:
                invalid()
            try:
                audio = decode_base64(outputs.get("audio_base64"), "audio_base64")
                with wave.open(io.BytesIO(audio)) as source:
                    if source.getframerate() != outputs["sample_rate"] or source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getnframes() <= 0:
                        invalid()
            except (ServiceError, wave.Error, EOFError):
                invalid()
        return outputs


class ChatTask(PassthroughTask):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None

    def validate(self, payload: dict, capability: str) -> None:
        if capability != "chat":
            raise ServiceError("unsupported_capability", "qwen_chat supports chat only")
        super().validate(payload, capability)

    def prepare(self, payload):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.path, local_files_only=True, trust_remote_code=False)
        prompt = self.tokenizer.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = dict(self.tokenizer(prompt, return_tensors="pt"))
        if inputs["input_ids"].shape[-1] > int(self.config.options.get("max_input_tokens", 1024)):
            bad("Prompt exceeds configured token limit")
        inputs["max_new_tokens"] = payload.get("max_new_tokens", int(self.config.options.get("max_new_tokens", 256)))
        return Prepared(inputs, {"prompt_tokens": inputs["input_ids"].shape[-1]})

    def finish(self, outputs, context):
        ids = outputs["token_ids"]
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True), "usage": {"prompt_tokens": context["prompt_tokens"], "completion_tokens": len(ids)}}


    def finish_chunk(self, outputs, context):
        ids = outputs["token_ids"]
        final = bool(outputs.get("final"))
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Byte-fallback tokenizers can decode a partial UTF-8 sequence as U+FFFD.
        # Keep the incomplete suffix until another token completes it.
        safe = decoded if final else decoded.split("\ufffd", 1)[0]
        emitted = context.get("emitted_text", "")
        if not safe.startswith(emitted):
            raise ServiceError("invalid_model_output", "Incremental tokenizer output changed an emitted prefix", 502)
        delta = safe[len(emitted):]
        context["emitted_text"] = safe
        if not delta and not final:
            return None
        result = {"delta": delta, "streaming_mode": "incremental", "finish_reason": outputs.get("finish_reason", "stop") if final else None}
        if final:
            result["text"] = decoded
            result["usage"] = {"prompt_tokens": context["prompt_tokens"], "completion_tokens": len(ids)}
        return result


class SpeechTask(PassthroughTask):
    def __init__(self, config):
        super().__init__(config)
        self.processor = None

    def validate(self, payload: dict, capability: str) -> None:
        expected = "asr" if self.config.task == "whisper_asr" else "tts"
        if capability != expected:
            raise ServiceError("unsupported_capability", f"{self.config.task} supports {expected} only")
        super().validate(payload, capability)

    def prepare(self, payload):
        if self.config.task == "vits_tts":
            return Prepared({"text": payload["text"], "speaker_id": payload.get("speaker_id", 0), "speed": payload.get("speed", 1.0)}, {})
        import numpy as np
        from transformers import WhisperProcessor
        if self.processor is None:
            self.processor = WhisperProcessor.from_pretrained(self.config.path, local_files_only=True)
        with wave.open(io.BytesIO(decode_base64(payload["audio_base64"], "audio_base64"))) as source:
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        encoded = self.processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        return Prepared(dict(encoded), {})

    def finish(self, outputs, context):
        if self.config.task == "whisper_asr":
            return {"text": self.processor.batch_decode(outputs["token_ids"], skip_special_tokens=True)[0]}
        import numpy as np
        samples = np.asarray(outputs["samples"], dtype=np.float32)
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ServiceError("invalid_model_output", "TTS produced empty/nonfinite audio")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setparams((1, 2, outputs["sample_rate"], 0, "NONE", "not compressed"))
            output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        return {"audio_base64": base64.b64encode(buffer.getvalue()).decode(), "sample_rate": outputs["sample_rate"], "format": "wav", "duration_s": len(samples) / outputs["sample_rate"]}


class MockTask(PassthroughTask):
    """Deterministic fake results. Never named after or advertised as a real model."""
    def finish_chunk(self, outputs: dict, context: dict) -> dict:
        index = outputs["chunk_index"]
        if "chunk" in outputs:
            chunk = outputs["chunk"]
            return chunk if isinstance(chunk, dict) else {"mock": True, "delta": chunk, "chunk_index": index}
        return {**self.finish(outputs, context), "chunk_index": index}

    def finish(self, outputs: dict, context: dict) -> dict:
        payload = outputs.get("payload", outputs)
        capability = self.config.capabilities[0]
        result: dict[str, Any] = {"mock": True, "notice": "Synthetic test output; no real model inference"}
        if capability in {"embeddings", "image_embeddings"}:
            values = payload["images"] if capability == "image_embeddings" else texts(payload)
            vectors = []
            for value in values:
                raw = hashlib.sha256(value.encode()).digest()[:8]
                vec = [(n - 127.5) / 127.5 for n in raw]
                norm = math.sqrt(sum(x * x for x in vec))
                vectors.append([x / norm for x in vec])
            result["embeddings"] = vectors
        elif capability == "rerank":
            result["results"] = [{"index": i, "score": 1 / (i + 1)} for i in range(len(payload["documents"]))]
        elif capability == "sparse_embeddings":
            result["sparse_embeddings"] = [{"indices": [0], "values": [1.0]} for _ in texts(payload)]
        elif capability == "chat":
            result["text"] = "MOCK: " + payload["messages"][-1]["content"]
        elif capability == "asr":
            result["text"] = "MOCK transcription"
        elif capability == "tts":
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\x00\x00" * 1600)
            result.update(audio_base64=base64.b64encode(buffer.getvalue()).decode(), sample_rate=16000, format="wav")
        return result
