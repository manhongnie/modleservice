"""Bounded speech payloads and JSON output; no native runtime in control process."""
from __future__ import annotations

import base64
import io
import math
import wave

from model_service.contracts import ModelConfig, Prepared, ServiceError
from .core import bad, decode_base64


TASK_CAPABILITIES = {
    "sherpa_sensevoice": "asr", "sherpa_matcha": "tts",
    "sherpa_speaker": "speaker_embeddings", "sherpa_zipvoice": "voice_clone",
}


def read_pcm(value, field, maximum_seconds=30, minimum_seconds=0.1):
    # Avoid decoding an oversized allocation even when called outside the HTTP API.
    if not isinstance(value, str) or len(value) > math.ceil(maximum_seconds * 48000 * 2 * 4 / 3) + 16384:
        bad(f"{field} exceeds the audio limit")
    raw = decode_base64(value, field)
    try:
        with wave.open(io.BytesIO(raw)) as audio:
            rate, frames = audio.getframerate(), audio.getnframes()
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getcomptype() != "NONE":
                bad(f"{field} must be mono PCM16 WAV")
            if not 8000 <= rate <= 48000 or not minimum_seconds <= frames / rate <= maximum_seconds:
                bad(f"{field} must contain {minimum_seconds}..{maximum_seconds} seconds at 8000..48000 Hz")
            samples = audio.readframes(frames)
            if len(samples) != frames * 2:
                bad(f"{field} WAV is truncated")
            return rate, samples
    except (wave.Error, EOFError, ValueError) as error:
        bad(f"{field} is not a valid PCM WAV: {error}")


def _waveform(value, field, maximum_seconds=30, minimum_seconds=0.1):
    import numpy as np
    rate, pcm = read_pcm(value, field, maximum_seconds, minimum_seconds)
    return rate, np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def encode_audio(outputs):
    import numpy as np
    samples = np.asarray(outputs.get("samples"), dtype=np.float32)
    rate = outputs.get("sample_rate")
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all() or type(rate) is not int or not 8000 <= rate <= 48000:
        raise ServiceError("invalid_model_output", "Speech model returned invalid audio", 502)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        audio.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return {"audio_base64": base64.b64encode(buffer.getvalue()).decode(), "format": "wav", "sample_rate": rate,
            "duration_s": samples.size / rate, "streaming_mode": "buffered"}


class SherpaSpeechTask:
    def __init__(self, config: ModelConfig):
        self.config = config
        if config.task not in TASK_CAPABILITIES or config.capabilities != [TASK_CAPABILITIES[config.task]]:
            raise ServiceError("unsupported_capability", "Sherpa task requires its single declared capability")

    def validate(self, payload, capability):
        if capability != TASK_CAPABILITIES[self.config.task]:
            raise ServiceError("unsupported_capability", "Capability does not match the speech task")
        allowed = {
            "asr": {"audio_base64"}, "tts": {"text", "speed", "speaker_id"},
            "speaker_embeddings": {"audio_base64", "compare_audio_base64"},
            "voice_clone": {"text", "reference_audio_base64", "reference_text", "speed", "num_steps"},
        }[capability]
        if not isinstance(payload, dict) or set(payload) - allowed:
            bad("Speech input contains unsupported fields; model paths and destinations are not accepted")
        if capability in {"asr", "speaker_embeddings"}:
            minimum = 0.5 if capability == "speaker_embeddings" else 0.1
            maximum = self.config.options.get("max_audio_seconds", 30)
            read_pcm(payload.get("audio_base64"), "audio_base64", maximum, minimum)
            if "compare_audio_base64" in payload:
                read_pcm(payload["compare_audio_base64"], "compare_audio_base64", maximum, minimum)
        if capability in {"tts", "voice_clone"}:
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > self.config.options.get("max_text_chars", 200):
                bad("text must be nonempty and within the configured character limit")
            speed = payload.get("speed", 1.0)
            if type(speed) not in {int, float} or not math.isfinite(speed) or not 0.5 <= speed <= 2:
                bad("speed must be between 0.5 and 2")
        if capability == "tts" and (type(payload.get("speaker_id", 0)) is not int or payload.get("speaker_id", 0) != 0):
            bad("This Matcha model has one voice; speaker_id must be 0")
        if capability == "voice_clone":
            reference = payload.get("reference_text")
            if not isinstance(reference, str) or not reference.strip() or len(reference) > 300:
                bad("reference_text must contain the transcript of the reference audio, up to 300 characters")
            read_pcm(payload.get("reference_audio_base64"), "reference_audio_base64", self.config.options.get("max_reference_seconds", 15), 1.0)
            steps = payload.get("num_steps", 4)
            if type(steps) is not int or not 1 <= steps <= self.config.options.get("max_steps", 8):
                bad("num_steps exceeds the configured diffusion step limit")

    def prepare(self, payload):
        capability = TASK_CAPABILITIES[self.config.task]
        # Repeat bounded validation in the worker for callers using Executor directly.
        self.validate(payload, capability)
        if capability in {"asr", "speaker_embeddings"}:
            minimum = 0.5 if capability == "speaker_embeddings" else 0.1
            maximum = self.config.options.get("max_audio_seconds", 30)
            rate, samples = _waveform(payload["audio_base64"], "audio_base64", maximum, minimum)
            inputs = {"sample_rate": rate, "samples": samples}
            if "compare_audio_base64" in payload:
                other_rate, other = _waveform(payload["compare_audio_base64"], "compare_audio_base64", maximum, minimum)
                inputs.update(compare_sample_rate=other_rate, compare_samples=other)
            return Prepared(inputs, {})
        inputs = {"text": payload["text"], "speed": payload.get("speed", 1.0)}
        if capability == "voice_clone":
            rate, samples = _waveform(payload["reference_audio_base64"], "reference_audio_base64", self.config.options.get("max_reference_seconds", 15), 1.0)
            inputs.update(reference_sample_rate=rate, reference_samples=samples,
                          reference_text=payload["reference_text"], num_steps=payload.get("num_steps", 4))
        return Prepared(inputs, {})

    def finish(self, outputs, context):
        capability = TASK_CAPABILITIES[self.config.task]
        if capability == "asr":
            if not isinstance(outputs.get("text"), str):
                raise ServiceError("invalid_model_output", "ASR must return text", 502)
            language = outputs.get("language", "")
            if not isinstance(language, str):
                raise ServiceError("invalid_model_output", "ASR language must be a string", 502)
            if language.startswith("<|") and language.endswith("|>"):
                language = language[2:-2]
            return {"text": outputs["text"], "language": language, "streaming_mode": "buffered"}
        if capability == "speaker_embeddings":
            import numpy as np
            def normalized(value):
                vector = np.asarray(value, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if vector.ndim != 1 or not 1 <= vector.size <= 4096 or not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 1e-12:
                    raise ServiceError("invalid_model_output", "Invalid speaker embedding", 502)
                return vector / norm
            embedding = normalized(outputs.get("embedding"))
            result = {"embedding": embedding.tolist(), "dimension": embedding.size}
            if "compare_embedding" in outputs:
                other = normalized(outputs["compare_embedding"])
                if other.shape != embedding.shape:
                    raise ServiceError("invalid_model_output", "Speaker embedding dimensions differ", 502)
                result["cosine_similarity"] = float(np.clip(np.dot(embedding, other), -1, 1))
            return result
        return encode_audio(outputs)
