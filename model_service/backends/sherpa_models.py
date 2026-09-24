"""Native sherpa-onnx runtimes; all execution settles before returning or cancelling."""
from __future__ import annotations

import gc
from pathlib import Path

from model_service.contracts import ServiceError


REQUIRED = {
    "sherpa_sensevoice": ("model.int8.onnx", "tokens.txt"),
    "sherpa_matcha": ("model-steps-3.onnx", "vocos-16khz-univ.onnx", "tokens.txt", "lexicon.txt", "espeak-ng-data"),
    "sherpa_speaker": ("model.onnx",),
    "sherpa_zipvoice": ("encoder.int8.onnx", "decoder.int8.onnx", "vocos_24khz.onnx", "tokens.txt", "lexicon.txt", "espeak-ng-data"),
}


def validate_sherpa(config, settings=None):
    if config.task not in REQUIRED:
        raise ServiceError("invalid_task_backend", "sherpa_onnx requires a registered sherpa speech task")
    if config.device.upper() != "CPU" or config.concurrency != 1:
        raise ServiceError("invalid_config", "Initial sherpa-onnx deployment requires CPU and concurrency=1")
    allowed = {"threads", "max_audio_seconds", "max_reference_seconds", "max_text_chars", "max_steps", "language", "use_itn"}
    if set(config.options) - allowed:
        raise ServiceError("invalid_config", "Unsupported sherpa-onnx option")
    for key, lower, upper, default in [("threads", 1, 8, 2), ("max_audio_seconds", 1, 60, 30),
                                      ("max_reference_seconds", 1, 30, 15), ("max_text_chars", 1, 500, 200), ("max_steps", 1, 16, 8)]:
        value = config.options.get(key, default)
        if type(value) is not int or not lower <= value <= upper:
            raise ServiceError("invalid_config", f"{key} must be an integer in {lower}..{upper}")
    language = config.options.get("language", "auto")
    if not isinstance(language, str) or language not in {"auto", "zh", "en", "ja", "ko", "yue"}:
        raise ServiceError("invalid_config", "Unsupported SenseVoice language")
    if type(config.options.get("use_itn", True)) is not bool:
        raise ServiceError("invalid_config", "use_itn must be boolean")
    root = Path(config.path).resolve()
    names = list(REQUIRED[config.task])
    if config.task == "sherpa_matcha":
        names += [name for name in ("date-zh.fst", "number-zh.fst", "phone-zh.fst") if (root / name).exists()]
    for name in names:
        asset = (root / name).resolve()
        if not asset.is_relative_to(root):
            raise ServiceError("invalid_model_path", "Speech assets must remain within the model directory")
        if not asset.exists() or (name != "espeak-ng-data" and not asset.is_file()) or (name == "espeak-ng-data" and not asset.is_dir()):
            raise ServiceError("model_files_missing", f"Missing sherpa-onnx model asset: {name}")


def _check_cancel(cancel):
    if cancel.is_set():
        raise ServiceError("cancelled", "Speech inference stopped after cancellation", 499)


class SherpaONNXBackend:
    def __init__(self, config):
        self.config, self.model = config, None

    def load(self):
        validate_sherpa(self.config)
        try:
            import sherpa_onnx as sherpa
        except ImportError as error:
            raise ServiceError("missing_dependency", "Install sherpa-onnx speech extras", 503) from error
        root = Path(self.config.path).resolve()
        asset = lambda name: str(root / name)
        threads = self.config.options.get("threads", 2)
        task = self.config.task
        if task == "sherpa_sensevoice":
            self.model = sherpa.OfflineRecognizer.from_sense_voice(
                model=asset("model.int8.onnx"), tokens=asset("tokens.txt"), num_threads=threads, provider="cpu",
                language=self.config.options.get("language", "auto"), use_itn=self.config.options.get("use_itn", True))
        elif task == "sherpa_speaker":
            cfg = sherpa.SpeakerEmbeddingExtractorConfig(model=asset("model.onnx"), num_threads=threads, provider="cpu")
            if not cfg.validate():
                raise ServiceError("invalid_model_config", "sherpa rejected speaker extractor configuration")
            self.model = sherpa.SpeakerEmbeddingExtractor(cfg)
        else:
            if task == "sherpa_matcha":
                specific = {"matcha": sherpa.OfflineTtsMatchaModelConfig(
                    acoustic_model=asset("model-steps-3.onnx"), vocoder=asset("vocos-16khz-univ.onnx"),
                    tokens=asset("tokens.txt"), lexicon=asset("lexicon.txt"), data_dir=asset("espeak-ng-data"))}
            else:
                specific = {"zipvoice": sherpa.OfflineTtsZipvoiceModelConfig(
                    encoder=asset("encoder.int8.onnx"), decoder=asset("decoder.int8.onnx"), vocoder=asset("vocos_24khz.onnx"),
                    tokens=asset("tokens.txt"), lexicon=asset("lexicon.txt"), data_dir=asset("espeak-ng-data"))}
            rules = [asset(name) for name in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
                     if task == "sherpa_matcha" and (root / name).is_file()]
            cfg = sherpa.OfflineTtsConfig(model=sherpa.OfflineTtsModelConfig(**specific, num_threads=threads, provider="cpu"),
                                         rule_fsts=",".join(rules), max_num_sentences=1)
            if not cfg.validate():
                raise ServiceError("invalid_model_config", "sherpa rejected speech synthesis configuration")
            self.model = sherpa.OfflineTts(cfg)
        return {"backend": "sherpa_onnx", "runtime_version": sherpa.__version__, "task": task, "streaming": "buffered"}

    def infer(self, inputs, cancel):
        _check_cancel(cancel)
        if self.model is None:
            raise ServiceError("model_not_loaded", "Speech model has not loaded", 503)
        task = self.config.task
        if task == "sherpa_sensevoice":
            stream = self.model.create_stream()
            stream.accept_waveform(inputs["sample_rate"], inputs["samples"])
            self.model.decode_stream(stream)
            result = {"text": stream.result.text, "language": getattr(stream.result, "lang", "")}
        elif task == "sherpa_speaker":
            def extract(rate, samples):
                stream = self.model.create_stream()
                stream.accept_waveform(rate, samples)
                stream.input_finished()
                if not self.model.is_ready(stream):
                    raise ServiceError("invalid_input", "Recording contains insufficient audio for speaker embedding", 422)
                return self.model.compute(stream)
            result = {"embedding": extract(inputs["sample_rate"], inputs["samples"])}
            _check_cancel(cancel)
            if "compare_samples" in inputs:
                result["compare_embedding"] = extract(inputs["compare_sample_rate"], inputs["compare_samples"])
        else:
            # Native Matcha/ZipVoice callbacks return 1 to continue, 0 to stop.
            # Even if a sentence cannot be interrupted, wait for generate() to
            # return before exposing cancellation to the execution coordinator.
            callback = lambda samples, progress: 0 if cancel.is_set() else 1
            if task == "sherpa_matcha":
                audio = self.model.generate(inputs["text"], sid=0, speed=inputs["speed"], callback=callback)
            else:
                import sherpa_onnx
                generation = sherpa_onnx.GenerationConfig()
                generation.reference_audio = inputs["reference_samples"]
                generation.reference_sample_rate = inputs["reference_sample_rate"]
                generation.reference_text = inputs["reference_text"]
                generation.num_steps = inputs["num_steps"]
                generation.speed = inputs["speed"]
                audio = self.model.generate(inputs["text"], generation, callback=callback)
            result = {"samples": audio.samples, "sample_rate": audio.sample_rate}
        _check_cancel(cancel)
        return result

    def close(self):
        self.model = None
        gc.collect()
        return {"released": True}
