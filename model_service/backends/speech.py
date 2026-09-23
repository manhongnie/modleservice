"""Optional local sherpa-onnx VITS execution; text frontend belongs to model runtime."""
from pathlib import Path
from model_service.contracts import ServiceError


class SherpaTTSBackend:
    def __init__(self, config):
        self.config, self.model = config, None

    def load(self):
        try:
            import sherpa_onnx
        except ImportError as error:
            raise ServiceError("missing_dependency", "Install speech extras: sherpa-onnx", 503) from error
        if self.config.device.upper() != "CPU":
            raise ServiceError("unsupported_device", "Initial sherpa TTS backend supports CPU only")
        root = Path(self.config.path)
        required = {name: root / name for name in ("model.onnx", "lexicon.txt", "tokens.txt")}
        if any(not path.resolve().is_relative_to(root.resolve()) for path in required.values()):
            raise ServiceError("invalid_model_path", "TTS assets must remain within model directory")
        if any(not path.is_file() for path in required.values()):
            raise ServiceError("model_files_missing", "VITS directory must include model.onnx, lexicon.txt and tokens.txt")
        # All auxiliary assets are resolved within the already validated model root.
        rules = [root / name for name in ("phone.fst", "date.fst", "number.fst") if (root / name).is_file()]
        if any(not path.resolve().is_relative_to(root.resolve()) for path in rules):
            raise ServiceError("invalid_model_path", "TTS rules must remain within model directory")
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(required["model.onnx"]), lexicon=str(required["lexicon.txt"]), tokens=str(required["tokens.txt"]))
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=int(self.config.options.get("threads", 2)), provider="cpu"),
            rule_fsts=",".join(map(str, rules)), max_num_sentences=1)
        if not config.validate():
            raise ServiceError("invalid_model_config", "sherpa-onnx rejected VITS configuration")
        self.model = sherpa_onnx.OfflineTts(config)
        return {"backend": "sherpa_tts", "sample_rate": self.model.sample_rate, "streaming": "buffered"}

    def infer(self, inputs, cancel):
        if cancel.is_set():
            raise ServiceError("cancelled", "Request cancelled", 499)
        # Runtime cooperatively calls callback between synthesized segments. If one
        # segment cannot be interrupted, keep its reservation until generate returns.
        audio = self.model.generate(inputs["text"], sid=inputs["speaker_id"], speed=inputs["speed"], callback=lambda samples, progress: 0 if cancel.is_set() else 1)
        if cancel.is_set():
            raise ServiceError("cancelled", "Speech generation stopped after cancellation", 499)
        return {"samples": audio.samples, "sample_rate": audio.sample_rate}

    def close(self):
        self.model = None
        import gc
        gc.collect()
        return {"released": True}
