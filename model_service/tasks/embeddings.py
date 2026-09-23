"""Named tensor export boundaries; tokenizers/processors are worker-local and lazy."""
from pathlib import Path
import io
from threading import Lock

from model_service.contracts import ModelConfig, Prepared, ServiceError
from .core import PassthroughTask, decode_base64, texts


def artifact_dir(config: ModelConfig) -> str:
    path = Path(config.path)
    return str(path if path.is_dir() else path.parent)


def output_tensor(outputs, name):
    import numpy as np
    if name not in outputs:
        raise ServiceError("unsupported_format", f"Export must provide tensor '{name}', received {list(outputs)}")
    return np.asarray(outputs[name], dtype=np.float32)


def normalize(vectors):
    import numpy as np
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ServiceError("invalid_model_output", "Expected finite rank-2 embedding tensor")
    return (vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)).tolist()


class DenseTask(PassthroughTask):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self._init_lock = Lock()

    def validate(self, payload, capability):
        if capability != "embeddings":
            raise ServiceError("unsupported_capability", "BGE dense task supports embeddings only")
        super().validate(payload, capability)

    def get_tokenizer(self):
        with self._init_lock:
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(artifact_dir(self.config), local_files_only=True, trust_remote_code=False)
        return self.tokenizer

    def prepare(self, payload):
        values = texts(payload)
        encoded = self.get_tokenizer()(values, padding=True, truncation=True,
                                       max_length=int(self.config.options.get("max_length", 512)), return_tensors="np")
        return Prepared(dict(encoded), {"count": len(values)})

    def finish(self, outputs, context):
        tensor = output_tensor(outputs, self.config.options.get("output_name", "last_hidden_state"))
        if tensor.ndim != 3:
            raise ServiceError("unsupported_format", "BGE-M3 dense expects last_hidden_state [batch, tokens, hidden]")
        if "count" in context and tensor.shape[0] != context["count"]:
            raise ServiceError("invalid_model_output", "BGE dense output batch size does not match input")
        return {"embeddings": normalize(tensor[:, 0, :]), "pooling": "cls_l2"}


class RerankTask(DenseTask):
    def validate(self, payload, capability):
        if capability != "rerank":
            raise ServiceError("unsupported_capability", "BGE reranker supports rerank only")
        PassthroughTask.validate(self, payload, capability)

    def prepare(self, payload):
        encoded = self.get_tokenizer()([payload["query"]] * len(payload["documents"]), payload["documents"],
                                      padding=True, truncation=True, max_length=int(self.config.options.get("max_length", 512)), return_tensors="np")
        return Prepared(dict(encoded), {"count": len(payload["documents"])})

    def finish(self, outputs, context):
        import numpy as np
        scores = output_tensor(outputs, self.config.options.get("output_name", "logits")).reshape(-1)
        if len(scores) != context["count"] or not np.isfinite(scores).all():
            raise ServiceError("invalid_model_output", "Reranker must return one finite logit per document")
        if self.config.options.get("sigmoid", True):
            scores = 1 / (1 + np.exp(-np.clip(scores, -80, 80)))
        return {"results": sorted([{"index": i, "score": float(v)} for i, v in enumerate(scores)], key=lambda row: row["score"], reverse=True)}


class ChineseClipTask(PassthroughTask):
    def __init__(self, config):
        super().__init__(config)
        self.processor = None
        self._init_lock = Lock()

    def validate(self, payload, capability):
        expected = "image_embeddings" if self.config.options.get("mode", "text") == "image" else "embeddings"
        if capability != expected:
            raise ServiceError("unsupported_capability", f"This CLIP export provides {expected} only")
        super().validate(payload, capability)

    def prepare(self, payload):
        from transformers import ChineseCLIPProcessor
        with self._init_lock:
            if self.processor is None:
                self.processor = ChineseCLIPProcessor.from_pretrained(artifact_dir(self.config), local_files_only=True)
        if self.config.options.get("mode", "text") == "image":
            from PIL import Image
            images = []
            for raw in payload["images"]:
                with Image.open(io.BytesIO(decode_base64(raw, "image"))) as image:
                    if image.width * image.height > int(self.config.options.get("max_image_pixels", 16777216)):
                        raise ServiceError("invalid_input", "Decoded image exceeds pixel limit", 422)
                    images.append(image.convert("RGB"))
            encoded = self.processor(images=images, return_tensors="np")
        else:
            encoded = self.processor(text=texts(payload), padding=True, truncation=True, max_length=52, return_tensors="np")
        count = len(payload["images"]) if self.config.options.get("mode", "text") == "image" else len(texts(payload))
        return Prepared(dict(encoded), {"count": count})

    def finish(self, outputs, context):
        default = "image_embeds" if self.config.options.get("mode", "text") == "image" else "text_embeds"
        tensor = output_tensor(outputs, self.config.options.get("output_name", default))
        if "count" in context and (tensor.ndim != 2 or tensor.shape[0] != context["count"]):
            raise ServiceError("invalid_model_output", "CLIP output batch size does not match input")
        return {"embeddings": normalize(tensor)}
