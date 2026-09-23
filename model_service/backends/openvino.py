"""OpenVINO Runtime adapter. The control process never imports OpenVINO."""
from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from model_service.contracts import ModelConfig, ServiceError


class OpenVINOBackend:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.compiled = None
        self.core = None

    def load(self) -> dict[str, Any]:
        try:
            import openvino as ov
        except ImportError as exc:
            raise ServiceError("missing_dependency", "Install the openvino optional dependencies", 503) from exc
        model_path = Path(self.config.path)
        if model_path.is_dir():
            root = model_path.resolve()
            model_path = (root / str(self.config.options.get("model_file", "openvino_model.xml"))).resolve()
            if not model_path.is_relative_to(root):
                raise ServiceError("invalid_config", "OpenVINO model_file must remain within model directory")
        self.core = ov.Core()
        try:
            self.compiled = self.core.compile_model(
                str(model_path), self.config.device, self.config.options.get("compile_config", {}),
            )
        except Exception as exc:
            raise ServiceError("load_failed", f"OpenVINO could not compile model: {type(exc).__name__}", 503) from exc
        return {"backend": "openvino", "management": "local", "device": self.config.device,
                "inputs": [port.any_name for port in self.compiled.inputs],
                "outputs": [port.any_name for port in self.compiled.outputs]}

    def infer(self, inputs: dict[str, Any], cancel: Event) -> dict[str, Any]:
        import numpy as np
        if self.compiled is None:
            raise ServiceError("not_loaded", "OpenVINO model is not loaded", 503)
        if cancel.is_set():
            raise ServiceError("cancelled", "Cancelled before OpenVINO execution", 499)
        request = self.compiled.create_infer_request()
        tensors = {}
        for port in self.compiled.inputs:
            name = port.any_name
            if name not in inputs:
                raise ServiceError("invalid_input", f"Required OpenVINO input is missing: {name}")
            tensors[name] = np.asarray(inputs[name], dtype=port.element_type.to_dtype())
        started = False
        try:
            request.start_async(tensors)
            started = True
            while not request.wait_for(20):
                if cancel.is_set():
                    request.cancel()
                    # cancel() is a request; wait() is the stop acknowledgement.
                    request.wait()
                    raise ServiceError("cancelled", "OpenVINO execution stopped", 499)
            if cancel.is_set():
                raise ServiceError("cancelled", "OpenVINO execution finished after cancellation", 499)
            return {port.any_name: request.get_output_tensor(i).data.copy()
                    for i, port in enumerate(self.compiled.outputs)}
        except ServiceError:
            raise
        except Exception as exc:
            if started:
                try:
                    request.cancel()
                finally:
                    try:
                        request.wait()
                    except Exception:
                        # wait raises when the request was cancelled, after it stopped.
                        pass
            if cancel.is_set():
                raise ServiceError("cancelled", "OpenVINO execution stopped", 499) from exc
            raise ServiceError("inference_failed", f"OpenVINO inference failed: {type(exc).__name__}", 503) from exc

    def close(self) -> dict[str, Any]:
        self.compiled = None
        self.core = None
        return {"backend": "openvino", "unloaded": True, "management": "local"}
