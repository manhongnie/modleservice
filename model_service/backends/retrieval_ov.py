"""OpenVINO adapters for retrieval graphs requiring more than one IR."""
from pathlib import Path

from model_service.contracts import ServiceError
from .openvino import OpenVINOBackend


class DualClipOpenVINOBackend:
    """Both projected towers share a single worker, lifecycle and reservation."""
    def __init__(self, config):
        self.config = config
        self.towers = {}

    def load(self):
        try:
            metadata = {}
            for tower in ('text', 'image'):
                model_file = self.config.options.get(f'{tower}_model_file', f'openvino_{tower}.xml')
                root = Path(self.config.path).resolve()
                if not (root / model_file).resolve().is_relative_to(root):
                    raise ServiceError('invalid_config', 'CLIP tower files must stay inside their model directory')
                options = {**self.config.options, 'model_file': model_file}
                backend = OpenVINOBackend(self.config.model_copy(update={'options': options}))
                self.towers[tower] = backend
                metadata[tower] = backend.load()
            return {'backend': 'openvino_clip', 'management': 'local', 'device': self.config.device, 'towers': metadata}
        except BaseException:
            self.close()
            raise

    def infer(self, inputs, cancel):
        if not self.towers:
            raise ServiceError('not_loaded', 'CLIP towers have not been loaded', 503)
        output = {}
        for tower, tensors in inputs.items():
            if tower not in self.towers:
                raise ServiceError('invalid_input', 'Unknown CLIP tower')
            output.update(self.towers[tower].infer(tensors, cancel))
        return output

    def close(self):
        for backend in self.towers.values(): backend.close()
        self.towers.clear()
        return {'backend': 'openvino_clip', 'management': 'local', 'unloaded': True}
