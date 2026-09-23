"""Deployment-independent model/input policy; plugin semantics are injected."""
import json

from .assets import check_assets
from .contracts import ModelConfig, PluginCatalog, ServiceError, Settings


class ConfigurationPolicy:
    def __init__(self, settings: Settings, catalog: PluginCatalog):
        self.settings, self.catalog = settings, catalog

    def validate(self, config: ModelConfig) -> None:
        features = self.catalog.describe(config)
        if features.synthetic and not self.settings.allow_mock:
            raise ServiceError("mock_disabled", "Synthetic models are disabled in this deployment", 422)
        if len(set(config.capabilities)) != len(config.capabilities):
            raise ServiceError("invalid_config", "Capabilities must be unique", 422)
        if features.requires_artifact and not config.path:
            raise ServiceError("model_files_missing", "This backend requires local model artifacts", 422)
        check_assets(config, self.settings.model_roots)
        self.catalog.validate(config, self.settings)
        for capability in config.capabilities:
            self.validate_input(config, config.validation_input, capability)

    def validate_input(self, config: ModelConfig, payload: dict, capability: str) -> None:
        if self.catalog.describe(config).synthetic and not self.settings.allow_mock:
            raise ServiceError("mock_disabled", "Synthetic models are disabled in this deployment", 422)
        if capability not in config.capabilities:
            raise ServiceError("unsupported_capability", "Selected model does not support this capability", 422)
        try:
            size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
        except (ValueError, TypeError) as exc:
            raise ServiceError("invalid_input", "Input must be finite JSON data", 422) from exc
        if size > config.max_input_bytes:
            raise ServiceError("input_too_large", "Input exceeds the model limit", 413)
        self.catalog.validate_input(config, payload, capability)
