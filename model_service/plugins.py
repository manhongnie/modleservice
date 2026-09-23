"""Application assembly for trusted task/backend plugins, with no model SDK imports."""
from model_service.backends import describe_backend, validate_backend
from model_service.contracts import BackendFeatures, ModelConfig, Settings
from model_service.tasks import create_task, validate_input


class BuiltinPluginCatalog:
    def describe(self, config: ModelConfig) -> BackendFeatures:
        return describe_backend(config)

    def validate(self, config: ModelConfig, settings: Settings) -> None:
        validate_backend(config, settings)
        create_task(config)
        for capability in config.capabilities:
            validate_input(config, config.validation_input, capability)

    def validate_input(self, config: ModelConfig, payload: dict, capability: str) -> None:
        validate_input(config, payload, capability)
