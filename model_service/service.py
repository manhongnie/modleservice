"""Stable application facade; use cases and runtime each own their state."""
from .contracts import Executor, ModelConfig, PluginCatalog, Settings
from .runtime import ApplicationRuntime


class Service:
    def __init__(self, settings: Settings, executor: Executor | None = None,
                 catalog: PluginCatalog | None = None):
        self.settings = settings
        self.runtime = ApplicationRuntime(settings, executor, catalog)

    @property
    def registry(self):
        return self.runtime.registry

    @property
    def scheduler(self):
        return self.runtime.scheduler

    @property
    def lifecycle(self):
        return self.runtime.lifecycle

    async def start(self):
        await self.runtime.start()

    async def close(self):
        await self.runtime.close()

    def infer(self, capability, model, payload, cancel, request_id, stream=False):
        return self.runtime.coordinator.infer(capability, model, payload, cancel, request_id, stream)

    async def add(self, config: ModelConfig, enable=True):
        return await self.runtime.manager.add(config, enable)

    async def validate(self, model_id, enable=True):
        return await self.runtime.manager.validate(model_id, enable)

    async def enable(self, model_id):
        return await self.runtime.manager.enable(model_id)

    async def load(self, model_id):
        return await self.runtime.manager.load(model_id)

    async def operate(self, model_id, operation, timeout_s=None):
        return await self.runtime.manager.operate(model_id, operation, timeout_s)

    async def set_alias(self, alias, target):
        return await self.runtime.manager.set_alias(alias, target)

    async def delete_alias(self, alias):
        return await self.runtime.manager.delete_alias(alias)

    async def add_dependency(self, dependency, model_id):
        return await self.runtime.manager.add_dependency(dependency, model_id)

    async def delete_dependency(self, dependency):
        return await self.runtime.manager.delete_dependency(dependency)

    async def reconcile(self, request_id, confirmed_stopped):
        return await self.runtime.manager.reconcile(request_id, confirmed_stopped)

    async def status(self):
        return await self.runtime.manager.status()

    async def readiness(self):
        return await self.runtime.readiness()

    async def metrics(self):
        return await self.runtime.metrics()
