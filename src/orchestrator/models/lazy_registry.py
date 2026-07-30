"""A model registry that discovers providers only when a model is demanded.

Populating a registry is not a free, local operation: it reads the user's
credentials from ``~/.orchestrator/.env``, imports provider SDKs and probes the
machine for a local Ollama install. A pipeline assembled only from
deterministic local tools (filesystem, data-processing, validation, ...) needs
none of that, and must not touch provider credentials at all.

So the population step is deferred to the first call that actually has to
return a :class:`~orchestrator.core.model.Model`. Constructing the registry,
compiling a pipeline and executing tool-only steps all stay hermetic; the
moment a step asks for a model, discovery runs once.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from .model_registry import ModelRegistry


class LazyModelRegistry(ModelRegistry):
    """A :class:`ModelRegistry` whose contents are discovered on demand.

    Args:
        populate: Callable invoked once, with this registry, the first time a
            model is actually requested. It is responsible for registering
            whatever models the environment can serve.
    """

    def __init__(self, populate: Callable[[ModelRegistry], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._populate = populate
        self._populated = False

    @property
    def populated(self) -> bool:
        """Whether provider discovery has already run."""
        return self._populated

    def ensure_populated(self) -> None:
        """Run provider discovery once. Subsequent calls are no-ops.

        The flag is set *before* populating so that a failing discovery is not
        retried on every model request; the failure surfaces to the caller that
        triggered it.
        """
        if self._populated:
            return
        self._populated = True
        self._populate(self)

    def can_provide_models(self) -> bool:
        """A lazy registry may yield models; discovery has not run yet."""
        return True

    # -- demand points -----------------------------------------------------
    # Only methods that must return a Model trigger discovery. Introspection
    # (``models``, ``list_models``, ``list_providers``) deliberately does not,
    # so that reporting code cannot cause a credential read.

    def get_model(self, model_name: str, provider: str = ""):
        self.ensure_populated()
        return super().get_model(model_name, provider)

    async def get_model_async(self, model_name: str, provider: str = ""):
        self.ensure_populated()
        return await super().get_model_async(model_name, provider)

    async def select_model(self, requirements: Dict[str, Any]):
        self.ensure_populated()
        return await super().select_model(requirements)

    async def get_available_models(self) -> List[str]:
        self.ensure_populated()
        return await super().get_available_models()
