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

import threading
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
        # Two flags, deliberately. `_populated` means "discovery was
        # attempted", and gates retries. `_population_done` means "discovery
        # has finished", and is the only safe basis for a lock-free fast path:
        # checking `_populated` there let a second thread sail past while the
        # first was still registering models.
        self._populated = False
        self._population_done = False
        # Reentrant: population registers models, and a registration path may
        # reach back into a demand point on this same thread. A plain Lock
        # would deadlock there; an RLock lets the owning thread through, where
        # the flag below already short-circuits it.
        self._populate_lock = threading.RLock()

    @property
    def populated(self) -> bool:
        """Whether provider discovery has already run."""
        return self._populated

    def ensure_populated(self) -> None:
        """Run provider discovery once. Subsequent calls are no-ops.

        The flag is set *before* populating so that a failing discovery is not
        retried on every model request; the failure surfaces to the caller that
        triggered it.

        The whole step is serialised. Without the lock a second thread saw
        ``_populated == True`` the instant the first thread set it and went on
        to read a registry that was still filling up -- reporting "model not
        found" for a model that was about to exist. Concurrent callers now
        block until discovery has actually finished.
        """
        # Fast path only once discovery has *finished*. Testing `_populated`
        # here would return the instant the flag flipped, handing the caller a
        # registry that was still filling up.
        if self._population_done:
            return
        with self._populate_lock:
            # Either another thread finished while this one waited, or this is
            # a re-entrant call from inside populate() on the same thread.
            if self._populated:
                return
            self._populated = True
            try:
                self._populate(self)
            finally:
                # Set even when populate() raises, so a failing discovery is
                # not re-run on every subsequent model request.
                self._population_done = True

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
