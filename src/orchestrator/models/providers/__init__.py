"""Provider abstractions for unified model management.

Concrete providers are resolved lazily so that importing the model registry
does not require provider SDKs. ``DartmouthProvider`` needs no extra at
all -- Dartmouth Chat is an OpenAI-compatible HTTP gateway and the adapter
speaks it with ``aiohttp``, which is already a core dependency. It also
serves several models at zero cost per token, so it is the cheapest way to
run this project against real models.

The supported providers are Dartmouth Chat and the HuggingFace Inference API
(#484). The Anthropic provider was retired (#430) and is deliberately absent.
"""

from ..._lazy import lazy_exports

_EXPORTS = {
    "ModelProvider": ".base",
    "ProviderConfig": ".base",
    "ProviderError": ".base",
    "DartmouthProvider": ".dartmouth_provider",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
