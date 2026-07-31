"""Provider abstractions for unified model management.

Concrete providers are resolved lazily so that importing the model registry
does not require every provider SDK to be installed. ``AnthropicProvider``
needs the ``anthropic`` extra; accessing it without that extra raises the
underlying ``ImportError`` naming the missing package.

``DartmouthProvider`` needs no extra at all -- Dartmouth Chat is an
OpenAI-compatible HTTP gateway and the adapter speaks it with ``aiohttp``,
which is already a core dependency. It also serves several models at zero
cost per token, so it is the cheapest way to run this project against real
models.
"""

from ..._lazy import lazy_exports

_EXPORTS = {
    "ModelProvider": ".base",
    "ProviderConfig": ".base",
    "ProviderError": ".base",
    "AnthropicProvider": ".anthropic_provider",
    "DartmouthProvider": ".dartmouth_provider",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
