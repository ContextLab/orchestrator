"""Provider abstractions for unified model management.

Concrete providers are resolved lazily so that importing the model registry
does not require every provider SDK to be installed. ``AnthropicProvider``
needs the ``anthropic`` extra; accessing it without that extra raises the
underlying ``ImportError`` naming the missing package.
"""

from ..._lazy import lazy_exports

_EXPORTS = {
    "ModelProvider": ".base",
    "ProviderConfig": ".base",
    "ProviderError": ".base",
    "AnthropicProvider": ".anthropic_provider",
}

__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())
