"""Shared helper for lazy package exports (PEP 562).

Several package ``__init__`` modules used to import every submodule eagerly.
That made ``import orchestrator`` pull in provider SDKs, browsers, plotting and
multimedia libraries, so a single missing optional dependency broke the whole
package.

``lazy_exports`` builds the ``__getattr__``/``__dir__`` pair implementing
on-demand resolution, so each package declares only a name -> submodule map:

    from .._lazy import lazy_exports

    _EXPORTS = {"AnthropicProvider": ".anthropic_provider"}
    __all__ = sorted(_EXPORTS)
    __getattr__, __dir__ = lazy_exports(__name__, _EXPORTS, globals())

Resolved values are cached into the module's namespace, so each name costs one
import at most and later lookups bypass ``__getattr__`` entirely.
"""

from importlib import import_module
from typing import Any, Callable


def lazy_exports(
    package: str,
    exports: dict[str, str],
    namespace: dict[str, Any],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Return ``(__getattr__, __dir__)`` implementing lazy name resolution.

    Args:
        package: The importing package's ``__name__``, used to anchor relative
            module paths.
        exports: Map of exported name -> module containing it. Relative paths
            (starting with ``.``) are resolved against ``package``.
        namespace: The module's ``globals()``, used to cache resolved values.

    The raised ``AttributeError`` matches CPython's wording so that
    ``hasattr``, ``inspect`` and ``copy``/``pickle`` protocol probes behave
    normally. An ``ImportError`` from a genuinely missing optional dependency
    is allowed to propagate unchanged, so the error names the real package.
    """

    def __getattr__(name: str) -> Any:
        module_path = exports.get(name)
        if module_path is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module = import_module(module_path, package)
        value = getattr(module, name)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(exports) | set(namespace))

    return __getattr__, __dir__
