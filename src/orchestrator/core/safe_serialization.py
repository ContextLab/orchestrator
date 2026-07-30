"""Guard for pickle-based persistence.

Unpickling executes code embedded in the data. Anything the process later
reads back -- checkpoints, cached entries, persisted execution state -- is a
code-execution vector if an attacker (or an unlucky path collision with another
project's files) can influence it.

The persistence layers here already default to JSON. This module makes the
remaining pickle *read* paths fail closed: they refuse unless the operator has
explicitly opted in with ``ORCHESTRATOR_ALLOW_PICKLE=1``.

Writing is unaffected -- only loading attacker-influenceable bytes is dangerous.
"""

from __future__ import annotations

import os

__all__ = ["PickleDisabledError", "pickle_allowed", "ensure_pickle_allowed"]

#: Environment variable that opts in to loading pickled data.
ALLOW_PICKLE_ENV_VAR = "ORCHESTRATOR_ALLOW_PICKLE"


class PickleDisabledError(RuntimeError):
    """Raised when pickle deserialization is attempted while disabled."""


def pickle_allowed() -> bool:
    """Whether loading pickled data is permitted (default: no)."""
    return os.environ.get(ALLOW_PICKLE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def ensure_pickle_allowed(source: str) -> None:
    """Raise :class:`PickleDisabledError` unless pickle loading is enabled.

    Args:
        source: Human-readable description of what is being loaded, used in the
            error message so the operator can tell which file triggered it.
    """
    if pickle_allowed():
        return
    raise PickleDisabledError(
        f"Refusing to unpickle {source}: unpickling executes arbitrary code. "
        f"Use a JSON persistence format, or set {ALLOW_PICKLE_ENV_VAR}=1 if the "
        "data is known to be trusted."
    )
