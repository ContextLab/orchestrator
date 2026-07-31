"""Credential resolution for the Dartmouth Chat API.

Dartmouth Chat is an OpenAI-compatible gateway that serves several models at
**zero cost per token**, which makes it the cheapest way to run this project
against real models. It needs one bearer token.

Resolution order, highest priority first:

1. ``DARTMOUTH_CHAT_API_KEY`` in the environment (the CI path).
2. ``~/.orchestrator/.env`` -- this project's own credential store.
3. ``~/.config/llmxive/credentials.toml`` -- the llmxive project's store,
   read only if the key is not already available. Sharing one token between
   sibling projects on the same machine beats a second copy that has to be
   rotated separately.

Nothing here ever logs, prints, or returns a key inside an error message.
:func:`mask_key` exists so callers can report *which* credential was found
without disclosing it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DARTMOUTH_KEY_ENV_VAR",
    "DartmouthCredentialError",
    "ResolvedCredential",
    "mask_key",
    "resolve_dartmouth_api_key",
]

DARTMOUTH_KEY_ENV_VAR = "DARTMOUTH_CHAT_API_KEY"

#: This project's own store, shared with the rest of the CLI configuration.
_ORCHESTRATOR_ENV_FILE = Path.home() / ".orchestrator" / ".env"

#: The sibling llmxive project's store. Same machine, same user, same token.
_LLMXIVE_CREDENTIALS_FILE = Path.home() / ".config" / "llmxive" / "credentials.toml"
_LLMXIVE_KEY_FIELD = "dartmouth_chat_api_key"


class DartmouthCredentialError(RuntimeError):
    """Raised when no Dartmouth Chat credential can be found."""


@dataclass(frozen=True)
class ResolvedCredential:
    """A credential plus where it came from, for diagnostics."""

    key: str
    source: str

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never let a key reach a traceback, log line, or pytest diff.
        return f"ResolvedCredential(key={mask_key(self.key)!r}, source={self.source!r})"


def mask_key(key: str | None) -> str:
    """Render a key safe to log: ``sk-014...9f2`` or ``<unset>``."""
    if not key:
        return "<unset>"
    if len(key) <= 10:
        return "<set>"
    return f"{key[:6]}...{key[-3:]}"


def _read_env_file(path: Path, variable: str) -> str | None:
    """Read ``variable`` from a ``KEY=value`` file, ignoring comments."""
    if not path.is_file():
        return None
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == variable:
                return value.strip().strip('"').strip("'") or None
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
    return None


def _read_llmxive_credentials(path: Path) -> str | None:
    """Read the Dartmouth key from llmxive's TOML credential store."""
    if not path.is_file():
        return None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - tomllib is stdlib on 3.11+
        return None
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError) as exc:
        logger.debug("Could not parse %s: %s", path, exc)
        return None
    value = data.get(_LLMXIVE_KEY_FIELD)
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_dartmouth_api_key(*, required: bool = True) -> ResolvedCredential | None:
    """Find a Dartmouth Chat API key.

    Args:
        required: Raise :class:`DartmouthCredentialError` when nothing is
            found. Pass ``False`` to probe availability without handling an
            exception.

    Returns:
        The credential and its source, or ``None`` when absent and
        ``required`` is ``False``.
    """
    env_value = os.environ.get(DARTMOUTH_KEY_ENV_VAR)
    if env_value and env_value.strip():
        return ResolvedCredential(env_value.strip(), f"${DARTMOUTH_KEY_ENV_VAR}")

    orchestrator_value = _read_env_file(_ORCHESTRATOR_ENV_FILE, DARTMOUTH_KEY_ENV_VAR)
    if orchestrator_value:
        return ResolvedCredential(orchestrator_value, str(_ORCHESTRATOR_ENV_FILE))

    llmxive_value = _read_llmxive_credentials(_LLMXIVE_CREDENTIALS_FILE)
    if llmxive_value:
        return ResolvedCredential(llmxive_value, str(_LLMXIVE_CREDENTIALS_FILE))

    if not required:
        return None
    raise DartmouthCredentialError(
        "No Dartmouth Chat API key found. Set "
        f"{DARTMOUTH_KEY_ENV_VAR}, or add "
        f"'{DARTMOUTH_KEY_ENV_VAR}=<key>' to {_ORCHESTRATOR_ENV_FILE}, or "
        f"'{_LLMXIVE_KEY_FIELD} = \"<key>\"' to {_LLMXIVE_CREDENTIALS_FILE}. "
        "Get a key from https://chat.dartmouth.edu/ (Settings -> Account -> "
        "API keys)."
    )
