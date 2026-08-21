"""Credential resolution for the HuggingFace Inference API.

The HuggingFace router (https://router.huggingface.co/v1) is an
OpenAI-compatible gateway to the Inference Providers network. It needs one
bearer token -- a fine-grained token with "Make calls to Inference Providers"
permission.

Resolution order, highest priority first:

1. ``HF_TOKEN`` in the environment (the CI path).
2. ``~/.orchestrator/.env`` -- this project's own credential store.
3. ``~/.cache/huggingface/token`` -- the HuggingFace CLI's own store, written
   by ``hf auth login``. Sharing that copy beats a second one that has to be
   rotated separately (the same reason the Dartmouth resolver reads the
   sibling llmxive store).

Nothing here ever logs, prints, or returns a token inside an error message.
``ResolvedCredential`` and ``mask_key`` are shared with the Dartmouth adapter
rather than copied: one secret-hygiene implementation, audited once.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .dartmouth_credentials import ResolvedCredential, mask_key  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "HF_TOKEN_ENV_VAR",
    "HuggingFaceCredentialError",
    "ResolvedCredential",
    "mask_key",
    "resolve_huggingface_api_key",
]

HF_TOKEN_ENV_VAR = "HF_TOKEN"

#: This project's own store, shared with the rest of the CLI configuration.
_ORCHESTRATOR_ENV_FILE = Path.home() / ".orchestrator" / ".env"

#: The HuggingFace CLI's store. Same machine, same user, same token.
_HF_CLI_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"


class HuggingFaceCredentialError(RuntimeError):
    """Raised when no HuggingFace API token can be found."""


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


def _read_hf_cli_token(path: Path) -> str | None:
    """Read the HF CLI token store: a plain file holding the raw token."""
    if not path.is_file():
        return None
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None
    return value or None


def resolve_huggingface_api_key(*, required: bool = True) -> ResolvedCredential | None:
    """Find a HuggingFace API token.

    Args:
        required: Raise :class:`HuggingFaceCredentialError` when nothing is
            found. Pass ``False`` to probe availability without handling an
            exception.

    Returns:
        The credential and its source, or ``None`` when absent and
        ``required`` is ``False``.
    """
    env_value = os.environ.get(HF_TOKEN_ENV_VAR)
    if env_value and env_value.strip():
        return ResolvedCredential(env_value.strip(), f"${HF_TOKEN_ENV_VAR}")

    orchestrator_value = _read_env_file(_ORCHESTRATOR_ENV_FILE, HF_TOKEN_ENV_VAR)
    if orchestrator_value:
        return ResolvedCredential(orchestrator_value, str(_ORCHESTRATOR_ENV_FILE))

    cli_value = _read_hf_cli_token(_HF_CLI_TOKEN_FILE)
    if cli_value:
        return ResolvedCredential(cli_value, str(_HF_CLI_TOKEN_FILE))

    if not required:
        return None
    raise HuggingFaceCredentialError(
        "No HuggingFace API token found. Set "
        f"{HF_TOKEN_ENV_VAR}, or add "
        f"'{HF_TOKEN_ENV_VAR}=<token>' to {_ORCHESTRATOR_ENV_FILE}, or run "
        f"'hf auth login' (which writes {_HF_CLI_TOKEN_FILE}). "
        "Create a fine-grained token with 'Make calls to Inference Providers' "
        "permission at https://huggingface.co/settings/tokens."
    )
