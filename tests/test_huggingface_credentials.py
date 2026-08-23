"""Credential resolution and secret-hygiene for the HuggingFace adapter.

The token is a real secret. The tests that matter most here are the ones
asserting it never reaches a log line, a repr, or an exception message -- a
credential that leaks into a traceback ends up in CI logs and issue reports.
"""

import logging

import pytest

from orchestrator.models.huggingface_credentials import (
    HF_TOKEN_ENV_VAR,
    HuggingFaceCredentialError,
    resolve_huggingface_api_key,
)

pytestmark = pytest.mark.unit

FAKE_KEY = "hf_0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Never read the developer's real credential stores during tests."""
    monkeypatch.delenv(HF_TOKEN_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "orchestrator.models.huggingface_credentials._ORCHESTRATOR_ENV_FILE",
        tmp_path / "orchestrator.env",
    )
    monkeypatch.setattr(
        "orchestrator.models.huggingface_credentials._HF_CLI_TOKEN_FILE",
        tmp_path / "hf-cli-token",
    )


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_environment_variable_wins(monkeypatch, tmp_path):
    (tmp_path / "orchestrator.env").write_text(f"{HF_TOKEN_ENV_VAR}=from-env-file\n")
    (tmp_path / "hf-cli-token").write_text("from-cli-store\n")
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, FAKE_KEY)

    resolved = resolve_huggingface_api_key()

    assert resolved.key == FAKE_KEY
    assert resolved.source == f"${HF_TOKEN_ENV_VAR}"


def test_orchestrator_env_file_is_second(tmp_path):
    (tmp_path / "orchestrator.env").write_text(
        f"# a comment\n\n{HF_TOKEN_ENV_VAR}={FAKE_KEY}\nOTHER=x\n"
    )
    (tmp_path / "hf-cli-token").write_text("from-cli-store\n")

    resolved = resolve_huggingface_api_key()

    assert resolved.key == FAKE_KEY
    assert "orchestrator.env" in resolved.source


def test_hf_cli_token_store_is_the_fallback(tmp_path):
    """`hf auth login` is how HF users actually get a token; share that copy."""
    (tmp_path / "hf-cli-token").write_text(f"{FAKE_KEY}\n")

    resolved = resolve_huggingface_api_key()

    assert resolved.key == FAKE_KEY
    assert "hf-cli-token" in resolved.source


def test_quoted_env_file_values_are_unwrapped(tmp_path):
    (tmp_path / "orchestrator.env").write_text(f'{HF_TOKEN_ENV_VAR}="{FAKE_KEY}"\n')
    assert resolve_huggingface_api_key().key == FAKE_KEY


def test_cli_store_tolerates_surrounding_whitespace(tmp_path):
    (tmp_path / "hf-cli-token").write_text(f"  {FAKE_KEY}  \n")
    assert resolve_huggingface_api_key().key == FAKE_KEY


def test_blank_environment_value_is_not_a_credential(monkeypatch, tmp_path):
    """An exported-but-empty variable must not shadow a real stored key."""
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "   ")
    (tmp_path / "hf-cli-token").write_text(f"{FAKE_KEY}\n")
    assert resolve_huggingface_api_key().key == FAKE_KEY


def test_empty_cli_store_is_not_a_credential(tmp_path):
    (tmp_path / "hf-cli-token").write_text("\n")
    assert resolve_huggingface_api_key(required=False) is None


def test_missing_credential_raises_with_actionable_guidance():
    with pytest.raises(HuggingFaceCredentialError) as excinfo:
        resolve_huggingface_api_key()
    message = str(excinfo.value)
    assert HF_TOKEN_ENV_VAR in message
    assert "huggingface.co/settings/tokens" in message, (
        "the error must say where to get a token"
    )


def test_optional_resolution_returns_none_instead_of_raising():
    assert resolve_huggingface_api_key(required=False) is None


def test_unreadable_env_file_does_not_crash_resolution(monkeypatch, tmp_path):
    """A store that cannot be read falls through to the next source."""
    env_file = tmp_path / "orchestrator.env"
    env_file.write_text(f"{HF_TOKEN_ENV_VAR}={FAKE_KEY}\n")
    monkeypatch.setattr(
        "orchestrator.models.huggingface_credentials._ORCHESTRATOR_ENV_FILE",
        tmp_path / "missing-dir" / "orchestrator.env",
    )
    (tmp_path / "hf-cli-token").write_text(f"{FAKE_KEY}\n")
    assert resolve_huggingface_api_key().key == FAKE_KEY


# ---------------------------------------------------------------------------
# Secret hygiene -- the token must never be disclosed
# ---------------------------------------------------------------------------

def test_repr_does_not_leak_the_key(tmp_path):
    """A repr lands in tracebacks and pytest diffs, so it must be masked."""
    (tmp_path / "hf-cli-token").write_text(f"{FAKE_KEY}\n")
    credential = resolve_huggingface_api_key()
    assert FAKE_KEY not in repr(credential)


def test_resolution_does_not_log_the_key(tmp_path, caplog):
    (tmp_path / "hf-cli-token").write_text(f"{FAKE_KEY}\n")
    with caplog.at_level(logging.DEBUG):
        resolve_huggingface_api_key()
    assert FAKE_KEY not in caplog.text


def test_error_message_does_not_echo_a_partial_key(monkeypatch):
    """Even a rejected value must not be quoted back into the message."""
    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "")
    with pytest.raises(HuggingFaceCredentialError) as excinfo:
        resolve_huggingface_api_key()
    assert FAKE_KEY not in str(excinfo.value)
