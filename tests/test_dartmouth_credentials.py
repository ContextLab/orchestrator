"""Credential resolution and secret-hygiene for the Dartmouth Chat adapter.

The key is a real secret shared with a sibling project. The tests that matter
most here are the ones asserting it never reaches a log line, a repr, or an
exception message -- a credential that leaks into a traceback ends up in CI
logs and issue reports.
"""

import logging

import pytest

from orchestrator.models.dartmouth_credentials import (
    DARTMOUTH_KEY_ENV_VAR,
    DartmouthCredentialError,
    ResolvedCredential,
    mask_key,
    resolve_dartmouth_api_key,
)

pytestmark = pytest.mark.unit

FAKE_KEY = "sk-0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Never read the developer's real credential stores during tests."""
    monkeypatch.delenv(DARTMOUTH_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "orchestrator.models.dartmouth_credentials._ORCHESTRATOR_ENV_FILE",
        tmp_path / "orchestrator.env",
    )
    monkeypatch.setattr(
        "orchestrator.models.dartmouth_credentials._LLMXIVE_CREDENTIALS_FILE",
        tmp_path / "llmxive.toml",
    )


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_environment_variable_wins(monkeypatch, tmp_path):
    (tmp_path / "orchestrator.env").write_text(
        f"{DARTMOUTH_KEY_ENV_VAR}=from-env-file\n"
    )
    monkeypatch.setenv(DARTMOUTH_KEY_ENV_VAR, FAKE_KEY)

    resolved = resolve_dartmouth_api_key()

    assert resolved.key == FAKE_KEY
    assert resolved.source == f"${DARTMOUTH_KEY_ENV_VAR}"


def test_orchestrator_env_file_is_second(tmp_path):
    (tmp_path / "orchestrator.env").write_text(
        f"# a comment\n\n{DARTMOUTH_KEY_ENV_VAR}={FAKE_KEY}\nOTHER=x\n"
    )
    (tmp_path / "llmxive.toml").write_text(
        'dartmouth_chat_api_key = "from-llmxive"\n'
    )

    resolved = resolve_dartmouth_api_key()

    assert resolved.key == FAKE_KEY
    assert "orchestrator.env" in resolved.source


def test_llmxive_store_is_the_fallback(tmp_path):
    """Sharing one token beats a second copy that rotates separately."""
    (tmp_path / "llmxive.toml").write_text(
        f'dartmouth_chat_api_key = "{FAKE_KEY}"\n'
        'semantic_scholar_api_key = "unrelated"\n'
    )

    resolved = resolve_dartmouth_api_key()

    assert resolved.key == FAKE_KEY
    assert "llmxive" in resolved.source


def test_quoted_values_are_unwrapped(tmp_path):
    (tmp_path / "orchestrator.env").write_text(
        f'{DARTMOUTH_KEY_ENV_VAR}="{FAKE_KEY}"\n'
    )
    assert resolve_dartmouth_api_key().key == FAKE_KEY


def test_blank_environment_value_is_not_a_credential(monkeypatch, tmp_path):
    """An exported-but-empty variable must not shadow a real stored key."""
    monkeypatch.setenv(DARTMOUTH_KEY_ENV_VAR, "   ")
    (tmp_path / "llmxive.toml").write_text(
        f'dartmouth_chat_api_key = "{FAKE_KEY}"\n'
    )
    assert resolve_dartmouth_api_key().key == FAKE_KEY


def test_missing_credential_raises_with_actionable_guidance():
    with pytest.raises(DartmouthCredentialError) as excinfo:
        resolve_dartmouth_api_key()
    message = str(excinfo.value)
    assert DARTMOUTH_KEY_ENV_VAR in message
    assert "chat.dartmouth.edu" in message, "the error must say where to get a key"


def test_optional_resolution_returns_none_instead_of_raising():
    assert resolve_dartmouth_api_key(required=False) is None


def test_malformed_toml_does_not_crash_resolution(tmp_path):
    """A corrupt sibling store must not break us; fall through to the error."""
    (tmp_path / "llmxive.toml").write_text("this is not [valid toml\n")
    assert resolve_dartmouth_api_key(required=False) is None


# ---------------------------------------------------------------------------
# Secret hygiene -- the key must never be disclosed
# ---------------------------------------------------------------------------

def test_mask_key_never_reveals_the_middle():
    masked = mask_key(FAKE_KEY)
    assert FAKE_KEY not in masked
    assert masked.startswith("sk-012")
    assert masked.endswith(FAKE_KEY[-3:])


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "<unset>"), ("", "<unset>"), ("short", "<set>")],
)
def test_mask_key_handles_absent_and_short_values(value, expected):
    assert mask_key(value) == expected


def test_repr_does_not_leak_the_key():
    """A repr lands in tracebacks and pytest diffs, so it must be masked."""
    credential = ResolvedCredential(FAKE_KEY, "$TEST")
    assert FAKE_KEY not in repr(credential)


def test_resolution_does_not_log_the_key(tmp_path, caplog):
    (tmp_path / "llmxive.toml").write_text(
        f'dartmouth_chat_api_key = "{FAKE_KEY}"\n'
    )
    with caplog.at_level(logging.DEBUG):
        resolve_dartmouth_api_key()
    assert FAKE_KEY not in caplog.text


def test_error_message_does_not_echo_a_partial_key(monkeypatch):
    """Even a rejected value must not be quoted back into the message."""
    monkeypatch.setenv(DARTMOUTH_KEY_ENV_VAR, "")
    with pytest.raises(DartmouthCredentialError) as excinfo:
        resolve_dartmouth_api_key()
    assert FAKE_KEY not in str(excinfo.value)
