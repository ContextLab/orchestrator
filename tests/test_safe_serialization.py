"""Tests for the pickle deserialization guard.

Unpickling executes code carried in the data, so every read path that consumes
persisted bytes (execution state, cache entries, compressed checkpoints) must
refuse by default and say why.
"""

import pickle

import pytest

from orchestrator.core.safe_serialization import (
    ALLOW_PICKLE_ENV_VAR,
    PickleDisabledError,
    ensure_pickle_allowed,
    pickle_allowed,
)

pytestmark = pytest.mark.unit


def test_pickle_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    assert pickle_allowed() is False
    with pytest.raises(PickleDisabledError) as excinfo:
        ensure_pickle_allowed("some/file.pkl")
    # The message must name the offending source and the way out.
    assert "some/file.pkl" in str(excinfo.value)
    assert ALLOW_PICKLE_ENV_VAR in str(excinfo.value)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_pickle_can_be_opted_into(monkeypatch, value):
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, value)
    assert pickle_allowed() is True
    ensure_pickle_allowed("trusted.pkl")  # must not raise


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_unrecognized_values_do_not_enable_pickle(monkeypatch, value):
    """Anything that is not an explicit yes must leave pickle disabled."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, value)
    assert pickle_allowed() is False


def test_execution_state_refuses_to_load_pickle_by_default(monkeypatch, tmp_path):
    """The state manager's pickle path fails closed on a real file."""
    from orchestrator.execution.state import FileStateManager

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)

    payload = tmp_path / "state.pkl"
    payload.write_bytes(pickle.dumps({"execution_id": "x"}))

    persistence = FileStateManager(state_dir=str(tmp_path))
    with pytest.raises(PickleDisabledError):
        persistence._load_pickle(payload)


def test_execution_state_loads_pickle_when_enabled(monkeypatch, tmp_path):
    from orchestrator.execution.state import FileStateManager

    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")

    payload = tmp_path / "state.pkl"
    payload.write_bytes(pickle.dumps({"execution_id": "x"}))

    persistence = FileStateManager(state_dir=str(tmp_path))
    assert persistence._load_pickle(payload) == {"execution_id": "x"}


def test_json_persistence_is_unaffected(monkeypatch, tmp_path):
    """The default JSON path must keep working with pickle disabled."""
    from orchestrator.execution.state import FileStateManager

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)

    persistence = FileStateManager(state_dir=str(tmp_path))
    target = tmp_path / "state.json"
    persistence._save_json({"execution_id": "x"}, target)
    assert persistence._load_json(target) == {"execution_id": "x"}
