"""Acceptance suite: the three #492 demonstrations through the public API.

Reduced matrix (1 seen + 1 held-out repair variant) keeps this hermetic and
fast; `python -m sherpa.benchmarks.harness` runs the full preregistered
matrix. Everything here is real: SQLite WAL, SIGKILL fault injection, pytest
subprocesses, FTS5 retrieval.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sherpa.benchmarks.harness import run_all

pytestmark = [pytest.mark.e2e]


def test_acceptance_scenarios_meet_gates(tmp_path: Path) -> None:
    out = run_all(tmp_path / "bench", b_seeds=[11], b_heldout=[401])
    suite = out["suite"]

    assert suite["scenario_a"]["killed_by_sigkill"] is True
    assert suite["scenario_a"]["exactly_once_effects"] is True
    assert suite["scenario_a"]["projection_equivalent"] is True

    assert suite["decomposition_decisions"] >= 50
    assert suite["claimed_atomic_admissions"] >= 30
    assert suite["task_success_rate"] == 1.0
    assert suite["m_upper_bound_max"] < 1.0
    assert suite["scenario_c"]["needle_recall"] >= 0.95
    assert suite["scenario_c"]["supported_claims"] == suite["scenario_c"]["claims"]
    assert "GO" in out["go_no_go"]
