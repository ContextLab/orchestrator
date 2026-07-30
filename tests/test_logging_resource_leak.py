"""Regression test: quality logging setup must not leak threads.

`configure_quality_logging` installs handlers on `logging.getLogger(
"orchestrator.quality")`, which returns the *same* logger object every call.
It previously appended a fresh handler set on each invocation without closing
the old one. `AsyncQualityHandler` starts three threads per instance (two
workers and a batch processor), so repeated setup leaked threads without
bound.

That is not a theoretical concern: it wedged a full test run, whose fault dump
showed ~900 live `_worker_loop` / `_batch_processor_loop` threads.
"""

import threading

import pytest

pytestmark = pytest.mark.unit


def test_repeated_setup_does_not_leak_threads(tmp_path, monkeypatch):
    from orchestrator.quality.logging import configure_quality_logging

    monkeypatch.chdir(tmp_path)
    baseline = threading.active_count()

    for _ in range(15):
        configure_quality_logging(log_dir=str(tmp_path / "logs"))

    leaked = threading.active_count() - baseline
    # One live handler set is expected; anything proportional to the loop count
    # means handlers are accumulating again.
    assert leaked <= 6, (
        f"{leaked} threads still alive after 15 setups - handlers are "
        "accumulating instead of being closed and replaced"
    )


def test_repeated_setup_does_not_stack_handlers(tmp_path, monkeypatch):
    import logging

    from orchestrator.quality.logging import configure_quality_logging

    monkeypatch.chdir(tmp_path)
    logger = logging.getLogger("orchestrator.quality")

    configure_quality_logging(log_dir=str(tmp_path / "logs"))
    first = len(logger.handlers)

    for _ in range(5):
        configure_quality_logging(log_dir=str(tmp_path / "logs"))

    assert len(logger.handlers) == first, (
        f"handler count grew from {first} to {len(logger.handlers)}; "
        "each setup must replace the previous handlers, not stack onto them"
    )
