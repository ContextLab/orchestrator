"""Shared hermetic fixtures for the sherpa test suite (issue #492 MVP)."""

from __future__ import annotations

import sys

import pytest

if sys.version_info < (3, 11):
    # The repo declares requires-python >= 3.11; on older interpreters Pydantic
    # cannot evaluate the IR's PEP 604 annotations and collection dies with
    # cryptic TypeError noise. Fail visibly, with the working invocation.
    pytest.exit(
        f"\nsherpa tests require Python >= 3.11 (you are running {sys.version.split()[0]}).\n"
        "Run them with the repository virtualenv instead:\n"
        "  .venv/bin/python -m pytest tests/sherpa -q\n",
        returncode=1,
    )

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[object]:
    """Real SQLite store in WAL mode backed by a real content-addressed blob dir."""
    from sherpa.store import Store

    s = Store(tmp_path / "runs.db")
    yield s
    s.close()


@pytest.fixture()
def blobs(tmp_path: Path) -> object:
    from sherpa.store import BlobStore

    return BlobStore(tmp_path / "blobs")


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws
