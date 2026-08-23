"""Shared hermetic fixtures for the sherpa test suite (issue #492 MVP)."""

from __future__ import annotations

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
