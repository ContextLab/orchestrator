"""How refused and failed pickle deserialization must be handled.

The gate itself (``ORCHESTRATOR_ALLOW_PICKLE``) is covered by
``tests/test_safe_serialization.py``. This file covers what happens *after* the
gate says no, or after ``pickle.load`` blows up, because those two conditions
are not the same thing:

* **Refusal** is a configuration state the operator can correct. The stored
  data is intact. Nothing may be deleted, and the error must say which
  environment variable turns it back on.
* **Corruption** is a damaged or hostile entry. It is unusable, so it is
  evicted and logged.

Two concrete regressions are pinned here:

1. ``PerformanceOptimizer._decompress_state`` used to ``return compressed_data``
   from its ``except`` block, handing the caller the undeserialized,
   attacker-controlled blob as if it were restored checkpoint state.
2. ``DiskCache.get``/``DiskCache.cleanup_expired`` used one ``except Exception``
   for both conditions, so a policy refusal silently *deleted* every entry it
   could not read.

Everything below writes real bytes to real files and toggles the real
environment variable. No mocks.
"""

import gzip
import json
import pickle

import pytest

from orchestrator.checkpointing.performance_optimizer import (
    CompressionMethod,
    PerformanceOptimizer,
    StateDecompressionError,
)
from orchestrator.core.cache import CacheEntry, DiskCache, DistributedCache
from orchestrator.core.safe_serialization import (
    ALLOW_PICKLE_ENV_VAR,
    PickleDisabledError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_STATE = {"execution_id": "run-1", "step": "analyze", "values": [1, 2, 3]}


def _hostile_pickle_bytes(tmp_path, name="payload.pkl"):
    """Write a pickled payload to disk and read it back as real bytes.

    Going through the filesystem keeps the test honest: these are bytes that
    arrived from storage, exactly like a checkpoint blob would.
    """
    target = tmp_path / name
    target.write_bytes(pickle.dumps(SAMPLE_STATE, protocol=pickle.HIGHEST_PROTOCOL))
    return target.read_bytes()


@pytest.fixture
async def make_optimizer():
    """Build real PerformanceOptimizers and shut them down afterwards.

    ``langgraph_manager`` is genuinely unused by the compression code path
    under test, and ``shutdown()`` cancels the background cleanup task so no
    task is left dangling.
    """
    created = []

    def _make(method: CompressionMethod) -> PerformanceOptimizer:
        optimizer = PerformanceOptimizer(
            langgraph_manager=None,
            compression_method=method,
            performance_monitoring=False,
        )
        created.append(optimizer)
        return optimizer

    yield _make

    for optimizer in created:
        await optimizer.shutdown()


@pytest.fixture
def disk_cache(tmp_path):
    """A real DiskCache rooted in a real temporary directory."""
    return DiskCache(cache_dir=str(tmp_path / "diskcache"), max_size=100)


async def _store(cache: DiskCache, key: str, value) -> None:
    """Write an entry. Writing is never gated -- only loading is."""
    assert await cache.set(key, value) is True, f"failed to store {key!r}"


# ---------------------------------------------------------------------------
# Defect 1: the checkpoint optimizer must never hand back raw bytes
# ---------------------------------------------------------------------------


async def test_refused_pickle_checkpoint_raises_and_names_the_env_var(
    monkeypatch, tmp_path, make_optimizer
):
    """A refused checkpoint payload must not come back as raw bytes."""
    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    payload = _hostile_pickle_bytes(tmp_path)
    optimizer = make_optimizer(CompressionMethod.PICKLE)

    with pytest.raises(PickleDisabledError) as excinfo:
        await optimizer._decompress_state(payload)

    message = str(excinfo.value)
    print(f"refusal message: {message}")
    assert ALLOW_PICKLE_ENV_VAR in message, (
        "the refusal must name the environment variable that re-enables it; "
        f"got: {message}"
    )


async def test_refused_pickle_checkpoint_never_returns_the_blob(
    monkeypatch, tmp_path, make_optimizer
):
    """Pin the exact regression: no code path may return ``compressed_data``."""
    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    payload = _hostile_pickle_bytes(tmp_path)
    optimizer = make_optimizer(CompressionMethod.PICKLE)

    result = sentinel = object()
    try:
        result = await optimizer._decompress_state(payload)
    except PickleDisabledError:
        pass

    assert result is sentinel, (
        "_decompress_state returned instead of raising; it produced "
        f"{type(result).__name__}: {result!r}"
    )


async def test_unreadable_checkpoint_raises_instead_of_returning_bytes(
    monkeypatch, make_optimizer
):
    """A payload that simply fails to decompress must also fail loudly."""
    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    optimizer = make_optimizer(CompressionMethod.GZIP)
    garbage = b"\x00\x01not-gzip-at-all\xff\xfe"

    with pytest.raises(StateDecompressionError) as excinfo:
        await optimizer._decompress_state(garbage)

    message = str(excinfo.value)
    print(f"decompression failure message: {message}")
    assert "gzip" in message, f"the failure should name the method tried: {message}"
    assert str(len(garbage)) in message, (
        f"the failure should report the payload size: {message}"
    )


async def test_truncated_pickle_raises_when_pickle_is_enabled(
    monkeypatch, tmp_path, make_optimizer
):
    """Enabled + damaged is corruption, not refusal, and still never returns bytes."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    full = _hostile_pickle_bytes(tmp_path)
    truncated = full[: len(full) // 2]
    optimizer = make_optimizer(CompressionMethod.PICKLE)

    with pytest.raises(StateDecompressionError):
        await optimizer._decompress_state(truncated)


async def test_enabled_pickle_checkpoint_round_trips(
    monkeypatch, tmp_path, make_optimizer
):
    """With the opt-in set, the real payload really does come back."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    payload = _hostile_pickle_bytes(tmp_path)
    optimizer = make_optimizer(CompressionMethod.PICKLE)

    assert await optimizer._decompress_state(payload) == SAMPLE_STATE


async def test_gzip_checkpoints_are_unaffected_by_the_pickle_gate(
    monkeypatch, tmp_path, make_optimizer
):
    """The non-pickle formats must keep working with pickle disabled."""
    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    blob = tmp_path / "state.gz"
    blob.write_bytes(gzip.compress(json.dumps(SAMPLE_STATE).encode("utf-8")))
    optimizer = make_optimizer(CompressionMethod.GZIP)

    assert await optimizer._decompress_state(blob.read_bytes()) == SAMPLE_STATE


# ---------------------------------------------------------------------------
# Defect 2: refusal and corruption must not share a handler in the cache
# ---------------------------------------------------------------------------


async def test_refusal_does_not_evict_the_entry(monkeypatch, disk_cache):
    """A policy refusal must leave the index and the file completely alone."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    await _store(disk_cache, "keep-me", {"answer": 42})

    file_path = disk_cache._get_file_path("keep-me")
    index_before = dict(disk_cache._index)

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    with pytest.raises(PickleDisabledError) as excinfo:
        await disk_cache.get("keep-me")

    message = str(excinfo.value)
    print(f"cache refusal message: {message}")
    assert ALLOW_PICKLE_ENV_VAR in message, f"not actionable: {message}"
    assert file_path in message, f"the refusal should name the file: {message}"
    assert disk_cache._index == index_before, (
        "a refusal mutated the index: "
        f"{index_before} -> {disk_cache._index}"
    )
    import os

    assert os.path.exists(file_path), "a refusal deleted the cache file"


async def test_data_survives_a_refusal_and_is_readable_once_re_enabled(
    monkeypatch, disk_cache
):
    """The whole point: refusing must not destroy recoverable data."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    await _store(disk_cache, "survivor", {"answer": 42})

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    with pytest.raises(PickleDisabledError):
        await disk_cache.get("survivor")

    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    entry = await disk_cache.get("survivor")
    assert entry is not None, "the entry was destroyed by the earlier refusal"
    assert entry.value == {"answer": 42}


async def test_corruption_evicts_the_entry(monkeypatch, disk_cache):
    """A damaged file is unusable, so it is dropped from index and disk."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    await _store(disk_cache, "broken", {"answer": 42})

    file_path = disk_cache._get_file_path("broken")
    with open(file_path, "wb") as handle:
        handle.write(b"this is not a pickle stream")

    assert await disk_cache.get("broken") is None
    assert "broken" not in disk_cache._index, "corrupt entry stayed in the index"

    import os

    assert not os.path.exists(file_path), "corrupt file was left on disk"


async def test_a_pickle_of_the_wrong_type_is_treated_as_corruption(
    monkeypatch, disk_cache
):
    """A well-formed pickle of hostile bytes must never be returned as an entry."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    await _store(disk_cache, "wrong-type", {"answer": 42})

    file_path = disk_cache._get_file_path("wrong-type")
    with open(file_path, "wb") as handle:
        pickle.dump(b"raw attacker bytes", handle)

    result = await disk_cache.get("wrong-type")
    assert result is None, (
        f"cache returned a non-CacheEntry object: {type(result).__name__}: {result!r}"
    )
    assert not isinstance(result, bytes), "cache returned raw bytes"
    assert "wrong-type" not in disk_cache._index


async def test_cleanup_expired_refuses_without_deleting_anything(
    monkeypatch, disk_cache
):
    """Refusal must not make every entry look expired."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    for key in ("a", "b", "c"):
        await _store(disk_cache, key, {"key": key})

    index_before = dict(disk_cache._index)

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    with pytest.raises(PickleDisabledError) as excinfo:
        await disk_cache.cleanup_expired()

    assert ALLOW_PICKLE_ENV_VAR in str(excinfo.value)
    assert disk_cache._index == index_before, (
        f"cleanup_expired deleted entries on refusal: "
        f"{sorted(index_before)} -> {sorted(disk_cache._index)}"
    )

    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    for key in ("a", "b", "c"):
        entry = await disk_cache.get(key)
        assert entry is not None and entry.value == {"key": key}


async def test_cleanup_expired_removes_corrupt_and_expired_entries(
    monkeypatch, disk_cache
):
    """With pickle enabled, cleanup evicts what is actually unusable."""
    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    await _store(disk_cache, "good", {"ok": True})
    await _store(disk_cache, "corrupt", {"ok": False})

    expired = CacheEntry(key="expired", value={"ok": False}, ttl=1)
    expired.created_at -= 3600  # a real timestamp in the past
    assert await disk_cache.set_entry(expired) is True

    with open(disk_cache._get_file_path("corrupt"), "wb") as handle:
        handle.write(b"not a pickle")

    await disk_cache.cleanup_expired()

    assert sorted(disk_cache._index) == ["good"], (
        f"unexpected surviving keys: {sorted(disk_cache._index)}"
    )
    assert (await disk_cache.get("good")).value == {"ok": True}


async def test_distributed_cache_reports_refusal_instead_of_a_miss(
    monkeypatch, tmp_path
):
    """A refusal is not a cache miss and must not be reported as ``None``."""
    cache = DistributedCache(cache_dir=str(tmp_path / "distributed"))

    monkeypatch.setenv(ALLOW_PICKLE_ENV_VAR, "1")
    assert await cache.disk_cache.set("dkey", {"answer": 42}) is True

    monkeypatch.delenv(ALLOW_PICKLE_ENV_VAR, raising=False)
    with pytest.raises(PickleDisabledError) as excinfo:
        await cache.get("dkey")

    assert ALLOW_PICKLE_ENV_VAR in str(excinfo.value)
    assert "dkey" in cache.disk_cache._index, "refusal evicted the disk entry"
