"""Hermetic tests for the durable store (real SQLite WAL + real blob files)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sherpa.events import Event
from sherpa.store import BlobStore, Store, content_hash

pytestmark = [pytest.mark.unit]


def _ev(run_id: str, kind: str, node_key: str | None = None, **payload: object) -> Event:
    return Event(kind=kind, run_id=run_id, node_key=node_key, payload=payload)


def _scripted(store: Store, run_id: str) -> None:
    """A sequence touching every projection-bearing event kind."""
    store.append(_ev(run_id, "run_started", problem_sha="abc", status="running"))
    store.append(_ev(run_id, "node_created", "a", state="pending", depth=0))
    store.append(_ev(run_id, "node_created", "b", state="pending", depth=1))
    store.cas_node_state(run_id, "a", "pending", "running", owner_session="s1")
    store.add_usage(run_id, tokens=100.0, nodes=1.0)
    store.enqueue_message(run_id, "b", {"kind": "scope_change"}, sender="root")
    store.take_messages(run_id, "b")
    store.cas_node_state(run_id, "a", "running", "completed", owner_session="s1")
    store.set_run_status(run_id, "completed")


class TestEventLog:
    def test_seq_monotonic_and_roundtrip(self, store: Store) -> None:
        e1 = store.append(_ev("r1", "run_started"))
        e2 = store.append(_ev("r1", "node_created", "a"))
        assert e2.seq == e1.seq + 1
        back = store.events(run_id="r1")
        assert [e.kind for e in back] == ["run_started", "node_created"]
        assert isinstance(back[0].payload, dict)

    def test_unknown_kind_rejected(self, store: Store) -> None:
        with pytest.raises(ValueError, match="unknown event kind"):
            store.append(_ev("r1", "not_a_kind"))

    def test_kind_filter(self, store: Store) -> None:
        _scripted(store, "r1")
        kinds = {e.kind for e in store.events(run_id="r1", kinds=["usage_checkpoint"])}
        assert kinds == {"usage_checkpoint"}


class TestProjections:
    def test_projection_equals_replay(self, store: Store) -> None:
        _scripted(store, "r1")
        live = store.projection("r1")
        replayed = store.replay_projection("r1")
        assert replayed["status"] == live["status"] == "completed"
        assert replayed["nodes"]["a"]["state"] == live["nodes"]["a"]["state"] == "completed"
        assert set(replayed["nodes"]) == set(live["nodes"])
        assert replayed["messages_pending"] == live["messages_pending"] == 0

    def test_terminal_states_recorded(self, store: Store) -> None:
        _scripted(store, "r1")
        store.set_run_status("r1", "budget_exhausted")
        assert store.projection("r1")["status"] == "budget_exhausted"

    def test_invalid_run_status_rejected(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.set_run_status("rX", "sort-of-done")


class TestCasAndLeases:
    def test_cas_success_then_stale_failure(self, store: Store) -> None:
        store.create_run("r1", problem_sha="x")
        store.upsert_node("r1", "a")
        assert store.cas_node_state("r1", "a", "pending", "leased")
        assert not store.cas_node_state("r1", "a", "pending", "running")

    def test_lease_ttl_expiry(self, store: Store) -> None:
        assert store.acquire_lease("r1", "a", "s1", ttl_s=50)
        assert not store.acquire_lease("r1", "a", "s2", ttl_s=50)
        assert store.expired_leases(now=time.time() + 100) == [("r1", "a", "s1")]
        assert store.acquire_lease("r1", "a", "s2", ttl_s=10, now=time.time() + 101)

    def test_release_only_by_owner(self, store: Store) -> None:
        store.acquire_lease("r1", "a", "s1")
        assert not store.release_lease("r1", "a", "s2")
        assert store.release_lease("r1", "a", "s1")


import time  # noqa: E402 - used by lease tests above


class TestMessages:
    def test_take_once_semantics(self, store: Store) -> None:
        store.create_run("r1", problem_sha="x")
        store.upsert_node("r1", "b")
        store.enqueue_message("r1", "b", {"kind": "scope_change"}, sender="root")
        first = store.take_messages("r1", "b")
        second = store.take_messages("r1", "b")
        assert len(first) == 1 and first[0]["kind"] == "scope_change"
        assert second == []
        assert store.projection("r1")["messages_pending"] == 0


class TestUsage:
    def test_accumulates(self, store: Store) -> None:
        store.create_run("r1", problem_sha="x")
        store.add_usage("r1", tokens=10)
        store.add_usage("r1", tokens=5, attempts=1)
        u = store.usage("r1")
        assert u["tokens"] == 15 and u["attempts"] == 1


class TestSolutionCache:
    def test_budget_exhaustion_never_negative_evidence(self, store: Store) -> None:
        with pytest.raises(ValueError):
            store.cache_put("sig1", {"status_class": "failed", "inconclusive": True})
        store.cache_put("sig1", {"status_class": "inconclusive", "inconclusive": True})
        store.cache_put("sig2", {"status_class": "failed", "outcome": "wrong answer"})
        assert store.cache_get("sig1")["status_class"] == "inconclusive"
        assert store.cache_get("sig2")["status_class"] == "failed"
        assert store.cache_get("missing") is None


class TestFTS:
    def _seed(self, store: Store) -> None:
        docs = {
            "d1": "the launch code is ZEBRA-77 hidden in plain text",
            "d2": "totally unrelated quarterly earnings grew steadily",
            "d3": "meeting notes about the coffee machine repair",
            "d4": "another distractor paragraph about weather patterns",
            "d5": "the launch code was mentioned again by the team",
        }
        for i, (doc_id, text) in enumerate(docs.items()):
            store.index_chunk(
                {
                    "chunk_id": f"{doc_id}:0",
                    "doc_id": doc_id,
                    "text": text,
                    "ordinal": i,
                    "start": 0,
                    "end": len(text),
                    "sha": content_hash(text),
                }
            )

    def test_needle_ranks_above_distractors(self, store: Store) -> None:
        self._seed(store)
        hits = store.fts_search("launch code", k=3)
        assert hits[0]["doc_id"] in ("d1", "d5")
        assert all(h["doc_id"] in ("d1", "d5") for h in hits)

    def test_doc_prefix_filter(self, store: Store) -> None:
        self._seed(store)
        hits = store.fts_search("launch", k=5, doc_prefix="d1")
        assert hits and {h["doc_id"] for h in hits} == {"d1"}


class TestBlobs:
    def test_roundtrip_layout(self, blobs: BlobStore, tmp_path: Path) -> None:
        sha = blobs.put_text("hello evidence")
        assert blobs.exists(sha)
        assert blobs.get_text(sha) == "hello evidence"
        assert blobs.path(sha).parent.name == sha[:2]
        with pytest.raises(KeyError):
            blobs.get_bytes("ff" * 32)


class TestFindings:
    def test_roundtrip_and_dispositions(self, store: Store) -> None:
        store.add_finding({"id": "find_1", "subject": "plan_sha", "criterion": "c", "blocking": True})
        assert len(store.findings(subject="plan_sha")) == 1
        assert store.set_finding_disposition("find_1", "fixed", "patched")
        f = store.findings()[0]
        assert f["disposition"] == "fixed"
        assert not store.set_finding_disposition("missing", "fixed")
        with pytest.raises(ValueError):
            store.set_finding_disposition("find_1", "whatever")


class TestConcurrency:
    def test_wal_reader_during_writer_transaction(self, tmp_path: Path) -> None:
        s1 = Store(tmp_path / "runs.db")
        s2 = Store(tmp_path / "runs.db")
        s1.create_run("r1", problem_sha="x")
        s1.append(Event(kind="run_started", run_id="r2", payload={}))
        # reader sees committed rows while writer connection stays open
        ids = {e.run_id for e in s2.events()}
        assert {"r1", "r2"} <= ids
        s1.close()
        s2.close()
