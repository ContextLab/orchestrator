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


# --------------------------------------------------------------------------------------
# Regression coverage for defects D1-D7 (store hardening).
# --------------------------------------------------------------------------------------

import hashlib  # noqa: E402 - grouped with the regression suite below
import re  # noqa: E402
import sqlite3  # noqa: E402


class TestContentAddressing:
    """D1: the content address must be a *full* SHA-256, not a prefix of one."""

    def test_digest_is_full_sha256_of_payload(self, blobs: BlobStore) -> None:
        payload = b"sherpa content addressing fixture\n"
        expected = hashlib.sha256(payload).hexdigest()
        assert len(expected) == 64  # sanity: the oracle itself is a full digest

        sha = blobs.put_bytes(payload)
        assert sha == expected, f"blob digest {sha!r} != sha256 {expected!r}"
        assert content_hash(payload) == expected
        assert content_hash(payload.decode("utf-8")) == expected
        assert len(sha) == 64, f"digest must be 64 hex chars, got {len(sha)}: {sha!r}"
        assert re.fullmatch(r"[0-9a-f]{64}", sha), f"digest not lowercase hex: {sha!r}"

    def test_blob_path_carries_the_whole_digest(self, blobs: BlobStore) -> None:
        sha = blobs.put_text("evidence artifact")
        p = blobs.path(sha)
        assert p.parent.name == sha[:2]
        assert p.name == sha
        assert len(p.name) == 64, f"blob filename truncated: {p.name!r}"

    def test_distinct_payloads_never_share_a_path(self, blobs: BlobStore) -> None:
        a = blobs.put_bytes(b"payload-A")
        b = blobs.put_bytes(b"payload-B")
        assert a != b
        assert blobs.path(a) != blobs.path(b)
        assert blobs.get_bytes(a) == b"payload-A"
        assert blobs.get_bytes(b) == b"payload-B"

    def test_identical_payloads_deduplicate(self, blobs: BlobStore) -> None:
        first = blobs.put_bytes(b"same bytes")
        second = blobs.put_bytes(b"same bytes")
        assert first == second
        assert blobs.path(first) == blobs.path(second)
        objects = list((blobs.root / "objects").rglob("*"))
        files = [p for p in objects if p.is_file()]
        assert len(files) == 1, f"deduplication failed, wrote {files}"


class TestFindingsProjection:
    """D2/D3: findings are scoped per run and rebuildable from the event log."""

    def test_run_without_findings_reports_none(self, store: Store) -> None:
        store.create_run("r1", problem_sha="x")
        store.create_run("r2", problem_sha="y")
        store.add_finding(
            {"id": "f_r2", "run_id": "r2", "subject": "r2/plan", "criterion": "c", "blocking": True}
        )
        assert store.projection("r2")["findings"] == ["f_r2"]
        assert store.projection("r1")["findings"] == [], (
            "a run with no findings must not inherit other runs' findings"
        )

    def test_replay_rebuilds_findings(self, store: Store) -> None:
        _scripted(store, "r1")
        store.add_finding(
            {"id": "f_2", "run_id": "r1", "subject": "r1/plan", "criterion": "d"}
        )
        store.add_finding(
            {"id": "f_1", "run_id": "r1", "subject": "r1/plan", "criterion": "c", "blocking": True}
        )
        store.add_finding(
            {"id": "f_other", "run_id": "r9", "subject": "r9/plan", "criterion": "c"}
        )
        live = store.projection("r1")
        assert live["findings"] == ["f_1", "f_2"]
        assert store.replay_projection("r1") == live, (
            "projection != replay_projection; the ADR claims every structure is "
            "independently rebuildable by replay"
        )
        assert store.replay_projection("r9")["findings"] == ["f_other"]

    def test_replay_deduplicates_repeated_finding_events(self, store: Store) -> None:
        store.create_run("r1", problem_sha="x")
        payload = {"id": "f_1", "run_id": "r1", "subject": "r1/plan", "criterion": "c"}
        store.add_finding(payload)
        store.add_finding(payload)  # INSERT OR IGNORE in the table; log gets two events
        assert store.projection("r1")["findings"] == ["f_1"]
        assert store.replay_projection("r1")["findings"] == ["f_1"]


class TestFTSRobustness:
    """D4: no plan-supplied query may leak a sqlite3.OperationalError to the caller."""

    QUERIES = [
        "foo:bar",
        "col:*",
        '"unterminated',
        'ZEBRA-77',
        "NEAR",
        "NEAR(launch code, 3)",
        "AND",
        "OR",
        "NOT",
        "launch AND code",
        "launch OR code",
        "launch NOT code",
        "",
        "   ",
        "*",
        "launch*",
        "(",
        ")()",
        "-",
        "^launch",
        '""',
        'say "hi"',
        "café",
        "☕",
        "naïve résumé",
        "日本語",
        "a" * 500,
    ]

    def _seed(self, store: Store) -> None:
        docs = {
            "d1": "the launch code is ZEBRA-77 hidden in plain text",
            "d2": "foo bar baz appears here as a colon-free phrase",
            "d3": "café naïve résumé unicode sample 日本語 text",
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

    @pytest.mark.parametrize("query", QUERIES)
    def test_hostile_queries_never_raise(self, store: Store, query: str) -> None:
        self._seed(store)
        try:
            hits = store.fts_search(query, k=3)
        except sqlite3.OperationalError as exc:  # pragma: no cover - the defect
            pytest.fail(f"query {query!r} leaked sqlite error: {exc}")
        assert isinstance(hits, list)
        assert all(isinstance(h["doc_id"], str) for h in hits)

    @pytest.mark.parametrize("query", QUERIES)
    def test_hostile_queries_never_raise_with_doc_prefix(self, store: Store, query: str) -> None:
        self._seed(store)
        try:
            hits = store.fts_search(query, k=3, doc_prefix="d")
        except sqlite3.OperationalError as exc:  # pragma: no cover - the defect
            pytest.fail(f"query {query!r} leaked sqlite error: {exc}")
        assert isinstance(hits, list)

    def test_punctuated_queries_still_retrieve(self, store: Store) -> None:
        self._seed(store)
        assert [h["doc_id"] for h in store.fts_search("ZEBRA-77", k=3)] == ["d1"]
        assert [h["doc_id"] for h in store.fts_search("foo:bar", k=3)] == ["d2"]
        assert [h["doc_id"] for h in store.fts_search("café", k=3)] == ["d3"]
        assert [h["doc_id"] for h in store.fts_search("日本語", k=3)] == ["d3"]

    def test_operator_words_are_literal_terms_not_syntax(self, store: Store) -> None:
        self._seed(store)
        # "launch AND code" must behave as the three literal terms; no doc holds "and".
        assert store.fts_search("launch AND code", k=3) == []
        assert [h["doc_id"] for h in store.fts_search("launch code", k=3)] == ["d1"]

    def test_empty_query_returns_no_hits(self, store: Store) -> None:
        self._seed(store)
        assert store.fts_search("", k=3) == []
        assert store.fts_search("   ", k=3) == []


class TestLeaseOwnership:
    """D5/D6: the owner may resume its own lease; nobody else may."""

    def test_owner_may_reacquire_its_live_lease(self, store: Store) -> None:
        base = time.time()
        assert store.acquire_lease("r1", "a", "s1", ttl_s=50, now=base)
        assert store.acquire_lease("r1", "a", "s1", ttl_s=50, now=base + 1), (
            "the current owner must be able to resume its own live lease"
        )
        assert not store.acquire_lease("r1", "a", "s2", ttl_s=50, now=base + 2)
        assert store.expired_leases(now=base + 2) == []

    def test_owner_reacquire_extends_expiry(self, store: Store) -> None:
        base = time.time()
        assert store.acquire_lease("r1", "a", "s1", ttl_s=10, now=base)
        assert store.acquire_lease("r1", "a", "s1", ttl_s=100, now=base + 5)
        assert store.expired_leases(now=base + 20) == []
        assert store.expired_leases(now=base + 200) == [("r1", "a", "s1")]

    def test_reacquire_still_refuses_a_different_live_session(self, store: Store) -> None:
        base = time.time()
        assert store.acquire_lease("r1", "a", "s1", ttl_s=50, now=base)
        assert store.acquire_lease("r1", "a", "s2", ttl_s=50, now=base + 1) is False

    def test_renew_lease_owner_only(self, store: Store) -> None:
        base = time.time()
        assert store.acquire_lease("r1", "a", "s1", ttl_s=10, now=base)
        assert not store.renew_lease("r1", "a", "s2"), "a non-owner must not renew"
        assert store.renew_lease("r1", "a", "s1", ttl_s=1000)
        assert store.expired_leases(now=time.time() + 100) == []

    def test_renew_lease_unknown_node(self, store: Store) -> None:
        assert not store.renew_lease("r1", "nosuch", "s1")

    def test_renew_after_release_fails(self, store: Store) -> None:
        store.acquire_lease("r1", "a", "s1", ttl_s=10)
        assert store.release_lease("r1", "a", "s1")
        assert not store.renew_lease("r1", "a", "s1")


class TestEventLogIntegrity:
    """D7: causal_seq round-trips, and the log is append-only in fact and in code."""

    def test_causal_seq_round_trip(self, store: Store) -> None:
        cause = store.append(_ev("r1", "run_started"))
        effect = Event(
            kind="node_created", run_id="r1", node_key="a", payload={}, causal_seq=cause.seq
        )
        store.append(effect)
        back = store.events(run_id="r1")
        assert back[0].causal_seq is None
        assert back[1].causal_seq == cause.seq, (
            f"causal_seq dropped on write: read back {back[1].causal_seq!r}"
        )

    def test_causal_chain_survives_a_reopen(self, store: Store, tmp_path: Path) -> None:
        c1 = store.append(_ev("r1", "run_started"))
        c2 = store.append(Event(kind="attempt_started", run_id="r1", causal_seq=c1.seq))
        store.append(Event(kind="attempt_finished", run_id="r1", causal_seq=c2.seq))
        reopened = Store(store.path)
        try:
            chain = [(e.seq, e.causal_seq) for e in reopened.events(run_id="r1")]
        finally:
            reopened.close()
        assert chain == [(c1.seq, None), (c2.seq, c1.seq), (c2.seq + 1, c2.seq)]

    def test_log_prefix_is_immutable_under_further_writes(self, store: Store) -> None:
        _scripted(store, "r1")
        before = [
            (e.seq, e.ts, e.run_id, e.node_key, e.kind, e.payload, e.causal_seq)
            for e in store.events()
        ]
        assert before, "scripted run must have produced events"
        # Exercise every writer path that could conceivably rewrite history.
        store.add_finding({"id": "f_1", "run_id": "r1", "subject": "r1/x", "criterion": "c"})
        store.set_finding_disposition("f_1", "fixed", "patched")
        store.upsert_node("r1", "a", state="running")
        store.cas_node_state("r1", "a", "running", "completed")
        store.acquire_lease("r1", "a", "s1")
        store.release_lease("r1", "a", "s1")
        store.add_usage("r1", tokens=3.0)
        store.set_run_status("r1", "failed", error="boom")
        after = [
            (e.seq, e.ts, e.run_id, e.node_key, e.kind, e.payload, e.causal_seq)
            for e in store.events()
        ]
        assert after[: len(before)] == before, "events table was mutated in place"
        assert len(after) > len(before)

    def test_no_update_or_delete_against_events_in_source(self) -> None:
        import sherpa.store as store_mod

        src_dir = Path(store_mod.__file__).parent
        offenders: list[str] = []
        pattern = re.compile(r"(UPDATE\s+events\b|DELETE\s+FROM\s+events\b)", re.IGNORECASE)
        for py in sorted(src_dir.rglob("*.py")):
            for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{py}:{lineno}: {line.strip()}")
        assert not offenders, "events must be append-only; found:\n" + "\n".join(offenders)
