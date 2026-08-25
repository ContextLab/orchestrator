"""Hermetic tests: admission control, planners, context substrate (#492)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sherpa.admission import AdmissionChecker, check_io
from sherpa.channel import RecordedChannel
from sherpa.capabilities import (
    AuthorityDenied,
    Capability,
    CapabilityContext,
    CapabilityRegistry,
    CapabilitySpec,
    FsReadFile,
    ProbeFailed,
)
from sherpa.context import (
    JOURNAL_KINDS,
    JournalError,
    build_summary,
    chunk_document,
    default_summary_id,
    journal,
    retrieve,
    scoped_snapshot,
    summarize_with_channel,
)
from sherpa.ir import Authority, Budgets, InvokeCapability, Plan, validate_plan
from sherpa.planner import (
    LLMPlanner,
    NoPlanTemplate,
    PlanAuthoringError,
    StubPlanner,
    plan_signature,
)
from sherpa.store import content_hash

pytestmark = [pytest.mark.unit]


def _ctx(store, workspace: Path) -> CapabilityContext:
    return CapabilityContext(
        workspace=workspace,
        store=store,
        run_id="r1",
        node_key="n1",
        channel_factory=lambda: RecordedChannel({"summarizer": ["S"], "llm": []}),
        granted=Authority(fs_read=("**",), fs_write=("**",), subprocess_allow=("**",)),
    )


class _BoomCap(Capability):
    spec = CapabilitySpec(
        name="boom.explode",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )

    def run(self, inputs: dict, ctx: CapabilityContext) -> dict:
        return {}

    def probe(self, ctx: CapabilityContext) -> bytes:
        raise RuntimeError("cannot execute")


class TestCheckIo:
    def test_types_and_required(self) -> None:
        schema = {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}}
        assert check_io({"a": 3}, schema)[0] is True
        ok, errs = check_io({"b": "x"}, schema)
        assert not ok and any("a" in e for e in errs)
        assert check_io(5, {"type": "integer"})[0] is True
        assert not check_io(True, {"type": "integer"})[0]
        assert check_io([1, 2], {"type": "array", "items": {"type": "integer"}})[0] is True


class TestAdmission:
    def _checker(self, store, workspace: Path, registry: CapabilityRegistry | None = None):
        reg = registry or CapabilityRegistry()
        if not reg.names():
            reg.register(FsReadFile())
        return AdmissionChecker(store, reg, store.blob), reg

    def test_happy_path_admits_with_evidence(self, store, workspace: Path) -> None:
        checker, reg = self._checker(store, workspace)
        ctx = _ctx(store, workspace)
        step = InvokeCapability(kind="invoke_capability", id="s1", capability="fs.read_file")
        verdict = checker.check(step, {"path": "x.txt"}, ctx.granted, ctx)
        assert verdict.decision == "admitted"
        assert verdict.probe_ok and verdict.evidence_sha and store.blob.exists(verdict.evidence_sha)
        kinds = [e.kind for e in store.events(run_id="r1")]
        assert kinds.count("admission_checked") == 1

    def test_unknown_capability_escalates(self, store, workspace: Path) -> None:
        checker, _ = self._checker(store, workspace)
        step = InvokeCapability(kind="invoke_capability", id="s1", capability="ghost.op")
        verdict = checker.check(step, {}, Authority(), _ctx(store, workspace))
        assert verdict.decision == "escalate"

    def test_bad_inputs_escalate(self, store, workspace: Path) -> None:
        checker, _ = self._checker(store, workspace)
        ctx = _ctx(store, workspace)
        step = InvokeCapability(kind="invoke_capability", id="s1", capability="fs.read_file")
        verdict = checker.check(step, {"path": 42}, ctx.granted, ctx)
        assert verdict.decision == "escalate" and verdict.io_compatible is False

    def test_missing_authority_escalates(self, store, workspace: Path) -> None:
        checker, _ = self._checker(store, workspace)
        ctx = _ctx(store, workspace)
        step = InvokeCapability(kind="invoke_capability", id="s1", capability="fs.read_file")
        verdict = checker.check(step, {"path": "x"}, Authority(), ctx)
        assert verdict.decision == "escalate"
        with pytest.raises(AuthorityDenied):
            from sherpa.capabilities import run_capability

            run_capability(registry_get(checker, "fs.read_file"), {"path": __file__}, ctx, Authority())

    def test_probe_failure_reclassifies(self, store, workspace: Path) -> None:
        reg = CapabilityRegistry()
        reg.register(_BoomCap())
        checker, _ = self._checker(store, workspace, reg)
        step = InvokeCapability(kind="invoke_capability", id="s1", capability="boom.explode")
        verdict = checker.check(step, {}, Authority(), _ctx(store, workspace))
        assert verdict.decision == "reclassify_decompose"
        assert verdict.probe_ok is False
        ev = [e for e in store.events(run_id="r1") if e.kind == "admission_checked"][0]
        assert ev.payload["decision"] == "reclassify_decompose"


def registry_get(checker, name):  # tiny helper for the authority-denied assertion
    return checker.registry.get(name)


class TestPlanSignature:
    def test_stable_and_sensitive(self) -> None:
        a = plan_signature("g", {}, Authority(), Budgets())
        b = plan_signature("g", {}, Authority(), Budgets())
        c = plan_signature("g2", {}, Authority(), Budgets())
        d = plan_signature("g", {}, Authority(fs_read=("x/",)), Budgets())
        e = plan_signature("g", {}, Authority(), Budgets(max_nodes=1))
        assert a == b
        assert len({a, c, d, e}) == 4


def _plan_dict() -> dict:
    return {
        "id": "p",
        "authority": {},
        "budgets": {"max_fanout": 2},
        "root": [
            {"kind": "invoke_capability", "id": "a", "capability": "fs.read_file",
             "inputs": {"path": "in.txt"}},
            {"kind": "return", "id": "out", "outputs": {}},
        ],
    }


class TestStubPlanner:
    def test_selects_by_requested_capability(self) -> None:
        planner = StubPlanner()
        hints = {"requested_capability": "fs.read_file",
                 "plan_library": [{"match": {"capability": "fs.read_file"}, "plan": _plan_dict()}]}
        plan = planner.author_plan("read it", hints, Authority(), Budgets(), "s")
        assert isinstance(plan, Plan) and plan.notes["authored_by"] == "stub"

    def test_selects_by_pattern(self) -> None:
        planner = StubPlanner()
        hints = {"plan_library": [{"match": {"pattern_contains": "repair"}, "plan": _plan_dict()}]}
        plan = planner.author_plan("please repair the module", hints, Authority(), Budgets(), "s")
        assert plan.id == "p"

    def test_miss_raises(self) -> None:
        planner = StubPlanner()
        with pytest.raises(NoPlanTemplate):
            planner.author_plan("nothing matches", {"plan_library": []}, Authority(), Budgets(), "s")

    def test_library_invalid_plan_raises(self) -> None:
        bad = _plan_dict()
        bad["root"][0]["capability"] = "not_registered"
        planner = StubPlanner(registry_names={"fs.read_file"})
        hints = {"requested_capability": "fs.read_file", "plan_library": [{"match": {"capability": "fs.read_file"}, "plan": bad}]}
        with pytest.raises(PlanAuthoringError):
            planner.author_plan("g", hints, Authority(), Budgets(), "s")


class TestLLMPlanner:
    def _channel(self, responses: list[str]) -> RecordedChannel:
        return RecordedChannel({"planner": responses})

    def test_happy_path_single_call(self) -> None:
        plan_json = json.dumps(_plan_dict())
        ch = self._channel([plan_json])
        planner = LLMPlanner(ch, registry_names={"fs.read_file"})
        plan = planner.author_plan("g", {}, Authority(), Budgets(), "planner")
        assert planner.calls == 1
        assert plan.notes["authored_by"] == "llm"

    def test_repair_round_then_success(self) -> None:
        ch = self._channel(["this is not json", json.dumps(_plan_dict())])
        planner = LLMPlanner(ch, registry_names={"fs.read_file"})
        plan = planner.author_plan("g", {}, Authority(), Budgets(), "planner")
        assert planner.calls == 2

    def test_persistent_failure_raises(self) -> None:
        ch = self._channel(["nope", "still nope"])
        planner = LLMPlanner(ch)
        with pytest.raises(PlanAuthoringError):
            planner.author_plan("g", {}, Authority(), Budgets(), "planner")

    def test_authority_violating_plan_rejected(self) -> None:
        bad = _plan_dict()
        bad["authority"] = {"fs_write": ("/etc/**",)}
        parent = Authority()  # grants nothing; child claims write on /etc
        assert not parent.allows(Plan(**bad).authority)
        ch = self._channel([json.dumps(bad), json.dumps(bad)])
        planner = LLMPlanner(ch, registry_names={"fs.read_file"})
        with pytest.raises(PlanAuthoringError, match="authority"):
            planner.author_plan("g", {}, parent, Budgets(), "planner")
        assert planner.calls == 2


class TestJournal:
    def test_roundtrip_and_kinds(self, store) -> None:
        store.create_run("r1", problem_sha="x")
        journal(store, "r1", "n1", "decision", "chose binary search", refs=["artifact:sha"])
        entries = [e for e in store.events(run_id="r1") if e.kind == "journal_appended"]
        assert entries[0].payload["kind"] == "decision"
        with pytest.raises(JournalError):
            journal(store, "r1", None, "ranting", "nope")

    def test_every_declared_kind_is_actually_accepted(self, store) -> None:
        """JOURNAL_KINDS is a behavioural contract, not a literal to restate."""
        store.create_run("r2", problem_sha="x")
        for kind in JOURNAL_KINDS:
            journal(store, "r2", "n1", kind, f"entry for {kind}")
        recorded = [
            e.payload["kind"] for e in store.events(run_id="r2") if e.kind == "journal_appended"
        ]
        assert recorded == list(JOURNAL_KINDS), (
            f"declared kinds {JOURNAL_KINDS} but journal recorded {recorded}"
        )
        # Anything outside the declared tuple must be refused, not silently stored.
        for bogus in ("thought", "chain_of_thought", "Decision", "", "results"):
            assert bogus not in JOURNAL_KINDS
            with pytest.raises(JournalError):
                journal(store, "r2", "n1", bogus, "nope")
        after = [e for e in store.events(run_id="r2") if e.kind == "journal_appended"]
        assert len(after) == len(JOURNAL_KINDS), "a rejected kind leaked an event into the log"


DOC = (
    "# Alpha\nintro line one which is fairly long to survive the merge threshold easily.\n\n"
    "detail paragraph about alpha internals with enough words to stand alone as a chunk.\n\n"
    "# Beta\nbeta overview sentence that also stretches past the eighty character merge limit now.\n\n"
    "tiny\n\n"
    "closing paragraph carrying the merged tiny fragment forward into a real chunk here."
)


class TestChunking:
    def test_paragraph_offsets_exact(self) -> None:
        chunks = chunk_document("doc", DOC)
        # DOC has 5 blank-line blocks, one of which ("tiny") merges forward -> exactly 4.
        assert [c.ordinal for c in chunks] == [0, 1, 2, 3], (
            f"expected 4 paragraph chunks, got {[(c.ordinal, c.start, c.end) for c in chunks]}"
        )
        for c in chunks:
            assert DOC[c.start : c.end] == c.text
        # Spans are ordered and never overlap.
        for prev, nxt in zip(chunks, chunks[1:]):
            assert prev.end <= nxt.start, f"{prev.chunk_id} overlaps {nxt.chunk_id}"
        # Chunking drops no document content, only inter-chunk whitespace.
        assert "\n".join(c.text for c in chunks).split() == DOC.split()

    def test_tiny_merged_forward(self) -> None:
        text = "short\n\ntiny\n\n" + ("long enough paragraph " * 8)
        chunks = chunk_document("doc", text)
        assert all(len(c.text) >= 80 or i == len(chunks) - 1 for i, c in enumerate(chunks))

    def test_heading_mode_spans(self) -> None:
        chunks = chunk_document("doc", DOC, strategy="heading")
        texts = [c.text for c in chunks]
        assert any(t.startswith("# Alpha") for t in texts)
        assert any(t.startswith("# Beta") for t in texts)
        for c in chunks:
            assert DOC[c.start : c.end] == c.text

    def test_determinism(self) -> None:
        a = chunk_document("doc", DOC)
        b = chunk_document("doc", DOC)
        assert [(c.chunk_id, c.sha) for c in a] == [(c.chunk_id, c.sha) for c in b]


def _fake_summarize(text: str) -> str:
    return f"SUM({len(text)})"


def _resolve_span(store, blobs, docs: dict[str, str], span: dict) -> tuple[str, str]:
    """Resolve one span in ITS OWN declared coordinate frame.

    Returns ``(cited_bytes, source_bytes)``. A span that does not declare a
    frame cannot be resolved at all -- that ambiguity is the defect this
    helper exists to detect.
    """
    frame = span.get("of")
    cited = blobs.get_text(span["sha"])
    if frame == "document":
        source = docs[span["source_id"]]
    elif frame == "summary":
        parent = store.get_summary(span["source_id"])
        assert parent is not None, f"summary-frame span cites unknown summary {span['source_id']!r}"
        source = parent["text"]
    else:
        raise AssertionError(
            f"span {span!r} declares no resolvable coordinate frame; "
            "a consumer cannot tell document offsets from summary offsets"
        )
    return cited, source[span["start"] : span["end"]]


class TestSummaryDag:
    def _index(self, store, blobs, doc_id: str, n_paras: int):
        doc_text = "\n\n".join(f"{i}. " + ("filler sentence " * 15) for i in range(n_paras))
        chunks = chunk_document(doc_id, doc_text)
        for c in chunks:
            store.index_chunk(
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "ordinal": c.ordinal,
                 "start": c.start, "end": c.end, "sha": c.sha}
            )
            blobs.put_text(c.text)
        return doc_text, chunks

    def test_every_span_cites_exact_source_bytes(self, store, blobs, workspace) -> None:
        """THE decisive contract test (#492 §5): lossless source addressability.

        Build 8 chunks -> 2 level-1 summaries -> 1 level-2 summary, then walk
        EVERY span in the top summary and assert the cited bytes are byte-equal
        to the source bytes at the cited offsets, in the frame the span declares.
        """
        doc_text, chunks = self._index(store, blobs, "corpus", 8)
        assert len(chunks) == 8, f"fixture expected 8 chunks, got {len(chunks)}"
        docs = {"corpus": doc_text}

        level1 = [
            build_summary(store, blobs, "corpus", 1, [c.chunk_id for c in chunks[:4]], _fake_summarize),
            build_summary(store, blobs, "corpus", 1, [c.chunk_id for c in chunks[4:]], _fake_summarize),
        ]
        top = build_summary(store, blobs, "corpus", 2, level1, _fake_summarize)
        summary = store.get_summary(top)
        spans = summary["spans"]
        assert spans, "top summary carries no spans at all"

        failures: list[str] = []
        for sp in spans:
            cited, source = _resolve_span(store, blobs, docs, sp)
            if cited != source:
                failures.append(
                    f"span child_id={sp.get('child_id')!r} of={sp.get('of')!r} "
                    f"source_id={sp.get('source_id')!r} [{sp['start']}:{sp['end']}] "
                    f"cited={cited[:40]!r} but source says {source[:40]!r}"
                )
        rate = (len(spans) - len(failures)) / len(spans)
        assert not failures, (
            f"span round-trip pass rate {len(spans) - len(failures)}/{len(spans)} = {rate:.2%}\n"
            + "\n".join(failures)
        )

        # Every direct child is cited, and every LEAF is document-addressable.
        assert set(level1) <= {sp["child_id"] for sp in spans}
        doc_spans = [sp for sp in spans if sp["of"] == "document"]
        assert {sp["child_id"] for sp in doc_spans} == {c.chunk_id for c in chunks}, (
            "level-2 summary lost transitive document addressability for some leaf chunk"
        )
        for sp in doc_spans:
            assert doc_text[sp["start"] : sp["end"]] == blobs.get_text(sp["sha"])

        # Summary-frame spans exist and are NOT mistakable for document offsets.
        sum_spans = [sp for sp in spans if sp["of"] == "summary"]
        assert {sp["source_id"] for sp in sum_spans} >= set(level1)
        for sp in sum_spans:
            assert store.get_summary(sp["source_id"]) is not None
            assert doc_text[sp["start"] : sp["end"]] != blobs.get_text(sp["sha"]), (
                "a summary-frame span happens to match document offsets; the frame tag "
                "is what keeps a consumer from silently mis-resolving it"
            )

    def test_sibling_sets_sharing_endpoints_do_not_collide(self, store, blobs, workspace) -> None:
        """D2: two child sets with identical first/last children must not overwrite.

        ``store.add_summary`` uses INSERT OR REPLACE, so an id that ignores the
        middle children silently destroys a summary a parent may already cite.
        """
        doc_text, chunks = self._index(store, blobs, "collide", 4)
        assert len(chunks) == 4
        first, middle_a, middle_b, last = [c.chunk_id for c in chunks]

        set_a = [first, middle_a, last]
        set_b = [first, middle_b, last]
        assert set_a[0] == set_b[0] and set_a[-1] == set_b[-1]

        id_a = default_summary_id("collide", 1, set_a)
        id_b = default_summary_id("collide", 1, set_b)
        assert id_a != id_b, f"summary ids collide for distinct child sets: {id_a!r}"

        sid_a = build_summary(store, blobs, "collide", 1, set_a, lambda t: f"A:{len(t)}")
        sid_b = build_summary(store, blobs, "collide", 1, set_b, lambda t: f"B:{len(t)}")
        assert sid_a != sid_b
        stored_a, stored_b = store.get_summary(sid_a), store.get_summary(sid_b)
        assert stored_a is not None and stored_b is not None
        assert stored_a["children"] == set_a, f"summary {sid_a} was overwritten: {stored_a}"
        assert stored_b["children"] == set_b, f"summary {sid_b} was overwritten: {stored_b}"
        assert stored_a["text"].startswith("A:") and stored_b["text"].startswith("B:")

        # Deterministic: the same child list always yields the same id.
        assert default_summary_id("collide", 1, list(set_a)) == id_a
        # ...and order is part of the identity.
        assert default_summary_id("collide", 1, [first, last, middle_a]) != id_a


class TestSummarizerTruncation:
    """D4: silent truncation of evidence is exactly what #492 forbids."""

    def test_overlong_summary_records_the_truncation(self) -> None:
        long_text = "x" * 3000
        summarize = summarize_with_channel(
            lambda: RecordedChannel({"summarizer": [long_text]}), max_chars=100
        )
        out = summarize("source material")
        assert out.startswith("x" * 100)
        assert "truncated" in out, f"truncation was silent: {out[-120:]!r}"
        assert "2900" in out and "3000" in out, (
            f"truncation notice must state how much was dropped, got {out[-120:]!r}"
        )
        # The retained summary content is never shortened by its own bookkeeping.
        assert out[:100] == long_text[:100]

    def test_short_summary_is_returned_verbatim(self) -> None:
        summarize = summarize_with_channel(
            lambda: RecordedChannel({"summarizer": ["a tidy summary"]}), max_chars=100
        )
        out = summarize("source material")
        assert out == "a tidy summary"
        assert "truncated" not in out


class TestRetrieveAndSnapshot:
    def _seed(self, store) -> None:
        needle = "the launch key is PERIDOT-9"
        distractor = "quarterly revenue rose again this quarter"
        for i, (doc, t) in enumerate([("needle_doc", needle), *[(f"distr_{i}", distractor) for i in range(4)]]):
            store.index_chunk(
                {"chunk_id": f"{doc}:0", "doc_id": doc, "text": t, "ordinal": 0,
                 "start": 0, "end": len(t), "sha": content_hash(t)}
            )

    def test_needle_beats_distractors(self, store) -> None:
        self._seed(store)
        hits = retrieve(store, "launch key PERIDOT", k=5)
        assert hits[0].doc_id == "needle_doc"

    def test_scoped_snapshot_includes_children(self, store) -> None:
        from sherpa.events import Event

        store.append(Event(kind="run_started", run_id="parent", payload={}))
        store.append(Event(kind="run_started", run_id="child_a", payload={"parent_run_id": "parent"}))
        store.append(Event(kind="journal_appended", run_id="child_a", node_key=None,
                           payload={"kind": "observation", "text": "t", "refs": []}))
        snap = scoped_snapshot(store, "parent")
        runs = {e.run_id for e in snap}
        assert runs == {"parent", "child_a"}
        only_journal = scoped_snapshot(store, "parent", kinds=["journal_appended"])
        assert all(e.kind == "journal_appended" for e in only_journal)

    def test_scoped_snapshot_follows_real_store_run_linkage(self, store) -> None:
        """D5: descendants created through ``store.create_run(parent_run_id=...)``.

        Exercises the real linkage path (not a hand-written run_started payload)
        and proves the traversal is transitive, excludes unrelated runs, and
        returns a seq-ordered snapshot.
        """
        store.create_run("root", problem_sha="p")
        store.create_run("kid", problem_sha="p", parent_run_id="root")
        store.create_run("grandkid", problem_sha="p", parent_run_id="kid")
        store.create_run("stranger", problem_sha="p")
        store.create_run("stranger_kid", problem_sha="p", parent_run_id="stranger")

        journal(store, "root", None, "intent", "root intent")
        journal(store, "grandkid", None, "result", "deep finding")
        journal(store, "stranger_kid", None, "result", "unrelated finding")

        snap = scoped_snapshot(store, "root")
        assert {e.run_id for e in snap} == {"root", "kid", "grandkid"}, (
            f"descendant traversal wrong: {sorted({e.run_id for e in snap})}"
        )
        assert [e.seq for e in snap] == sorted(e.seq for e in snap), "snapshot is not seq-ordered"

        texts = [e.payload["text"] for e in snap if e.kind == "journal_appended"]
        assert texts == ["root intent", "deep finding"], texts

        # A leaf run's snapshot contains only itself.
        assert {e.run_id for e in scoped_snapshot(store, "grandkid")} == {"grandkid"}
        # Filtering composes with descendant discovery.
        filtered = scoped_snapshot(store, "root", kinds=["journal_appended"])
        assert [e.payload["text"] for e in filtered] == ["root intent", "deep finding"]


class TestValidatePlanIntegration:
    def test_full_valid_plan(self) -> None:
        plan = Plan(**{**_plan_dict(), "id": "ok"})
        assert validate_plan(plan, registry_names={"fs.read_file"}) == []
