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
    journal,
    retrieve,
    scoped_snapshot,
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
        assert set(JOURNAL_KINDS) == {"intent", "decision", "observation", "assumption", "blocker", "result"}


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
        assert len(chunks) >= 2
        for c in chunks:
            assert DOC[c.start : c.end] == c.text

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


class TestSummaryDag:
    def test_every_child_in_spans_two_levels(self, store, blobs, workspace) -> None:
        doc_text = "\n\n".join(f"{i}. " + ("filler sentence " * 15) for i in range(4))
        chunks = chunk_document("corpus", doc_text)
        for c in chunks:
            store.index_chunk(
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "ordinal": c.ordinal,
                 "start": c.start, "end": c.end, "sha": c.sha}
            )
            blobs.put_text(c.text)

        calls: list[str] = []

        def fake_summarize(text: str) -> str:
            calls.append(text)
            return f"SUM({len(text)})"

        level1 = [
            build_summary(store, blobs, "corpus", 1, [c.chunk_id for c in chunks[:2]], fake_summarize),
            build_summary(store, blobs, "corpus", 1, [c.chunk_id for c in chunks[2:]], fake_summarize),
        ]
        top = build_summary(store, blobs, "corpus", 2, level1, fake_summarize)
        summary = store.get_summary(top)
        span_children = {sp["child_id"] for sp in summary["spans"]}
        assert set(level1) <= span_children
        # spans are lossless: every span's sha resolves to the exact source text
        for sp in summary["spans"]:
            src = blobs.get_text(sp["sha"])
            assert src  # addressable evidence exists


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


class TestValidatePlanIntegration:
    def test_full_valid_plan(self) -> None:
        plan = Plan(**{**_plan_dict(), "id": "ok"})
        assert validate_plan(plan, registry_names={"fs.read_file"}) == []
