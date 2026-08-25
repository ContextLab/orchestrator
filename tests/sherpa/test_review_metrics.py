"""Hermetic tests for bounded independent review and metrics projections."""

from __future__ import annotations

import json
import math

import pytest

from sherpa.channel import RecordedChannel
from sherpa.ir import AcceptanceCheck, Authority, Budgets, InvokeCapability, Plan, ProblemSpec, Return, validate_plan
from sherpa.metrics import aggregate_run_reports, bootstrap_ci, render_report_md, run_metrics
from sherpa.review import (
    Finding,
    ReviewPolicy,
    Reviewer,
    SeparationOfDutyError,
    concern_ledger,
    ledger_sha,
)
from sherpa.store import Store

pytestmark = [pytest.mark.unit]


def _problem(**kw) -> ProblemSpec:
    defaults = dict(
        id="prob",
        goal="do the thing",
        acceptance=[
            AcceptanceCheck(id="acc_tests", kind="pytest", spec={"cmd": "pytest -q"}),
        ],
        budgets=Budgets(max_tokens=1000),
    )
    defaults.update(kw)
    return ProblemSpec(**defaults)


def _plan(authority: Authority | None = None) -> Plan:
    return Plan(
        id="p1",
        authority=authority or Authority(),
        budgets=Budgets(max_tokens=900, max_fanout=2),
        root=[
            InvokeCapability(kind="invoke_capability", id="a", capability="fs.read_file"),
            Return(kind="return", id="out", outputs={"ok": True}),
        ],
    )


#: The reviewer session the Reviewer must derive for author session "a". Hard-coded
#: on purpose: if production changes how it names the reviewer session, the recorded
#: responses stop matching and every test that stocks them fails loudly.
REVIEWER_SESSION_FOR_A = "reviewer::a"


def _finding_json(criterion: str, evidence: str | None, *, blocking: bool = False,
                  subject: str = "plan:p1@1") -> str:
    return json.dumps(
        {"findings": [{"criterion": criterion, "subject": subject,
                       "evidence": evidence, "blocking": blocking}]}
    )


class _CapturingRecordedChannel(RecordedChannel):
    """A real RecordedChannel that also remembers the messages it was handed.

    Not a mock: replay behaviour is unchanged, the recorded boundary still does the
    work. Capturing the prompt is the only way to assert the frozen ledger actually
    reaches the reviewer instead of being hashed and thrown away.
    """

    def __init__(self, recordings: dict) -> None:
        super().__init__(recordings)
        self.seen: list[list[dict]] = []

    def complete(self, messages, **kw):
        self.seen.append([dict(m) for m in messages])
        return super().complete(messages, **kw)


def _reviewer(store: Store, recordings: dict | None = None) -> Reviewer:
    return Reviewer(
        store=store,
        blob=store.blob,
        channel_factory=lambda: RecordedChannel(recordings or {}),
        policy=ReviewPolicy(max_rounds=2, max_seconds=30),
    )


class TestSeparation:
    def test_same_session_rejected(self) -> None:
        with pytest.raises(SeparationOfDutyError):
            Reviewer.ensure_separate("author", "author")

    def test_model_call_uses_the_reviewer_session_not_the_authors(self, store) -> None:
        """D6: falsifiable separation.

        RecordedChannel is keyed by session role, so stocking a response under the
        AUTHOR's key and nothing else proves which session the model call used: if
        review_plan reviewed under "author_1" it would consume this response and
        finish a round; using its own session exhausts the recording instead.
        """
        rev = _reviewer(store, {"author_1": [_finding_json("acc_tests", "evidence: x")]})
        report = rev.review_plan(_problem(), _plan(), author_session="author_1")
        assert report.rounds == 0, "the reviewer consumed the AUTHOR's session recordings"
        assert report.verdict == "escalated_review_incomplete"
        assert "author_1" in report.channel_error

    def test_model_call_consumes_the_reviewer_session_recordings(self, store) -> None:
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [_finding_json("acc_tests", "")]})
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.channel_error == ""
        assert report.rounds == 1

    def test_review_round_event_records_a_session_distinct_from_the_author(self, store) -> None:
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [_finding_json("acc_tests", "")]})
        rev.review_plan(_problem(), _plan(), author_session="a")
        rounds = [e for e in store.events(kinds=["review_round"])]
        assert len(rounds) == 1
        session = rounds[0].payload["reviewer_session"]
        assert session != "a"
        assert session == REVIEWER_SESSION_FOR_A


class TestPlanReview:
    def test_structural_defect_blocks_with_evidence(self, store) -> None:
        bad = Plan(
            id="p",
            authority=Authority(),
            budgets=Budgets(),
            root=[InvokeCapability(kind="invoke_capability", id="dup", capability="c"),
                  InvokeCapability(kind="invoke_capability", id="dup", capability="c")],
        )
        report = _reviewer(store).review_plan(_problem(), bad, author_session="a")
        assert report.verdict == "blocked_escalated"
        blocking = [f for f in report.findings if f.blocking]
        assert any("plan_validity" in f.criterion for f in blocking)
        assert all(f.evidence_ref for f in blocking)

    def test_authority_violation_blocks(self, store) -> None:
        problem = _problem()
        plan = _plan(authority=Authority(fs_write=("**",)))
        report = _reviewer(store).review_plan(problem, plan, author_session="a")
        assert report.verdict == "blocked_escalated"
        assert any(f.criterion == "authority_containment" for f in report.findings)

    def test_budget_violation_blocks(self, store) -> None:
        problem = _problem(budgets=Budgets(max_tokens=10))
        plan = _plan()
        report = _reviewer(store).review_plan(problem, plan, author_session="a")
        assert any(f.criterion == "budget_containment" for f in report.findings)

    def test_hallucinated_blocking_downgraded_to_risk(self, store) -> None:
        hallucination = _finding_json("acc_tests", None, blocking=True)
        report = _reviewer(store, {REVIEWER_SESSION_FOR_A: [hallucination]}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict == "pass_with_risk", (
            f"an unevidenced concern blocked the run: {[f.model_dump() for f in report.findings]}"
        )
        downgraded = [f for f in report.findings if f.criterion == "acc_tests"]
        assert len(downgraded) == 1
        assert downgraded[0].blocking is False
        assert downgraded[0].disposition == "invalid"
        assert downgraded[0].rationale == "downgraded: no reproducible evidence provided"
        assert downgraded[0].id in {f.id for f in report.residual_risks}

    def test_evidenced_model_finding_stays_blocking(self, store) -> None:
        evidenced = _finding_json(
            "acc_tests", "inputs/in.txt referenced by acc_tests but absent from fixture",
            blocking=True,
        )
        report = _reviewer(store, {REVIEWER_SESSION_FOR_A: [evidenced] * 2}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict == "blocked_escalated"
        blocking = [f for f in report.findings if f.blocking]
        assert [f.criterion for f in blocking] == ["acc_tests"] * len(blocking)
        assert all(f.evidence_ref for f in blocking)

    def test_pass_reports_residual_not_clean(self, store) -> None:
        """A pass is ``pass_with_risk`` and it must carry the risks it passed over."""
        report = _reviewer(
            store, {REVIEWER_SESSION_FOR_A: [_finding_json("acc_tests", "  ", blocking=True)]}
        ).review_plan(_problem(), _plan(), author_session="a")
        assert report.verdict == "pass_with_risk"
        assert report.rounds == 1
        assert report.channel_error == ""
        assert len(report.residual_risks) == 1, (
            f"pass reported no residual risk despite {len(report.findings)} findings"
        )
        risk = report.residual_risks[0]
        assert risk.criterion == "acc_tests"
        assert risk.blocking is False
        assert risk.disposition == "invalid"

    def test_max_rounds_zero_runs_no_model_round(self, store) -> None:
        """D2: the cap is honoured exactly, proved against a STOCKED channel.

        The channel holds three usable responses; if max_rounds=0 were floored to 1
        the reviewer would consume one and report rounds == 1.
        """
        blocking = _finding_json("acc_tests", "evidence: acc_tests never ran", blocking=True)
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [blocking] * 3})
        rev.policy = ReviewPolicy(max_rounds=0, max_seconds=5)
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.rounds == 0
        assert report.tokens == 0
        assert report.channel_error == "", "no round was requested, so no channel failure is possible"
        assert not [f for f in report.findings if f.criterion == "acc_tests"]

    def test_max_rounds_caps_a_reviewer_that_keeps_blocking(self, store) -> None:
        """Rounds stop at the cap even while every round returns a blocking finding."""
        blocking = _finding_json("acc_tests", "evidence: acc_tests never ran", blocking=True)
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [blocking] * 5})
        rev.policy = ReviewPolicy(max_rounds=2, max_seconds=30)
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.rounds == 2
        assert report.verdict == "blocked_escalated"

    def test_max_rounds_one_stops_after_one_round(self, store) -> None:
        blocking = _finding_json("acc_tests", "evidence: acc_tests never ran", blocking=True)
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [blocking] * 5})
        rev.policy = ReviewPolicy(max_rounds=1, max_seconds=30)
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.rounds == 1

    def test_identical_problems_hash_to_the_same_ledger(self, store) -> None:
        l1 = concern_ledger(_problem())
        l2 = concern_ledger(_problem())
        assert [c.model_dump() for c in l1] == [c.model_dump() for c in l2]
        assert ledger_sha(l1) == ledger_sha(l2)
        assert any(c.source == "contract" for c in l1)
        assert any(c.source == "generic" for c in l1)

    def test_different_ledgers_hash_differently(self, store) -> None:
        """A hash that cannot distinguish two ledgers cannot freeze anything."""
        base = _problem()
        extra = _problem(acceptance=[
            AcceptanceCheck(id="acc_tests", kind="pytest", spec={"cmd": "pytest -q"}),
            AcceptanceCheck(id="acc_lint", kind="pytest", spec={"cmd": "ruff check"}),
        ])
        renamed = _problem(acceptance=[
            AcceptanceCheck(id="acc_other", kind="pytest", spec={"cmd": "pytest -q"}),
        ])
        empty = _problem(acceptance=[])
        shas = {
            "base": ledger_sha(concern_ledger(base)),
            "extra": ledger_sha(concern_ledger(extra)),
            "renamed": ledger_sha(concern_ledger(renamed)),
            "empty": ledger_sha(concern_ledger(empty)),
        }
        assert len(set(shas.values())) == 4, f"ledger hashes collided: {shas}"

    def test_report_carries_the_ledger_sha_of_the_problem_reviewed(self, store) -> None:
        problem = _problem()
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [json.dumps({"findings": []})]})
        report = rev.review_plan(problem, _plan(), author_session="a")
        assert report.ledger_sha == ledger_sha(concern_ledger(problem))
        assert report.ledger_sha != ledger_sha(concern_ledger(_problem(acceptance=[])))

    def test_disposition_roundtrip(self, store) -> None:
        """Unconditional: the review below is constructed to always raise a finding."""
        rev = _reviewer(store, {REVIEWER_SESSION_FOR_A: [json.dumps({"findings": []})]})
        report = rev.review_plan(_problem(budgets=Budgets(max_tokens=10)), _plan(),
                                 author_session="a")
        assert report.findings, "fixture no longer produces a finding to dispose of"
        fid = report.findings[0].id
        assert fid in {f["id"] for f in store.findings()}
        assert rev.disposition(fid, "accepted_risk", "known limitation") is True
        stored = {f["id"]: f for f in store.findings()}[fid]
        assert stored["disposition"] == "accepted_risk"
        assert stored["rationale"] == "known limitation"

    def test_disposition_of_unknown_finding_is_reported_false(self, store) -> None:
        rev = _reviewer(store)
        assert rev.disposition("find_does_not_exist", "fixed", "nope") is False

    def test_disposition_rejects_values_outside_the_vocabulary(self, store) -> None:
        rev = _reviewer(store)
        with pytest.raises(ValueError):
            rev.disposition("find_whatever", "looks_fine_to_me")


class TestFrozenLedgerConstrainsReview:
    """D3: the ledger frozen at review start is the reviewer's entire scope."""

    def test_offledger_blocking_finding_cannot_block(self, store) -> None:
        drift = _finding_json("vibes", "I ran it and disliked the variable names", blocking=True)
        report = _reviewer(store, {REVIEWER_SESSION_FOR_A: [drift] * 3}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict == "pass_with_risk", (
            "a concern outside the frozen ledger blocked the plan: "
            f"{[f.model_dump() for f in report.findings]}"
        )
        drifted = [f for f in report.findings if f.criterion == "vibes"]
        assert len(drifted) == 1
        assert drifted[0].blocking is False
        assert drifted[0].disposition == "invalid"
        assert "ledger" in drifted[0].rationale
        assert drifted[0].evidence_ref, "the out-of-scope finding is recorded, not discarded"
        assert drifted[0].id in {f.id for f in report.residual_risks}

    def test_onledger_finding_with_evidence_still_blocks(self, store) -> None:
        """The constraint must not be a blanket downgrade of every model finding."""
        on_ledger = _finding_json("budgets_respected", "evidence: 9000 tokens > 1000", blocking=True)
        report = _reviewer(store, {REVIEWER_SESSION_FOR_A: [on_ledger] * 3}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict == "blocked_escalated"

    def test_ledger_is_sent_to_the_reviewer(self, store) -> None:
        channel = _CapturingRecordedChannel({REVIEWER_SESSION_FOR_A: [json.dumps({"findings": []})]})
        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: channel,
                       policy=ReviewPolicy(max_rounds=1, max_seconds=30))
        problem = _problem()
        rev.review_plan(problem, _plan(), author_session="a")
        assert channel.seen, "no model round was made"
        prompt = "\n".join(str(m["content"]) for m in channel.seen[0])
        for concern in concern_ledger(problem):
            assert concern.id in prompt, f"{concern.id} was never shown to the reviewer"
        assert ledger_sha(concern_ledger(problem)) in prompt

    def test_ledger_scope_is_identical_in_every_round(self, store) -> None:
        blocking = _finding_json("acc_tests", "evidence: acc_tests never ran", blocking=True)
        channel = _CapturingRecordedChannel({REVIEWER_SESSION_FOR_A: [blocking, blocking]})
        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: channel,
                       policy=ReviewPolicy(max_rounds=2, max_seconds=30))
        rev.review_plan(_problem(), _plan(), author_session="a")
        assert len(channel.seen) == 2
        assert channel.seen[0][0] == channel.seen[1][0], "the frozen ledger drifted between rounds"


class TestOutputReview:
    def test_failed_acceptance_blocks(self, store) -> None:
        problem = _problem()
        results = {"acc_tests": False}
        report = _reviewer(store).review_output(problem, {}, results, author_session="a")
        assert report.verdict == "blocked_escalated"

    def test_missing_check_blocks_skipped_concern(self, store) -> None:
        report = _reviewer(store).review_output(_problem(), {}, {}, author_session="a")
        assert any(f.blocking for f in report.findings)

    def test_all_pass_is_pass_with_risk(self, store) -> None:
        clean = json.dumps({"findings": []})
        report = _reviewer(store, {"reviewer::a": [clean]}).review_output(
            _problem(), {"answer": "42"}, {"acc_tests": True}, author_session="a"
        )
        assert report.verdict == "pass_with_risk"
        assert report.rounds == 1
        assert report.findings == []
        assert report.channel_error == ""


class TestGenericConcernsAreEnforced:
    """D4: the two generic contract clauses are checked, not merely declared."""

    SCHEMA = {"type": "object", "required": ["answer"],
              "properties": {"answer": {"type": "string"}}}

    def _reviewer_no_rounds(self, store, run_id: str = "") -> Reviewer:
        return Reviewer(store=store, blob=store.blob,
                        channel_factory=lambda: RecordedChannel({}),
                        policy=ReviewPolicy(max_rounds=0, max_seconds=30), run_id=run_id)

    def test_outputs_missing_a_required_field_block(self, store) -> None:
        problem = _problem(output_schema=self.SCHEMA)
        report = self._reviewer_no_rounds(store).review_output(
            problem, {"outputs_json": json.dumps({"unrelated": 1})},
            {"acc_tests": True}, author_session="a",
        )
        schema_findings = [f for f in report.findings if f.criterion == "outputs_conform_to_schema"]
        assert len(schema_findings) == 1, f"schema concern unenforced: {report.findings}"
        assert schema_findings[0].blocking is True
        assert schema_findings[0].evidence_ref, "a blocking finding must carry evidence"
        assert report.verdict == "blocked_escalated"

    def test_outputs_with_wrong_type_block(self, store) -> None:
        problem = _problem(output_schema=self.SCHEMA)
        report = self._reviewer_no_rounds(store).review_output(
            problem, {"outputs_json": json.dumps({"answer": 42})},
            {"acc_tests": True}, author_session="a",
        )
        assert any(f.criterion == "outputs_conform_to_schema" and f.blocking for f in report.findings)

    def test_conforming_outputs_raise_no_schema_finding(self, store) -> None:
        problem = _problem(output_schema=self.SCHEMA)
        report = self._reviewer_no_rounds(store).review_output(
            problem, {"outputs_json": json.dumps({"answer": "42"})},
            {"acc_tests": True}, author_session="a",
        )
        assert not [f for f in report.findings if f.criterion == "outputs_conform_to_schema"]
        assert report.verdict == "pass_with_risk"

    def test_unparseable_outputs_block(self, store) -> None:
        problem = _problem(output_schema=self.SCHEMA)
        report = self._reviewer_no_rounds(store).review_output(
            problem, {"outputs_json": "{not json"}, {"acc_tests": True}, author_session="a",
        )
        assert any(f.criterion == "outputs_conform_to_schema" and f.blocking for f in report.findings)

    def test_recorded_usage_over_budget_blocks(self, store) -> None:
        problem = _problem(budgets=Budgets(max_tokens=1000))
        store.create_run("run_b", problem_sha="sha_b")
        store.add_usage("run_b", tokens=4200.0)
        report = self._reviewer_no_rounds(store, run_id="run_b").review_output(
            problem, {"answer": "42"}, {"acc_tests": True}, author_session="a",
        )
        budget = [f for f in report.findings if f.criterion == "budgets_respected"]
        assert len(budget) == 1, f"budget concern unenforced: {report.findings}"
        assert budget[0].blocking is True
        assert "4200" in budget[0].evidence_ref and "1000" in budget[0].evidence_ref
        assert report.verdict == "blocked_escalated"

    def test_recorded_usage_within_budget_raises_no_finding(self, store) -> None:
        problem = _problem(budgets=Budgets(max_tokens=1000))
        store.create_run("run_c", problem_sha="sha_c")
        store.add_usage("run_c", tokens=999.0)
        report = self._reviewer_no_rounds(store, run_id="run_c").review_output(
            problem, {"answer": "42"}, {"acc_tests": True}, author_session="a",
        )
        assert not [f for f in report.findings if f.criterion == "budgets_respected"]
        assert report.verdict == "pass_with_risk"

    def test_every_declared_generic_concern_is_reachable(self, store) -> None:
        """No dead clauses: each generic concern id is raised by some real condition."""
        from sherpa.review import _GENERIC_CONCERNS

        declared = {cid for cid, _ in _GENERIC_CONCERNS}
        problem = _problem(output_schema=self.SCHEMA)
        store.create_run("run_d", problem_sha="sha_d")
        store.add_usage("run_d", tokens=9_000.0)
        report = self._reviewer_no_rounds(store, run_id="run_d").review_output(
            problem, {"outputs_json": json.dumps({"nope": 1})}, {}, author_session="a",
        )
        raised = {f.criterion for f in report.findings}
        assert declared <= raised, f"never enforced: {sorted(declared - raised)}"


class TestReviewIsAttributedToTheRun:
    """D7: findings and round events must be filed under the real run id."""

    def test_findings_and_round_events_are_filed_under_the_run_id(self, store) -> None:
        problem = _problem()
        store.create_run("run_x", problem_sha="sha_x")
        rev = Reviewer(store=store, blob=store.blob,
                       channel_factory=lambda: RecordedChannel({}),
                       policy=ReviewPolicy(max_rounds=0, max_seconds=30),
                       run_id="run_x")
        plan = Plan(id="p", authority=Authority(), budgets=Budgets(),
                    root=[InvokeCapability(kind="invoke_capability", id="dup", capability="c"),
                          InvokeCapability(kind="invoke_capability", id="dup", capability="c")])
        report = rev.review_plan(problem, plan, author_session="a")
        assert report.verdict == "blocked_escalated"

        raised = store.events(run_id="run_x", kinds=["finding_raised"])
        assert len(raised) == len(report.findings) > 0, (
            "findings were not attributed to the run: "
            f"{[e.model_dump() for e in store.events(kinds=['finding_raised'])]}"
        )
        rounds = store.events(run_id="run_x", kinds=["review_round"])
        assert len(rounds) == 1, "the review_round event is missing from the run-filtered listing"
        assert not store.events(run_id=problem.id, kinds=["review_round"]), (
            "review_round was filed under the problem id instead of the run id"
        )

    def test_budget_finding_keeps_its_rationale(self, store) -> None:
        """The rationale used to be passed as an ignored ``evidence=`` argument."""
        report = _reviewer(store).review_plan(
            _problem(budgets=Budgets(max_tokens=10)), _plan(), author_session="a"
        )
        budget = [f for f in report.findings if f.criterion == "budget_containment"]
        assert len(budget) == 1
        assert budget[0].rationale == "plan budgets exceed parent budgets"
        stored = {f["id"]: f for f in store.findings()}[budget[0].id]
        assert stored["rationale"] == "plan budgets exceed parent budgets"


class TestFindingIdentity:
    def test_stable_ids(self) -> None:
        from sherpa.review import _finding_id

        a = _finding_id("crit", "subj", "ev")
        b = _finding_id("crit", "subj", "ev")
        c = _finding_id("crit2", "subj", "ev")
        assert a == b and a != c


def _events_for_metrics(store: Store) -> list:
    from sherpa.events import Event

    evs = [
        Event(kind="run_started", run_id="r", payload={}),
        Event(kind="admission_checked", run_id="r", node_key="n1",
              payload={"capability": "fs.read_file", "decision": "reclassify_decompose",
                       "atomic_claimed": True}),
        Event(kind="admission_checked", run_id="r", node_key="n2",
              payload={"capability": "fs.read_file", "decision": "admitted", "atomic_claimed": True}),
        Event(kind="admission_checked", run_id="r", node_key="n3",
              payload={"capability": "text.summarize", "decision": "escalate", "atomic_claimed": False}),
        Event(kind="decompose_outcome", run_id="r", node_key="d1",
              payload={"children_declared": 3, "children_ambiguous": 1}),
        Event(kind="decompose_outcome", run_id="r", node_key="d2",
              payload={"children_declared": 2, "children_ambiguous": 0}),
        Event(kind="usage_checkpoint", run_id="r", payload={"tokens": 120.0}),
        Event(kind="run_terminal", run_id="r", payload={"status": "completed"}),
    ]
    for e in evs:
        store.append(e)
    return store.events(run_id="r")


class TestMetrics:
    def test_overclaim_and_branching(self, store) -> None:
        m = run_metrics(_events_for_metrics(store))
        assert m["admission"]["claimed_atomic"] == 2
        assert m["admission"]["rejected_or_reclassified"] == 1
        assert m["admission"]["overclaim_rate"] == 0.5
        assert m["branching"]["b_declared"] == 2.5
        # `f_declared` is the planner's self-report: 1 declared-ambiguous child
        # out of 5 declared children.
        assert abs(m["branching"]["f_declared"] - 0.2) < 1e-9
        # `f_ambiguous` is the CORRECTED figure and must incorporate admission
        # outcomes: n3 escalated so it was never a viable child (5 - 1 = 4), and
        # n1's atomic claim was reclassified. This assertion previously pinned
        # 0.2 -- the self-report -- which is precisely the defect that let
        # `m_corrected` read 0.0 across every benchmark run.
        assert abs(m["branching"]["f_ambiguous"] - 0.25) < 1e-9
        assert m["branching"]["children_viable"] == 4
        assert m["branching"]["m_corrected"] == 0.5
        assert m["terminal_status"] == "completed"
        assert m["usage"]["tokens"] == 120.0

    def test_aggregate_synthetic(self) -> None:
        reports = [
            {"terminal_status": "completed", "admission": {"overclaim_rate": 0.4},
             "branching": {"m_corrected": 0.8}, "usage": {"tokens": 100}},
            {"terminal_status": "failed", "admission": {"overclaim_rate": 0.6},
             "branching": {"m_corrected": 1.2}, "usage": {"tokens": 300}},
            {"terminal_status": "completed", "admission": {"overclaim_rate": 0.2},
             "branching": {"m_corrected": 0.4}, "usage": {"tokens": 200}},
        ]
        agg = aggregate_run_reports(reports)
        assert agg["runs_aggregated"] == 3
        assert agg["task_success_rate"] == pytest.approx(2 / 3)
        assert agg["mean_overclaim_rate"] == pytest.approx(0.4)

    def test_render_report_md(self) -> None:
        suite = {
            "runs_aggregated": 1, "task_success_rate": 1.0, "task_success_ci95": (0.9, 1.0),
            "mean_overclaim_rate": 0.25, "overclaim_ci95": (0.1, 0.4), "total_tokens": 500.0,
            "m_upper_bound_max": 0.8,
        }
        runs = [{
            "run_id": "r1", "terminal_status": "completed",
            "admission": {"checked": 4, "overclaim_rate": 0.25},
            "branching": {"m_corrected": 0.8}, "usage": {"tokens": 500},
        }]
        md = render_report_md(suite, runs)
        assert "| r1 | completed | 4 | 25.0% | 0.800 | 500 |" in md
        assert "subcritical (<1)" in md


class TestChannelFailureIsLoud:
    """D1: a review that needed a model and could not get one must never pass."""

    def test_missing_channel_does_not_masquerade_as_pass(self, store) -> None:
        from sherpa.channel import EchoChannel

        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: EchoChannel(),
                       policy=ReviewPolicy(max_rounds=2, max_seconds=30))
        report = rev.review_plan(_problem(), _plan(), author_session="author_1")
        assert report.verdict != "pass_with_risk", (
            f"channel failure silently passed: verdict={report.verdict!r} rounds={report.rounds}"
        )
        assert report.verdict == "escalated_review_incomplete"
        assert "no channel configured" in report.channel_error
        assert report.rounds == 0
        recorded = [f for f in report.findings if f.criterion == "review_channel_unavailable"]
        assert len(recorded) == 1, "the channel failure was not recorded as an outcome"
        assert recorded[0].evidence_ref == report.channel_error
        assert recorded[0].id in {f["id"] for f in store.findings()}, "not persisted"
        assert recorded[0].id in {f.id for f in report.residual_risks}

    def test_exhausted_recordings_mid_review_escalate(self, store) -> None:
        """Round 1 lands a blocking finding, round 2 finds the tape empty."""
        blocking = _finding_json("acc_tests", "evidence: acc_tests never ran", blocking=True)
        # One shared tape across rounds (the kernel reuses a single channel), so the
        # second round really does run out of recorded responses.
        channel = RecordedChannel({REVIEWER_SESSION_FOR_A: [blocking]})
        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: channel,
                       policy=ReviewPolicy(max_rounds=2, max_seconds=30))
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.rounds == 1
        assert "RecordingExhausted" in report.channel_error
        # A real blocking finding outranks the incomplete review.
        assert report.verdict == "blocked_escalated"

    def test_output_review_channel_failure_escalates_too(self, store) -> None:
        from sherpa.channel import EchoChannel

        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: EchoChannel(),
                       policy=ReviewPolicy(max_rounds=1, max_seconds=30))
        report = rev.review_output(_problem(), {"answer": "42"}, {"acc_tests": True},
                                   author_session="a")
        assert report.verdict == "escalated_review_incomplete"
        assert any(f.criterion == "review_channel_unavailable" for f in report.findings)

    def test_unexpected_channel_error_is_not_swallowed(self, store) -> None:
        """Only the declared boundary failures are handled; anything else must surface."""

        class Exploding(RecordedChannel):
            def complete(self, messages, **kw):
                raise RuntimeError("provider adapter blew up")

        rev = Reviewer(store=store, blob=store.blob, channel_factory=lambda: Exploding({}),
                       policy=ReviewPolicy(max_rounds=1, max_seconds=30))
        with pytest.raises(RuntimeError, match="provider adapter blew up"):
            rev.review_plan(_problem(), _plan(), author_session="a")


class TestEvidenceMustBeReproducible:
    """D5: a blocking finding needs a criterion PLUS evidence someone can re-run."""

    @pytest.mark.parametrize("evidence", ["   ", "\t", "\n", " \t\n ", ""])
    def test_blank_evidence_cannot_block(self, store, evidence: str) -> None:
        report = _reviewer(
            store, {REVIEWER_SESSION_FOR_A: [_finding_json("acc_tests", evidence, blocking=True)]}
        ).review_plan(_problem(), _plan(), author_session="a")
        assert report.verdict == "pass_with_risk", f"{evidence!r} was accepted as evidence"
        finding = [f for f in report.findings if f.criterion == "acc_tests"][0]
        assert finding.blocking is False
        assert finding.evidence_ref == ""
        assert finding.disposition == "invalid"

    def test_padded_evidence_is_kept_but_normalised(self, store) -> None:
        report = _reviewer(
            store,
            {REVIEWER_SESSION_FOR_A: [_finding_json("acc_tests", "  pytest -q exits 1  ",
                                                    blocking=True)] * 2},
        ).review_plan(_problem(), _plan(), author_session="a")
        assert report.verdict == "blocked_escalated"
        finding = [f for f in report.findings if f.criterion == "acc_tests"][0]
        assert finding.evidence_ref == "pytest -q exits 1"


class TestBootstrapCI:
    """A resample of a constant list is that same constant, so a constant fixture
    cannot tell a real bootstrap from ``return (x, x)``. Every case below uses a
    sample whose resamples genuinely differ.
    """

    SKEWED = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]

    def test_interval_is_nondegenerate_and_brackets_the_sample_mean(self) -> None:
        lo, hi = bootstrap_ci(self.SKEWED)
        assert lo < hi, "a bootstrap over a varying sample must produce a real interval"
        mean = math.fsum(self.SKEWED) / len(self.SKEWED)
        assert lo <= mean <= hi
        assert min(self.SKEWED) <= lo and hi <= max(self.SKEWED)

    def test_same_seed_is_reproducible_and_different_seeds_resample_differently(self) -> None:
        assert bootstrap_ci(self.SKEWED, seed=0) == bootstrap_ci(self.SKEWED, seed=0)
        assert bootstrap_ci(self.SKEWED, seed=0) != bootstrap_ci(self.SKEWED, seed=1), (
            "the seed does not reach the resampler"
        )

    @pytest.mark.parametrize("narrow,wide", [(0.5, 0.2), (0.2, 0.05), (0.05, 0.01)])
    def test_smaller_alpha_widens_the_interval(self, narrow: float, wide: float) -> None:
        n_lo, n_hi = bootstrap_ci(self.SKEWED, alpha=narrow)
        w_lo, w_hi = bootstrap_ci(self.SKEWED, alpha=wide)
        assert (w_hi - w_lo) > (n_hi - n_lo), (
            f"alpha={wide} interval ({w_lo}, {w_hi}) is not wider than "
            f"alpha={narrow} ({n_lo}, {n_hi}) — alpha is being ignored"
        )

    def test_median_statistic_differs_from_mean_on_a_skewed_sample(self) -> None:
        mean_ci = bootstrap_ci(self.SKEWED, statistic="mean")
        median_ci = bootstrap_ci(self.SKEWED, statistic="median")
        assert mean_ci != median_ci, "the statistic argument is ignored"
        median = sorted(self.SKEWED)[len(self.SKEWED) // 2]
        assert median_ci[0] <= median <= median_ci[1]
        assert median_ci[1] < mean_ci[1], "the outlier must drag the mean CI above the median CI"

    def test_more_resamples_change_the_estimate(self) -> None:
        assert bootstrap_ci(self.SKEWED, n_boot=50) != bootstrap_ci(self.SKEWED, n_boot=1000)

    def test_empty_sample_has_no_interval(self) -> None:
        assert bootstrap_ci([]) is None

    def test_single_observation_collapses_to_that_observation(self) -> None:
        assert bootstrap_ci([3.5]) == (3.5, 3.5)

    def test_constant_series_keeps_its_exact_value(self) -> None:
        # [0.1]*10 sums to 0.9999999999999999 under naive accumulation on every
        # CPython; exact-rounded means keep the constant-series invariant.
        assert bootstrap_ci([0.1] * 10) == (0.1, 0.1)
        assert bootstrap_ci([0.4] * 20) == (0.4, 0.4)
