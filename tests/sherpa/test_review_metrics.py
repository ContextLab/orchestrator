"""Hermetic tests for bounded independent review and metrics projections."""

from __future__ import annotations

import json

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

    def test_reviewer_session_differs_from_author(self, store) -> None:
        rev = _reviewer(store)
        report = rev.review_plan(_problem(), _plan(), author_session="author_1")
        assert report.verdict in ("pass_with_risk", "blocked_escalated")


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
        hallucination = json.dumps(
            {"findings": [{"criterion": "vibes", "subject": "plan:p1@1", "evidence": None, "blocking": True}]}
        )
        report = _reviewer(store, {"reviewer": [hallucination]}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict != "blocked_escalated" or any(
            f.blocking and f.evidence_ref for f in report.findings
        )
        downgraded = [f for f in report.findings if f.criterion == "vibes"]
        assert downgraded and downgraded[0].blocking is False

    def test_evidenced_model_finding_stays_blocking(self, store) -> None:
        evidenced = json.dumps(
            {"findings": [{"criterion": "missing_input_file",
                           "subject": "plan:p1@1",
                           "evidence": "inputs/in.txt referenced but absent from fixture",
                           "blocking": True}]}
        )
        report = _reviewer(store, {"reviewer": [evidenced]}).review_plan(
            _problem(), _plan(), author_session="a"
        )
        assert report.verdict == "blocked_escalated"

    def test_pass_reports_residual_not_clean(self, store) -> None:
        report = _reviewer(store).review_plan(_problem(), _plan(), author_session="a")
        assert report.verdict == "pass_with_risk"
        assert isinstance(report.residual_risks, list)

    def test_caps_stop_rounds(self, store) -> None:
        rev = _reviewer(store)
        rev.policy = ReviewPolicy(max_rounds=0, max_seconds=5)
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        assert report.rounds == 0

    def test_ledger_frozen_and_hashed(self, store) -> None:
        p = _problem()
        l1 = concern_ledger(p)
        l2 = concern_ledger(_problem())
        assert ledger_sha(l1) == ledger_sha(l2)
        assert any(c.source == "contract" for c in l1)

    def test_disposition_roundtrip(self, store) -> None:
        rev = _reviewer(store)
        report = rev.review_plan(_problem(), _plan(), author_session="a")
        if report.findings:
            fid = report.findings[0].id
            assert rev.disposition(fid, "accepted_risk", "known limitation")
            stored = {f["id"]: f for f in store.findings()}[fid]
            assert stored["disposition"] == "accepted_risk"


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
        report = _reviewer(store).review_output(_problem(), {"answer": "42"}, {"acc_tests": True}, author_session="a")
        assert report.verdict == "pass_with_risk"


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
        assert abs(m["branching"]["f_ambiguous"] - 0.2) < 1e-9
        assert m["branching"]["m_corrected"] < 1.0
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
        ci1 = bootstrap_ci([0.4] * 20)
        ci2 = bootstrap_ci([0.4] * 20)
        assert ci1 == ci2
        assert ci1[0] <= 0.4 <= ci1[1]

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
