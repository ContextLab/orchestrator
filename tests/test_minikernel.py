"""Tests for the #485 design kernel under scripts/prototypes/minikernel.

Real SQLite databases on disk, real subprocesses, real signals, real capability
implementations. Nothing is mocked and nothing is simulated: where a test needs
a crash it kills a process, and where it needs a failing capability it calls
one that really fails.

The network-backed planner is exercised by scripts/prototypes/measure_ambiguity.py
and probe_optimism.py, which cache their measurements; those are deliberately
not run here so the suite stays offline and free.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

PROTO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "scripts", "prototypes")
if PROTO not in sys.path:
    sys.path.insert(0, PROTO)

from minikernel import (  # noqa: E402
    ArtifactVersion, Authority, BugReport, BugTracker, Budget, Capability,
    CapabilityRegistry, Criterion, Finding, InsightPool, LLMPlanner, Message,
    Plan, ReviewBoard, Runtime, SeparationOfDuty, SolvedProblemLibrary, Step,
    Store, StubPlanner, validate)
from minikernel.library import signature_compatible, similarity  # noqa: E402
from minikernel.store import n_tokens  # noqa: E402

ROOT_AUTH = Authority(net=frozenset({"example.com"}), spend_usd=5.0)


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "k.db"), segment_tokens=400)
    yield s
    s.close()


@pytest.fixture()
def registry(store):
    reg = CapabilityRegistry(store)
    reg.seed(Capability("echo", 1, lambda a: str(a.get("text", "")), "str", "str",
                        tests=[({"text": "x"}, "x")]))
    reg.seed(Capability("upper", 1, lambda a: str(a.get("text", "")).upper(),
                        "str", "str", tests=[({"text": "x"}, "X")]))
    return reg


class ScriptedPlanner:
    """A planner whose output the test specifies.

    StubPlanner is seeded but its draw depends on the capability list, so tests
    written against it are hostage to a lucky roll -- three of these tests
    silently tested nothing until that was noticed. A test about recursion
    should state the recursion it means.
    """

    def __init__(self, by_depth: dict[int, list[tuple[str, str]]]):
        self.by_depth = by_depth
        self.calls = 0

    def decompose(self, problem, signature, capabilities, depth):
        from minikernel.planner import PlanDraft
        self.calls += 1
        spec = self.by_depth.get(depth, self.by_depth[max(self.by_depth)])
        steps = []
        for i, (kind, arg) in enumerate(spec):
            if kind == "capability":
                steps.append(Step(f"s{i}", "capability", ref=arg,
                                  args={"text": f"{problem}#{i}"},
                                  output_schema="str"))
            else:
                steps.append(Step(f"s{i}", "decompose",
                                  problem=f"{problem} :: {arg}", outputs=("y",),
                                  output_schema="str"))
        return PlanDraft(Plan(f"scripted-d{depth}", problem, tuple(steps),
                              signature=signature), "scripted")


def make_runtime(store, registry, **kw):
    planner = kw.pop("planner", StubPlanner(seed=1, b=4, f=0.15,
                                            capability_pool=("echo@1",)))
    return Runtime(store, registry, SolvedProblemLibrary(store), planner, **kw)


# ------------------------------------------------------------------- store

def test_token_estimate_is_conservative():
    text = "hello world " * 100
    assert n_tokens(text) >= len(text.split())


def test_blobs_are_content_addressed_and_deduplicated(store):
    a = store.put_blob("same body")
    b = store.put_blob("same body")
    assert a == b
    assert store.get_blob(a) == "same body"


def test_events_are_append_only_with_per_run_sequence(store):
    for i in range(5):
        store.append_event("r", f"n{i}", "node_started", {"i": i})
    seqs = [e["seq"] for e in store.events("r")]
    assert seqs == [1, 2, 3, 4, 5]


def test_descendant_runs_are_reachable_from_the_root(store):
    store.append_event("root", "root", "run_created", {})
    store.append_event("root/a", "root/a", "run_created", {}, parent_run_id="root")
    store.append_event("root/a/b", "root/a/b", "run_created", {},
                       parent_run_id="root/a")
    runs = {e["run_id"] for e in store.events("root")}
    assert runs == {"root", "root/a", "root/a/b"}


def test_sealing_summarises_each_range_exactly_once(store):
    calls = []

    def summarize(text, level):
        calls.append(level)
        return f"L{level}: {text[:40]}"

    for i in range(120):
        store.append_note("r", "n", "intent", f"note number {i} about calibration")
    first = store.seal(summarize)
    n_first = len(calls)
    second = store.seal(summarize)
    assert first, "expected segments to be sealed"
    assert second == [], "re-sealing with no new notes must do nothing"
    assert len(calls) == n_first, "no range may be summarised twice"


def test_a_summary_can_always_be_exchanged_for_its_source(store):
    for i in range(120):
        store.append_note("r", "n", "intent", f"note {i} about calibration drift")
    segs = store.seal(lambda t, lvl: f"summary L{lvl}")
    leaves = store.expand(segs[0])
    assert leaves and any("calibration drift" in x for x in leaves)


def test_context_window_respects_its_budget_and_reports_a_manifest(store):
    for i in range(200):
        store.append_note("r", "n", "intent", f"widget calibration note {i}")
    store.seal(lambda t, lvl: f"summary L{lvl} of widget calibration")
    win = store.compile_window(500, query="widget calibration")
    assert win.tokens <= 500
    assert win.manifest()
    assert {e["lane"] for e in win.manifest()} <= {
        "tail", "summaries", "retrieved", "insights"}


def test_the_four_views_read_from_one_substrate(store, registry):
    store.append_note("r", "n", "intent", "a plain thought")
    store.append_note("r", "n", "insight", "a durable insight about drift")
    registry.invoke("echo@1", {"text": "hi"}, "r", "n", "s", ROOT_AUTH)
    assert len(store.view_scratchpad()) == 2
    assert len(store.view_insights()) == 1
    assert len(store.view_tool_history("echo@1")) == 1


# ---------------------------------------------------------------------- IR

@pytest.mark.parametrize("step,expected", [
    (Step("l", "loop", guard="true", then=(Step("i", "capability", ref="echo@1"),)),
     "static iteration bound"),
    (Step("u", "capability", ref="ghost@9"), "unknown capability"),
    (Step("p", "capability", ref="echo@1", authority=Authority(subprocess=True)),
     "beyond its parent"),
    (Step("b", "branch"), "branch has no guard"),
    (Step("d", "decompose", problem="x"), "declares no outputs"),
])
def test_validator_rejects_each_unsafe_form(registry, step, expected):
    errors = validate(Plan("p", "x", (step,)), registry.maturities(), ROOT_AUTH)
    assert any(expected in e for e in errors), errors


def test_validator_accepts_a_well_formed_plan(registry):
    plan = Plan("p", "x", (Step("a", "capability", ref="echo@1",
                                output_schema="str"),))
    assert validate(plan, registry.maturities(), ROOT_AUTH) == []


def test_plan_hash_is_stable_and_input_sensitive(registry):
    a = Plan("p", "x", (Step("a", "capability", ref="echo@1", output_schema="str"),))
    b = Plan("p", "x", (Step("a", "capability", ref="echo@1", output_schema="str"),))
    c = Plan("p", "y", (Step("a", "capability", ref="echo@1", output_schema="str"),))
    assert a.hash() == b.hash() and a.hash() != c.hash()


def test_authority_narrows_but_never_widens():
    parent = Authority(net=frozenset({"a", "b"}), spend_usd=1.0)
    child = Authority(net=frozenset({"b", "c"}), spend_usd=0.5)
    assert parent.narrow(child).net == frozenset({"b"})
    assert not parent.covers(Authority(subprocess=True))


def test_budget_split_holds_a_reserve_back():
    b = Budget(1000, 1.0, 10, 60.0, reserve_frac=0.3)
    share = b.child_share(2)
    assert share.tokens == 350
    assert 2 * share.tokens < b.tokens


def test_budget_top_up_moves_credit_and_records_the_spend():
    parent, child = Budget(1000, 1.0, 10, 60.0), Budget(100, 0.1, 2, 6.0)
    moved = child.top_up(parent, 400)
    assert moved == 400 and child.tokens == 500
    assert parent.remaining().tokens == 600


# --------------------------------------------------------------- capabilities

def test_a_draft_capability_cannot_be_invoked(store, registry):
    registry.register(Capability("x", 1, lambda a: 1, "str", "int",
                                 tests=[({}, 1)], author_session="A"))
    with pytest.raises(PermissionError):
        registry.invoke("x@1", {}, "r", "n", "s", ROOT_AUTH)


def test_an_author_cannot_qualify_their_own_capability(store, registry):
    registry.register(Capability("x", 1, lambda a: 1, "str", "int",
                                 tests=[({}, 1)], author_session="A"))
    with pytest.raises(SeparationOfDuty):
        registry.qualify("x@1", "A")


def test_qualification_actually_runs_the_tests(store, registry):
    registry.register(Capability("good", 1, lambda a: 2, "str", "int",
                                 tests=[({}, 2)], author_session="A"))
    registry.register(Capability("bad", 1, lambda a: 3, "str", "int",
                                 tests=[({}, 2)], author_session="A"))
    assert registry.qualify("good@1", "B")[0] is True
    ok, failures = registry.qualify("bad@1", "B")
    assert ok is False and "observed 3" in failures[0]
    assert registry.get("bad@1").maturity == "quarantined"


def test_a_capability_without_tests_is_quarantined_not_trusted(store, registry):
    registry.register(Capability("untested", 1, lambda a: 1, "str", "int",
                                 author_session="A"))
    ok, why = registry.qualify("untested@1", "B")
    assert ok is False and registry.get("untested@1").maturity == "quarantined"
    assert "no tests" in why[0]


def test_a_capability_cannot_exceed_the_callers_authority(store, registry):
    registry.seed(Capability("net", 1, lambda a: "d", "str", "str",
                             authority=Authority(net=frozenset({"evil.test"})),
                             tests=[({}, "d")]))
    with pytest.raises(PermissionError):
        registry.invoke("net@1", {}, "r", "n", "s", ROOT_AUTH)


def test_capability_failures_are_recorded_before_being_raised(store, registry):
    registry.seed(Capability("boom", 1, lambda a: 1 / 0, "str", "int",
                             tests=[({}, 0)]))
    with pytest.raises(RuntimeError):
        registry.invoke("boom@1", {}, "r", "n", "s", ROOT_AUTH)
    history = store.view_tool_history("boom@1")
    assert len(history) == 1
    assert "ZeroDivisionError" in json.loads(history[0]["payload"])["error"]


def test_bug_triage_is_independent_and_evidence_driven(store, registry):
    registry.seed(Capability("flaky", 1, lambda a: 1 / 0, "str", "int",
                             tests=[({}, 0)]))
    for _ in range(5):
        with pytest.raises(RuntimeError):
            registry.invoke("flaky@1", {}, "r", "n", "s", ROOT_AUTH)
    tracker = BugTracker(store, registry)
    rid = tracker.file(BugReport("B", "flaky@1", "use", "ok", "raises", "R"))
    with pytest.raises(SeparationOfDuty):
        tracker.triage(rid, "R")
    assert tracker.triage(rid, "Q") == "revoked"
    assert registry.get("flaky@1").maturity == "revoked"


# -------------------------------------------------------------------- review

def test_an_author_cannot_review_their_own_artifact(store):
    board = ReviewBoard(store)
    art = ArtifactVersion("a", 1, "body", "A")
    with pytest.raises(SeparationOfDuty):
        board.run(art, [], "A", revise=lambda a, f: a)


def test_evidential_finding_blocks_and_a_revision_clears_it(store):
    board = ReviewBoard(store)
    art = ArtifactVersion("a", 1, "the answer is 41", "A")
    crit = [Criterion("C1", "must say 42", lambda b: ("42" in b, f"body={b!r}"))]
    final, outcome = board.run(
        art, crit, "B",
        revise=lambda a, f: ArtifactVersion(a.artifact_id, a.version + 1,
                                            "the answer is 42", "A"))
    assert outcome.passed and final.version == 2 and outcome.rounds == 2


def test_a_pass_reports_residual_risk_rather_than_certifying_clean(store):
    board = ReviewBoard(store, detection_rate_prior=0.6)
    art = ArtifactVersion("a", 1, "42", "A")
    _, outcome = board.run(art, [Criterion("C1", "ok", lambda b: (True, ""))],
                           "B", revise=lambda a, f: a)
    assert outcome.passed and outcome.residual_risk == pytest.approx(0.4)


def test_a_finding_outside_the_frozen_criteria_cannot_block(store):
    board = ReviewBoard(store)
    art = ArtifactVersion("a", 1, "body", "A")
    board.freeze("a", [Criterion("C1", "in scope")])
    drift = board.file(art, "B", Finding("F1", "C_OTHER", "blocker", "scope creep",
                                         "evidence!"))
    assert drift.status == "deferred" and not drift.blocking


def test_a_prose_worry_is_recorded_as_a_risk_not_a_blocker(store):
    board = ReviewBoard(store)
    art = ArtifactVersion("a", 1, "body", "A")
    board.freeze("a", [Criterion("C1", "in scope")])
    worry = board.file(art, "B", Finding("F1", "C1", "blocker", "feels off", None))
    assert worry.status == "risk" and not worry.blocking


def test_review_escalates_instead_of_looping_forever(store):
    board = ReviewBoard(store, max_rounds=3)
    art = ArtifactVersion("a", 1, "never fixed", "A")
    crit = [Criterion("C1", "impossible", lambda b: (False, "still wrong"))]
    _, outcome = board.run(art, crit, "B", revise=lambda a, f: ArtifactVersion(
        a.artifact_id, a.version + 1, a.body, a.author_session))
    assert not outcome.passed and outcome.escalated and outcome.rounds == 3


def test_contradicting_insights_are_refused_at_insertion(store):
    pool = InsightPool(store, ReviewBoard(store))
    ok, _ = pool.propose("r", "n", "the calibration drifts above 40 degrees",
                         "A", "B")
    assert ok
    ok2, why = pool.propose("r", "n",
                            "the calibration does not drift above 40 degrees",
                            "C", "B")
    assert not ok2 and "contradicts" in why


def test_an_author_cannot_approve_their_own_insight(store):
    pool = InsightPool(store, ReviewBoard(store))
    with pytest.raises(SeparationOfDuty):
        pool.propose("r", "n", "a thought", "A", "A")


# ------------------------------------------------------------------- library

@pytest.mark.parametrize("want,have,ok", [
    ("str->int", "str->int", True),
    ("str->int", "str->str", False),
    ("str->int", "any->int", True),
    ("str,int->bool", "str->bool", False),
])
def test_signature_compatibility(want, have, ok):
    assert signature_compatible(want, have) is ok


def test_similarity_ignores_stopwords_and_order():
    assert similarity("summarise the sales corpus",
                      "corpus of sales to summarise") > 0.9


def test_an_untyped_solution_is_never_published(store):
    lib = SolvedProblemLibrary(store)
    plan = Plan("p", "x", (Step("a", "capability", ref="echo@1",
                                output_schema="str"),))
    assert lib.publish("some problem", "any->any", plan, {}) is None
    assert len(lib) == 0


def test_a_typed_solution_is_retrieved_only_on_a_compatible_signature(store):
    lib = SolvedProblemLibrary(store)
    plan = Plan("p", "x", (Step("a", "capability", ref="echo@1",
                                output_schema="str"),))
    lib.publish("summarise the sales corpus for the north region", "any->str",
                plan, {})
    assert lib.lookup("summarise the sales corpus for the north region",
                      "any->str") is not None
    assert lib.lookup("summarise the sales corpus for the north region",
                      "any->int") is None


def test_budget_exhaustion_does_not_count_against_a_cached_plan(store):
    lib = SolvedProblemLibrary(store)
    plan = Plan("p", "x", (Step("a", "capability", ref="echo@1",
                                output_schema="str"),))
    key = lib.publish("a solvable problem statement", "any->str", plan, {})
    before = lib.entries[key].reliability
    lib.record_use(key, "budget_exhausted")
    assert lib.entries[key].reliability == before
    lib.record_use(key, "failed")
    assert lib.entries[key].reliability < before


def test_a_dead_end_is_remembered_only_at_the_allowance_that_failed(store):
    lib = SolvedProblemLibrary(store)
    lib.record_intractable("an unreachable problem statement here", "any->str",
                           "depth cap", depth_allowance=2)
    assert lib.lookup_intractable("an unreachable problem statement here",
                                  "any->str", 2) is not None
    # more room than last time is a legitimate reason to try again
    assert lib.lookup_intractable("an unreachable problem statement here",
                                  "any->str", 5) is None


# ------------------------------------------------------------------- runtime

def test_child_plans_get_their_own_run_with_a_recorded_parent(store, registry):
    rt = make_runtime(store, registry, max_depth=3, review_plans=False,
                      planner=ScriptedPlanner({
                          0: [("capability", "echo@1"), ("decompose", "hard bit")],
                          1: [("capability", "echo@1"), ("capability", "upper@1")],
                      }))
    rt.run("build a widget report", Budget(200_000, 5.0, 400, 600.0), ROOT_AUTH,
           run_id="r1", signature="any->str")
    proj = rt.projection("r1")
    children = [r for r, v in proj["tree"].items() if v["parent"]]
    assert children, "expected at least one child run"
    assert all(v["parent"] in proj["tree"] for v in proj["tree"].values()
               if v["parent"])


def test_a_supercritical_planner_never_reports_success(store, registry):
    # every level emits two ambiguous children: m = 2 > 1, so the tree can only
    # ever end at the depth cap.
    rt = make_runtime(store, registry, max_depth=3, review_plans=False,
                      planner=ScriptedPlanner({
                          0: [("decompose", "left"), ("decompose", "right")]}))
    res = rt.run("summarise the sales corpus for region north",
                 Budget(2_000_000, 50.0, 20_000, 600.0), ROOT_AUTH, run_id="h",
                 signature="any->str")
    assert res.status == "escalated" and res.partial


def test_a_repeated_dead_end_costs_less_the_second_time(store, registry):
    rt = make_runtime(store, registry, max_depth=3, review_plans=False,
                      planner=ScriptedPlanner({
                          0: [("decompose", "left"), ("decompose", "right")]}))
    costs = []
    for i in range(3):
        before = rt.stats.nodes
        rt.run("summarise the sales corpus for region north",
               Budget(400_000, 5.0, 2000, 600.0), ROOT_AUTH, run_id=f"h{i}",
               signature="any->str")
        costs.append(rt.stats.nodes - before)
    assert costs[-1] < costs[0] / 2, costs
    assert rt.stats.dead_end_hits > 0


def test_an_escalating_sibling_does_not_cancel_the_others(store, registry):
    rt = make_runtime(store, registry, max_depth=1, review_plans=False)
    plan = Plan("p", "x", (
        Step("a", "capability", ref="echo@1", args={"text": "one"},
             output_schema="str"),
        Step("b", "decompose", problem="unreachable", outputs=("y",),
             output_schema="str"),
        Step("c", "capability", ref="upper@1", args={"text": "three"},
             output_schema="str"),
    ))
    store.append_event("rr", "rr", "run_created", {})
    res = rt._execute_plan(plan, Budget(100_000, 1.0, 100, 60.0), ROOT_AUTH,
                           "rr", 1, {})
    done = {e["node_id"] for e in store.events("rr")
            if e["type"] == "node_completed"}
    assert res.status == "escalated" and "unresolved steps" in res.reason
    assert {"a", "c"} <= done, "siblings after the escalation must still run"


def test_budget_exhaustion_is_loud_and_never_a_green_checkmark(store, registry):
    rt = make_runtime(store, registry, max_depth=4, review_plans=False,
                      planner=ScriptedPlanner({
                          0: [("capability", "echo@1"), ("decompose", "deeper"),
                              ("decompose", "deeper still")]}))
    res = rt.run("an expensive recursive problem", Budget(4_000, 0.05, 12, 60.0),
                 ROOT_AUTH, run_id="rb", signature="any->str")
    kinds = [e["type"] for e in store.events("rb")]
    assert res.status != "completed" and res.partial
    assert "node_budget_exhausted" in kinds or "node_escalated" in kinds


def test_a_message_is_delivered_at_a_step_boundary_and_recorded(store, registry):
    rt = make_runtime(store, registry, max_depth=1, review_plans=False)
    plan = Plan("p", "x", (Step("s1", "capability", ref="upper@1",
                                args={"text": "original"}, output_schema="str"),))
    rt.bus.send("rm", "root", Message("m1", "rm/s1", "scope_change",
                                      {"args": {"text": "redirected"}}))
    store.append_event("rm", "rm", "run_created", {})
    res = rt._execute_plan(plan, Budget(100_000, 1.0, 50, 60.0), ROOT_AUTH,
                           "rm", 0, {})
    assert res.value == "REDIRECTED"
    assert any(e["type"] == "message_delivered" for e in store.events("rm"))


def test_a_message_over_its_hop_budget_is_dead_lettered(store, registry):
    rt = make_runtime(store, registry)
    rt.bus.send("r", "root", Message("m", "r/x", "instruction", {}, hops=99,
                                     max_hops=8))
    assert len(rt.bus.dead_letters) == 1
    assert rt.bus.take("r/x") == []


def test_admission_control_demotes_an_overclaimed_atomic_step(store, registry):
    registry.seed(Capability("guess", 1, lambda a: "PLAUSIBLE", "str", "str",
                             tests=[({}, "PLAUSIBLE")]))

    class Optimistic:
        def decompose(self, problem, signature, capabilities, depth):
            from minikernel.planner import PlanDraft
            return PlanDraft(Plan("o", problem, tuple(
                Step(f"s{i}", "capability", ref="guess@1",
                     args={"text": problem}, output_schema="str")
                for i in range(3)), signature=signature))

    unchecked = Runtime(store, registry, SolvedProblemLibrary(store), Optimistic(),
                        max_depth=1, review_plans=False)
    res = unchecked.run("reproduce figure 3", Budget(200_000, 5.0, 400, 600.0),
                        ROOT_AUTH, run_id="u", signature="any->str")
    assert res.status == "completed" and res.value == "PLAUSIBLE"
    assert unchecked.stats.m_measured == 0.0

    checked = Runtime(store, registry, SolvedProblemLibrary(store), Optimistic(),
                      max_depth=1, review_plans=False,
                      admission=lambda step, run: (False, "cannot reproduce a figure")
                      if step.ref == "guess@1" else (True, ""))
    res2 = checked.run("reproduce figure 3", Budget(200_000, 5.0, 400, 600.0),
                       ROOT_AUTH, run_id="c", signature="any->str")
    assert res2.status != "completed"
    # each demoted step becomes a decomposition, whose own plan is demoted too,
    # so the count compounds down the tree -- that is the point.
    assert checked.stats.demoted >= 3
    assert checked.stats.m_declared == 0.0 and checked.stats.m_measured == 3.0


def test_the_offspring_mean_is_not_the_product_of_the_means(store, registry):
    """The estimator that got this wrong once already."""
    rt = make_runtime(store, registry, max_depth=2, review_plans=False,
                      planner=StubPlanner(seed=2, b=6, f=0.3,
                                          capability_pool=("echo@1",)))
    rt.run("x", Budget(400_000, 5.0, 4000, 600.0), ROOT_AUTH, run_id="e",
           signature="any->str")
    assert rt.stats.m_measured == pytest.approx(
        rt.stats.ambiguous / len(rt.stats.fan_outs))


# ------------------------------------------------------------- crash / resume

def test_a_killed_run_resumes_to_an_identical_projection(tmp_path):
    """Kills a real child process with a real exit code, then resumes."""
    script = os.path.join(PROTO, "run_scenarios.py")
    env = dict(os.environ, MK_DB=str(tmp_path / "c.db"),
               MK_CLEAN=str(tmp_path / "clean.db"), PYTHONPATH=PROTO)
    clean = subprocess.run([sys.executable, script, "--child", "clean"], env=env,
                           capture_output=True, text=True, timeout=300)
    assert clean.returncode == 0, clean.stderr
    crashed = subprocess.run([sys.executable, script, "--child", "crash"], env=env,
                             capture_output=True, text=True, timeout=300)
    assert crashed.returncode == 137, "expected a hard kill mid-run"
    resumed = subprocess.run([sys.executable, script, "--child", "resume"], env=env,
                             capture_output=True, text=True, timeout=300)
    assert resumed.returncode == 0, resumed.stderr
    a = json.loads(resumed.stdout.strip().splitlines()[-1])
    b = json.loads(clean.stdout.strip().splitlines()[-1])
    assert a["skipped"] > 0, "resume must reuse completed work from the log"
    assert a["projection"] == b["projection"]


# ------------------------------------------------------------------- planner

def test_stub_planner_is_a_function_of_its_inputs():
    p = StubPlanner(seed=4, b=5, f=0.4, capability_pool=("echo@1",))
    a = p.decompose("the same problem", "any->str", {"echo@1": "trusted"}, 0)
    b = p.decompose("the same problem", "any->str", {"echo@1": "trusted"}, 0)
    assert [s.kind for s in a.plan.steps] == [s.kind for s in b.plan.steps]


@pytest.mark.parametrize("reply,steps", [
    ('{"steps": [], "rationale": "none"}', 0),
    ('```json\n{"steps": [{"id":"s1","kind":"capability","ref":"echo@1"}]}\n```', 1),
    ('here you go {"steps": [{"id":"s1","kind":"decompose","problem":"p"}]} ok', 1),
])
def test_llm_reply_parsing_handles_the_shapes_models_actually_emit(reply, steps):
    assert len(LLMPlanner._extract_json(reply).get("steps", [])) == steps


def test_llm_reply_parsing_refuses_to_guess_at_garbage():
    with pytest.raises(ValueError):
        LLMPlanner._extract_json("I'm afraid I can't do that.")
