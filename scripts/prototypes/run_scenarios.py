#!/usr/bin/env python3
"""End-to-end scenarios against the #485 kernel.

Nothing here is mocked. The crash scenario kills a real child process with a
real signal; the store is a real SQLite database on disk; the capabilities do
real work. Run:

    .venv/bin/python scripts/prototypes/run_scenarios.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minikernel import (ArtifactVersion, Authority, BugReport, BugTracker,
                        Budget, Capability, CapabilityRegistry, Crash,
                        Criterion, Finding, InsightPool, Message, ReviewBoard,
                        Runtime, SeparationOfDuty, SolvedProblemLibrary,
                        StubPlanner, Store, Step, Plan, validate)

PASS, FAIL = "  PASS", "  FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL}  {name}" + (f"  --  {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------ fixtures

def seed_capabilities(reg: CapabilityRegistry) -> None:
    reg.seed(Capability("echo", 1, lambda a: str(a.get("text", "")), "str", "str",
                        tests=[({"text": "x"}, "x")], source="return text"))
    reg.seed(Capability("upper", 1, lambda a: str(a.get("text", "")).upper(),
                        "str", "str", tests=[({"text": "x"}, "X")],
                        source="return text.upper()"))
    reg.seed(Capability("wordcount", 1,
                        lambda a: len(str(a.get("text", "")).split()),
                        "str", "int", tests=[({"text": "a b"}, 2)],
                        source="return len(text.split())"))


def fresh(tmp: str, name: str, **kw):
    store = Store(os.path.join(tmp, f"{name}.db"), segment_tokens=800)
    reg = CapabilityRegistry(store)
    seed_capabilities(reg)
    lib = SolvedProblemLibrary(store)
    planner = kw.pop("planner", None) or StubPlanner(seed=7, b=4, f=0.25,
                                                     capability_pool=("echo@1",))
    rt = Runtime(store, reg, lib, planner, **kw)
    return store, reg, lib, rt


ROOT_AUTH = Authority(net=frozenset({"example.com"}), subprocess=False, spend_usd=5.0)


# ----------------------------------------------------------------- scenarios

def s1_nested_runs(tmp: str) -> None:
    print("\nS1  durable nested runs (P1)")
    # m = b*f = 4*0.15 = 0.6: subcritical, so the recursion is finite.
    store, reg, lib, rt = fresh(tmp, "s1", max_depth=3, review_plans=False)
    rt.planner = StubPlanner(seed=7, b=4, f=0.15, capability_pool=("echo@1",))
    res = rt.run("build a widget report", Budget(200_000, 5.0, 400, 600.0),
                 ROOT_AUTH, run_id="r1", signature="any->str")
    proj = rt.projection("r1")
    children = [r for r, v in proj["tree"].items() if v["parent"]]
    linked = all(v["parent"] in proj["tree"] or v["parent"] is None
                 for v in proj["tree"].values())
    check("root run completes", res.status == "completed", res.status)
    check("child runs exist with their own run_id", len(children) > 0,
          f"{len(children)} child runs, depth {rt.stats.max_depth}")
    check("every child records its parent_run_id", linked)
    check("org tree is a projection of the log, not an object graph",
          len(proj["tree"]) == len(children) + 1,
          f"{len(proj['tree'])} runs rebuilt from {len(store.events('r1'))} events")
    store.close()

    # The same kernel at the critical point (m = 4*0.25 = 1.0) must NOT quietly
    # produce an answer -- it must run into the depth cap and say so.
    store2, _, _, rt2 = fresh(tmp, "s1crit", max_depth=3, review_plans=False)
    rt2.planner = StubPlanner(seed=7, b=4, f=0.25, capability_pool=("echo@1",))
    crit = rt2.run("build a widget report", Budget(200_000, 5.0, 400, 600.0),
                   ROOT_AUTH, run_id="rc", signature="any->str")
    # m=1.0 is the critical point: a SINGLE realisation may well terminate --
    # it is the expectation that diverges. So the invariant is not "it fails",
    # it is "it costs sharply more, and whatever it returns is an honest
    # terminal status".
    check("at m=1.0 the run still reaches an honest terminal status",
          crit.status in ("completed", "escalated", "budget_exhausted"),
          f"status={crit.status}, depth reached {rt2.stats.max_depth}")
    store2.close()
    # One realisation says nothing about a branching process. Average over
    # seeds instead.
    def sweep(f_: float, seeds: int = 16) -> tuple[float, float]:
        total, escalated = 0, 0
        for sd in range(seeds):
            st, _, _, r = fresh(tmp, f"s1sweep{f_}{sd}", max_depth=4,
                                review_plans=False)
            r.planner = StubPlanner(seed=sd, b=4, f=f_, capability_pool=("echo@1",))
            res = r.run("build a widget report", Budget(400_000, 5.0, 4000, 600.0),
                        ROOT_AUTH, run_id="x", signature="any->str")
            total += r.stats.nodes
            escalated += res.status != "completed"
            st.close()
        return total / seeds, escalated / seeds
    (n_sub, e_sub), (n_crit, e_crit) = sweep(0.15), sweep(0.30)
    # With a depth cap in place, crossing m=1 does NOT mostly show up as cost:
    # the cap truncates the tree. It shows up as ANSWERS NOT PRODUCED.
    check("crossing m=1 shows up as escalation rate, not as cost (16 seeds)",
          e_crit > 2 * max(e_sub, 0.01) and n_crit > n_sub,
          f"m=0.6: {n_sub:.1f} nodes / {e_sub:.0%} escalated  ->  "
          f"m=1.2: {n_crit:.1f} nodes / {e_crit:.0%} escalated")


def s2_crash_resume(tmp: str) -> None:
    print("\nS2  crash and resume (P2)")
    db = os.path.join(tmp, "s2.db")
    clean_db = os.path.join(tmp, "s2_clean.db")
    env = dict(os.environ, MK_DB=db, MK_CLEAN=clean_db,
               PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    # 1. clean reference run in its own process
    r0 = subprocess.run([sys.executable, __file__, "--child", "clean"], env=env,
                        capture_output=True, text=True)
    # 2. a run that really dies mid-flight
    r1 = subprocess.run([sys.executable, __file__, "--child", "crash"], env=env,
                        capture_output=True, text=True)
    check("child process died with a hard exit code", r1.returncode == 137,
          f"returncode={r1.returncode}")
    # 3. resume in a third process
    r2 = subprocess.run([sys.executable, __file__, "--child", "resume"], env=env,
                        capture_output=True, text=True)
    if r2.returncode != 0:
        check("resume ran", False, r2.stderr.strip().splitlines()[-1:] or "")
        return
    resumed = json.loads(r2.stdout.strip().splitlines()[-1])
    reference = json.loads(r0.stdout.strip().splitlines()[-1])
    check("resumed run reaches a terminal status",
          resumed["status"] in ("completed", "budget_exhausted"), resumed["status"])
    check("resumed projection == uninterrupted projection",
          resumed["projection"] == reference["projection"],
          "byte-identical results/statuses/tree")
    check("resume skipped already-completed work",
          resumed["skipped"] > 0, f"{resumed['skipped']} nodes replayed from log")


def _child(mode: str) -> None:
    db, clean = os.environ["MK_DB"], os.environ["MK_CLEAN"]
    path = clean if mode == "clean" else db
    store = Store(path, segment_tokens=800)
    reg = CapabilityRegistry(store)
    seed_capabilities(reg)
    rt = Runtime(store, reg, SolvedProblemLibrary(store),
                 StubPlanner(seed=11, b=4, f=0.3, capability_pool=("echo@1",)),
                 max_depth=2, review_plans=False)
    budget = Budget(200_000, 5.0, 400, 600.0)
    if mode == "crash":
        rt.crash_at = "r1/s2"
        try:
            rt.run("assemble the quarterly digest", budget, ROOT_AUTH, run_id="r1")
        except Crash:
            os._exit(137)
        os._exit(0)
    skipped = len(rt.completed_nodes("r1")) if mode == "resume" else 0
    res = rt.run("assemble the quarterly digest", budget, ROOT_AUTH, run_id="r1",
                 resume=(mode == "resume"))
    print(json.dumps({"status": res.status, "skipped": skipped,
                      "projection": rt.projection("r1")}))
    store.close()


def s3_budget(tmp: str) -> None:
    print("\nS3  budget exhaustion is loud (P3)")
    store, reg, lib, rt = fresh(tmp, "s3", max_depth=4, review_plans=False)
    rt.planner = StubPlanner(seed=3, b=6, f=0.5, capability_pool=("echo@1",))
    tight = Budget(4_000, 0.05, 12, 60.0)
    res = rt.run("an expensive recursive problem", tight, ROOT_AUTH, run_id="rb")
    ev = [e["type"] for e in store.events("rb")]
    check("tight budget does NOT report success",
          res.status != "completed", res.status)
    check("exhaustion is recorded as an event, not swallowed",
          "node_budget_exhausted" in ev or "node_escalated" in ev,
          f"{ev.count('node_budget_exhausted')} exhaustion events")
    check("result is flagged partial", res.partial, res.reason)
    check("parent reallocated from reserve before giving up",
          "budget_reallocated" in ev, f"{ev.count('budget_reallocated')} top-ups")
    # Money is not the binding constraint when m > 1. Same problem, same huge
    # budget, two planners either side of the critical point.
    def generous() -> Budget:
        return Budget(2_000_000, 50.0, 20_000, 600.0)

    store2, _, _, rt2 = fresh(tmp, "s3b", max_depth=4, review_plans=False)
    rt2.planner = StubPlanner(seed=3, b=6, f=0.5, capability_pool=("echo@1",))
    g2 = generous()
    res2 = rt2.run("an expensive recursive problem", g2, ROOT_AUTH, run_id="rb",
                   signature="any->str")
    check("supercritical (m=3.0) does NOT complete even on a huge budget",
          res2.status != "completed",
          f"status={res2.status}, {rt2.stats.nodes} nodes, {g2.spent_tokens} tokens")
    store3, _, _, rt3 = fresh(tmp, "s3c", max_depth=4, review_plans=False)
    rt3.planner = StubPlanner(seed=3, b=6, f=0.1, capability_pool=("echo@1",))
    g3 = generous()
    res3 = rt3.run("an expensive recursive problem", g3, ROOT_AUTH, run_id="rb",
                   signature="any->str")
    check("subcritical (m=0.6) completes on the same budget",
          res3.status == "completed",
          f"{rt3.stats.nodes} nodes, {g3.spent_tokens} tokens")
    for st in (store, store2, store3):
        st.close()


def s4_messages(tmp: str) -> None:
    print("\nS4  messages at deterministic boundaries (P4)")
    store, reg, lib, rt = fresh(tmp, "s4", max_depth=1, review_plans=False)
    plan = Plan("p", "greet", (
        Step("s0", "capability", ref="echo@1", args={"text": "original"},
             output_schema="str"),
        Step("s1", "capability", ref="upper@1", args={"text": "original"},
             output_schema="str"),
    ))
    rt.bus.send("rm", "root", Message("m1", "rm/s1", "scope_change",
                                      {"args": {"text": "redirected"}}))
    budget = Budget(100_000, 1.0, 50, 60.0)
    store.append_event("rm", "rm", "run_created", {"problem": "greet"})
    res = rt._execute_plan(plan, budget, ROOT_AUTH, "rm", 0, {})
    delivered = [e for e in store.events("rm") if e["type"] == "message_delivered"]
    check("message delivered at the step boundary", len(delivered) == 1)
    check("message actually changed the node's inputs",
          res.value == "REDIRECTED", str(res.value))
    check("delivery is an auditable event, not prompt injection",
          json.loads(delivered[0]["payload"])["kind"] == "scope_change")
    rt.bus.send("rm", "root", Message("m2", "rm/nobody", "instruction", {},
                                      hops=99, max_hops=8))
    check("over-budget hop count dead-letters instead of looping",
          len(rt.bus.dead_letters) == 1)
    store.close()


def s5_review(tmp: str) -> None:
    print("\nS5  review protocol: separation of duty, evidence, ledger")
    store, reg, lib, rt = fresh(tmp, "s5")
    board = ReviewBoard(store, max_rounds=5)
    author = "sess-author-1"
    art = ArtifactVersion("report", 1, "the answer is 41", author)
    crit = [Criterion("C1", "answer must be 42",
                      lambda b: ("42" in b, f"observed body: {b!r}"))]
    try:
        board.run(art, crit, author, revise=lambda a, f: a)
        check("author cannot review own work", False, "no exception raised")
    except SeparationOfDuty:
        check("author cannot review own work", True)

    revisions = {"n": 0}

    def revise(a, findings):
        revisions["n"] += 1
        return ArtifactVersion(a.artifact_id, a.version + 1,
                               "the answer is 42", "sess-author-2")

    final, outcome = board.run(art, crit, "sess-reviewer-9", revise=revise)
    check("evidential finding blocks, then is fixed",
          outcome.passed and revisions["n"] == 1,
          f"{outcome.rounds} rounds, v{final.version}")
    check("a passing review reports residual risk, not 'clean'",
          outcome.residual_risk > 0, f"residual_risk={outcome.residual_risk}")

    # out-of-scope prose worry must not be able to block
    art2 = ArtifactVersion("report2", 1, "the answer is 42", author)
    board.freeze("report2", crit)
    drifty = board.file(art2, "sess-reviewer-9",
                        Finding("F99", "C_NEW", "blocker",
                                "we should also rewrite the intro", None))
    check("finding citing no frozen criterion is auto-deferred",
          drifty.status == "deferred" and not drifty.blocking, drifty.status)
    prose = board.file(art2, "sess-reviewer-9",
                       Finding("F98", "C1", "blocker", "feels underspecified", None))
    check("in-scope finding with no evidence becomes a non-blocking risk",
          prose.status == "risk" and not prose.blocking, prose.status)
    store.close()


def s6_capabilities(tmp: str) -> None:
    print("\nS6  capability lifecycle")
    store, reg, lib, rt = fresh(tmp, "s6")
    broken = Capability("flaky", 1, lambda a: 1 / 0 if a.get("boom") else "ok",
                        "str", "str", tests=[({"boom": False}, "ok")],
                        author_session="sess-author-3", source="1/0")
    reg.register(broken)
    try:
        reg.invoke("flaky@1", {}, "r", "n", "s", ROOT_AUTH)
        check("draft capability cannot be invoked", False)
    except PermissionError:
        check("draft capability cannot be invoked", True)
    try:
        reg.qualify("flaky@1", "sess-author-3")
        check("author cannot qualify own capability", False)
    except SeparationOfDuty:
        check("author cannot qualify own capability", True)
    ok, failures = reg.qualify("flaky@1", "sess-reviewer-4")
    check("independent qualification RUNS the tests and promotes",
          ok and reg.get("flaky@1").maturity == "trusted", str(failures))

    privileged = Capability("fetch", 1, lambda a: "data", "str", "str",
                            authority=Authority(net=frozenset({"evil.test"})),
                            tests=[({}, "data")], author_session="sess-author-5")
    reg.register(privileged)
    reg.qualify("fetch@1", "sess-reviewer-4")
    try:
        reg.invoke("fetch@1", {}, "r", "n", "s", ROOT_AUTH)
        check("capability cannot exceed caller authority", False)
    except PermissionError as exc:
        check("capability cannot exceed caller authority", True, str(exc)[:60])

    tracker = BugTracker(store, reg)
    for _ in range(5):
        try:
            reg.invoke("flaky@1", {"boom": True}, "r", "n", "s", ROOT_AUTH)
        except RuntimeError:
            pass
    rid = tracker.file(BugReport("B1", "flaky@1", "call with boom",
                                 "returns ok", "raises ZeroDivisionError",
                                 "sess-worker-6"))
    try:
        tracker.triage(rid, "sess-worker-6")
        check("reporter cannot triage own bug", False)
    except SeparationOfDuty:
        check("reporter cannot triage own bug", True)
    decision = tracker.triage(rid, "sess-reviewer-7")
    check("triage uses the call history as evidence",
          decision == "revoked" and reg.get("flaky@1").maturity == "revoked",
          decision)
    store.close()


def s7_library(tmp: str) -> None:
    print("\nS7  the library is the termination mechanism")
    # (a) subcritical: repeated work should get cheaper via cache hits.
    store, reg, lib, rt = fresh(tmp, "s7", max_depth=3, review_plans=False)
    rt.planner = StubPlanner(seed=5, b=5, f=0.15, capability_pool=("echo@1",))
    nodes, plans = [], []
    for i in range(4):
        b = Budget(400_000, 5.0, 2000, 600.0)
        before, before_p = rt.stats.nodes, rt.stats.planner_calls
        res = rt.run("summarise the sales corpus for region north", b, ROOT_AUTH,
                     run_id=f"m{i}", signature="any->str")
        nodes.append(rt.stats.nodes - before)
        plans.append(rt.stats.planner_calls - before_p)
    check("subcritical mission completes", res.status == "completed", res.status)
    check("library grows as problems are solved", len(lib) > 0, f"{len(lib)} entries")
    check("a repeated problem needs strictly less PLANNING",
          plans[-1] < plans[0], f"planner calls per mission: {plans}")
    # The finding this scenario actually produced: a library hit removes the
    # decomposition, not the work. Reuse drives `f` down (which is what
    # termination needs) but it does NOT make execution cheaper unless results
    # are memoised for identical inputs -- a separate mechanism #485 does not
    # mention and the round-2 cost model quietly folded in.
    check("but execution cost is UNCHANGED by a plan-level cache hit",
          nodes[-1] == nodes[0], f"nodes per mission: {nodes}")
    check("cache hits are recorded with similarity and reliability",
          rt.stats.library_hits > 0,
          f"{rt.stats.library_hits} hits / {rt.stats.library_misses} misses")
    stmt = next(iter(lib.entries.values())).statement
    check("identical text with an incompatible signature does not hit",
          lib.lookup(stmt, "any->int") is None
          and lib.lookup(stmt, "any->str") is not None)
    check("an untyped solution is never published at all",
          lib.publish("some untyped problem", "any->any", _plan_stub(), {}) is None)
    store.close()

    # (b) supercritical: the mission CANNOT succeed. What must still improve is
    #     the cost of finding that out, and the honesty of the answer.
    store2, _, lib2, rt2 = fresh(tmp, "s7b", max_depth=3, review_plans=False)
    rt2.planner = StubPlanner(seed=5, b=5, f=0.35, capability_pool=("echo@1",))
    hard = []
    for i in range(4):
        b = Budget(400_000, 5.0, 2000, 600.0)
        before = rt2.stats.nodes
        res2 = rt2.run("summarise the sales corpus for region north", b, ROOT_AUTH,
                       run_id=f"h{i}", signature="any->str")
        hard.append(rt2.stats.nodes - before)
    check("supercritical mission never reports success",
          res2.status == "escalated", res2.status)
    check("failure is remembered, so the retry is cheap",
          hard[-1] < hard[0] / 2, f"nodes per attempt: {hard}")
    check("negative results are first-class library entries",
          len(lib2.dead_ends) > 0 and rt2.stats.dead_end_hits > 0,
          f"{len(lib2.dead_ends)} dead ends, {rt2.stats.dead_end_hits} hits")
    check("a dead end names how many attempts have hit it",
          next(iter(lib2.dead_ends.values())).attempts >= 1)
    store2.close()


def _plan_stub() -> Plan:
    return Plan("stub", "x", (Step("a", "capability", ref="echo@1",
                                   output_schema="str"),))


def s8_validator(tmp: str) -> None:
    print("\nS8  validator refuses plans the runtime cannot bound")
    store, reg, lib, rt = fresh(tmp, "s8")
    bad = Plan("p", "x", (
        Step("l", "loop", guard="true", then=(Step("i", "capability",
                                                   ref="echo@1"),)),
        Step("u", "capability", ref="ghost@9"),
        Step("p", "capability", ref="echo@1",
             authority=Authority(subprocess=True)),
    ))
    errs = validate(bad, reg.maturities(), ROOT_AUTH)
    check("unbounded loop rejected", any("static iteration bound" in e for e in errs))
    check("unknown capability rejected", any("unknown capability" in e for e in errs))
    check("authority widening rejected", any("beyond its parent" in e for e in errs))
    good = Plan("p", "x", (Step("a", "capability", ref="echo@1",
                                output_schema="str"),))
    check("valid plan passes", validate(good, reg.maturities(), ROOT_AUTH) == [])
    store.close()


def s9_insights(tmp: str) -> None:
    print("\nS9  insight pool gating")
    store, reg, lib, rt = fresh(tmp, "s9")
    pool = InsightPool(store, ReviewBoard(store))
    ok, ref = pool.propose("r", "n", "the calibration drifts above 40 degrees",
                           "sess-a", "sess-b")
    check("insight accepted after independent review", ok, ref)
    try:
        pool.propose("r", "n", "another thought", "sess-a", "sess-a")
        check("author cannot approve own insight", False)
    except SeparationOfDuty:
        check("author cannot approve own insight", True)
    ok2, why = pool.propose("r", "n",
                            "the calibration does not drift above 40 degrees",
                            "sess-c", "sess-b")
    check("contradicting insight is refused at insertion", not ok2, why)
    store.close()


class OptimisticPlanner:
    """A planner that marks everything atomic. Not a straw man: the best real
    planner measured (probe_optimism.py) overclaimed 35% of its atomic steps,
    and reported an apparent m of 0.12 while its corrected m was 1.62."""

    def __init__(self, b: int = 4):
        self.b = b

    def decompose(self, problem, signature, capabilities, depth):
        from minikernel.planner import PlanDraft
        steps = tuple(
            Step(id=f"s{i}", kind="capability", ref="guess@1",
                 args={"text": f"{problem}#{i}"}, output_schema="str")
            for i in range(self.b)
        )
        return PlanDraft(Plan(f"opt-{depth}", problem, steps,
                              signature=signature), "everything looks easy")


def s10_admission(tmp: str) -> None:
    print("\nS10  admission control: is 'atomic' a claim or a label?")

    def build(with_admission: bool):
        store = Store(os.path.join(tmp, f"s10-{with_admission}.db"))
        reg = CapabilityRegistry(store)
        seed_capabilities(reg)
        # A capability that always answers, and is always wrong on hard input.
        reg.seed(Capability("guess", 1, lambda a: "PLAUSIBLE-BUT-UNVERIFIED",
                            "str", "str", tests=[({"text": "x"},
                                                  "PLAUSIBLE-BUT-UNVERIFIED")]))
        admission = None
        if with_admission:
            def admission(step, run_id):
                # Stands in for the independent judge used in probe_optimism.py.
                if step.ref == "guess@1" and "reproduce" in str(step.args):
                    return False, "guess@1 cannot reproduce a published figure"
                return True, ""
        rt = Runtime(store, reg, SolvedProblemLibrary(store), OptimisticPlanner(),
                     max_depth=2, review_plans=False, admission=admission)
        res = rt.run("reproduce figure 3 from the released dataset",
                     Budget(200_000, 5.0, 400, 600.0), ROOT_AUTH,
                     run_id="ro", signature="any->str")
        return store, rt, res

    st0, rt0, r0 = build(False)
    check("without admission control the run reports SUCCESS",
          r0.status == "completed", r0.status)
    check("...and its answer is an unverified placeholder",
          r0.value == "PLAUSIBLE-BUT-UNVERIFIED", str(r0.value))
    check("...and its measured m looks perfectly safe",
          rt0.m_measured_safe() < 0.05 if hasattr(rt0, "m_measured_safe")
          else rt0.stats.m_measured < 0.05,
          f"m_measured={rt0.stats.m_measured:.2f} "
          f"(declared {rt0.stats.m_declared:.2f})")

    st1, rt1, r1 = build(True)
    check("with admission control the same run does NOT report success",
          r1.status != "completed", r1.status)
    check("overclaimed steps are demoted to decomposition, and recorded",
          rt1.stats.demoted > 0,
          f"{rt1.stats.demoted} steps demoted; "
          f"declared m={rt1.stats.m_declared:.2f} -> "
          f"corrected m={rt1.stats.m_measured:.2f}")
    check("the demotion is an auditable event",
          any(e["type"] == "atomicity_rejected" for e in st1.events("ro")))
    for st in (st0, st1):
        st.close()


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(sys.argv[2])
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        for scenario in (s1_nested_runs, s2_crash_resume, s3_budget,
                         s4_messages, s5_review, s6_capabilities, s7_library,
                         s8_validator, s9_insights, s10_admission):
            scenario(tmp)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*70}\n{n_pass}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
